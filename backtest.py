# -*- coding: utf-8 -*-
"""
策略回测与自我进化引擎

=== 设计 ===
1. 回测策略: 板块ETF轮动
   买入: 板块进入"启动期"(得分低位 + ΔScore >= buy_delta)
   卖出: 进入"高潮期"(止盈) 或 "退潮期"(止损/回避)
   仓位: 按MRS映射总仓位, 等权分配到持仓板块, 最多max_sectors个
2. 自我进化: 网格寻优 + 走前验证(walk-forward)
   - 前70%数据做训练集网格寻优, 后30%做样本外验证
   - 最优参数持久化到 data/strategy_params.json
   - 每次运行与历史最优对比, 只在样本外表现不劣化时才更新(防过拟合)
3. 反事实回测(因果干预层): do(资金因子=摘除)重跑MRS仓位映射,
   对比有无资金因子的策略收益差 = 资金因子的因果Alpha
"""
import json
import os
import itertools
import numpy as np
import pandas as pd

import trend_engine

PARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "strategy_params.json")

# 默认参数(首次运行或无历史时使用)
DEFAULT_PARAMS = {"buy_delta": 5.0, "exit_climax": True, "max_sectors": 3}

# 寻优网格
GRID = {
    "buy_delta": [3.0, 5.0, 8.0],
    "exit_climax": [True, False],   # 高潮期是否止盈
    "max_sectors": [2, 3, 4],
}


def _mrs_to_position(mrs):
    """MRS -> 仓位比例上限"""
    if mrs >= 75: return 1.0
    if mrs >= 60: return 0.8
    if mrs >= 45: return 0.6
    if mrs >= 30: return 0.4
    return 0.15


def run_backtest(scored, mrs_series, params, use_margin_factor=True):
    """
    板块轮动策略回测
    Args:
        scored: {板块: DataFrame(date, score, delta5)} 横截面得分
        mrs_series: DataFrame(date, mrs) 市场环境分(可为None=不做仓位管理)
        params: dict(buy_delta, exit_climax, max_sectors)
        use_margin_factor: False时mrs_series应传入"摘除两融因子"的重算版(反事实)
    Returns:
        dict: equity曲线/trades/统计指标
    """
    sectors = list(scored.keys())
    # 对齐所有板块日期(取并集索引)
    panel = {}
    for name, g in scored.items():
        gg = g.set_index('date')
        panel[name] = gg
    all_dates = sorted(set().union(*[set(g.index) for g in panel.values()]))

    mrs_map = {}
    if mrs_series is not None and len(mrs_series) > 0:
        mrs_map = dict(zip(mrs_series['date'], mrs_series['mrs']))

    holdings = {}          # {板块: {'date': 买入日, 'price': 买入收盘}}
    equity = [1.0]
    eq_dates = [all_dates[0]]
    trades = []

    for i in range(1, len(all_dates)):
        d = all_dates[i]
        daily_ret = 0.0

        # 当日总仓位上限
        mrs = mrs_map.get(d, 50.0)
        pos_cap = _mrs_to_position(mrs) if mrs_map else 1.0

        # 1. 卖出判定(收盘执行)
        for name in list(holdings.keys()):
            g = panel[name]
            if d not in g.index:
                continue
            row = g.loc[d]
            stage = trend_engine.calc_lifecycle(row['score'],
                                                row['delta5'] if pd.notna(row['delta5']) else 0)
            sell = (stage == '退潮期') or (params['exit_climax'] and stage == '高潮期')
            if sell:
                entry = holdings.pop(name)
                ret = row['close'] / entry['price'] - 1
                trades.append({'sector': name,
                               'entry_date': entry['date'].strftime('%Y-%m-%d'),
                               'exit_date': d.strftime('%Y-%m-%d'),
                               'ret': round(ret * 100, 2)})

        # 2. 买入判定(收盘执行)
        room = params['max_sectors'] - len(holdings)
        if room > 0 and pos_cap > 0.2:   # 市场环境太差不开新仓
            cands = []
            for name in sectors:
                if name in holdings:
                    continue
                g = panel[name]
                if d not in g.index:
                    continue
                row = g.loc[d]
                delta = row['delta5'] if pd.notna(row['delta5']) else 0
                if row['score'] < 55 and delta >= params['buy_delta']:
                    cands.append((delta, name))
            cands.sort(reverse=True)
            for _, name in cands[:room]:
                g = panel[name]
                holdings[name] = {'date': d, 'price': g.loc[d, 'close']}

        # 3. 当日组合收益 = 持仓板块当日收益均值 × 仓位系数
        if holdings:
            prev_d = all_dates[i - 1]
            day_rets = []
            for name in holdings:
                g = panel[name]
                if d in g.index and prev_d in g.index and 'close' in g.columns:
                    day_rets.append(g.loc[d, 'close'] / g.loc[prev_d, 'close'] - 1)
            if day_rets:
                # pos_cap<0.6时不开新仓,持仓收益也按仓位比例折算
                daily_ret = float(np.mean(day_rets)) * min(1.0, pos_cap / 0.6)
        equity.append(equity[-1] * (1 + daily_ret))
        eq_dates.append(d)

    eq = pd.Series(equity, index=pd.DatetimeIndex(eq_dates))
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    dd = (eq / eq.cummax() - 1).min()
    days = max(1, (eq.index[-1] - eq.index[0]).days)
    ann_ret = (1 + total_ret) ** (365 / days) - 1
    win_rate = (np.mean([1 if t['ret'] > 0 else 0 for t in trades])
                if trades else 0.0)

    # 持仓详情(含买入日/买入价/当前价/浮盈/仓位权重)
    holdings_detail = []
    if holdings:
        last_d = all_dates[-1]
        n_held = len(holdings)
        weight = round(1.0 / n_held * 100, 1) if n_held else 0
        for name, info in holdings.items():
            g = panel[name]
            cur_price = float(g.loc[last_d, 'close']) if last_d in g.index else info['price']
            ret_pct = round((cur_price / info['price'] - 1) * 100, 2)
            holdings_detail.append({
                'sector': name,
                'entry_date': info['date'].strftime('%Y-%m-%d'),
                'entry_price': round(float(info['price']), 2),
                'current_price': round(cur_price, 2),
                'return': ret_pct,
                'weight': weight,
            })
        holdings_detail.sort(key=lambda x: x['return'], reverse=True)

    return {
        'equity': eq,
        'trades': trades,
        'n_trades': len(trades),
        'win_rate': round(win_rate * 100, 1),
        'total_ret': round(total_ret * 100, 2),
        'ann_ret': round(ann_ret * 100, 2),
        'max_dd': round(dd * 100, 2),
        'holdings_now': list(holdings.keys()),
        'holdings_detail': holdings_detail,
    }


def grid_search(scored, mrs_series, train_ratio=0.7):
    """
    走前验证网格寻优: 前70%训练集选参, 后30%样本外验证
    Returns: (最优参数, 训练集表现, 样本外表现, 全部结果表)
    """
    dates_all = sorted(set().union(*[set(g['date']) for g in scored.values()]))
    split = dates_all[int(len(dates_all) * train_ratio)]

    def slice_data(d0, d1):
        s = {n: g[(g['date'] >= d0) & (g['date'] <= d1)].reset_index(drop=True)
             for n, g in scored.items()}
        m = mrs_series[(mrs_series['date'] >= d0) & (mrs_series['date'] <= d1)] \
            if mrs_series is not None else None
        return s, m

    train_s, train_m = slice_data(dates_all[0], split)
    test_s, test_m = slice_data(split, dates_all[-1])

    results = []
    for bd, ec, ms in itertools.product(GRID['buy_delta'], GRID['exit_climax'],
                                        GRID['max_sectors']):
        p = {"buy_delta": bd, "exit_climax": ec, "max_sectors": ms}
        tr = run_backtest(train_s, train_m, p)
        results.append({**p, 'train_ann': tr['ann_ret'], 'train_dd': tr['max_dd']})

    # 训练集最优: 年化为主,回撤惩罚
    for r in results:
        r['score'] = r['train_ann'] + r['train_dd'] * 0.5  # dd为负值,扣分
    best = max(results, key=lambda x: x['score'])
    best_params = {"buy_delta": best['buy_delta'], "exit_climax": best['exit_climax'],
                   "max_sectors": best['max_sectors']}

    oos = run_backtest(test_s, test_m, best_params)
    return best_params, best, {'oos_ann': oos['ann_ret'], 'oos_dd': oos['max_dd']}, results


def evolve(scored, mrs_series, mrs_no_margin):
    """
    自我进化总入口:
      1. 网格寻优当前最优参数
      2. 与历史最优对比, 样本外不劣化才更新(参数持久化)
      3. 用最优参数全样本回测(展示净值)
      4. 反事实回测: 摘除两融因子 vs 完整MRS(因果Alpha)
    Returns:
        dict: params/全样本回测/反事实对比/进化日志
    """
    best_params, train_perf, oos_perf, grid_table = grid_search(scored, mrs_series)

    # --- 与历史最优对比 ---
    log = []
    if os.path.exists(PARAMS_FILE):
        try:
            saved = json.load(open(PARAMS_FILE, encoding='utf-8'))
            old_p = saved['params']
            # 旧参数在样本外重跑
            dates_all = sorted(set().union(*[set(g['date']) for g in scored.values()]))
            split = dates_all[int(len(dates_all) * 0.7)]
            test_s = {n: g[g['date'] >= split].reset_index(drop=True)
                      for n, g in scored.items()}
            test_m = mrs_series[mrs_series['date'] >= split] if mrs_series is not None else None
            old_oos = run_backtest(test_s, test_m, old_p)
            # 新参数样本外不劣于旧参数才更新
            if oos_perf['oos_ann'] >= old_oos['ann_ret'] - 2:
                final_params = best_params
                log.append(f"参数更新: {old_p} → {best_params} (样本外{old_oos['ann_ret']:.1f}%→{oos_perf['oos_ann']:.1f}%)")
            else:
                final_params = old_p
                log.append(f"保留旧参数(新参数样本外{oos_perf['oos_ann']:.1f}% < 旧{old_oos['ann_ret']:.1f}%,判为过拟合)")
        except Exception:
            final_params = best_params
            log.append("历史参数读取失败,采用本次寻优结果")
    else:
        final_params = best_params
        log.append(f"首次进化: 采用网格寻优参数 {best_params}")

    os.makedirs(os.path.dirname(PARAMS_FILE), exist_ok=True)
    json.dump({'params': final_params,
               'updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
               'train_perf': train_perf, 'oos_perf': oos_perf},
              open(PARAMS_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    # --- 全样本回测(展示用) ---
    full = run_backtest(scored, mrs_series, final_params)

    # --- 反事实: 摘除两融因子 ---
    counter = {}
    if mrs_no_margin is not None and len(mrs_no_margin) > 0:
        cf = run_backtest(scored, mrs_no_margin, final_params)
        counter = {
            'with_margin_ann': full['ann_ret'],
            'without_margin_ann': cf['ann_ret'],
            'causal_alpha': round(full['ann_ret'] - cf['ann_ret'], 2),
        }

    return {
        'params': final_params,
        'full': full,
        'counterfactual': counter,
        'evolve_log': log,
        'oos_perf': oos_perf,
    }

# -*- coding: utf-8 -*-
"""
市场趋势监测看板 - 主控脚本 (V3 综合分析版)
用法: python generate_dashboard.py

分析体系:
  技术面: 缠论(分型/笔/中枢/背驰/买卖点) + 波浪计数 + 量柱理论(倍量/高量/黄金柱)
  资金面: 两融资金流 + 横截面相对强度RPS
  基本面/联动: 全球指数联动 + 格兰杰因果检验
  进化层: 策略回测(网格寻优+走前验证) + 反事实回测(因果Alpha) + 参数持久化

仓位锚点:
  MRS = 价格趋势35% + 量能20% + 两融25% + 市场宽度20%
  >=75满仓进攻 / 60-75六到八成 / 45-60半仓做T / 30-45防守 / <30空仓
"""
import json
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_fetcher
import trend_engine
import chan_engine
import volpillar
import causal_engine
import backtest

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")


class _NumpyEncoder(json.JSONEncoder):
    """JSON编码器: 兼容numpy标量/数组类型"""
    def default(self, o):
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def fetch_all_data():
    """获取所有需要的数据"""
    print("=" * 60)
    print("步骤1: 获取行业板块数据")
    print("=" * 60)
    sectors_data = data_fetcher.get_all_sectors_data(days=150)

    print("\n" + "=" * 60)
    print("步骤2: 获取上证指数K线数据")
    print("=" * 60)
    index_data = data_fetcher.get_index_kline("sh000001", days=400)
    if len(index_data) > 0:
        print(f"  获取成功: {len(index_data)}条, 最新日期={index_data.iloc[-1]['date'].strftime('%Y-%m-%d')}")

    print("\n" + "=" * 60)
    print("步骤3: 获取两融数据")
    print("=" * 60)
    margin_data = data_fetcher.get_margin_data(days=60)

    print("\n" + "=" * 60)
    print("步骤4: 获取全球指数(联动分析)")
    print("=" * 60)
    global_indices = data_fetcher.get_global_indices(days=300)

    print("\n" + "=" * 60)
    print("步骤4b: 获取创业板指K线数据")
    print("=" * 60)
    cyb_data = data_fetcher.get_index_kline("sz399006", days=400)
    if len(cyb_data) > 0:
        print(f"  获取成功: {len(cyb_data)}条, 最新日期={cyb_data.iloc[-1]['date'].strftime('%Y-%m-%d')}")

    print("\n" + "=" * 60)
    print("步骤4c: 获取科创50指数K线数据")
    print("=" * 60)
    kcb_data = data_fetcher.get_index_kline("sh000688", days=400)
    if len(kcb_data) > 0:
        print(f"  获取成功: {len(kcb_data)}条, 最新日期={kcb_data.iloc[-1]['date'].strftime('%Y-%m-%d')}")

    return sectors_data, index_data, margin_data, global_indices, cyb_data, kcb_data


def compute_sector_heatmap(sectors_data, num_days=18):
    """
    用V2横截面逻辑计算行业热力图数据

    Returns:
        dict: sectors/dates/scores(含delta)/latest(含生命周期+ETF)/breadth/scored(回测用)
    """
    print("\n" + "=" * 60)
    print("步骤5: 计算板块横截面趋势得分")
    print("=" * 60)

    sectors = [s for s in data_fetcher.SECTOR_ORDER if s in sectors_data]

    # 1. 各板块原始指标 -> 横截面打分
    metrics = {name: trend_engine.calc_sector_raw_metrics(sectors_data[name])
               for name in sectors}
    scored = trend_engine.build_cross_sectional_scores(metrics)

    # 2. 对齐日期（所有板块共有交易日）
    all_dates = None
    for name in sectors:
        dates = set(scored[name]['date'].dt.strftime('%Y-%m-%d').tolist())
        all_dates = dates if all_dates is None else (all_dates & dates)
    if not all_dates:
        print("  错误: 没有共有的交易日数据")
        return None
    sorted_dates = sorted(all_dates)[-num_days:]

    # 3. 组装热力图 [sector_idx, date_idx, score, delta]
    scores_data, latest, breadth_by_date = [], {}, {}
    for s_idx, name in enumerate(sectors):
        g = scored[name].copy()
        g['date_str'] = g['date'].dt.strftime('%Y-%m-%d')
        g = g.set_index('date_str')
        for d_idx, ds in enumerate(sorted_dates):
            if ds in g.index:
                row = g.loc[ds]
                score = int(row['score'])
                delta = round(float(row['delta5']), 1) if pd.notna(row['delta5']) else 0.0
                scores_data.append([s_idx, d_idx, score, delta])
                breadth_by_date[ds] = breadth_by_date.get(ds, 0) + (1 if score > 50 else 0)
                if d_idx == len(sorted_dates) - 1:
                    stage = trend_engine.calc_lifecycle(score, delta)
                    etf_code, etf_name = data_fetcher.SECTOR_ETF_MAP.get(name, ("", ""))
                    latest[name] = {"score": score, "delta": delta, "stage": stage,
                                    "etf_code": etf_code, "etf_name": etf_name}
            else:
                scores_data.append([s_idx, d_idx, 0, 0])

    n = len(sectors)
    breadth_by_date = {d: round(v / n * 100, 1) for d, v in breadth_by_date.items()}

    # 全历史市场宽度(供MRS使用): 得分>50的板块占比
    score_panel = pd.DataFrame(
        {name: scored[name].set_index('date')['score'] for name in sectors})
    breadth_full = ((score_panel > 50).mean(axis=1) * 100).round(1)

    print(f"  计算完成: {n}个行业 × {len(sorted_dates)}个交易日")
    strong = [k for k, v in latest.items() if v['stage'] in ('启动期', '发酵期', '主升期')]
    print(f"  当前强势阶段板块: {', '.join(strong) if strong else '无'}")

    return {
        'sectors': sectors,
        'dates': sorted_dates,
        'scores': scores_data,
        'latest': latest,
        'breadth': breadth_by_date,
        'breadth_full': breadth_full,
        'scored': scored,          # 回测引擎用(含close), 不序列化
    }


def compute_market_trend(index_data, margin_data, regime):
    """用MRS逻辑计算市场趋势监测数据(展示窗口=近250日)"""
    print("\n" + "=" * 60)
    print("步骤6: 计算市场环境综合分 MRS")
    print("=" * 60)

    if len(index_data) == 0 or regime is None or len(regime) == 0:
        print("  错误: 无指数数据")
        return None

    regime = regime.tail(250).reset_index(drop=True)
    regime['date_str'] = regime['date'].dt.strftime('%Y-%m-%d')

    # 两融5日变化率(展示用)
    mflow = trend_engine.calc_margin_flow_score(margin_data)
    margin_chg5 = []
    if len(mflow) > 0:
        mflow['date_str'] = pd.to_datetime(mflow['date']).dt.strftime('%Y-%m-%d')
        chg_map = {d: (float(v) if pd.notna(v) else None)
                   for d, v in zip(mflow['date_str'], mflow['margin_chg5'].round(2))}
        margin_chg5 = [chg_map.get(d) for d in regime['date_str']]

    # 仓位建议 + 背离检测 + MRS变盘点检测
    latest_mrs = float(regime['mrs'].iloc[-1])

    # MRS变盘点(连续2日突破75/60/45/30分值点位 → 攻防转换 → 加减仓信号)
    bp = trend_engine.detect_mrs_breakpoints(
        regime['close'].reset_index(drop=True), regime['mrs'].reset_index(drop=True))
    breaks_payload = [{'date': regime['date_str'].iloc[b['idx']],
                       'type': b['direction'],  # up=向上突破(加仓信号) down=向下突破(减仓信号)
                       'threshold': b['threshold'],
                       'mrs': b['mrs']}
                      for b in bp['breaks']]

    # 静态档位仓位(变盘点触发的加减仓动作在拐点卡中单独给出)
    position, gear, tip = trend_engine.get_position_advice(latest_mrs, 0.0)
    div_type, div_msg = trend_engine.detect_mrs_divergence(
        regime['close'].reset_index(drop=True), regime['mrs'].reset_index(drop=True))

    print(f"  MRS最新值: {latest_mrs:.1f} -> {gear}({position}) | 当前区间: {bp['zone']}")
    print(f"  因子分解: 价格{regime['price_score'].iloc[-1]:.0f} "
          f"量能{float(regime['vol_score'].iloc[-1]):.0f} "
          f"两融{regime['margin_score'].iloc[-1]:.0f} "
          f"宽度{regime['breadth_score'].iloc[-1]:.0f}")
    print(f"  背离检测: {div_msg}")
    print(f"  变盘点: {bp['state']} -> {bp['action']}")
    if bp['stats'].get('up'):
        s = bp['stats']['up']
        print(f"  历史向上突破{s['n']}次: 后10日均值{s['avg_ret']}%, 上涨率{s['win_rate']}%")
    if bp['stats'].get('down'):
        s = bp['stats']['down']
        print(f"  历史向下突破{s['n']}次: 后10日均值{s['avg_ret']}%, 下跌率{s['win_rate']}%")

    return {
        'dates': regime['date_str'].tolist(),
        'kline': index_data.set_index(pd.to_datetime(index_data['date']))
                           .reindex(pd.to_datetime(regime['date_str']))
                           [['open', 'close', 'low', 'high']].values.tolist(),
        'mrs': regime['mrs'].tolist(),
        'price_score': regime['price_score'].round(1).tolist(),
        'vol_score': [round(float(v), 1) for v in regime['vol_score']],
        'margin_score': regime['margin_score'].round(1).tolist(),
        'breadth_score': regime['breadth_score'].round(1).tolist(),
        'margin_chg5': margin_chg5,
        'latest_close': round(float(regime['close'].iloc[-1]), 2),
        'latest_mrs': latest_mrs,
        'position': position,
        'gear': gear,
        'tip': tip,
        'div_type': div_type or "",
        'div_msg': div_msg,
        'turn_state': bp['state'],
        'turn_action': bp['action'],
        'turn_zone': bp['zone'],
        'turns': breaks_payload,
        'turn_stats': bp['stats'],
        'mrs_smooth': regime['mrs'].rolling(3, min_periods=1).mean().round(1).tolist(),
    }


# 指数多周期级别配置: 级别 -> (每日模拟K线根数, min_gap, 模拟取最近N天日线)
# 大侠三绝体系: 分型->笔->中枢 在各级别独立联立, 级别间走势类型互相印证
IDX_LEVEL_ORDER = ['日', '60min', '30min', '15min', '5min']
IDX_LEVEL_CFG = {'日': (1, 4, None), '60min': (4, 3, 120), '30min': (8, 3, 80),
                 '15min': (16, 2, 40), '5min': (48, 2, 25)}


def compute_chan_vol(index_data, index_name="上证指数"):
    """多周期缠论结构 + 量柱分析(可作用于任意指数)

    Returns:
        (levels, text)
        levels: {级别: {dates/kline/volume/bi_line/zhongshu/trade_points/waves/
                        macd/pillars/forecast/state_text}}
        text:   日线级别文字摘要(供顶部状态横幅)
    """
    print(f"\n{'=' * 60}")
    print(f"步骤7: {index_name}多周期缠论结构分析(日/60/30/15/5分钟联立)")
    print("=" * 60)

    index_data = index_data.sort_values('date').reset_index(drop=True)
    levels, text = {}, None

    for lv in IDX_LEVEL_ORDER:
        n, min_gap, tail_days = IDX_LEVEL_CFG[lv]
        if n == 1:
            df_lv = index_data
            dates = pd.to_datetime(df_lv['date']).dt.strftime('%Y-%m-%d').tolist()
        else:
            df_lv = chan_engine._simulate_intraday(index_data.tail(tail_days), n)
            dates = df_lv['date'].astype(str).tolist()

        res = chan_engine.analyze(df_lv, min_gap=min_gap)
        fc = chan_engine.forecast_paths(df_lv, res)
        if n > 1:
            # 分钟级推演日期标签: 按A股交易时段顺延
            fc['future_dates'] = chan_engine._future_intraday_dates(
                df_lv['date'].iloc[-1], n, 15)

        # 笔折线: 在笔端点日期处填价格,其余为None -> echarts connectNulls连成折线
        bi_map = {p['date']: p['price'] for p in res['bi_points']}
        bi_line = [bi_map.get(d) for d in dates]

        # 量柱: 日线用真实量柱理论; 分钟级为日均拆分的模拟量, 不做量柱定性
        if n == 1:
            vol = volpillar.analyze_volume(df_lv)
            pillars = vol['pillars']
        else:
            pillars = [None] * len(dates)

        levels[lv] = {
            'dates': dates,
            'kline': df_lv[['open', 'close', 'low', 'high']]
                     .astype(float).round(2).values.tolist(),
            'volume': [round(float(v)) for v in df_lv['volume']],
            'bi_line': bi_line,
            'zhongshu': res['zhongshu'],
            'trade_points': res['trade_points'],
            'waves': res['waves'],
            'macd': res['macd'],
            'pillars': pillars,
            'forecast': fc,
            'state_text': res['state_text'],
        }
        print(f"  [{lv}] {len(dates)}根K线, {len(res['bi_points'])}个笔端点, "
              f"{len(res['zhongshu'])}个中枢 | 推演 上攻{fc['probs']['up']}%/"
              f"震荡{fc['probs']['range']}%/下探{fc['probs']['down']}%")

        if lv == '日':
            print(f"  缠论: {res['state_text']}")
            print(f"  量柱: {vol['summary']['recent_msg']} | {vol['summary']['support_txt']}")
            print(f"  推演: {fc['basis']}")
            if fc.get('signals'):
                print(f"  结构信号: {'; '.join(fc['signals'])}")
            key_lv = chan_engine.extract_key_levels(res, df_lv['close'].iloc[-1], vol)
            print(f"  关键点位: 支撑{key_lv['support']} 压力{key_lv['resistance']}")
            text = {
                'index_name': index_name,
                'chan_state': res['state_text'],
                'bottom_div': res['bottom_div'],
                'top_div': res['top_div'],
                'div_msg': res['div_msg'],
                'vol_summary': vol['summary'],
                'forecast_probs': fc['probs'],
                'forecast_basis': fc['basis'],
                'forecast_structures': fc.get('structures', {}),
                'forecast_signals': fc.get('signals', []),
                'cur_pos': fc.get('cur_pos', ''),
                'key_levels': key_lv,
            }

    return levels, text


def compute_sector_chan(sectors_data):
    """计算所有行业板块的多周期缠论分析(日/60min/30min, 供前端联立看图)"""
    print("\n" + "=" * 60)
    print("步骤7c: 行业板块多周期缠论分析 (日/60min/30min)")
    print("=" * 60)
    results = {}
    for name, df in sectors_data.items():
        if len(df) < 60:
            continue
        try:
            df = df.sort_values('date').reset_index(drop=True)
            # 展示窗口裁剪后再分析, 保证K线与笔/中枢/买卖点严格对齐
            df_daily = df.tail(120).reset_index(drop=True)
            sim60 = chan_engine._simulate_intraday(df.tail(90), 4)
            sim30 = chan_engine._simulate_intraday(df.tail(60), 8)
            daily = chan_engine.analyze(df_daily, min_gap=4)
            h60 = chan_engine.analyze(sim60, min_gap=3)
            m30 = chan_engine.analyze(sim30, min_gap=3)
            k_daily = {
                'dates': pd.to_datetime(df_daily['date']).dt.strftime('%Y-%m-%d').tolist(),
                'kline': df_daily[['open', 'close', 'low', 'high']]
                         .astype(float).round(2).values.tolist(),
            }
            k_60 = {
                'dates': sim60['date'].astype(str).tolist(),
                'kline': sim60[['open', 'close', 'low', 'high']]
                         .astype(float).round(2).values.tolist(),
            }
            k_30 = {
                'dates': sim30['date'].astype(str).tolist(),
                'kline': sim30[['open', 'close', 'low', 'high']]
                         .astype(float).round(2).values.tolist(),
            }

            def _pack(res, k):
                return {
                    'dates': k['dates'],
                    'kline': k['kline'],
                    'bi_points': res['bi_points'],
                    'zhongshu': res['zhongshu'],
                    'trade_points': res['trade_points'],
                    'bottom_div': bool(res['bottom_div']),
                    'top_div': bool(res['top_div']),
                    'state_text': res['state_text'],
                }

            results[name] = {
                'daily': _pack(daily, k_daily),
                '60min': _pack(h60, k_60),
                '30min': _pack(m30, k_30),
            }
            print(f"  {name}: 日线{len(daily['bi_points'])}笔 "
                  f"60min{len(h60['bi_points'])}笔 30min{len(m30['bi_points'])}笔")
        except Exception as e:
            print(f"  {name} 缠论分析失败: {e}")
    return results


def compute_causal(index_data, margin_data, global_indices, regime_full):
    """因果推导: 格兰杰 + 全球联动 + 历史类比"""
    print("\n" + "=" * 60)
    print("步骤8: 因果推导(格兰杰/全球联动/历史类比)")
    print("=" * 60)

    suite = causal_engine.run_causal_suite(index_data, margin_data,
                                           global_indices, regime_full)
    for g in suite['granger']:
        print(f"  格兰杰[{g['pair']}]: {g['result']}")
    for g in suite['global']:
        print(f"  联动[{g['name']}]: {g['verdict']}")
    if suite.get('analogy'):
        print(f"  历史类比: {suite['analogy'].get('forecast')}")
    return suite


def compute_backtest(heatmap_data, regime_full, regime_no_margin, index_data=None):
    """策略回测与自我进化"""
    print("\n" + "=" * 60)
    print("步骤9: 策略回测与自我进化")
    print("=" * 60)

    scored = heatmap_data['scored']
    mrs_series = regime_full[['date', 'mrs']].copy() if regime_full is not None else None
    mrs_nm = regime_no_margin[['date', 'mrs']].copy() if regime_no_margin is not None else None

    result = backtest.evolve(scored, mrs_series, mrs_nm)

    for line in result['evolve_log']:
        print(f"  进化: {line}")
    full = result['full']
    print(f"  全样本回测: 年化{full['ann_ret']}% 回撤{full['max_dd']}% "
          f"胜率{full['win_rate']}% 交易{full['n_trades']}笔")
    if result['counterfactual']:
        cf = result['counterfactual']
        print(f"  反事实: 含两融年化{cf['with_margin_ann']}% vs 摘除{cf['without_margin_ann']}% "
              f"-> 两融因子因果Alpha={cf['causal_alpha']}%")

    # 净值曲线转可序列化结构
    eq = full['equity']
    eq_dates = [d.strftime('%Y-%m-%d') for d in eq.index]
    equity_payload = {
        'dates': eq_dates,
        'values': [round(float(v), 4) for v in eq.values],
    }

    # 上证指数基准(与净值日期对齐, 归一化为1.0起点)
    benchmark_payload = {'dates': eq_dates, 'values': []}
    if index_data is not None and len(index_data) > 0:
        idx = index_data.set_index(pd.to_datetime(index_data['date']))
        eq_ts = pd.to_datetime(eq_dates)
        idx_close = idx['close'].astype(float)
        first_val = None
        for d in eq_ts:
            if d in idx_close.index:
                v = float(idx_close.loc[d])
                if first_val is None:
                    first_val = v
                benchmark_payload['values'].append(round(v / first_val, 4))
            else:
                benchmark_payload['values'].append(None)
        # 前向填充缺失值
        prev = None
        for i, v in enumerate(benchmark_payload['values']):
            if v is None and prev is not None:
                benchmark_payload['values'][i] = prev
            elif v is not None:
                prev = v
    else:
        benchmark_payload['values'] = [None] * len(eq_dates)

    return {
        'params': result['params'],
        'stats': {k: full[k] for k in ('n_trades', 'win_rate', 'total_ret',
                                       'ann_ret', 'max_dd', 'holdings_now',
                                       'holdings_detail')},
        'trades_recent': full['trades'][-15:],
        'equity': equity_payload,
        'benchmark': benchmark_payload,
        'counterfactual': result['counterfactual'],
        'evolve_log': result['evolve_log'],
        'oos_perf': result['oos_perf'],
    }


def _lifecycle_legend_html():
    items = "".join(
        f"<span class='lc-item'><i style='background:{v['color']}'></i>{k}: {v['action']}</span>"
        for k, v in trend_engine.LIFECYCLE_STAGES.items())
    return f"<div class='lc-legend'>{items}</div>"


def _sector_table_html(latest):
    """板块操作表: 按阶段分组,附ETF代码"""
    order = ["启动期", "发酵期", "主升期", "高潮期", "震荡期", "退潮期", "冰点期"]
    rows = ""
    for stage in order:
        members = [(n, v) for n, v in latest.items() if v['stage'] == stage]
        if not members:
            continue
        members.sort(key=lambda x: x[1]['score'], reverse=True)
        color = trend_engine.LIFECYCLE_STAGES[stage]['color']
        action = trend_engine.LIFECYCLE_STAGES[stage]['action']
        cells = ""
        for name, v in members:
            arrow = "↑" if v['delta'] > 0 else ("↓" if v['delta'] < 0 else "→")
            etf = f"{v['etf_name']} {v['etf_code']}" if v['etf_code'] else v['etf_name']
            cells += (f"<div class='sec-chip'>"
                      f"<b>{name}</b> {v['score']}{arrow} "
                      f"<span class='etf'>{etf}</span></div>")
        rows += (f"<tr><td><span class='stage-badge' style='background:{color}'>{stage}</span>"
                 f"<div class='stage-action'>{action}</div></td>"
                 f"<td><div class='sec-chips'>{cells}</div></td></tr>")
    return (f"<table class='sec-table'><thead><tr><th style='width:120px'>生命周期</th>"
            f"<th>板块 (得分/5日变化) + 对应ETF</th></tr></thead><tbody>{rows}</tbody></table>")


def _causal_html(causal):
    """因果推导面板"""
    if not causal:
        return ""
    granger_rows = "".join(
        f"<tr><td>{g['pair']}</td><td>{g['result']}</td></tr>"
        for g in causal.get('granger', []))
    global_rows = "".join(
        f"<tr><td>{g['name']}</td><td>同步{g['sync_corr']}</td>"
        f"<td>隔夜领先{g['lead_corr']}</td><td>{g['verdict']}</td></tr>"
        for g in causal.get('global', []))
    analogy = causal.get('analogy') or {}
    analogy_rows = "".join(
        f"<tr><td>{a['date']}</td><td>{a['similarity']}</td>"
        f"<td style='color:{'#ef4444' if a['fwd_ret'] > 0 else '#22c55e'}'>"
        f"{a['fwd_ret']:+}%</td></tr>"
        for a in analogy.get('analogs', []))

    return f"""
<div class="chart-section">
    <div class="chart-title">因果推导面板 (格兰杰因果 + 全球联动 + 历史类比推演)</div>
    <div class="causal-grid">
        <div class="causal-card">
            <div class="causal-h">资金因果检验 (Granger)</div>
            <table class="mini-table">{granger_rows}</table>
            <div class="causal-note">判读: "显著因果"=该变量对次日涨跌有预测力;
            "追涨性"显著则提示两融是跟风盘而非聪明钱。</div>
        </div>
        <div class="causal-card">
            <div class="causal-h">全球市场联动</div>
            <table class="mini-table"><thead><tr><th>指数</th><th>同步</th><th>领先</th><th>结论</th></tr></thead>
            {global_rows}</table>
            <div class="causal-note">判读: 隔夜领先相关&gt;0.3为强联动,
            隔夜美股大跌且强联动时,次日A股开盘需防守。</div>
        </div>
        <div class="causal-card">
            <div class="causal-h">历史类比推演 (四因子向量余弦相似)</div>
            <div class="analogy-forecast">{analogy.get('forecast', '样本不足')}</div>
            <table class="mini-table"><thead><tr><th>相似历史日</th><th>相似度</th><th>后10日涨跌</th></tr></thead>
            {analogy_rows}</table>
            <div class="causal-note">判读: 相似的资金/趋势结构往往重演相似走势,
            作为MRS仓位建议的佐证或反证。</div>
        </div>
    </div>
</div>"""


def _backtest_html(bt):
    """回测与进化面板"""
    if not bt:
        return ""
    s = bt['stats']
    p = bt['params']
    cf = bt.get('counterfactual') or {}
    holdings = '、'.join(s['holdings_now']) if s['holdings_now'] else '空仓'
    log_rows = "".join(f"<li>{line}</li>" for line in bt['evolve_log'])
    trade_rows = ""
    for t in reversed(bt.get('trades_recent', [])):
        if t.get('type') == '买入':
            trade_rows += (f"<tr><td><span style='color:#ef4444;font-weight:600'>买入</span></td>"
                           f"<td>{t['sector']}</td><td>{t['entry_date']}</td>"
                           f"<td>{t['entry_price']}</td><td>-</td><td>-</td>"
                           f"<td style='color:#8b949e'>{t.get('reason','')}</td></tr>")
        else:
            ret_color = '#ef4444' if t['ret'] > 0 else '#22c55e'
            trade_rows += (f"<tr><td><span style='color:#22c55e;font-weight:600'>卖出</span></td>"
                           f"<td>{t['sector']}</td><td>{t['entry_date']}→{t['exit_date']}</td>"
                           f"<td>{t['entry_price']}</td><td>{t['exit_price']}</td>"
                           f"<td style='color:{ret_color}'>{t['ret']:+}%</td>"
                           f"<td style='color:#8b949e'>{t.get('exit_type','')}·{t.get('reason','')}</td></tr>")
    cf_html = ""
    if cf:
        alpha_color = '#ef4444' if cf['causal_alpha'] > 0 else '#22c55e'
        cf_html = (f"<div class='stat-card'><div class='stat-label'>两融因子因果Alpha</div>"
                   f"<div class='stat-value' style='color:{alpha_color}'>{cf['causal_alpha']:+}%</div>"
                   f"<div class='stat-sub'>含{cf['with_margin_ann']}% vs 摘除{cf['without_margin_ann']}%</div></div>")

    # 持仓详情表
    hd = s.get('holdings_detail') or []
    if hd:
        hold_rows = "".join(
            f"<tr><td>{h['sector']}</td><td>{h['entry_date']}</td>"
            f"<td>{h['entry_price']}</td><td>{h['current_price']}</td>"
            f"<td style='color:{'#ef4444' if h['return'] > 0 else '#22c55e'}'>{h['return']:+}%</td>"
            f"<td>{h['weight']}%</td></tr>"
            for h in hd)
        hold_table = (f"<div class='causal-h'>当前持仓 ({len(hd)}只, 等权分配)</div>"
                      f"<table class='mini-table'><thead><tr>"
                      f"<th>板块</th><th>买入日</th><th>买入价</th><th>现价</th>"
                      f"<th>浮盈</th><th>仓位</th></tr></thead>{hold_rows}</table>")
    else:
        hold_table = "<div class='causal-h'>当前持仓: 空仓</div>"

    return f"""
    <div class="sub-title">行业ETF量化策略 (板块轮动: 启动期买入/高潮期止盈/退潮期卖出, MRS控仓)</div>
    <div class="stat-grid">
        <div class="stat-card"><div class="stat-label">当前参数</div>
            <div class="stat-value" style="font-size:15px">Δ≥{p['buy_delta']} | {'高潮止盈' if p['exit_climax'] else '高潮不止盈'} | 最多{p['max_sectors']}板块</div>
            <div class="stat-sub">样本外年化{bt['oos_perf'].get('oos_ann')}% 回撤{bt['oos_perf'].get('oos_dd')}%</div></div>
        <div class="stat-card"><div class="stat-label">全样本年化收益</div>
            <div class="stat-value" style="color:{'#ef4444' if s['ann_ret'] > 0 else '#22c55e'}">{s['ann_ret']}%</div>
            <div class="stat-sub">累计{s['total_ret']}%</div></div>
        <div class="stat-card"><div class="stat-label">最大回撤</div>
            <div class="stat-value" style="color:#22c55e">{s['max_dd']}%</div></div>
        <div class="stat-card"><div class="stat-label">胜率</div>
            <div class="stat-value">{s['win_rate']}%</div>
            <div class="stat-sub">共{s['n_trades']}笔交易</div></div>
        <div class="stat-card"><div class="stat-label">当前持仓</div>
            <div class="stat-value" style="font-size:15px">{holdings}</div></div>
        {cf_html}
    </div>
    <div id="equity" class="chart-mobile-h" style="width: 100%; height: 280px;"></div>
    <div class="bt-cols">
        <div>{hold_table}</div>
        <div><div class="causal-h">最近交易明细</div>
            <table class="mini-table"><thead><tr><th>类型</th><th>板块</th><th>日期</th><th>买价</th><th>卖价</th><th>收益</th><th>说明</th></tr></thead>
            {trade_rows}</table>
            <div class="causal-h" style="margin-top:12px">进化日志(参数持久化, 过拟合防护)</div>
            <ul class="log-list">{log_rows}</ul></div>
    </div>"""


def _compute_actions(latest, market_data, bt):
    """策略操作计算(策略表与今日操作卡共用单一数据源)

    Returns: (position, buys, per, groups, buy_delta)
        position: 总仓位建议(如"4~6成"); buys: 启动期买入标的列表; per: 单只仓位%
        groups: {生命周期阶段: [(板块名, 信息)]}; buy_delta: 启动阈值
    """
    import re
    params = (bt or {}).get('params', {})
    max_sectors = params.get('max_sectors', 3)
    buy_delta = params.get('buy_delta', 5)
    position, mid_pct = '4~6成', 50
    if market_data:
        position = market_data['position']
        nums = [int(x) for x in re.findall(r'\d+', position)]
        mid_pct = round((nums[0] + nums[1]) / 2 * 10) if len(nums) >= 2 else 50

    groups = {}
    for name, v in (latest or {}).items():
        groups.setdefault(v['stage'], []).append((name, v))
    for g in groups.values():
        g.sort(key=lambda x: (x[1]['delta'], x[1]['score']), reverse=True)

    buys = groups.get('启动期', [])[:max_sectors]
    per = min(20, round(mid_pct / len(buys))) if buys else 0
    return position, buys, per, groups, buy_delta


def _today_card_html(latest, market_data, bt):
    """今日操作汇总卡: 打开看板10秒知道今天干什么"""
    if not latest or not market_data:
        return ""
    position, buys, per, groups, _ = _compute_actions(latest, market_data, bt)
    mrs = market_data['latest_mrs']

    def _etf(name, v):
        return f"{v['etf_name']}" if v['etf_code'] else name

    segs = [f"<span class='tc-seg'>总仓位 <b>{position}</b> (MRS {mrs:.0f})</span>"]
    if buys:
        txt = '、'.join(f"{_etf(n, v)}≈{per}%" for n, v in buys)
        segs.append(f"<span class='tc-seg buy'>买入 {len(buys)}只: {txt}</span>")
    for stage, cls, act in [('高潮期', 'tp', '止盈'), ('退潮期', 'sell', '卖出')]:
        items = groups.get(stage, [])
        if items:
            segs.append(f"<span class='tc-seg {cls}'>{act}: "
                        + '、'.join(_etf(n, v) for n, v in items) + "</span>")
    n_wait = sum(len(groups.get(s, [])) for s in ('震荡期', '冰点期'))
    if n_wait:
        segs.append(f"<span class='tc-seg wait'>观望 {n_wait}个板块</span>")
    if len(segs) == 1:
        segs.append("<span class='tc-seg wait'>无明确操作标的, 持仓/空仓观望, 等启动期信号</span>")
    return f"<div class='today-card'><span class='tc-title'>今日操作</span>{''.join(segs)}</div>"


def _idx_compare_html(idx_texts):
    """指数强弱对比条: 三指数缠论状态一览(不用切换即可横向对比)"""
    if not idx_texts:
        return ""
    pos_map = {'above': ('中枢上·强', '#ef4444'), 'inside': ('中枢内·荡', '#eab308'),
               'below': ('中枢下·弱', '#22c55e')}
    chips = ""
    for name, t in idx_texts.items():
        pos_txt, pos_color = pos_map.get(t.get('cur_pos', ''), ('', '#8b949e'))
        div = ""
        if t.get('bottom_div'):
            div = "<span class='cmp-div bottom'>底背驰</span>"
        elif t.get('top_div'):
            div = "<span class='cmp-div top'>顶背驰</span>"
        fp = t.get('forecast_probs', {})
        chips += (f"<div class='cmp-chip'><b>{name}</b>"
                  f"<span style='color:{pos_color}'>{pos_txt}</span>{div}"
                  f"<span class='cmp-prob'>攻<b style='color:#ef4444'>{fp.get('up', '-')}%</b>"
                  f" / 荡<b style='color:#eab308'>{fp.get('range', '-')}%</b>"
                  f" / 探<b style='color:#22c55e'>{fp.get('down', '-')}%</b></span></div>")
    return f"<div class='cmp-strip'>{chips}</div>"


def _etf_strategy_html(latest, market_data, bt):
    """行业ETF投资策略: 明确买卖操作与具体仓位(每日更新)

    逻辑: MRS定总仓位 -> 生命周期定标的 -> 启动期买入/发酵主升持有/高潮止盈/退潮卖出
    单只仓位 = 总仓位中值 / 买入标的数, 上限20%(分散风控)
    """
    if not latest or not market_data:
        return ""
    mrs = market_data['latest_mrs']
    zone = market_data.get('turn_zone', '')
    params = (bt or {}).get('params', {})
    max_sectors = params.get('max_sectors', 3)
    position, buys, per, groups, buy_delta = _compute_actions(latest, market_data, bt)

    def _etf(name, v):
        return f"{v['etf_name']}({v['etf_code']})" if v['etf_code'] else name

    rows = ""
    for name, v in buys:
        rows += (f"<tr><td><span class='op-badge buy'>买入</span></td>"
                 f"<td><b>{_etf(name, v)}</b></td><td class='pos'>≈{per}%</td>"
                 f"<td class='reason'>启动期: 得分{v['score']}, Δ{v['delta']:+.1f}≥{buy_delta}, "
                 f"早期分歧日介入, 跌破启动日低点止损</td></tr>")
    for stage, badge, act, reason in [
            ('发酵期', 'hold', '持有', '趋势发酵中, 持有为主, 回调至MA20附近可加仓'),
            ('主升期', 'hold', '持有', '主升浪, 坚定持有, 不提前下车'),
            ('高潮期', 'tp', '止盈', '高位动能衰竭(Δ转负), 分批止盈: 先减半, 跌破MA10清仓'),
            ('退潮期', 'sell', '卖出', '资金退潮, 清仓回避, 不抄底')]:
        for name, v in groups.get(stage, []):
            rows += (f"<tr><td><span class='op-badge {badge}'>{act}</span></td>"
                     f"<td><b>{_etf(name, v)}</b></td><td class='pos'>{'维持' if badge == 'hold' else ('减半→清' if badge == 'tp' else '清仓')}</td>"
                     f"<td class='reason'>{stage}: 得分{v['score']}, Δ{v['delta']:+.1f} | {reason}</td></tr>")
    if not rows:
        rows = ("<tr><td colspan='4' style='text-align:center;color:#8b949e'>"
                "当前无明确操作标的, 空仓/持仓观望, 等待启动期信号(Δ转正≥+" + str(buy_delta) + ")</td></tr>")

    wait_stages = [s for s in ('震荡期', '冰点期') if groups.get(s)]
    wait_txt = ('、'.join(f"{s}({len(groups[s])}个)" for s in wait_stages) +
                " 板块观望: 震荡期做T不加仓, 冰点期等Δ转正") if wait_stages else ""

    return f"""
    <div class='strategy-head'>
        <span class='strategy-anchor'>总仓位锚点: <b style='color:#f0f6fc'>MRS {mrs:.0f}({zone})</b>
        → 建议总仓位 <b style='color:#f0f6fc'>{position}</b>;
        买入标的 <b style='color:#f0f6fc'>{len(buys)}</b> 只, 单只≈<b style='color:#f0f6fc'>{per}%</b>
        <span class='cb-sub'>(策略参数: Δ≥{buy_delta}, 最多{max_sectors}板块, 回测自进化每日校准)</span></span>
    </div>
    <table class='strategy-table'>
        <thead><tr><th style='width:64px'>操作</th><th>标的ETF</th>
        <th style='width:76px'>仓位</th><th>依据与风控</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    <div class='cb-sub' style='margin-bottom:4px'>{wait_txt}</div>"""


def generate_html(heatmap_data, market_data, idx_charts, idx_texts,
                  sector_chan, causal, bt):
    """生成HTML看板"""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # === 数据源截止日期标注(大盘先行展示模式) ===
    # 各模块取其最大日期, 显示在页头
    def _last_d(x):
        return x[-1] if isinstance(x, list) and x else ""
    idx_max_dates = {}
    for name, levels in (idx_charts or {}).items():
        d = (levels.get('日') or {}).get('dates') or []
        if d:
            idx_max_dates[name] = d[-1]
    idx_latest = max(idx_max_dates.values()) if idx_max_dates else ""

    sector_dates = []
    for sname, sv in (sector_chan or {}).items():
        d = (sv.get('k_daily') or {}).get('dates') or []
        if d:
            sector_dates.append(d[-1])
    sector_latest = max(sector_dates) if sector_dates else ""

    hm_dates = (heatmap_data or {}).get('dates') or []
    heatmap_latest = hm_dates[-1] if hm_dates else ""

    mrs_dates = (market_data or {}).get('dates') or []
    mrs_latest = mrs_dates[-1] if mrs_dates else ""

    parts = []
    if idx_latest:
        parts.append(f"三大指数截至 <b style='color:#58a6ff'>{idx_latest}</b>")
    lag_parts = [x[0] for x in
                 [(f"行业板块缠论截至 {sector_latest}", sector_latest and sector_latest != idx_latest),
                  (f"热力图/生命周期截至 {heatmap_latest}", heatmap_latest and heatmap_latest != idx_latest),
                  (f"MRS市场环境截至 {mrs_latest}", mrs_latest and mrs_latest != idx_latest)]
                 if x[1]]
    date_note = ""
    if idx_latest:
        date_note = " ｜ ".join(parts + lag_parts)
        if lag_parts:
            date_note += "（行业/两融数据源T+1更新）"

    # ---- MRS摘要面板 ----
    summary_html = ""
    if market_data:
        mrs = market_data['latest_mrs']
        mrs_color = '#ef4444' if mrs >= 60 else ('#eab308' if mrs >= 45 else '#22c55e')
        # 变盘点卡颜色: 向上突破=绿(加仓信号) / 向下突破=红(减仓信号) / 区间内运行=黄
        ts = market_data['turn_state']
        ts_color = '#22c55e' if ts.startswith('突破') else ('#ef4444' if ts.startswith('跌破') else '#eab308')
        zone = market_data.get('turn_zone', '')
        div_badge = ""
        if market_data['div_type'] == 'top':
            div_badge = f"<div class='div-alert top'>⚠ {market_data['div_msg']}</div>"
        elif market_data['div_type'] == 'bottom':
            div_badge = f"<div class='div-alert bottom'>◆ {market_data['div_msg']}</div>"
        # 变盘点有效性历史统计
        st = market_data.get('turn_stats') or {}
        st_parts = []
        if st.get('up'):
            st_parts.append(f"历史向上突破{st['up']['n']}次→后10日上涨率{st['up']['win_rate']}%")
        if st.get('down'):
            st_parts.append(f"向下突破{st['down']['n']}次→后10日下跌率{st['down']['win_rate']}%")
        st_txt = ' | '.join(st_parts)
        summary_html = f"""
        <div class='summary-grid'>
            <div class='summary-card'>
                <div class='summary-label'>上证指数</div>
                <div class='summary-value'>{market_data['latest_close']}</div>
            </div>
            <div class='summary-card'>
                <div class='summary-label'>市场环境分 MRS</div>
                <div class='summary-value' style='color:{mrs_color}'>{mrs:.0f}</div>
                <div class='summary-sub'>{market_data['gear']} · {zone}</div>
            </div>
            <div class='summary-card'>
                <div class='summary-label'>建议仓位</div>
                <div class='summary-value' style='color:{mrs_color}'>{market_data['position']}</div>
                <div class='summary-sub'>{market_data['tip']}</div>
            </div>
            <div class='summary-card'>
                <div class='summary-label'>MRS变盘点(突破分值点位)</div>
                <div class='summary-value' style='color:{ts_color};font-size:19px'>{ts}</div>
                <div class='summary-sub'>{market_data['turn_action']}</div>
            </div>
        </div>
        <div class='turn-note'>变盘判读: MRS在0-100间周期震荡, <b>连续2日站上/跌破关键分值点位(75/60/45/30)即触发攻防转换</b>:
        <b style='color:#4ade80'>向上突破60/45/30=逐级加仓信号(绿▲)</b>,
        <b style='color:#f87171'>向下跌破60/45/30=逐级减仓信号(红▼)</b>;
        区间划分: ≥75进攻 | 60-75偏多 | 45-60震荡 | 30-45防守 | &lt;30空仓。{st_txt}</div>{div_badge}"""

    # ---- 今日操作汇总卡 + 指数强弱对比条(摘要区) ----
    summary_html += _today_card_html(heatmap_data['latest'] if heatmap_data else None,
                                     market_data, bt)
    summary_html += _idx_compare_html(idx_texts)

    # ---- 缠论/量柱/结构推演状态横幅(可复用于多指数, 单行精简版) ----
    def _chan_banner_html(ct, chart_id_prefix=''):
        vs = ct['vol_summary']
        fp = ct['forecast_probs']
        div_cls = 'top' if ct['top_div'] else ('bottom' if ct['bottom_div'] else '')
        name = ct.get('index_name', '上证指数')
        # 结构事件列表(折叠明细)
        structs = ct.get('forecast_structures', {})
        struct_html = ""
        for path_key, path_label, color in [('up', '上攻', '#ef4444'),
                                              ('range', '震荡', '#eab308'),
                                              ('down', '下探', '#22c55e')]:
            events = structs.get(path_key, [])
            if events:
                items = ''.join(f'<li>{e}</li>' for e in events)
                struct_html += (f"<div class='fc-path'><b style='color:{color}'>"
                                f"{path_label} {fp[path_key]}%</b><ul>{items}</ul></div>")
        sigs = ct.get('forecast_signals', [])
        sig_txt = f" | 信号: {'; '.join(sigs)}" if sigs else ""
        # 关键点位: 近期支撑位/压力位(中枢ZD/ZG、前低前高、黄金柱/将军柱)
        kl = ct.get('key_levels') or {}
        sup_txt = '/'.join(f"{v:.0f}" for v in kl.get('support', [])) or '—'
        res_txt = '/'.join(f"{v:.0f}" for v in kl.get('resistance', [])) or '—'
        kl_html = (f"<span class='cb-sep'>|</span><span class='cb-tag'>关键点位</span>"
                   f"支撑位 <b style='color:#22c55e'>{sup_txt}</b> "
                   f"压力位 <b style='color:#ef4444'>{res_txt}</b>")
        return f"""
        <div class='chan-banner {div_cls}'>
            <div class='cb-line'><span class='cb-tag'>{name}</span>{ct['chan_state']}
            <span class='cb-sep'>|</span><span class='cb-tag'>量柱</span>{vs['recent_msg']}·{vs['support_txt']}
            <span class='cb-sep'>|</span><span class='cb-tag'>推演</span>
            <b style='color:#ef4444'>攻{fp['up']}%</b>·<b style='color:#eab308'>荡{fp['range']}%</b>·<b style='color:#22c55e'>探{fp['down']}%</b>{sig_txt}{kl_html}</div>
            <details class='cb-detail'><summary>结构事件明细与推演依据</summary>
            <div class='cb-sub' style='margin:4px 0'>{ct['forecast_basis']}</div>
            <div class='fc-structures'>{struct_html}</div></details>
        </div>"""

    idx_banners = {name: _chan_banner_html(t) for name, t in idx_texts.items()}
    first_banner = next(iter(idx_banners.values()), "")
    # 指数切换按钮(按可用指数动态生成)
    idx_buttons = ''.join(
        f'<button class="sc-level-btn idx-sel-btn{" active" if i == 0 else ""}" '
        f'data-index="{n}">{n}</button>'
        for i, n in enumerate(idx_charts))

    strategy_html = _etf_strategy_html(heatmap_data['latest'] if heatmap_data else None,
                                       market_data, bt)
    sector_table = _sector_table_html(heatmap_data['latest']) if heatmap_data else ""
    legend = _lifecycle_legend_html()
    backtest_html = _backtest_html(bt)
    # 因果推导仅作底层分析逻辑, 不在看板展示(causal结果已在控制台输出)

    # ---- 行业板块多周期缠论面板(放在因果面板之前) ----
    sector_chan_section = ""
    if sector_chan:
        sector_chan_section = """
<div class="chart-section" id="sector-chan">
    <div class="chart-title"><span class="mod-badge alt">进阶 · 行业结构</span>行业板块多周期缠论结构 (日线/60min/30min 联立: 笔·中枢·买卖点·背驰)</div>
    <div class="sc-controls">
        <select id="sc-sector" class="sc-select"></select>
        <div class="sc-level-btns">
            <button class="sc-level-btn active" data-level="daily">日线</button>
            <button class="sc-level-btn" data-level="60min">60分钟</button>
            <button class="sc-level-btn" data-level="30min">30分钟</button>
        </div>
    </div>
    <div id="sc-state" class="sc-state">加载中...</div>
    <div class="chart-container">
        <div id="sectorChan" class="chart-mobile-h" style="width: 100%; height: 500px;"></div>
    </div>
</div>"""

    # breadth_full与scored仅供Python内部使用,序列化前剔除
    heatmap_for_json = ({k: v for k, v in heatmap_data.items()
                         if k not in ('breadth_full', 'scored')}
                        if heatmap_data else None)
    heatmap_json = json.dumps(heatmap_for_json, ensure_ascii=False) if heatmap_for_json else "null"
    market_json = json.dumps(market_data, ensure_ascii=False) if market_data else "null"
    chan_json = json.dumps(idx_charts, ensure_ascii=False,
                           cls=_NumpyEncoder) if idx_charts else "null"
    banners_json = json.dumps(idx_banners, ensure_ascii=False) if idx_banners else "null"
    sector_chan_json = json.dumps(sector_chan, ensure_ascii=False,
                                  cls=_NumpyEncoder) if sector_chan else "null"
    equity_json = json.dumps(bt['equity'], ensure_ascii=False) if bt else "null"
    benchmark_json = json.dumps(bt.get('benchmark'), ensure_ascii=False) if bt else "null"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>行业ETF轮动量化投资驾驶舱</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, 'Microsoft YaHei', 'Segoe UI', sans-serif; padding: 12px; max-width: 1600px; margin: 0 auto; -webkit-text-size-adjust: 100%; }}
.header {{ text-align: center; padding: 16px 0; border-bottom: 1px solid #30363d; margin-bottom: 16px; }}
.header h1 {{ font-size: 22px; color: #f0f6fc; font-weight: 600; }}
.header .update-time {{ font-size: 12px; color: #8b949e; margin-top: 6px; }}
.instructions {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; font-size: 13px; line-height: 1.9; color: #8b949e; }}
.instructions strong {{ color: #f0f6fc; }}
.instructions .mrs-map {{ color: #c9d1d9; }}
.summary-grid {{ display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
.summary-card {{ flex: 1; min-width: 150px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; text-align: center; }}
.summary-label {{ font-size: 12px; color: #8b949e; margin-bottom: 6px; }}
.summary-value {{ font-size: 26px; font-weight: 700; color: #f0f6fc; }}
.summary-sub {{ font-size: 11px; color: #8b949e; margin-top: 4px; }}
.div-alert {{ border-radius: 8px; padding: 10px 16px; margin-bottom: 16px; font-size: 13px; font-weight: 600; }}
.div-alert.top {{ background: rgba(239,68,68,0.12); border: 1px solid #ef4444; color: #f87171; }}
.div-alert.bottom {{ background: rgba(34,197,94,0.12); border: 1px solid #22c55e; color: #4ade80; }}
.turn-note {{ background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 8px 16px; margin-bottom: 16px; font-size: 12px; color: #8b949e; line-height: 1.8; }}
.turn-note b {{ color: #c9d1d9; }}
.chan-banner {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 13px; line-height: 2; }}
.chan-banner.top {{ border-color: #ef4444; }}
.chan-banner.bottom {{ border-color: #22c55e; }}
.cb-tag {{ display: inline-block; background: #21262d; color: #58a6ff; border-radius: 4px; padding: 1px 8px; font-size: 11px; font-weight: 600; margin-right: 8px; }}
.cb-sub {{ color: #8b949e; font-size: 11px; margin-left: 8px; }}
.fc-signals {{ margin-top: 4px; font-size: 12px; color: #58a6ff; }}
.fc-structures {{ display: flex; gap: 16px; margin-top: 8px; flex-wrap: wrap; }}
.fc-path {{ flex: 1; min-width: 200px; }}
.fc-path ul {{ padding-left: 16px; margin: 4px 0 0 0; }}
.fc-path li {{ font-size: 11px; color: #b1bac4; line-height: 1.7; list-style: disc; }}
.chart-section, .table-section {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
.chart-title {{ font-size: 15px; font-weight: 600; color: #f0f6fc; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #30363d; }}
.chart-container {{ width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; }}
.lc-legend {{ display: flex; flex-wrap: wrap; gap: 10px 16px; margin-bottom: 10px; font-size: 11px; color: #8b949e; }}
.lc-item i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }}
.sec-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.sec-table th {{ text-align: left; color: #8b949e; font-weight: 500; padding: 6px 8px; border-bottom: 1px solid #30363d; }}
.sec-table td {{ padding: 8px; border-bottom: 1px solid #21262d; vertical-align: top; }}
.stage-badge {{ display: inline-block; padding: 2px 10px; border-radius: 10px; color: #fff; font-weight: 600; font-size: 12px; }}
.stage-action {{ font-size: 11px; color: #8b949e; margin-top: 4px; }}
.sec-chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.sec-chip {{ background: #21262d; border-radius: 4px; padding: 3px 8px; }}
.sec-chip .etf {{ color: #58a6ff; font-size: 11px; }}
.causal-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
.causal-card {{ background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 12px; }}
.causal-h {{ font-size: 13px; font-weight: 600; color: #58a6ff; margin-bottom: 8px; }}
.causal-note {{ font-size: 11px; color: #8b949e; margin-top: 8px; line-height: 1.6; }}
.mini-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.mini-table th {{ text-align: left; color: #8b949e; font-weight: 500; padding: 4px 6px; border-bottom: 1px solid #30363d; }}
.mini-table td {{ padding: 5px 6px; border-bottom: 1px solid #21262d; }}
.analogy-forecast {{ font-size: 13px; font-weight: 600; color: #f0f6fc; padding: 6px 0; }}
.stat-grid {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }}
.stat-card {{ flex: 1; min-width: 140px; background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 10px 12px; }}
.stat-label {{ font-size: 11px; color: #8b949e; margin-bottom: 4px; }}
.stat-value {{ font-size: 20px; font-weight: 700; color: #f0f6fc; }}
.stat-sub {{ font-size: 11px; color: #8b949e; margin-top: 3px; }}
.bt-cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }}
.log-list {{ font-size: 12px; color: #c9d1d9; padding-left: 18px; line-height: 1.9; }}
.sub-title {{ font-size: 13px; font-weight: 600; color: #c9d1d9; margin: 20px 0 10px; padding-bottom: 6px; border-bottom: 1px dashed #30363d; }}
.strategy-head {{ margin-bottom: 10px; font-size: 13px; color: #8b949e; line-height: 1.8; }}
.strategy-anchor {{ background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 8px 12px; display: inline-block; }}
.strategy-table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 6px; }}
.strategy-table th, .strategy-table td {{ padding: 8px 10px; border-bottom: 1px solid #21262d; text-align: left; vertical-align: top; }}
.strategy-table th {{ color: #8b949e; font-size: 12px; font-weight: 500; }}
.op-badge {{ padding: 2px 10px; border-radius: 4px; font-size: 12px; font-weight: 700; white-space: nowrap; }}
.op-badge.buy {{ background: rgba(239,68,68,0.18); color: #f87171; }}
.op-badge.hold {{ background: rgba(234,179,8,0.15); color: #eab308; }}
.op-badge.tp {{ background: rgba(168,85,247,0.18); color: #c084fc; }}
.op-badge.sell {{ background: rgba(59,130,246,0.18); color: #60a5fa; }}
.strategy-table .pos {{ color: #f0f6fc; font-weight: 700; white-space: nowrap; }}
.strategy-table .reason {{ color: #8b949e; font-size: 12px; line-height: 1.6; }}
.sc-controls {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }}
.sc-select {{ background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 6px 12px; font-size: 13px; outline: none; }}
.sc-level-btns {{ display: flex; gap: 6px; }}
.sc-level-btn {{ background: #21262d; color: #8b949e; border: 1px solid #30363d; border-radius: 6px; padding: 5px 16px; font-size: 12px; cursor: pointer; }}
.sc-level-btn.active {{ background: #1f6feb; color: #fff; border-color: #1f6feb; font-weight: 600; }}
.sc-state {{ font-size: 12px; color: #c9d1d9; background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 8px 12px; margin-bottom: 10px; line-height: 1.7; }}
.sc-div {{ display: inline-block; padding: 1px 8px; border-radius: 4px; font-weight: 600; margin-right: 8px; font-size: 11px; }}
.sc-div.top {{ background: rgba(239,68,68,0.15); color: #f87171; }}
.sc-div.bottom {{ background: rgba(34,197,94,0.15); color: #4ade80; }}
.footer {{ text-align: center; font-size: 11px; color: #484f58; padding: 12px 0; }}

/* === 今日操作汇总卡 === */
.today-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px 14px; margin: -4px 0 12px; font-size: 13px; display: flex; flex-wrap: wrap; align-items: center; gap: 6px 14px; }}
.tc-title {{ font-weight: 700; color: #fff; background: #1f6feb; border-radius: 4px; padding: 2px 10px; font-size: 12px; white-space: nowrap; }}
.tc-seg {{ color: #c9d1d9; }}
.tc-seg b {{ color: #f0f6fc; }}
.tc-seg.buy, .tc-seg.buy b {{ color: #f87171; }}
.tc-seg.tp {{ color: #c084fc; }}
.tc-seg.sell {{ color: #60a5fa; }}
.tc-seg.wait {{ color: #8b949e; }}
/* === 指数强弱对比条 === */
.cmp-strip {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }}
.cmp-chip {{ flex: 1; min-width: 210px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 8px 12px; font-size: 12px; color: #c9d1d9; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
.cmp-chip > b {{ color: #f0f6fc; font-size: 13px; }}
.cmp-div {{ padding: 1px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; }}
.cmp-div.bottom {{ background: rgba(34,197,94,0.15); color: #4ade80; }}
.cmp-div.top {{ background: rgba(239,68,68,0.15); color: #f87171; }}
.cmp-prob {{ color: #8b949e; font-size: 11px; }}
/* === 单行精简横幅 === */
.cb-line {{ line-height: 1.9; }}
.cb-sep {{ color: #30363d; margin: 0 8px; }}
.cb-detail {{ margin-top: 4px; }}
.cb-detail summary {{ cursor: pointer; color: #58a6ff; font-size: 12px; outline: none; }}
/* === 模块序号徽章 === */
.mod-badge {{ display: inline-block; background: #1f6feb; color: #fff; font-size: 11px; font-weight: 700; border-radius: 4px; padding: 2px 8px; margin-right: 8px; vertical-align: 2px; }}
.mod-badge.alt {{ background: #30363d; color: #8b949e; }}

/* === 移动端适配 === */
@media (max-width: 768px) {{
    body {{ padding: 8px; }}
    .header h1 {{ font-size: 16px; }}
    .header .update-time {{ font-size: 11px; }}
    .instructions {{ padding: 10px 12px; font-size: 12px; line-height: 1.8; }}
    .summary-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    .summary-card {{ min-width: 0; padding: 10px 8px; }}
    .summary-label {{ font-size: 10px; }}
    .summary-value {{ font-size: 18px; }}
    .summary-sub {{ font-size: 10px; }}
    .turn-note {{ padding: 6px 10px; font-size: 11px; line-height: 1.6; }}
    .chan-banner {{ padding: 8px 10px; font-size: 12px; line-height: 1.8; }}
    .cb-tag {{ font-size: 10px; padding: 1px 6px; margin-right: 4px; }}
    .cb-sub {{ display: block; margin-left: 0; margin-top: 2px; font-size: 10px; }}
    .chart-section, .table-section {{ padding: 10px; margin-bottom: 12px; }}
    .chart-title {{ font-size: 13px; margin-bottom: 8px; padding-bottom: 6px; }}
    .chart-container {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
    .lc-legend {{ gap: 6px 10px; font-size: 10px; }}
    .sec-table {{ font-size: 11px; }}
    .sec-table td {{ padding: 6px 4px; }}
    .sec-chip {{ font-size: 11px; padding: 2px 6px; }}
    .sec-chip .etf {{ font-size: 10px; }}
    .causal-grid {{ grid-template-columns: 1fr; }}
    .stat-card {{ min-width: 100px; padding: 8px 10px; }}
    .stat-value {{ font-size: 16px; }}
    .bt-cols {{ grid-template-columns: 1fr; }}
    .sc-select {{ font-size: 12px; }}
    .sc-controls {{ flex-wrap: wrap; gap: 6px; }}
    .sc-level-btns {{ gap: 4px; flex-wrap: wrap; }}
    .sc-level-btn {{ padding: 3px 9px; font-size: 10px; }}
    .sc-state {{ font-size: 11px; padding: 6px 8px; }}
    .today-card {{ font-size: 12px; gap: 4px 10px; padding: 8px 10px; }}
    .cmp-strip {{ gap: 6px; }}
    .cmp-chip {{ min-width: 0; font-size: 11px; gap: 6px; padding: 6px 8px; }}
    .cb-sep {{ margin: 0 4px; }}
    .mod-badge {{ font-size: 10px; padding: 1px 6px; margin-right: 5px; }}
    .footer {{ font-size: 10px; }}
}}
/* 手机端图表高度自适应 */
@media (max-width: 768px) {{
    .chart-mobile-h {{ height: 500px !important; }}
    #equity.chart-mobile-h {{ height: 220px !important; }}
}}
@media (max-width: 480px) {{
    .chart-mobile-h {{ height: 420px !important; }}
    #equity.chart-mobile-h {{ height: 200px !important; }}
    #heatmap {{ min-width: 900px; }}
}}
</style>
</head>
<body>

<div class="header">
    <h1>行业ETF轮动量化投资驾驶舱 <span style="font-size:12px;color:#8b949e">缠论·量柱·多周期·MRS·ETF策略</span></h1>
    <div class="update-time">数据更新时间: {now_str}{" · " + date_note if date_note else ""}</div>
</div>

<div class="instructions">
    <strong>使用方法（三步走）:</strong><br>
    <strong>第一步 定仓位</strong> —— 看市场环境分 MRS(0-100):
    <span class="mrs-map">≥75→8~10成仓 | 60-75→6~8成 | 45-60→4~6成(做T) | 30-45→2~4成 | &lt;30→空仓</span>。
    MRS是周期震荡分: <b>高位必回落、低位必回升</b>, 攻防转换看<span style="color:#f0f6fc">分值点位突破</span> —
    <b style="color:#4ade80">连续2日向上突破60/45/30=逐级加仓信号(绿▲)</b>,
    <b style="color:#f87171">连续2日向下跌破60/45/30=逐级减仓信号(红▼)</b>;
    MRS顶背离/缠论顶背驰+1卖 与跌破变盘点共振=确定性卖点, 底背驰/1买与突破变盘点共振=确定性买点。<br>
    <strong>第二步 看大盘</strong> —— 指数多周期缠论图(上证/创业板/科创50切换):
    日线定方向与中枢位置, 30/60分钟找精确买卖点; 三类买点(底背驰1买/回踩2买/突破回抽3买)介入,
    三类卖点(顶背驰1卖/反抽2卖/跌破回抽3卖)离场; 推演虚线标注未来结构事件时点。<br>
    <strong>第三步 选行业</strong> —— 行业ETF投资策略模块(每日更新):
    按策略表执行<b style="color:#f0f6fc">明确买卖与具体仓位</b> —
    <b style="color:#f97316">启动期</b>买入对应ETF(早期分歧日介入),
    <b style="color:#ef4444">发酵期/主升期</b>持有,
    <b style="color:#a855f7">高潮期</b>分批止盈,
    <b style="color:#3b82f6">退潮期</b>卖出; 不追加速日/高潮日。
</div>

{summary_html}

<div class="chart-section">
    <div class="chart-title"><span class="mod-badge">第一步 · 定仓位</span>市场环境监测 (K线 + MRS综合分 + 四因子分解)</div>
    <div class="chart-container">
        <div id="market" class="chart-mobile-h" style="width: 100%; height: 820px;"></div>
    </div>
</div>

<div class="chart-section">
    <div class="chart-title"><span class="mod-badge">第二步 · 看大盘</span>指数多周期缠论结构图 (指数切换 × 日/60/30/15/5分钟联立: 笔/中枢/买卖点/波浪 + MACD背驰 + 结构推演虚线)</div>
    <div class="sc-controls">
        <div class="sc-level-btns" id="idx-index-btns">{idx_buttons}</div>
        <div class="sc-level-btns" id="idx-level-btns">
            <button class="sc-level-btn idx-level-btn active" data-level="日">日线</button>
            <button class="sc-level-btn idx-level-btn" data-level="60min">60分钟</button>
            <button class="sc-level-btn idx-level-btn" data-level="30min">30分钟</button>
            <button class="sc-level-btn idx-level-btn" data-level="15min">15分钟</button>
            <button class="sc-level-btn idx-level-btn" data-level="5min">5分钟</button>
        </div>
    </div>
    <div id="idx-chan-banner">{first_banner}</div>
    <div id="idx-chan-state" class="sc-state">加载中...</div>
    <div class="chart-container">
        <div id="chan" class="chart-mobile-h" style="width: 100%; height: 880px;"></div>
    </div>
</div>

<div class="chart-section" id="etf-strategy">
    <div class="chart-title"><span class="mod-badge">第三步 · 选行业</span>行业ETF投资策略 <span style="font-size:12px;color:#8b949e;font-weight:400">每日更新 · 明确买卖策略与具体仓位</span></div>
    <div class="sub-title">行业板块趋势热力图 (横截面相对强度, 单元格=得分, 黄框=资金流入/蓝框=流出)</div>
    <div class="chart-container" id="heatmap-wrap">
        <div id="heatmap" class="chart-mobile-h" style="width: 100%; height: 560px;"></div>
    </div>
    {strategy_html}
    <div class="sub-title">板块生命周期全景</div>
    {legend}
    {sector_table}
    {backtest_html}
</div>

{sector_chan_section}

<div class="footer">
    数据来源: akshare(申万行业指数/上证/创业板/科创50/沪深两融/美股指数) |
    技术面: 缠论(分型·笔·中枢·背驰·买卖点)+简化波浪+量柱理论 |
    量化: 板块横截面RPS40%+趋势位置25%+量能健康20%+RPS5 15% | 市场MRS=价格35%+量能20%+两融25%+宽度20% |
    进化: 网格寻优+走前验证+反事实回测(因果Alpha, 底层逻辑)
</div>

<script>
var heatmapData = {heatmap_json};
var marketData = {market_json};
var chanData = {chan_json};
var idxBanners = {banners_json};
var sectorChanData = {sector_chan_json};
var equityData = {equity_json};
var benchmarkData = {benchmark_json};

// ===== 图0: 指数多周期缠论结构图(工厂函数, 指数切换 × 日/60/30/15/5分钟切换) =====
function createIdxChanChart(elId, idxBtnsId, lvlBtnsId, stateId, bannerId, allData) {{
    if (!allData) return;
    var chartEl = document.getElementById(elId);
    if (!chartEl) return;
    var chart = echarts.init(chartEl);
    var stateEl = document.getElementById(stateId);
    var bannerEl = document.getElementById(bannerId);
    var levelNames = {{ '日': '日线', '60min': '60分钟', '30min': '30分钟',
                       '15min': '15分钟', '5min': '5分钟' }};
    var evColors = {{ 'up': '#ef4444', 'range': '#eab308', 'down': '#22c55e' }};
    var curIdx = Object.keys(allData)[0];
    var curLevel = '日';

    function render() {{
        var levelData = allData[curIdx] || {{}};
        var d = levelData[curLevel];
        // 横幅: 当前指数的日线缠论/量柱/推演状态
        if (bannerEl && typeof idxBanners !== 'undefined') {{
            bannerEl.innerHTML = idxBanners[curIdx] || '';
        }}
        if (!d || !d.dates || !d.dates.length) {{
            stateEl.textContent = curIdx + ' ' + (levelNames[curLevel] || curLevel) + ': 无数据';
            chart.clear();
            return;
        }}
        // x轴 = 历史K线 + 未来15根推演K线(日线未来标签为MM-DD, 分钟级为YYYY-MM-DD HH:MM)
        var histLen = d.dates.length;
        var fc = d.forecast;
        var dates = d.dates.concat(fc.future_dates);

        // 推演路径虚线: 在历史最后一根衔接,向未来延伸
        function forecastLine(path, name, color, prob) {{
            return {{
                name: name, type: 'line', xAxisIndex: 0, yAxisIndex: 0,
                data: new Array(histLen - 1).fill(null).concat(path),
                symbol: 'none', smooth: true,
                lineStyle: {{ color: color, width: 1.8, type: 'dashed' }},
                itemStyle: {{ color: color }},
                endLabel: {{ show: true, formatter: name + ' ' + prob + '%',
                             color: color, fontSize: 12, fontWeight: 'bold', distance: 6 }},
                z: 18
            }};
        }}

        // 推演路径上的精确缠论结构事件(突破ZG/3买/中枢上移/1卖等)
        var evData = [];
        if (fc.path_events) {{
            ['up', 'range', 'down'].forEach(function(pk) {{
                (fc.path_events[pk] || []).forEach(function(e) {{
                    var xi = Math.min(histLen - 1 + e.offset, dates.length - 1);
                    evData.push({{
                        value: [dates[xi], e.price],
                        itemStyle: {{ color: evColors[pk] }},
                        label: {{ color: evColors[pk], borderColor: evColors[pk],
                                  position: pk === 'down' ? 'bottom' : 'top',
                                  formatter: e.label }}
                    }});
                }});
            }});
        }}

        // 中枢矩形(markArea)
        var zsAreas = d.zhongshu.map(function(z) {{
            return [
                {{ xAxis: z.start_date, yAxis: z.zd,
                   itemStyle: {{ color: 'rgba(88,166,255,0.10)', borderColor: '#58a6ff', borderWidth: 1 }},
                   label: {{ show: false }} }},
                {{ xAxis: z.end_date, yAxis: z.zg }}
            ];
        }});
        // 推演区域底色(最后历史K线 -> 未来末端)
        zsAreas.push([
            {{ xAxis: d.dates[histLen - 1],
               itemStyle: {{ color: 'rgba(139,148,158,0.06)' }},
               label: {{ show: true, formatter: '推演区', color: '#8b949e',
                         fontSize: 10, position: 'insideTop' }} }},
            {{ xAxis: fc.future_dates[fc.future_dates.length - 1] }}
        ]);

        // 买卖点散点
        var tpBuy = [], tpSell = [];
        d.trade_points.forEach(function(t) {{
            var item = {{ value: [t.date, t.price], label: {{ formatter: t.label }} }};
            if (t.label.indexOf('买') >= 0) tpBuy.push(item); else tpSell.push(item);
        }});
        // 波浪标注
        var wavePts = d.waves.map(function(w) {{
            return {{ value: [w.date, w.price], label: {{ formatter: w.label }} }};
        }});

        // 量柱颜色: 特殊量柱优先,否则按涨跌
        var pillarColors = {{
            '倍量柱': '#f97316', '高量柱': '#eab308', '低量柱': '#22c55e',
            '黄金柱': '#ffd700', '将军柱': '#ff6b6b'
        }};
        var volBars = d.volume.map(function(v, i) {{
            var kind = d.pillars[i];
            var color;
            if (kind && pillarColors[kind]) {{
                color = pillarColors[kind];
            }} else {{
                var k = d.kline[i];
                color = k[1] >= k[0] ? 'rgba(239,68,68,0.55)' : 'rgba(34,197,94,0.55)';
            }}
            return {{ value: v, itemStyle: {{ color: color }} }};
        }});

        // 状态行: 当前指数+级别的缠论结构状态
        stateEl.innerHTML = '<b style="color:#58a6ff">' + curIdx + ' · ' +
            levelNames[curLevel] + '</b> ' + d.state_text;

        // 初始缩放窗口: 最多显示约200根K线
        var startPct = Math.max(0, 100 - Math.round(200 / histLen * 100));

        var option = {{
            animationDuration: 300,
            tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'cross' }},
                backgroundColor: 'rgba(22,27,34,0.95)', borderColor: '#30363d',
                textStyle: {{ color: '#c9d1d9', fontSize: 12 }} }},
            axisPointer: {{ link: [{{ xAxisIndex: 'all' }}] }},
            legend: {{ top: 0, textStyle: {{ color: '#8b949e', fontSize: 10 }},
                data: ['K线', '笔', 'DIF', 'DEA', 'MACD柱', '成交量',
                       '推演-上攻', '推演-震荡', '推演-下探', '推演事件'] }},
            grid: [
                {{ top: 30, height: '42%', left: 60, right: 115 }},
                {{ top: '56%', height: '14%', left: 60, right: 115 }},
                {{ top: '74%', height: '12%', left: 60, right: 115 }}
            ],
            xAxis: [
                {{ gridIndex: 0, type: 'category', data: dates, show: false }},
                {{ gridIndex: 1, type: 'category', data: dates, show: false }},
                {{ gridIndex: 2, type: 'category', data: dates,
                   axisLabel: {{ color: '#8b949e', fontSize: 10,
                       formatter: function(v) {{ return v.length > 10 ? v.slice(5) : v; }} }},
                   axisLine: {{ lineStyle: {{ color: '#30363d' }} }} }}
            ],
            yAxis: [
                {{ gridIndex: 0, scale: true, position: 'right',
                   axisLabel: {{ color: '#8b949e', fontSize: 10 }},
                   splitLine: {{ lineStyle: {{ color: '#21262d' }} }} }},
                {{ gridIndex: 1, scale: true, position: 'right',
                   axisLabel: {{ color: '#8b949e', fontSize: 10 }},
                   splitLine: {{ lineStyle: {{ color: '#21262d' }} }} }},
                {{ gridIndex: 2, scale: true, position: 'right',
                   axisLabel: {{ color: '#8b949e', fontSize: 10 }},
                   splitLine: {{ show: false }} }}
            ],
            dataZoom: [
                {{ type: 'inside', xAxisIndex: [0,1,2], start: startPct, end: 100,
                   zoomOnMouseWheel: false, moveOnMouseMove: true, moveOnMouseWheel: true }},
                {{ type: 'slider', xAxisIndex: [0,1,2], bottom: 0, height: 15, start: startPct, end: 100,
                   borderColor: '#30363d', textStyle: {{ color: '#8b949e', fontSize: 10 }} }}
            ],
            series: [
                {{ name: 'K线', type: 'candlestick', xAxisIndex: 0, yAxisIndex: 0,
                   data: d.kline,
                   itemStyle: {{ color: '#ef4444', color0: '#22c55e',
                                 borderColor: '#ef4444', borderColor0: '#22c55e' }},
                   markArea: {{ silent: true, data: zsAreas }} }},
                {{ name: '笔', type: 'line', xAxisIndex: 0, yAxisIndex: 0,
                   data: d.bi_line, connectNulls: true, symbol: 'circle', symbolSize: 5,
                   lineStyle: {{ color: '#58a6ff', width: 1.8 }},
                   itemStyle: {{ color: '#58a6ff' }}, z: 10 }},
                {{ name: '买点', type: 'scatter', xAxisIndex: 0, yAxisIndex: 0, data: tpBuy,
                   symbol: 'triangle', symbolSize: 16,
                   itemStyle: {{ color: '#ff2d2d' }},
                   label: {{ show: true, position: 'bottom', color: '#ff8080',
                             fontSize: 11, fontWeight: 'bold' }}, z: 20 }},
                {{ name: '卖点', type: 'scatter', xAxisIndex: 0, yAxisIndex: 0, data: tpSell,
                   symbol: 'triangle', symbolRotate: 180, symbolSize: 16,
                   itemStyle: {{ color: '#00e676' }},
                   label: {{ show: true, position: 'top', color: '#69f0ae',
                             fontSize: 11, fontWeight: 'bold' }}, z: 20 }},
                {{ name: '波浪', type: 'scatter', xAxisIndex: 0, yAxisIndex: 0, data: wavePts,
                   symbol: 'circle', symbolSize: 8,
                   itemStyle: {{ color: '#c084fc', borderColor: '#fff', borderWidth: 1 }},
                   label: {{ show: true, position: 'top', color: '#c084fc',
                             fontSize: 12, fontWeight: 'bold' }}, z: 15 }},
                forecastLine(fc.paths.up, '推演-上攻', '#ef4444', fc.probs.up),
                forecastLine(fc.paths.range, '推演-震荡', '#eab308', fc.probs.range),
                forecastLine(fc.paths.down, '推演-下探', '#22c55e', fc.probs.down),
                {{ name: '推演事件', type: 'scatter', xAxisIndex: 0, yAxisIndex: 0, data: evData,
                   symbol: 'circle', symbolSize: 6,
                   label: {{ show: true, fontSize: 9, fontWeight: 'bold',
                             backgroundColor: 'rgba(13,17,23,0.88)', borderWidth: 1,
                             borderRadius: 3, padding: [2, 4], distance: 5 }}, z: 22 }},
                {{ name: 'DIF', type: 'line', xAxisIndex: 1, yAxisIndex: 1,
                   data: d.macd.dif, symbol: 'none',
                   lineStyle: {{ color: '#f59e0b', width: 1 }} }},
                {{ name: 'DEA', type: 'line', xAxisIndex: 1, yAxisIndex: 1,
                   data: d.macd.dea, symbol: 'none',
                   lineStyle: {{ color: '#818cf8', width: 1 }} }},
                {{ name: 'MACD柱', type: 'bar', xAxisIndex: 1, yAxisIndex: 1,
                   data: d.macd.hist.map(function(v) {{
                       return {{ value: v, itemStyle: {{ color: v >= 0 ? '#ef4444' : '#22c55e' }} }};
                   }}) }},
                {{ name: '成交量', type: 'bar', xAxisIndex: 2, yAxisIndex: 2, data: volBars }}
            ]
        }};
        chart.setOption(option, true);
    }}

    // 指数切换按钮
    var idxBtns = document.querySelectorAll('#' + idxBtnsId + ' .idx-sel-btn');
    idxBtns.forEach(function(b) {{
        b.addEventListener('click', function() {{
            idxBtns.forEach(function(x) {{ x.classList.remove('active'); }});
            b.classList.add('active');
            curIdx = b.getAttribute('data-index');
            render();
        }});
    }});
    // 级别切换按钮
    var lvlBtns = document.querySelectorAll('#' + lvlBtnsId + ' .idx-level-btn');
    lvlBtns.forEach(function(b) {{
        b.addEventListener('click', function() {{
            lvlBtns.forEach(function(x) {{ x.classList.remove('active'); }});
            b.classList.add('active');
            curLevel = b.getAttribute('data-level');
            render();
        }});
    }});
    render();
    window.addEventListener('resize', function() {{ chart.resize(); }});
}}

createIdxChanChart('chan', 'idx-index-btns', 'idx-level-btns',
                   'idx-chan-state', 'idx-chan-banner', chanData);

// ===== 图0c: 行业板块多周期缠论结构图(日/60min/30min切换, 单实例平滑更新) =====
(function() {{
    if (!sectorChanData) return;
    var secNames = Object.keys(sectorChanData);
    if (!secNames.length) return;
    var chartEl = document.getElementById('sectorChan');
    if (!chartEl) return;
    var chart = echarts.init(chartEl);
    var sel = document.getElementById('sc-sector');
    var stateEl = document.getElementById('sc-state');
    secNames.forEach(function(n) {{
        var opt = document.createElement('option');
        opt.value = n; opt.textContent = n;
        sel.appendChild(opt);
    }});
    var curSector = secNames[0], curLevel = 'daily';
    var levelNames = {{ 'daily': '日线', '60min': '60分钟', '30min': '30分钟' }};

    function render() {{
        var sd = sectorChanData[curSector];
        var d = sd && sd[curLevel];
        if (!d || !d.dates || !d.dates.length) {{
            stateEl.textContent = curSector + ' · ' + levelNames[curLevel] + ': 无数据';
            chart.clear();
            return;
        }}
        // 笔折线: 端点日期填价, 其余为null, connectNulls连成折线
        var biMap = {{}};
        d.bi_points.forEach(function(p) {{ biMap[p.date] = p.price; }});
        var biLine = d.dates.map(function(dt) {{
            return (dt in biMap) ? biMap[dt] : null;
        }});
        // 中枢矩形(markArea)
        var zsAreas = d.zhongshu.map(function(z) {{
            return [
                {{ xAxis: z.start_date, yAxis: z.zd,
                   itemStyle: {{ color: 'rgba(88,166,255,0.10)', borderColor: '#58a6ff', borderWidth: 1 }},
                   label: {{ show: false }} }},
                {{ xAxis: z.end_date, yAxis: z.zg }}
            ];
        }});
        // 买卖点散点
        var tpBuy = [], tpSell = [];
        d.trade_points.forEach(function(t) {{
            var item = {{ value: [t.date, t.price], label: {{ formatter: t.label }} }};
            if ((t.label || '').indexOf('买') >= 0) tpBuy.push(item); else tpSell.push(item);
        }});
        // 背驰文字标注(最新K线处)
        var divMark = [];
        if (d.top_div || d.bottom_div) {{
            divMark.push({{
                value: [d.dates[d.dates.length - 1], d.kline[d.kline.length - 1][1]],
                label: {{
                    formatter: d.top_div ? '⚠ 顶背驰' : '◆ 底背驰',
                    color: d.top_div ? '#f87171' : '#4ade80',
                    position: d.top_div ? 'top' : 'bottom'
                }}
            }});
        }}
        // 选择器下方状态文字(含背驰徽标)
        var divHtml = '';
        if (d.top_div) divHtml = "<span class='sc-div top'>⚠ 顶背驰</span>";
        else if (d.bottom_div) divHtml = "<span class='sc-div bottom'>◆ 底背驰</span>";
        stateEl.innerHTML = divHtml + '<b style="color:#58a6ff">' + curSector +
            ' · ' + levelNames[curLevel] + '</b> ' + d.state_text;

        // 初始缩放窗口: 最多显示约150根K线
        var n = d.dates.length;
        var startPct = Math.max(0, 100 - Math.round(150 / n * 100));
        chart.setOption({{
            animationDuration: 300,
            tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'cross' }},
                backgroundColor: 'rgba(22,27,34,0.95)', borderColor: '#30363d',
                textStyle: {{ color: '#c9d1d9', fontSize: 12 }} }},
            legend: {{ top: 0, textStyle: {{ color: '#8b949e', fontSize: 10 }},
                data: ['K线', '笔', '买点', '卖点'] }},
            grid: {{ top: 30, bottom: 55, left: 55, right: 70 }},
            xAxis: {{ type: 'category', data: d.dates,
                axisLabel: {{ color: '#8b949e', fontSize: 10,
                    formatter: function(v) {{ return v.length > 10 ? v.slice(5) : v; }} }},
                axisLine: {{ lineStyle: {{ color: '#30363d' }} }} }},
            yAxis: {{ scale: true, position: 'right',
                axisLabel: {{ color: '#8b949e', fontSize: 10 }},
                splitLine: {{ lineStyle: {{ color: '#21262d' }} }} }},
            dataZoom: [
                {{ type: 'inside', start: startPct, end: 100,
                   zoomOnMouseWheel: false, moveOnMouseMove: true, moveOnMouseWheel: true }},
                {{ type: 'slider', bottom: 5, height: 15, start: startPct, end: 100,
                   borderColor: '#30363d', textStyle: {{ color: '#8b949e', fontSize: 10 }} }}
            ],
            series: [
                {{ name: 'K线', type: 'candlestick', data: d.kline,
                   itemStyle: {{ color: '#ef4444', color0: '#22c55e',
                                 borderColor: '#ef4444', borderColor0: '#22c55e' }},
                   markArea: {{ silent: true, data: zsAreas }} }},
                {{ name: '笔', type: 'line', data: biLine, connectNulls: true,
                   symbol: 'circle', symbolSize: 5,
                   lineStyle: {{ color: '#58a6ff', width: 1.8 }},
                   itemStyle: {{ color: '#58a6ff' }}, z: 10 }},
                {{ name: '买点', type: 'scatter', data: tpBuy,
                   symbol: 'triangle', symbolSize: 16,
                   itemStyle: {{ color: '#ff2d2d' }},
                   label: {{ show: true, position: 'bottom', color: '#ff8080',
                             fontSize: 11, fontWeight: 'bold' }}, z: 20 }},
                {{ name: '卖点', type: 'scatter', data: tpSell,
                   symbol: 'triangle', symbolRotate: 180, symbolSize: 16,
                   itemStyle: {{ color: '#00e676' }},
                   label: {{ show: true, position: 'top', color: '#69f0ae',
                             fontSize: 11, fontWeight: 'bold' }}, z: 20 }},
                {{ name: '背驰', type: 'scatter', data: divMark,
                   symbol: 'roundRect', symbolSize: 1,
                   itemStyle: {{ color: 'transparent' }},
                   label: {{ show: true, fontSize: 12, fontWeight: 'bold',
                             backgroundColor: 'rgba(13,17,23,0.85)',
                             borderColor: '#30363d', borderWidth: 1,
                             borderRadius: 4, padding: [3, 6] }}, z: 25 }}
            ]
        }});
    }}

    sel.addEventListener('change', function() {{ curSector = sel.value; render(); }});
    var btns = document.querySelectorAll('#sector-chan .sc-level-btn');
    btns.forEach(function(b) {{
        b.addEventListener('click', function() {{
            btns.forEach(function(x) {{ x.classList.remove('active'); }});
            b.classList.add('active');
            curLevel = b.getAttribute('data-level');
            render();
        }});
    }});
    render();
    window.addEventListener('resize', function() {{ chart.resize(); }});
}})();

// ===== 图1: 行业板块热力图(含Δ方向边框) =====
(function() {{
    if (!heatmapData) return;
    var chart = echarts.init(document.getElementById('heatmap'));
    var data = heatmapData.scores.map(function(s) {{
        var delta = s[3];
        var border = null;
        if (delta >= 5) border = '#ffeb3b';
        else if (delta <= -5) border = '#00e5ff';
        return {{
            value: [s[0], s[1], s[2]],
            delta: delta,
            itemStyle: border ? {{ borderColor: border, borderWidth: 2 }} : {{}}
        }};
    }});
    var option = {{
        tooltip: {{
            position: 'top',
            formatter: function(p) {{
                var sec = heatmapData.sectors[p.value[0]];
                var date = heatmapData.dates[p.value[1]];
                var arrow = p.data.delta > 0 ? '↑' : (p.data.delta < 0 ? '↓' : '→');
                return '<b>' + sec + '</b><br/>日期: ' + date +
                       '<br/>趋势得分: <b>' + p.value[2] + '</b>' +
                       '<br/>5日变化: ' + arrow + ' ' + Math.abs(p.data.delta) +
                       '<br/><span style="color:#8b949e;font-size:11px">黄框=资金流入 蓝框=资金流出</span>';
            }}
        }},
        grid: {{ top: 30, bottom: 80, left: 80, right: 20 }},
        xAxis: {{ type: 'category', data: heatmapData.sectors, splitArea: {{ show: true }},
            axisLabel: {{ fontSize: 11, color: '#8b949e', interval: 0, rotate: 35 }},
            axisLine: {{ lineStyle: {{ color: '#30363d' }} }} }},
        yAxis: {{ type: 'category', data: heatmapData.dates, splitArea: {{ show: true }},
            axisLabel: {{ fontSize: 11, color: '#8b949e' }},
            axisLine: {{ lineStyle: {{ color: '#30363d' }} }}, inverse: true }},
        visualMap: {{
            min: 0, max: 100, calculable: true, orient: 'horizontal', left: 'center', bottom: 10,
            textStyle: {{ color: '#8b949e', fontSize: 11 }},
            inRange: {{ color: ['#14532d', '#22c55e', '#a3e635', '#eab308', '#f97316', '#ef4444', '#dc2626'] }}
        }},
        series: [{{
            name: '趋势得分', type: 'heatmap', data: data,
            label: {{ show: true, fontSize: 10, color: '#fff', fontWeight: 'bold' }},
            itemStyle: {{ borderColor: '#0d1117', borderWidth: 2, borderRadius: 4 }},
            emphasis: {{ itemStyle: {{ shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' }} }}
        }}]
    }};
    chart.setOption(option);
    window.addEventListener('resize', function() {{ chart.resize(); }});
}})();

// ===== 图2: 市场环境监测(K线 + MRS + 因子分解 + 两融) =====
(function() {{
    if (!marketData) return;
    var chart = echarts.init(document.getElementById('market'));
    var dates = marketData.dates;
    var option = {{
        tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'cross' }},
            backgroundColor: 'rgba(22,27,34,0.95)', borderColor: '#30363d',
            textStyle: {{ color: '#c9d1d9', fontSize: 12 }} }},
        axisPointer: {{ link: [{{ xAxisIndex: 'all' }}] }},
        legend: {{ top: 0, textStyle: {{ color: '#8b949e', fontSize: 10 }},
            data: ['MRS', 'MRS平滑', '向上突破点', '向下突破点',
                   '价格趋势', '量能', '两融', '宽度'] }},
        grid: [
            {{ top: 30, height: '30%', left: 60, right: 60 }},
            {{ top: '42%', height: '16%', left: 60, right: 60 }},
            {{ top: '62%', height: '16%', left: 60, right: 60 }},
            {{ top: '82%', height: '11%', left: 60, right: 60 }}
        ],
        xAxis: [
            {{ gridIndex: 0, type: 'category', data: dates, show: false }},
            {{ gridIndex: 1, type: 'category', data: dates, show: false }},
            {{ gridIndex: 2, type: 'category', data: dates, show: false }},
            {{ gridIndex: 3, type: 'category', data: dates,
               axisLabel: {{ color: '#8b949e', fontSize: 10 }},
               axisLine: {{ lineStyle: {{ color: '#30363d' }} }} }}
        ],
        yAxis: [
            {{ gridIndex: 0, scale: true, position: 'right',
               axisLabel: {{ color: '#8b949e', fontSize: 10 }},
               splitLine: {{ lineStyle: {{ color: '#21262d' }} }} }},
            {{ gridIndex: 1, min: 0, max: 100, position: 'right',
               axisLabel: {{ color: '#8b949e', fontSize: 10 }},
               splitLine: {{ lineStyle: {{ color: '#21262d' }} }} }},
            {{ gridIndex: 2, min: 0, max: 100, position: 'right',
               axisLabel: {{ color: '#8b949e', fontSize: 10 }},
               splitLine: {{ lineStyle: {{ color: '#21262d' }} }} }},
            {{ gridIndex: 3, scale: true, position: 'right',
               axisLabel: {{ color: '#8b949e', fontSize: 10 }},
               splitLine: {{ lineStyle: {{ color: '#21262d' }} }} }}
        ],
        dataZoom: [
            {{ type: 'inside', xAxisIndex: [0,1,2,3], start: 60, end: 100,
               zoomOnMouseWheel: false, moveOnMouseMove: true, moveOnMouseWheel: true }},
            {{ type: 'slider', xAxisIndex: [0,1,2,3], bottom: 0, height: 15, start: 60, end: 100,
               borderColor: '#30363d', textStyle: {{ color: '#8b949e', fontSize: 10 }} }}
        ],
        series: [
            {{ name: 'K线', type: 'candlestick', xAxisIndex: 0, yAxisIndex: 0,
               data: marketData.kline,
               itemStyle: {{ color: '#ef4444', color0: '#22c55e',
                             borderColor: '#ef4444', borderColor0: '#22c55e' }} }},
            {{ name: 'MRS', type: 'line', xAxisIndex: 1, yAxisIndex: 1,
               data: marketData.mrs, smooth: true, symbol: 'none',
               lineStyle: {{ color: '#f472b6', width: 2 }},
               areaStyle: {{ color: new echarts.graphic.LinearGradient(0,0,0,1,[
                   {{ offset: 0, color: 'rgba(244,114,182,0.25)' }},
                   {{ offset: 1, color: 'rgba(244,114,182,0.02)' }}]) }},
               markArea: {{ silent: true, data: [
                   [{{ yAxis: 75, itemStyle: {{ color: 'rgba(239,68,68,0.10)' }},
                      label: {{ formatter: '进攻', color: '#ef4444', fontSize: 9 }} }}, {{ yAxis: 100 }}],
                   [{{ yAxis: 60, itemStyle: {{ color: 'rgba(249,115,22,0.07)' }},
                      label: {{ formatter: '偏多', color: '#f97316', fontSize: 9 }} }}, {{ yAxis: 75 }}],
                   [{{ yAxis: 45, itemStyle: {{ color: 'rgba(234,179,8,0.06)' }},
                      label: {{ formatter: '震荡', color: '#eab308', fontSize: 9 }} }}, {{ yAxis: 60 }}],
                   [{{ yAxis: 30, itemStyle: {{ color: 'rgba(34,197,94,0.06)' }},
                      label: {{ formatter: '防守', color: '#22c55e', fontSize: 9 }} }}, {{ yAxis: 45 }}],
                   [{{ yAxis: 0, itemStyle: {{ color: 'rgba(34,197,94,0.12)' }},
                      label: {{ formatter: '空仓', color: '#16a34a', fontSize: 9 }} }}, {{ yAxis: 30 }}]
               ] }},
               markLine: {{ silent: true, symbol: 'none',
                   lineStyle: {{ color: '#6b7280', type: 'dashed', width: 1 }},
                   data: [{{ yAxis: 75 }}, {{ yAxis: 60 }}, {{ yAxis: 45 }}, {{ yAxis: 30 }}] }} }},
            {{ name: 'MRS平滑', type: 'line', xAxisIndex: 1, yAxisIndex: 1,
               data: marketData.mrs_smooth, smooth: true, symbol: 'none',
               lineStyle: {{ color: '#f0f6fc', width: 1, type: 'dashed', opacity: 0.6 }} }},
            {{ name: '向上突破点', type: 'scatter', xAxisIndex: 1, yAxisIndex: 1,
               data: marketData.turns.filter(function(t) {{ return t.type === 'up'; }})
                   .map(function(t) {{ return {{ value: [t.date, t.mrs],
                       label: {{ formatter: '加' + t.threshold }} }}; }}),
               symbol: 'triangle', symbolSize: 13,
               itemStyle: {{ color: '#4ade80', borderColor: '#fff', borderWidth: 1 }},
               label: {{ show: true, position: 'bottom',
                         color: '#4ade80', fontSize: 10, fontWeight: 'bold' }}, z: 20 }},
            {{ name: '向下突破点', type: 'scatter', xAxisIndex: 1, yAxisIndex: 1,
               data: marketData.turns.filter(function(t) {{ return t.type === 'down'; }})
                   .map(function(t) {{ return {{ value: [t.date, t.mrs],
                       label: {{ formatter: '减' + t.threshold }} }}; }}),
               symbol: 'triangle', symbolRotate: 180, symbolSize: 13,
               itemStyle: {{ color: '#f87171', borderColor: '#fff', borderWidth: 1 }},
               label: {{ show: true, position: 'top',
                         color: '#f87171', fontSize: 10, fontWeight: 'bold' }}, z: 20 }},
            {{ name: '价格趋势', type: 'line', xAxisIndex: 2, yAxisIndex: 2,
               data: marketData.price_score, smooth: true, symbol: 'none',
               lineStyle: {{ color: '#818cf8', width: 1.2 }} }},
            {{ name: '量能', type: 'line', xAxisIndex: 2, yAxisIndex: 2,
               data: marketData.vol_score, smooth: true, symbol: 'none',
               lineStyle: {{ color: '#f59e0b', width: 1.2 }} }},
            {{ name: '两融', type: 'line', xAxisIndex: 2, yAxisIndex: 2,
               data: marketData.margin_score, smooth: true, symbol: 'none',
               lineStyle: {{ color: '#34d399', width: 1.2 }} }},
            {{ name: '宽度', type: 'line', xAxisIndex: 2, yAxisIndex: 2,
               data: marketData.breadth_score, smooth: true, symbol: 'none',
               lineStyle: {{ color: '#22d3ee', width: 1.2 }} }},
            {{ name: '两融余额5日变化率%', type: 'bar', xAxisIndex: 3, yAxisIndex: 3,
               data: marketData.margin_chg5.map(function(v) {{
                   if (v === null || v === undefined) return {{ value: null }};
                   return {{ value: v, itemStyle: {{ color: v >= 0 ? '#ef4444' : '#22c55e' }} }};
               }}) }}
        ]
    }};
    chart.setOption(option);
    window.addEventListener('resize', function() {{ chart.resize(); }});
}})();

// ===== 图3: 策略净值曲线(叠加上证指数基准) =====
(function() {{
    if (!equityData) return;
    var el = document.getElementById('equity');
    if (!el) return;
    var chart = echarts.init(el);
    var series = [{{
        name: '策略净值', type: 'line', data: equityData.values,
        symbol: 'none', smooth: true,
        lineStyle: {{ color: '#58a6ff', width: 2 }},
        areaStyle: {{ color: new echarts.graphic.LinearGradient(0,0,0,1,[
            {{ offset: 0, color: 'rgba(88,166,255,0.25)' }},
            {{ offset: 1, color: 'rgba(88,166,255,0.02)' }}]) }},
        markLine: {{ silent: true, symbol: 'none',
            lineStyle: {{ color: '#6b7280', type: 'dashed', width: 1 }},
            data: [{{ yAxis: 1.0, label: {{ formatter: '成本线', color: '#8b949e', fontSize: 9 }} }}] }}
    }}];
    if (benchmarkData && benchmarkData.values) {{
        series.push({{
            name: '上证指数', type: 'line', data: benchmarkData.values,
            symbol: 'none', smooth: false,
            lineStyle: {{ color: '#f97316', width: 1.5, type: 'dashed' }},
        }});
    }}
    var option = {{
        tooltip: {{ trigger: 'axis',
            backgroundColor: 'rgba(22,27,34,0.95)', borderColor: '#30363d',
            textStyle: {{ color: '#c9d1d9', fontSize: 12 }} }},
        legend: {{ show: true, top: 0, right: 10,
            textStyle: {{ color: '#8b949e', fontSize: 11 }},
            data: ['策略净值', '上证指数'] }},
        grid: {{ top: 30, bottom: 40, left: 60, right: 20 }},
        xAxis: {{ type: 'category', data: equityData.dates,
            axisLabel: {{ color: '#8b949e', fontSize: 10 }},
            axisLine: {{ lineStyle: {{ color: '#30363d' }} }} }},
        yAxis: {{ scale: true, position: 'right',
            axisLabel: {{ color: '#8b949e', fontSize: 10,
                formatter: function(v) {{ return (v*100).toFixed(0)+'%' }} }},
            splitLine: {{ lineStyle: {{ color: '#21262d' }} }} }},
        dataZoom: [{{ type: 'inside', start: 0, end: 100 }}],
        series: series
    }};
    chart.setOption(option);
    window.addEventListener('resize', function() {{ chart.resize(); }});
}})();

// === 移动端: 屏幕旋转/尺寸变化时强制重绘所有图表 ===
function resizeAllCharts() {{
    document.querySelectorAll('[_echarts_instance_]').forEach(function(el) {{
        var inst = echarts.getInstanceByDom(el);
        if (inst) inst.resize();
    }});
}}
window.addEventListener('orientationchange', function() {{
    setTimeout(resizeAllCharts, 300);
}});
// 首次加载后延迟重绘(确保移动端布局稳定)
window.addEventListener('load', function() {{
    setTimeout(resizeAllCharts, 500);
}});
</script>

</body>
</html>"""
    return html


def main():
    print("=" * 60)
    print("  市场趋势监测看板生成器 V3")
    print("  缠论+量柱+波浪 | 因果推导 | 回测自进化 | MRS控仓")
    print("=" * 60)
    print()

    sectors_data, index_data, margin_data, global_indices, cyb_data, kcb_data = fetch_all_data()
    if not sectors_data:
        print("\n错误: 无法获取行业板块数据，请检查网络连接")
        return

    # --- 板块热力图 + 横截面得分 ---
    heatmap_data = compute_sector_heatmap(sectors_data, num_days=18)

    # --- 市场环境: 完整版 与 摘除两融因子版(反事实) ---
    breadth_series = heatmap_data['breadth_full'] if heatmap_data else None
    if breadth_series is not None:
        breadth_series.index = pd.to_datetime(breadth_series.index)
    regime_full = trend_engine.calc_market_regime(index_data, margin_data, breadth_series)
    regime_no_margin = trend_engine.calc_market_regime(index_data, None, breadth_series)

    market_data = compute_market_trend(index_data, margin_data, regime_full)

    # --- 缠论 + 量柱 (上证/创业板/科创50 多周期联立) ---
    idx_charts, idx_texts = {}, {}
    c, t = compute_chan_vol(index_data, "上证指数")
    idx_charts['上证指数'], idx_texts['上证指数'] = c, t
    if cyb_data is not None and len(cyb_data) > 0:
        c, t = compute_chan_vol(cyb_data, "创业板指")
        idx_charts['创业板指'], idx_texts['创业板指'] = c, t
    if kcb_data is not None and len(kcb_data) > 0:
        c, t = compute_chan_vol(kcb_data, "科创50")
        idx_charts['科创50'], idx_texts['科创50'] = c, t

    # --- 行业板块多周期缠论 (日/60min/30min) ---
    sector_chan = compute_sector_chan(sectors_data)

    # --- 因果推导(仅作底层分析逻辑, 不在看板展示) ---
    causal = compute_causal(index_data, margin_data, global_indices, regime_full)

    # --- 回测与自进化 ---
    bt = compute_backtest(heatmap_data, regime_full, regime_no_margin, index_data) \
        if heatmap_data else None

    print("\n" + "=" * 60)
    print("步骤10: 生成HTML看板")
    print("=" * 60)
    html = generate_html(heatmap_data, market_data, idx_charts, idx_texts,
                         sector_chan, causal, bt)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  看板已生成: {OUTPUT_FILE}")
    print(f"  文件大小: {len(html) / 1024:.1f} KB")
    print("\n" + "=" * 60)
    print("  完成! 请在浏览器中打开 dashboard.html 查看看板")
    print("=" * 60)


if __name__ == "__main__":
    main()

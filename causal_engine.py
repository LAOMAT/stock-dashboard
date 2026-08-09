# -*- coding: utf-8 -*-
"""
因果推导引擎
基于用户已建立的因果推断量化体系:
  关联层 P(Y|X)   = 格兰杰因果检验(时间先后方向性) + 全球联动相关性
  干预层 P(Y|do(X)) = 反事实回测(在backtest模块: 摘掉某因子重跑回测对比)
  反事实层         = 同上, 策略收益 vs 摘掉因子后的反事实基准

本模块负责"关联层"的因果方向判定:
  1. 格兰杰因果: 两融资金流 → 次日涨跌; 全球指数(美股) → 次日A股
  2. 全球联动: 隔夜美股收益 vs 次日上证收益 的领先-滞后相关
  3. 历史类比: 当前MRS四因子结构 vs 历史同期, 找最相似片段推演后续走势
"""
import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    HAS_SM = True
except ImportError:
    HAS_SM = False


def granger_cause(cause, effect, maxlag=5):
    """
    格兰杰因果检验: cause 是否 Granger-引起 effect
    Returns: (最优滞后阶数, 最小p值, 结论文本)
    """
    if not HAS_SM:
        return None, None, "statsmodels未安装,跳过"
    df = pd.concat([effect, cause], axis=1).dropna()
    if len(df) < maxlag * 4 + 10:
        return None, None, f"样本不足({len(df)}条)"
    try:
        res = grangercausalitytests(df, maxlag=maxlag, verbose=False)
        pvals = [(lag, res[lag][0]['ssr_ftest'][1]) for lag in res]
        best_lag, best_p = min(pvals, key=lambda x: x[1])
        if best_p < 0.01:
            verdict = f"显著因果(p={best_p:.3f},滞后{best_lag}日)"
        elif best_p < 0.05:
            verdict = f"弱因果(p={best_p:.3f},滞后{best_lag}日)"
        else:
            verdict = f"因果不显著(p={best_p:.3f})"
        return best_lag, best_p, verdict
    except Exception as e:
        return None, None, f"检验失败:{e}"


def global_linkage(index_df, global_dfs):
    """
    全球联动分析: 隔夜外盘收益 → 次日上证收益
    Args:
        index_df: 上证日线(date/close)
        global_dfs: {名称: DataFrame(date/close)}
    Returns:
        list of dict: 每个外盘的 相关系数/隔夜领先相关/结论
    """
    sh = index_df[['date', 'close']].copy()
    sh['sh_ret'] = sh['close'].pct_change()
    sh = sh.set_index('date')

    results = []
    for name, gdf in global_dfs.items():
        if gdf is None or len(gdf) < 30:
            continue
        g = gdf.copy()
        g['g_ret'] = g['close'].pct_change()
        g = g.set_index('date')
        # 外盘T日收盘 -> A股T+1日(外盘当日晚上交易,A股次日开盘反映)
        # 对齐: 外盘收益shift(-1) 对应A股次日
        joined = sh[['sh_ret']].join(g[['g_ret']], how='inner').dropna()
        if len(joined) < 30:
            continue
        # 同步相关
        sync_corr = joined['sh_ret'].corr(joined['g_ret'])
        # 领先相关: 外盘T日 vs A股T+1日
        lead = joined['sh_ret'].shift(-1)
        lead_corr = joined['g_ret'].corr(lead.dropna())
        strength = "强联动" if abs(lead_corr) > 0.3 else ("中联动" if abs(lead_corr) > 0.15 else "弱联动")
        results.append({
            'name': name,
            'sync_corr': round(sync_corr, 3),
            'lead_corr': round(lead_corr, 3),
            'verdict': f"{strength}: 隔夜{name}涨跌对次日A股解释力{abs(lead_corr)*100:.0f}%"
        })
    return results


def historical_analogy(regime_df, index_close, topn=3, window=10):
    """
    历史类比推演: 用MRS四因子(价格/量能/两融/宽度)构造市场状态向量,
    在历史中找余弦相似度最高的片段,统计其后window日的平均涨跌
    —— "相似的资金结构往往重演相似的走势"
    Args:
        regime_df: calc_market_regime输出(含四因子分)
        index_close: 指数收盘价Series(与regime同索引)
    Returns:
        dict: analogs列表 + 推演结论
    """
    feats = regime_df[['price_score', 'vol_score', 'margin_score', 'breadth_score']].values
    if len(feats) < window + 40:
        return {'analogs': [], 'forecast': "历史样本不足"}

    cur = feats[-1]
    norm_cur = np.linalg.norm(cur)
    sims = []
    # 排除最近window+5日(避免与自身重叠)
    for i in range(30, len(feats) - window - 5):
        hist_vec = feats[i]
        denom = np.linalg.norm(hist_vec) * norm_cur
        if denom == 0:
            continue
        sim = float(np.dot(cur, hist_vec) / denom)
        # 该片段后续window日收益
        fwd_ret = float(index_close.iloc[i + window] / index_close.iloc[i] - 1)
        sims.append({
            'date': regime_df['date'].iloc[i].strftime('%Y-%m-%d'),
            'similarity': round(sim, 3),
            'fwd_ret': round(fwd_ret * 100, 2),
        })
    if not sims:
        return {'analogs': [], 'forecast': "无有效类比样本"}

    sims.sort(key=lambda x: x['similarity'], reverse=True)
    top = sims[:topn]
    avg_ret = np.mean([t['fwd_ret'] for t in top])
    win_rate = np.mean([1 if t['fwd_ret'] > 0 else 0 for t in top])
    direction = "偏多" if avg_ret > 0.5 else ("偏空" if avg_ret < -0.5 else "中性")
    forecast = (f"最相似{topn}段历史后{window}日: 平均{avg_ret:+.2f}%, "
                f"上涨概率{win_rate*100:.0f}% → {direction}")
    return {'analogs': top, 'forecast': forecast}


def run_causal_suite(index_df, margin_df, global_dfs, regime_df):
    """
    因果分析总入口
    Returns:
        dict: granger结果列表/global联动列表/历史类比
    """
    idx = index_df.copy()
    idx['ret'] = idx['close'].pct_change()
    idx = idx.set_index('date')

    suite = {'granger': [], 'global': [], 'analogy': {}}

    # --- 格兰杰: 两融变化率 -> 次日收益 ---
    if margin_df is not None and len(margin_df) > 20:
        m = margin_df.set_index('date')
        m['chg'] = m['margin_balance'].pct_change()
        suite['granger'].append({
            'pair': '两融资金 → 次日涨跌',
            'result': granger_cause(m['chg'], idx['ret'])[2]
        })
        # 反向: 涨跌 -> 两融(检验是资金引领还是追涨杀跌)
        suite['granger'].append({
            'pair': '当日涨跌 → 次日两融(追涨性)',
            'result': granger_cause(idx['ret'], m['chg'])[2]
        })

    # --- 全球联动 ---
    if global_dfs:
        suite['global'] = global_linkage(index_df, global_dfs)
        # 格兰杰: 最强联动外盘 -> A股
        if suite['global']:
            best = max(suite['global'], key=lambda x: abs(x['lead_corr']))
            gname = best['name']
            g = global_dfs[gname].set_index('date')
            g_ret = g['close'].pct_change()
            suite['granger'].append({
                'pair': f'隔夜{gname} → 次日A股',
                'result': granger_cause(g_ret, idx['ret'])[2]
            })

    # --- 历史类比 ---
    if regime_df is not None and len(regime_df) > 60:
        regime_i = regime_df.reset_index(drop=True)
        close_i = regime_i['close']
        suite['analogy'] = historical_analogy(regime_i, close_i)

    return suite

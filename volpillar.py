# -*- coding: utf-8 -*-
"""
量柱理论分析引擎(黑马王子量柱理论)

=== 理论依据 ===
1. 倍量柱: 当日量 >= 前日量×1.9 —— 资金异动标志
2. 高量柱: 近N日最大量 —— 多空分歧极点,其后走势定强弱
3. 低量柱: 近N日最小量 —— 抛压衰竭标志,常现于底部
4. 黄金柱: 倍量/高量柱出现后,随后3日"价升量缩"(均价不破柱顶且量能逐日递减)
   —— 主力控盘标志,柱顶/柱底成支撑压力
5. 将军柱: 黄金柱基础上3日均价上移 —— 更强
6. 用法: 价涨量增=真涨; 价涨量缩=控盘好; 价跌量增=出货; 价跌量缩=洗盘/衰竭
"""
import numpy as np
import pandas as pd


def analyze_volume(df, lookback=20):
    """
    量柱分析
    Args:
        df: date/open/high/low/close/volume 日线
        lookback: 高量/低量柱的回看窗口
    Returns:
        dict: pillars(标记序列)/golden_pillars(黄金柱列表)/summary(状态文本)
    """
    df = df.sort_values('date').reset_index(drop=True).copy()
    vol = df['volume'].astype(float)
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    prev_vol = vol.shift(1)

    n = len(df)
    kinds = [''] * n          # 每根K线的量柱类型
    golden = []               # 黄金柱列表

    vol_max = vol.rolling(lookback, min_periods=5).max()
    vol_min = vol.rolling(lookback, min_periods=5).min()

    for i in range(1, n):
        v = vol.iloc[i]
        if pd.isna(prev_vol.iloc[i]):
            continue
        # 倍量柱
        if v >= prev_vol.iloc[i] * 1.9:
            kinds[i] = '倍量柱'
        # 高量柱(不覆盖倍量标记,倍量更具体)
        elif v >= vol_max.iloc[i] * 0.999:
            kinds[i] = '高量柱'
        # 低量柱
        elif v <= vol_min.iloc[i] * 1.001:
            kinds[i] = '低量柱'

    # 黄金柱/将军柱: 倍量或高量柱后3日价升量缩
    for i in range(1, n - 3):
        if kinds[i] not in ('倍量柱', '高量柱'):
            continue
        seg_v = vol.iloc[i + 1:i + 4].values
        seg_c = close.iloc[i + 1:i + 4].values
        base_c = close.iloc[i]
        # 价稳: 后3日收盘均值 >= 基柱收盘×0.99; 量缩: 3日均量 < 基柱量
        price_ok = seg_c.mean() >= base_c * 0.99
        vol_shrink = seg_v.mean() < vol.iloc[i] * 0.8
        if price_ok and vol_shrink:
            # 将军柱: 后3日重心明显上移
            is_general = seg_c[-1] > base_c * 1.01
            label = '将军柱' if is_general else '黄金柱'
            kinds[i] = label
            golden.append({
                'date': df['date'].iloc[i].strftime('%Y-%m-%d'),
                'top': round(float(high.iloc[i]), 2),
                'bottom': round(float(low.iloc[i]), 2),
                'label': label,
            })

    # --- 最近5日量价关系状态 ---
    recent_msg = []
    for i in range(max(1, n - 5), n):
        chg = close.iloc[i] / close.iloc[i - 1] - 1
        vchg = vol.iloc[i] / prev_vol.iloc[i] - 1 if prev_vol.iloc[i] > 0 else 0
        d = df['date'].iloc[i].strftime('%m-%d')
        if chg > 0.005 and vchg > 0.2:
            recent_msg.append(f"{d} 价涨量增(真涨)")
        elif chg > 0.005 and vchg < -0.1:
            recent_msg.append(f"{d} 价涨量缩(控盘/惜售)")
        elif chg < -0.005 and vchg > 0.2:
            recent_msg.append(f"{d} 价跌量增(出货警示)")
        elif chg < -0.005 and vchg < -0.1:
            recent_msg.append(f"{d} 价跌量缩(洗盘/衰竭)")

    # 最近的黄金柱支撑
    support_txt = ""
    if golden:
        g = golden[-1]
        pos = "上方" if close.iloc[-1] > g['top'] else ("下方" if close.iloc[-1] < g['bottom'] else "区间内")
        support_txt = f"最近{g['label']}({g['date']}): 支撑区间[{g['bottom']}-{g['top']}], 现价处于其{pos}"

    kinds_s = pd.Series(kinds, index=df.index)
    summary = {
        'bei_count': int((kinds_s == '倍量柱').sum()),
        'gao_count': int((kinds_s == '高量柱').sum()),
        'di_count': int((kinds_s == '低量柱').sum()),
        'golden_count': int(((kinds_s == '黄金柱') | (kinds_s == '将军柱')).sum()),
        'recent_msg': '; '.join(recent_msg[-3:]) if recent_msg else "近5日量价平稳",
        'support_txt': support_txt or "近期无黄金柱支撑",
    }

    return {
        'pillars': kinds,           # 每根K线的类型标记(''为普通)
        'golden': golden,           # 黄金柱列表(含支撑区间)
        'summary': summary,
    }

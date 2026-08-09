# -*- coding: utf-8 -*-
"""
趋势计算引擎
计算趋势得分(0-100)、趋势强度、量价指标等

=== 算法说明 ===

一、趋势得分 (0-100)
    综合四个因子加权计算，反映板块/指数的趋势强弱:
    - 均线趋势 (35%): 价格在MA5/10/20/60之上的比例 + 多头排列加成
    - RSI强弱  (20%): 14日RSI值，>50偏强，<50偏弱
    - 价格动量 (25%): 5日/10日/20日收益率加权归一化
    - 量价配合 (20%): 放量上涨加分，缩量下跌减分

    操作指引:
    - 0-30 (深绿): 趋势极弱，空仓观望
    - 30-50 (浅绿): 趋势偏弱，关注转折
    - 50-70 (橙黄): 趋势偏强，关注持有
    - 70-100 (深红): 趋势极强，持仓待涨/高位减仓

二、趋势强度
    衡量趋势的力度和加速度:
    - 基于价格偏离MA20的ATR倍数 × 成交量比率
    - 0: 无趋势，空仓观望
    - 100-200: 趋势启动，开始建仓
    - 500+: 趋势过热，减仓止盈

三、两融资金净买入
    融资余额的日变化值，正值=资金流入，负值=资金流出
    - 负值: 资金流出，空仓观望
    - 正值: 资金流入，结合趋势指标减仓
"""
import pandas as pd
import numpy as np


def calc_ma(series, periods):
    """计算移动平均线"""
    return series.rolling(window=periods, min_periods=1).mean()


def calc_rsi(close, period=14):
    """计算RSI指标"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # 使用EMA平滑（Wilder's方法）
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # 无数据时取中性值
    return rsi


def calc_atr(high, low, close, period=20):
    """计算ATR(真实波幅)"""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def calc_trend_score(df):
    """
    计算趋势得分(0-100)

    Args:
        df: 包含 close, high, low, volume 列的DataFrame

    Returns:
        Series: 0-100的趋势得分序列
    """
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)

    # === 因子1: 均线趋势 (35%) ===
    ma5 = calc_ma(close, 5)
    ma10 = calc_ma(close, 10)
    ma20 = calc_ma(close, 20)
    ma60 = calc_ma(close, 60)

    # 价格在均线之上得分
    ma_score = pd.Series(0.0, index=close.index)
    weights = [(ma5, 0.35), (ma10, 0.25), (ma20, 0.25), (ma60, 0.15)]
    for ma, w in weights:
        ma_score += (close > ma).astype(float) * w * 100

    # 多头排列加成
    bull_align = (ma5 > ma10) & (ma10 > ma20)
    bear_align = (ma5 < ma10) & (ma10 < ma20)
    ma_score = np.where(bull_align, ma_score * 1.15, ma_score)
    ma_score = np.where(bear_align, ma_score * 0.85, ma_score)
    ma_score = pd.Series(ma_score, index=close.index).clip(0, 100)

    # === 因子2: RSI (20%) ===
    rsi = calc_rsi(close, 14)
    rsi_score = rsi.clip(0, 100)

    # === 因子3: 动量 (25%) ===
    ret_5 = close.pct_change(5)
    ret_10 = close.pct_change(10)
    ret_20 = close.pct_change(20)

    # 加权动量，±8%映射到0-100
    raw_momentum = ret_5 * 0.5 + ret_10 * 0.3 + ret_20 * 0.2
    momentum_score = 50 + raw_momentum * 625  # 8% -> 100, -8% -> 0
    momentum_score = momentum_score.clip(0, 100)

    # === 因子4: 量价配合 (20%) ===
    vol_ma5 = calc_ma(volume, 5)
    vol_ratio = volume / vol_ma5.replace(0, np.nan)
    price_change = close.pct_change()

    # 放量上涨: 正分; 缩量下跌: 负分
    vol_price_raw = price_change * vol_ratio.fillna(1) * 100
    vol_price_score = 50 + vol_price_raw.clip(-50, 50)
    vol_price_score = vol_price_score.fillna(50).clip(0, 100)

    # === 综合得分 ===
    score = (ma_score * 0.35 +
             rsi_score * 0.20 +
             momentum_score * 0.25 +
             vol_price_score * 0.20)

    return score.clip(0, 100).round(0)


def calc_trend_strength(df):
    """
    计算趋势强度
    基于价格偏离均线的ATR倍数 × 成交量比率

    Args:
        df: 包含 close, high, low, volume 列的DataFrame

    Returns:
        Series: 趋势强度值（通常0-500+）
    """
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)

    atr = calc_atr(high, low, close, 20)
    ma20 = calc_ma(close, 20)

    # 价格偏离MA20的ATR倍数
    deviation = (close - ma20) / atr.replace(0, np.nan)

    # 成交量比率
    vol_ma20 = calc_ma(volume, 20)
    vol_ratio = volume / vol_ma20.replace(0, np.nan)

    # 趋势强度 = |偏离度| × 成交量比率 × 系数
    strength = (deviation.abs() * vol_ratio.fillna(1) * 100)

    return strength.fillna(0).round(0)


def calc_full_indicators(df):
    """
    一次性计算所有指标

    Returns:
        DataFrame: 原始数据 + trend_score, trend_strength, rsi, ma5, ma20 等列
    """
    result = df.copy()

    # 均线
    result['ma5'] = calc_ma(result['close'].astype(float), 5)
    result['ma10'] = calc_ma(result['close'].astype(float), 10)
    result['ma20'] = calc_ma(result['close'].astype(float), 20)
    result['ma60'] = calc_ma(result['close'].astype(float), 60)

    # RSI
    result['rsi'] = calc_rsi(result['close'].astype(float), 14)

    # 趋势得分和强度
    result['trend_score'] = calc_trend_score(result)
    result['trend_strength'] = calc_trend_strength(result)

    return result


def get_score_color(score):
    """根据趋势得分返回颜色（用于热力图）"""
    if score <= 20:
        return '#006400'  # 深绿
    elif score <= 35:
        return '#228B22'  # 中绿
    elif score <= 45:
        return '#7CCD7C'  # 浅绿
    elif score <= 55:
        return '#FFFF00'  # 黄色
    elif score <= 65:
        return '#FFA500'  # 橙色
    elif score <= 80:
        return '#FF6347'  # 浅红
    else:
        return '#DC143C'  # 深红


def get_score_action(score):
    """根据趋势得分返回操作建议"""
    if score <= 30:
        return "空仓观望"
    elif score <= 50:
        return "关注转折"
    elif score <= 70:
        return "持有关注"
    else:
        return "持仓待涨"


# ======================================================================
# === 重塑版底层逻辑 V2 ===
# 设计原则:
#   1. 因子低共线: 相对强度(横截面)+趋势位置+量能健康度,三个维度相互独立
#   2. 板块间可比: 动量类因子一律用横截面分位数(RPS),消除板块波动率差异
#   3. 反映"变化": 输出 ΔScore(较5日前变化),热力图同时呈现强弱与迁移方向
#   4. 生命周期定位: 同样的分数,启动期和高潮期操作相反,必须区分
# ======================================================================

# --- 板块生命周期阶段定义 ---
LIFECYCLE_STAGES = {
    "启动期": {"color": "#f97316", "action": "买入区(早期分歧日介入)"},
    "发酵期": {"color": "#ef4444", "action": "持有/回调加仓"},
    "主升期": {"color": "#dc2626", "action": "坚定持有"},
    "高潮期": {"color": "#a855f7", "action": "禁止追高,分批止盈"},
    "退潮期": {"color": "#3b82f6", "action": "卖出/回避"},
    "震荡期": {"color": "#eab308", "action": "做T降成本,不加仓"},
    "冰点期": {"color": "#22c55e", "action": "观望,等Δ转正再介入"},
}


def calc_sector_raw_metrics(df):
    """
    计算单板块原始指标(供横截面比较使用)

    Returns:
        DataFrame: 原数据 + ret_20, ret_5, trend_pos, vol_health 列
            - ret_20: 20日收益率(相对强度候选)
            - ret_5:  5日收益率(短期资金攻击方向)
            - trend_pos: 趋势位置分(0-100), 价格vs MA20/MA60 + 均线多头加成
            - vol_health: 量能健康度(0-100), 量比×方向的5日均值
    """
    result = df.copy()
    close = result['close'].astype(float)
    volume = result['volume'].astype(float)

    # 动量
    result['ret_20'] = close.pct_change(20)
    result['ret_5'] = close.pct_change(5)

    # 趋势位置: 波动率无关的0/1型指标,直接用绝对值即可板块间可比
    ma20 = calc_ma(close, 20)
    ma60 = calc_ma(close, 60)
    trend_pos = ((close > ma20).astype(float) * 60 +
                 (close > ma60).astype(float) * 40)
    # MA20在MA60之上(中期多头)额外加成,反之折扣
    trend_pos = np.where(ma20 > ma60, trend_pos * 1.1, trend_pos * 0.9)
    result['trend_pos'] = pd.Series(trend_pos, index=close.index).clip(0, 100)

    # 量能健康度: 5日均量/20日均量 结合价格5日方向
    # 放量上涨=100方向满分; 放量下跌=0; 缩量=50中性
    vol_ratio = calc_ma(volume, 5) / calc_ma(volume, 20).replace(0, np.nan)
    vol_ratio = vol_ratio.fillna(1).clip(0.6, 1.8)
    vol_level = (vol_ratio - 0.6) / 1.2 * 100          # 量比映射到0-100
    price_dir = np.sign(close.pct_change(5)).fillna(0)  # 方向: +1/-1/0
    # 方向权重: 涨=量比值, 跌=100-量比值(放量下跌最不健康), 平=50
    vol_health = np.where(price_dir > 0, vol_level,
                          np.where(price_dir < 0, 100 - vol_level, 50.0))
    result['vol_health'] = pd.Series(vol_health, index=close.index).clip(0, 100)

    return result


def build_cross_sectional_scores(metrics_dict):
    """
    横截面相对强度打分(RPS思想)
    对每一天, 把28个板块的 ret_20 / ret_5 / vol_health 做分位数排名,
    消除板块间波动率差异(计算机天然比银行波动大,绝对值打分失真)

    总分 = RPS20×40% + 趋势位置×25% + 量能健康度(横截面)×20% + RPS5×15%

    Args:
        metrics_dict: {板块名: calc_sector_raw_metrics的输出DataFrame}

    Returns:
        dict: {板块名: DataFrame(date, score, delta5, ret_20, ret_5)}
    """
    # 1. 汇总所有板块的每日指标到长表
    frames = []
    for name, mdf in metrics_dict.items():
        tmp = mdf[['date', 'close', 'ret_20', 'ret_5', 'trend_pos', 'vol_health']].copy()
        tmp['sector'] = name
        frames.append(tmp)
    panel = pd.concat(frames, ignore_index=True)

    # 2. 按日期分组做横截面分位数排名(pct=True -> 0~1)
    panel['rps20'] = panel.groupby('date')['ret_20'].rank(pct=True) * 100
    panel['rps5'] = panel.groupby('date')['ret_5'].rank(pct=True) * 100
    panel['rps_vol'] = panel.groupby('date')['vol_health'].rank(pct=True) * 100

    # 3. 加权合成
    panel['score'] = (panel['rps20'] * 0.40 +
                      panel['trend_pos'] * 0.25 +
                      panel['rps_vol'] * 0.20 +
                      panel['rps5'] * 0.15).clip(0, 100).round(0)

    # 4. 拆分回各板块并计算 ΔScore(较5个交易日前)
    out = {}
    for name, grp in panel.groupby('sector'):
        grp = grp.sort_values('date').reset_index(drop=True)
        grp['delta5'] = grp['score'] - grp['score'].shift(5)
        out[name] = grp[['date', 'close', 'score', 'delta5', 'ret_20', 'ret_5']]
    return out


def calc_lifecycle(score, delta5):
    """
    板块生命周期定位
    同样的分数在不同阶段操作相反: 60分+Δ>5是启动(买), 60分+Δ<-5是退潮(卖)

    Returns:
        str: 启动期/发酵期/主升期/高潮期/退潮期/震荡期/冰点期
    """
    d = 0 if pd.isna(delta5) else delta5
    if score >= 78 and d < 0:
        return "高潮期"      # 高位动能衰竭,最危险
    if score >= 70:
        return "主升期" if d >= 0 else "高潮期"
    if score >= 55:
        return "发酵期" if d >= 0 else "退潮期"
    if score >= 40:
        if d >= 5:
            return "启动期"  # 中低位动能转正,最佳介入窗口
        return "震荡期" if d > -5 else "退潮期"
    # score < 40
    return "启动期" if d >= 5 else "冰点期"  # 低位拐点=启动前夜


def calc_margin_flow_score(margin_df):
    """
    两融资金因子(相对化改造)
    旧逻辑用单日净买入绝对值,噪音极大;
    新逻辑用融资余额5日变化率(±1.5%映射0-100), 趋势化且与市场规模无关

    Returns:
        DataFrame: date + margin_score + margin_chg5(5日变化率%)
    """
    if margin_df is None or len(margin_df) == 0:
        return pd.DataFrame()
    m = margin_df.sort_values('date').reset_index(drop=True).copy()
    # 5日余额变化率(%)
    m['margin_chg5'] = (m['margin_balance'] / m['margin_balance'].shift(5) - 1) * 100
    # ±1.5% -> 0~100 (融资余额5日变化超过±1.5%属极端)
    m['margin_score'] = (50 + m['margin_chg5'] / 1.5 * 50).clip(0, 100)
    # 数据不足5日时给中性50
    m['margin_score'] = m['margin_score'].fillna(50)
    return m[['date', 'margin_score', 'margin_chg5']]


def calc_market_regime(index_df, margin_df, breadth_series):
    """
    市场环境综合分 MRS (0-100) —— 决定总仓位的唯一锚点

    四个低共线因子:
      价格趋势 35%: 指数 vs MA20/60/120 + MA20斜率(市场的"形")
      量能趋势 20%: 成交额5日/20日均值比 × 价格方向(市场的"气")
      两融资金 25%: 融资余额5日变化率(杠杆资金的"真金白银")
      市场宽度 20%: 得分>50的板块占比(行情的"群众基础")

    Args:
        index_df: 指数K线 (date, open, high, low, close, volume)
        margin_df: 两融数据 (date, margin_net, margin_balance)
        breadth_series: Series(date索引, 0-100宽度值) 或 None

    Returns:
        DataFrame: date + mrs + 各子因子分
    """
    df = index_df.sort_values('date').reset_index(drop=True).copy()
    close = df['close'].astype(float)

    # --- 因子1: 价格趋势 (35%) ---
    ma20, ma60, ma120 = calc_ma(close, 20), calc_ma(close, 60), calc_ma(close, 120)
    price_score = ((close > ma20).astype(float) * 40 +
                   (close > ma60).astype(float) * 30 +
                   (close > ma120).astype(float) * 30)
    # MA20斜率加成/折扣(5日变化±3%以内线性)
    slope = (ma20 / ma20.shift(5) - 1).fillna(0)
    price_score = price_score * (1 + slope.clip(-0.03, 0.03) * 5)
    df['price_score'] = pd.Series(price_score).clip(0, 100)

    # --- 因子2: 量能趋势 (20%) ---
    # 用成交额(比成交量更真实); 缺amount列时退化为volume
    vol_col = 'amount' if 'amount' in df.columns else 'volume'
    vr = (calc_ma(df[vol_col].astype(float), 5) /
          calc_ma(df[vol_col].astype(float), 20).replace(0, np.nan)).fillna(1)
    vol_score = ((vr - 0.8) / 0.45 * 100).clip(0, 100)   # 0.8~1.25 -> 0~100
    price_up = close.pct_change(5).fillna(0) > 0
    # 跌时量能分打折(放量下跌是坏量)
    df['vol_score'] = np.where(price_up, vol_score, vol_score * 0.5)

    # --- 因子3: 两融资金 (25%) ---
    mscore = calc_margin_flow_score(margin_df)
    if len(mscore) > 0:
        mscore['date'] = pd.to_datetime(mscore['date'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.merge(mscore[['date', 'margin_score']], on='date', how='left')
        df['margin_score'] = df['margin_score'].ffill().fillna(50)
    else:
        df['margin_score'] = 50.0

    # --- 因子4: 市场宽度 (20%) ---
    if breadth_series is not None and len(breadth_series) > 0:
        b = breadth_series.copy()
        df['breadth_score'] = df['date'].map(b).ffill().fillna(50)
    else:
        df['breadth_score'] = 50.0

    # --- 合成 ---
    df['mrs'] = (df['price_score'] * 0.35 +
                 df['vol_score'] * 0.20 +
                 df['margin_score'] * 0.25 +
                 df['breadth_score'] * 0.20).clip(0, 100).round(1)

    return df[['date', 'close', 'mrs', 'price_score', 'vol_score',
               'margin_score', 'breadth_score']]


def get_position_advice(mrs):
    """
    MRS -> 仓位配置映射(核心交付物)

    Returns:
        (仓位区间, 档位名称, 操作要点)
    """
    if mrs >= 75:
        return "8~10成", "进攻期", "主线板块ETF满仓轮动,持有待涨"
    elif mrs >= 60:
        return "6~8成", "偏多期", "持有主线,回调至MA20加仓"
    elif mrs >= 45:
        return "4~6成", "震荡期", "高抛低吸做T降成本,不追涨"
    elif mrs >= 30:
        return "2~4成", "防守期", "只留底仓,等MRS回升再加"
    else:
        return "0~2成", "空仓期", "空仓观望,保存本金等拐点"


def detect_mrs_turning_point(close, mrs, window=10):
    """
    MRS周期拐点检测(周期理论: MRS是0-100的震荡周期分,
    高位必然回落、低位必然回升 —— 拐点(方向转变的临界点)才是增减仓操作点,
    静态档位只反映"现在在哪",拐点反映"该动了吗")

    拐点定义(3日平滑MRS + 3日斜率):
      低位回升点(加仓窗口): 斜率由负转正 且 MRS<45
      高位回落点(减仓窗口): 斜率由正转负 且 MRS>55

    Args:
        close: 指数收盘价Series(与mrs同索引)
        mrs:   MRS序列Series
        window: 统计拐点后续收益的窗口(交易日)

    Returns:
        dict: state(当前状态)/action(操作建议)/slope(当前斜率)
              turns(历史拐点列表,供图上标注)/stats(拐点有效性统计)
    """
    m = pd.Series(mrs, dtype=float).reset_index(drop=True)
    c = pd.Series(close, dtype=float).reset_index(drop=True)
    ms = m.rolling(3, min_periods=1).mean()          # 3日平滑降噪
    slope = ms.diff(3)                                # 3日斜率

    # --- 历史拐点识别 + 后续收益统计 ---
    turns = []
    low_stats, high_stats = [], []
    for i in range(5, len(m) - window):
        s_prev, s_now = slope.iloc[i - 1], slope.iloc[i]
        if pd.isna(s_prev) or pd.isna(s_now):
            continue
        fwd = float(c.iloc[i + window] / c.iloc[i] - 1)
        if s_prev <= 0 < s_now and ms.iloc[i] < 45:
            turns.append({'idx': i, 'type': 'low_up', 'mrs': round(float(ms.iloc[i]), 1)})
            low_stats.append(fwd)
        elif s_prev >= 0 > s_now and ms.iloc[i] > 55:
            turns.append({'idx': i, 'type': 'high_down', 'mrs': round(float(ms.iloc[i]), 1)})
            high_stats.append(fwd)

    # --- 拐点有效性统计 ---
    stats = {}
    if low_stats:
        stats['low'] = {'n': len(low_stats),
                        'avg_ret': round(float(np.mean(low_stats)) * 100, 2),
                        'win_rate': round(float(np.mean([1 if r > 0 else 0
                                                       for r in low_stats])) * 100)}
    if high_stats:
        stats['high'] = {'n': len(high_stats),
                         'avg_ret': round(float(np.mean(high_stats)) * 100, 2),
                         'win_rate': round(float(np.mean([1 if r < 0 else 0
                                                        for r in high_stats])) * 100)}

    # --- 当前状态判定 ---
    cur_mrs = float(m.iloc[-1])
    cur_slope = float(slope.iloc[-1]) if not pd.isna(slope.iloc[-1]) else 0.0
    # 近3日斜率方向(避免单日噪声)
    recent_slope = float(ms.iloc[-1] - ms.iloc[-4]) if len(ms) > 4 else cur_slope

    if cur_mrs < 45 and recent_slope > 1.5:
        state, action = "低位回升拐点", "加仓窗口: MRS从低位拐头向上, 逐步加仓至4~6成"
    elif cur_mrs > 55 and recent_slope < -1.5:
        state, action = "高位回落拐点", "减仓窗口: MRS从高位拐头向下, 降至2~4成防守"
    elif cur_mrs >= 60 and recent_slope >= -1.5:
        state, action = "强势上行/高位钝化", "持有为主, 密切盯拐头, 斜率转负即减仓"
    elif cur_mrs <= 40 and recent_slope <= 1.5:
        state, action = "探底中/低位钝化", "空仓观望, 等斜率转正再加仓"
    elif recent_slope > 0:
        state, action = "中位回升", "跟随斜率方向, 维持或小幅加仓"
    else:
        state, action = "中位回落", "跟随斜率方向, 控制仓位做T"

    return {
        'state': state, 'action': action,
        'slope': round(cur_slope, 2),
        'mrs_smooth': [round(float(v), 1) for v in ms],
        'turns': turns,
        'stats': stats,
    }


def detect_mrs_divergence(close, mrs, window=20):
    """
    量价-资金背离检测(顶底预警)
    顶背离: 价格创20日新高但MRS低于20日前 -> 减仓预警
    底背离: 价格创20日新低但MRS高于20日前 -> 建仓提示

    Returns:
        (信号类型, 描述) 信号类型: top/bottom/None
    """
    if len(close) < window + 5:
        return None, "数据不足"
    c_now, c_prev = close.iloc[-1], close.iloc[-window - 1:-1].max()
    c_low_prev = close.iloc[-window - 1:-1].min()
    m_now, m_prev = mrs.iloc[-1], mrs.iloc[-window]

    if c_now >= c_prev and m_now < m_prev - 3:
        return "top", f"顶背离: 指数创{window}日新高但MRS({m_now:.0f})低于20日前({m_prev:.0f}), 减仓预警"
    if c_now <= c_low_prev and m_now > m_prev + 3:
        return "bottom", f"底背离: 指数创{window}日新低但MRS({m_now:.0f})高于20日前({m_prev:.0f}), 关注建仓"
    return None, "无背离"


if __name__ == "__main__":
    # 测试计算逻辑
    import data_fetcher

    print("=== 测试趋势计算引擎 ===\n")
    df = data_fetcher.get_sector_hist_data("银行", 120)
    if len(df) > 0:
        result = calc_full_indicators(df)
        print(f"银行板块最近5日趋势:")
        cols = ['date', 'close', 'trend_score', 'trend_strength', 'rsi']
        print(result[cols].tail(5).to_string(index=False))
        print(f"\n最新趋势得分: {result.iloc[-1]['trend_score']:.0f}")
        print(f"操作建议: {get_score_action(result.iloc[-1]['trend_score'])}")

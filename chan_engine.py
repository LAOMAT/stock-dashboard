# -*- coding: utf-8 -*-
"""
缠论结构分析引擎
实现: 包含关系处理 -> 分型 -> 笔 -> 中枢 -> MACD背驰 -> 一二三买卖点 -> 简化波浪计数

=== 理论依据(缠中说禅) ===
1. K线包含关系: 两根K线高低点互相包含时按方向合并(向上取高高,向下取低低)
2. 分型: 合并后三根K线,中间K线高低点同时最高=顶分型,同时最低=底分型
3. 笔: 相邻顶底分型交替连接,分型间至少间隔4根合并K线
4. 中枢: 连续三笔价格区间有重叠 [ZD=max(低点), ZG=min(高点)],ZG>ZD即中枢
5. 背驰: 同向两笔比较,价格新低(高)但MACD绿(红)柱面积缩小 -> 一买(一卖)先兆
6. 买卖点:
   1买=下跌末段底背驰; 2买=1买后回踩不破1买低点; 3买=突破中枢后回踩不进中枢(ZG之上)
   1卖/2卖/3卖镜像
7. 波浪计数(简化艾略特): 在笔序列上识别五浪推进+ABC调整,
   校验规则: 2浪不破1浪起点; 3浪非最短; 4浪不进1浪区间
"""
import numpy as np
import pandas as pd


# ---------------- 基础: MACD(背驰判断用) ----------------
def _macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist


# ---------------- 1. 包含关系处理 ----------------
def merge_inclusion(df):
    """K线包含关系合并,返回合并K线列表 [{high, low, oi(原始末根索引), date}]"""
    bars = []
    direction = 1  # 初始假设向上
    highs = df['high'].astype(float).values
    lows = df['low'].astype(float).values
    dates = df['date'].values
    for i in range(len(df)):
        h, l = highs[i], lows[i]
        if not bars:
            bars.append({'high': h, 'low': l, 'oi': i, 'date': dates[i]})
            continue
        last = bars[-1]
        contains = (h <= last['high'] and l >= last['low']) or \
                   (h >= last['high'] and l <= last['low'])
        if contains:
            if direction > 0:   # 向上: 取高点的高点, 低点的高点
                last['high'] = max(last['high'], h)
                last['low'] = max(last['low'], l)
            else:               # 向下: 取低点的低点, 高点的低点
                last['high'] = min(last['high'], h)
                last['low'] = min(last['low'], l)
            last['oi'] = i
            last['date'] = dates[i]
        else:
            direction = 1 if h > last['high'] else -1
            bars.append({'high': h, 'low': l, 'oi': i, 'date': dates[i]})
    return bars


# ---------------- 2. 分型识别 ----------------
def find_fractals(bars):
    """顶/底分型 [{type, price, mi(合并索引), oi, date}]"""
    fr = []
    for i in range(1, len(bars) - 1):
        a, b, c = bars[i - 1], bars[i], bars[i + 1]
        if b['high'] > a['high'] and b['high'] > c['high'] and \
           b['low'] > a['low'] and b['low'] > c['low']:
            fr.append({'type': 'top', 'price': b['high'], 'mi': i,
                       'oi': b['oi'], 'date': b['date']})
        elif b['low'] < a['low'] and b['low'] < c['low'] and \
                b['high'] < a['high'] and b['high'] < c['high']:
            fr.append({'type': 'bottom', 'price': b['low'], 'mi': i,
                       'oi': b['oi'], 'date': b['date']})
    return fr


# ---------------- 3. 笔 ----------------
def build_bi(fractals, min_gap=4):
    """顶底交替成笔,返回笔端点序列(同型取更极端者)"""
    pts = []
    for f in fractals:
        if not pts:
            pts.append(dict(f))
            continue
        last = pts[-1]
        if f['type'] == last['type']:
            # 同型: 顶取更高, 底取更低
            if (f['type'] == 'top' and f['price'] >= last['price']) or \
               (f['type'] == 'bottom' and f['price'] <= last['price']):
                pts[-1] = dict(f)
        else:
            if f['mi'] - last['mi'] >= min_gap:
                pts.append(dict(f))
            # 间隔不足则忽略该分型
    # 构造笔段
    bis = []
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        bis.append({
            'dir': 'up' if p1['price'] > p0['price'] else 'down',
            'start_date': p0['date'], 'end_date': p1['date'],
            'start_oi': p0['oi'], 'end_oi': p1['oi'],
            'lo': min(p0['price'], p1['price']),
            'hi': max(p0['price'], p1['price']),
            'start_price': p0['price'], 'end_price': p1['price'],
        })
    return pts, bis


# ---------------- 4. 中枢 ----------------
def build_zhongshu(pts, bis):
    """连续三笔重叠构成中枢; 后续笔与之重叠则延伸"""
    zs_list = []
    i = 0
    while i <= len(bis) - 3:
        zg = min(bis[i]['hi'], bis[i + 1]['hi'], bis[i + 2]['hi'])
        zd = max(bis[i]['lo'], bis[i + 1]['lo'], bis[i + 2]['lo'])
        if zg > zd:
            j = i + 3
            # 中枢延伸: 后续笔区间与[zd,zg]有重叠
            while j < len(bis) and bis[j]['lo'] <= zg and bis[j]['hi'] >= zd:
                j += 1
            zs_list.append({
                'zg': round(zg, 2), 'zd': round(zd, 2),
                'start_date': pts[i]['date'], 'end_date': pts[min(j, len(pts) - 1)]['date'],
            })
            i = j
        else:
            i += 1
    return zs_list


# ---------------- 5. MACD背驰 ----------------
def _bi_area(hist, bi):
    """笔区间内的MACD柱面积(向下笔取绿柱,向上笔取红柱)"""
    seg = hist.iloc[bi['start_oi']:bi['end_oi'] + 1]
    if bi['dir'] == 'down':
        return float(seg[seg < 0].abs().sum())
    return float(seg[seg > 0].sum())


def detect_divergence(bis, hist):
    """同向相邻两笔比较面积; 返回(底背驰bool, 顶背驰bool, 说明)"""
    downs = [b for b in bis if b['dir'] == 'down']
    ups = [b for b in bis if b['dir'] == 'up']
    bottom_div = top_div = False
    msg = []
    if len(downs) >= 2:
        p, c = downs[-2], downs[-1]
        pa, ca = _bi_area(hist, p), _bi_area(hist, c)
        if c['lo'] < p['lo'] and ca < pa * 0.85:
            bottom_div = True
            msg.append(f"底背驰: 价格新低({c['lo']:.0f}<{p['lo']:.0f})但绿柱面积缩小{pa:.0f}→{ca:.0f}")
    if len(ups) >= 2:
        p, c = ups[-2], ups[-1]
        pa, ca = _bi_area(hist, p), _bi_area(hist, c)
        if c['hi'] > p['hi'] and ca < pa * 0.85:
            top_div = True
            msg.append(f"顶背驰: 价格新高({c['hi']:.0f}>{p['hi']:.0f})但红柱面积缩小{pa:.0f}→{ca:.0f}")
    return bottom_div, top_div, '; '.join(msg) if msg else "无背驰"


# ---------------- 6. 买卖点 ----------------
def find_trade_points(df, pts, bis, zs_list, hist):
    """识别一二三买卖点(取每类最近若干次)"""
    tps = []
    downs = [b for b in bis if b['dir'] == 'down']
    ups = [b for b in bis if b['dir'] == 'up']

    # 1买: 下跌笔新低 + 面积背驰
    for k in range(1, len(downs)):
        p, c = downs[k - 1], downs[k]
        if c['lo'] < p['lo'] and _bi_area(hist, c) < _bi_area(hist, p) * 0.85:
            tps.append({'date': c['end_date'], 'price': c['lo'], 'label': '1买'})
    # 1卖: 镜像
    for k in range(1, len(ups)):
        p, c = ups[k - 1], ups[k]
        if c['hi'] > p['hi'] and _bi_area(hist, c) < _bi_area(hist, p) * 0.85:
            tps.append({'date': c['end_date'], 'price': c['hi'], 'label': '1卖'})

    # 2买: 最近1买之后首个不破1买低点的底分型
    last_b1 = [t for t in tps if t['label'] == '1买']
    if last_b1:
        b1 = last_b1[-1]
        for pt in pts:
            if pt['type'] == 'bottom' and pt['date'] > b1['date'] and pt['price'] > b1['price']:
                tps.append({'date': pt['date'], 'price': pt['price'], 'label': '2买'})
                break
    # 2卖: 镜像
    last_s1 = [t for t in tps if t['label'] == '1卖']
    if last_s1:
        s1 = last_s1[-1]
        for pt in pts:
            if pt['type'] == 'top' and pt['date'] > s1['date'] and pt['price'] < s1['price']:
                tps.append({'date': pt['date'], 'price': pt['price'], 'label': '2卖'})
                break

    # 3买/3卖: 相对最后一个中枢的突破回抽
    if zs_list:
        z = zs_list[-1]
        dates = df['date'].values
        closes = df['close'].astype(float).values
        broke_up = broke_down = None
        for i in range(len(df)):
            if dates[i] > z['end_date'] and broke_up is None and closes[i] > z['zg']:
                broke_up = dates[i]
            if dates[i] > z['end_date'] and broke_down is None and closes[i] < z['zd']:
                broke_down = dates[i]
        if broke_up:
            for pt in pts:
                if pt['type'] == 'bottom' and pt['date'] > broke_up and pt['price'] >= z['zg'] * 0.995:
                    tps.append({'date': pt['date'], 'price': pt['price'], 'label': '3买'})
                    break
        if broke_down:
            for pt in pts:
                if pt['type'] == 'top' and pt['date'] > broke_down and pt['price'] <= z['zd'] * 1.005:
                    tps.append({'date': pt['date'], 'price': pt['price'], 'label': '3卖'})
                    break

    tps.sort(key=lambda t: t['date'])
    # 每类只保留最近2次,避免图面过密
    out = []
    for lbl in ('1买', '2买', '3买', '1卖', '2卖', '3卖'):
        out.extend([t for t in tps if t['label'] == lbl][-2:])
    return sorted(out, key=lambda t: t['date'])


# ---------------- 7. 简化波浪计数 ----------------
def count_waves(pts):
    """
    在笔端点序列上识别最近的五浪结构(向上或向下)+后续ABC
    校验: 2浪不破1浪起点; 3浪非最短; 4浪不进1浪价格区间
    Returns: [{date, price, label}] label∈{0..5,a,b,c} 或 下1..下5
    """
    if len(pts) < 6:
        return []
    # 从最近的完整6端点倒序扫描
    for i in range(len(pts) - 6, max(-1, len(pts) - 30), -1):
        seg = pts[i:i + 6]
        types = [p['type'] for p in seg]
        prices = [p['price'] for p in seg]
        # 向上五浪: 底顶底顶底顶
        if types == ['bottom', 'top', 'bottom', 'top', 'bottom', 'top']:
            w1 = prices[1] - prices[0]
            w3 = prices[3] - prices[2]
            w5 = prices[5] - prices[4]
            if w1 > 0 and w3 > 0 and w5 > 0 and \
               prices[2] > prices[0] and \
               prices[4] > prices[1] and \
               w3 >= min(w1, w5):
                labels = ['0', '1', '2', '3', '4', '5']
                out = [{'date': p['date'], 'price': p['price'], 'label': l}
                       for p, l in zip(seg, labels)]
                # 后续ABC
                rest = pts[i + 6:i + 9]
                if len(rest) >= 2 and rest[0]['type'] == 'bottom':
                    out.append({'date': rest[0]['date'], 'price': rest[0]['price'], 'label': 'a'})
                    if len(rest) >= 2:
                        out.append({'date': rest[1]['date'], 'price': rest[1]['price'], 'label': 'b'})
                    if len(rest) >= 3 and rest[2]['type'] == 'bottom':
                        out.append({'date': rest[2]['date'], 'price': rest[2]['price'], 'label': 'c'})
                return out
        # 向下五浪: 顶底顶底顶底
        if types == ['top', 'bottom', 'top', 'bottom', 'top', 'bottom']:
            w1 = prices[0] - prices[1]
            w3 = prices[2] - prices[3]
            w5 = prices[4] - prices[5]
            if w1 > 0 and w3 > 0 and w5 > 0 and \
               prices[2] < prices[0] and \
               prices[4] < prices[1] and \
               w3 >= min(w1, w5):
                labels = ['下0', '下1', '下2', '下3', '下4', '下5']
                return [{'date': p['date'], 'price': p['price'], 'label': l}
                        for p, l in zip(seg, labels)]
    return []


# ---------------- 8. 走势推演(三路径+历史相似概率) ----------------
def _future_trade_dates(last_date, n):
    """生成未来n个交易日标签(跳过周末)"""
    out, d = [], pd.Timestamp(last_date)
    while len(out) < n:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5:
            out.append(d.strftime('%m-%d'))
    return out


def forecast_paths(df, chan_result, horizon=15):
    """
    缠论走势推演:
      1. 基于最近中枢(或60日高低点替代)构造三条推演路径:
         上攻(突破ZG)、震荡(中枢内波动)、下探(跌破ZD)
      2. 概率估计: 提取当前市场特征(区间位置/MACD柱变化/20日动量/5日动量),
         与全历史逐日特征做标准化欧氏距离匹配, 取最相似30天,
         统计其后horizon日收益分布 -> 上攻/震荡/下探概率
    Returns:
        dict: future_dates/paths{up,range,down}/probs{up,range,down}/basis(说明)
    """
    df = df.sort_values('date').reset_index(drop=True)
    close = df['close'].astype(float)
    cur = float(close.iloc[-1])
    last_date = pd.Timestamp(df['date'].iloc[-1])

    # --- 路径锚点: 最近中枢, 无中枢用60日高低点 ---
    zs = chan_result.get('zhongshu') or []
    if zs:
        zg, zd = zs[-1]['zg'], zs[-1]['zd']
    else:
        zg = float(df['high'].astype(float).tail(60).max())
        zd = float(df['low'].astype(float).tail(60).min())
    mid, rng = (zg + zd) / 2, zg - zd

    # --- 三条路径的关键节点(从当前价出发) ---
    up_wp = [cur, max(cur, zg), zg + rng * 0.5, zg + rng * 0.8]
    range_wp = [cur, mid, zg - rng * 0.15, zd + rng * 0.15, mid]
    down_wp = [cur, min(cur, zd), zd - rng * 0.5, zd - rng * 0.8]

    def expand(waypoints):
        """关键节点线性插值到horizon个点"""
        pts = [waypoints[0]]
        seg = horizon // (len(waypoints) - 1)
        for i in range(len(waypoints) - 1):
            a, b = waypoints[i], waypoints[i + 1]
            n = seg if i < len(waypoints) - 2 else horizon - seg * (len(waypoints) - 2)
            for j in range(1, n + 1):
                pts.append(a + (b - a) * j / n)
        return [round(v, 1) for v in pts[:horizon]]

    # --- 历史相似状态统计概率 ---
    high60 = df['high'].astype(float).rolling(60).max()
    low60 = df['low'].astype(float).rolling(60).min()
    pos = ((close - low60) / (high60 - low60).replace(0, np.nan))
    _, _, hist = _macd(close)
    feats = pd.DataFrame({
        'pos': pos * 100,
        'hist_chg': hist.diff(5),
        'ret20': close.pct_change(20) * 100,
        'ret5': close.pct_change(5) * 100,
    }).dropna()
    idx = feats.index
    fwd_ret = close.pct_change(horizon).shift(-horizon)

    if len(feats) < 80:
        return {'future_dates': _future_trade_dates(last_date, horizon),
                'paths': {'up': expand(up_wp), 'range': expand(range_wp),
                          'down': expand(down_wp)},
                'probs': {'up': 33, 'range': 34, 'down': 33},
                'basis': f'历史样本不足,概率按均势处理; 中枢[{zd:.0f}-{zg:.0f}]'}

    # 标准化后欧氏距离匹配
    mean, std = feats.mean(), feats.std().replace(0, 1)
    cur_feat = ((feats.iloc[-1] - mean) / std).values
    hist_mat = ((feats - mean) / std).values
    dists = np.linalg.norm(hist_mat - cur_feat, axis=1)
    # 排除最近horizon+5天(与自身重叠)
    cutoff = len(feats) - horizon - 5
    valid = np.argsort(dists[:cutoff])[:30]
    sim_dates = idx[valid]
    rets = fwd_ret.loc[sim_dates].dropna()

    p_up = float((rets > 0.02).mean()) if len(rets) else 0.33
    p_down = float((rets < -0.02).mean()) if len(rets) else 0.33
    p_range = max(0.0, 1 - p_up - p_down)
    total = p_up + p_down + p_range
    probs = {'up': round(p_up / total * 100), 'range': round(p_range / total * 100),
             'down': round(p_down / total * 100)}
    avg_ret = float(rets.mean() * 100) if len(rets) else 0.0

    basis = (f"锚定中枢[{zd:.0f}-{zg:.0f}]; 与历史最相似30段(特征: 区间位置/"
             f"MACD柱变化/20日动量/5日动量)后{horizon}日平均{avg_ret:+.2f}%")

    return {
        'future_dates': _future_trade_dates(last_date, horizon),
        'paths': {'up': expand(up_wp), 'range': expand(range_wp),
                  'down': expand(down_wp)},
        'probs': probs,
        'zg': zg, 'zd': zd,
        'basis': basis,
    }


# ---------------- 总入口 ----------------
def analyze(df, min_gap=4):
    """
    缠论完整分析
    Args:
        df: date/open/high/low/close 日线
    Returns:
        dict: bi_points/zhongshu/trade_points/waves/divergence/state_text/macd序列
    """
    df = df.sort_values('date').reset_index(drop=True).copy()
    close = df['close'].astype(float)
    dif, dea, hist = _macd(close)

    bars = merge_inclusion(df)
    fractals = find_fractals(bars)
    pts, bis = build_bi(fractals, min_gap=min_gap)
    zs_list = build_zhongshu(pts, bis) if len(bis) >= 3 else []
    bottom_div, top_div, div_msg = detect_divergence(bis, hist)
    tps = find_trade_points(df, pts, bis, zs_list, hist)
    waves = count_waves(pts)

    # ---- 当前状态描述 ----
    last_dir = bis[-1]['dir'] if bis else 'unknown'
    dir_cn = {'up': '上升笔', 'down': '下降笔', 'unknown': '未知'}[last_dir]
    cur_close = float(close.iloc[-1])
    if zs_list:
        z = zs_list[-1]
        if cur_close > z['zg']:
            z_pos = f"中枢上方(中枢[{z['zd']:.0f}-{z['zg']:.0f}])"
        elif cur_close < z['zd']:
            z_pos = f"中枢下方(中枢[{z['zd']:.0f}-{z['zg']:.0f}])"
        else:
            z_pos = f"中枢内部([{z['zd']:.0f}-{z['zg']:.0f}])"
    else:
        z_pos = "近期无中枢(单边行情)"
    last_tp = tps[-1] if tps else None
    tp_txt = f"最近信号: {last_tp['label']}({pd.Timestamp(last_tp['date']).strftime('%m-%d')})" if last_tp else "近期无买卖点信号"

    state = (f"当前{dir_cn}运行中; 价格处于{z_pos}; "
             f"{'⚠' + div_msg if (bottom_div or top_div) else div_msg}; {tp_txt}")

    def _fmt_pts(seq):
        return [{'date': pd.Timestamp(p['date']).strftime('%Y-%m-%d'),
                 'price': round(float(p['price']), 2),
                 **{k: v for k, v in p.items() if k in ('type', 'label')}}
                for p in seq]

    return {
        'bi_points': _fmt_pts(pts),
        'zhongshu': [{'zg': z['zg'], 'zd': z['zd'],
                      'start_date': pd.Timestamp(z['start_date']).strftime('%Y-%m-%d'),
                      'end_date': pd.Timestamp(z['end_date']).strftime('%Y-%m-%d')}
                     for z in zs_list],
        'trade_points': _fmt_pts(tps),
        'waves': _fmt_pts(waves),
        'bottom_div': bottom_div,
        'top_div': top_div,
        'div_msg': div_msg,
        'state_text': state,
        'macd': {
            'dif': [round(float(v), 2) for v in dif],
            'dea': [round(float(v), 2) for v in dea],
            'hist': [round(float(v), 2) for v in hist],
        },
    }

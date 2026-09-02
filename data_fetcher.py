# -*- coding: utf-8 -*-
"""
数据获取模块
负责从akshare获取行业板块指数、市场K线、两融数据
"""
import akshare as ak
import pandas as pd
import os
from datetime import datetime, timedelta

# 数据缓存目录
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(CACHE_DIR, exist_ok=True)

# 28个申万一级行业（与监测图一致）
# 格式: 显示名称 -> 申万行业代码
SECTOR_MAP = {
    "银行": "801780",
    "非银": "801790",
    "食品": "801120",
    "医药": "801150",
    "家电": "801110",
    "汽车": "801880",
    "轻工": "801140",
    "美容": "801980",
    "纺织": "801130",
    "商贸": "801200",
    "社服": "801210",
    "传媒": "801760",
    "计算机": "801750",
    "通信": "801770",
    "电力": "801730",
    "军工": "801740",
    "机械": "801890",
    "房地产": "801180",
    "建材": "801710",
    "建筑": "801720",
    "交通": "801170",
    "公用": "801160",
    "环保": "801970",
    "有色": "801050",
    "化工": "801030",
    "钢铁": "801040",
    "煤炭": "801950",
    "石油": "801960",
}

# 行业排列顺序（按图1从左到右）
SECTOR_ORDER = [
    "银行", "非银", "食品", "医药", "家电", "汽车", "轻工", "美容",
    "纺织", "商贸", "社服", "传媒", "计算机", "通信", "电力", "军工",
    "机械", "房地产", "建材", "建筑", "交通", "公用", "环保", "有色",
    "化工", "钢铁", "煤炭", "石油"
]

# 板块对应的主流行业ETF（跟踪误差小、流动性好的一只代表）
# 格式: 板块名 -> (ETF代码, ETF名称)
SECTOR_ETF_MAP = {
    "银行":   ("512800", "银行ETF"),
    "非银":   ("512880", "证券ETF"),
    "食品":   ("515170", "食品饮料ETF"),
    "医药":   ("512170", "医疗ETF"),
    "家电":   ("159996", "家电ETF"),
    "汽车":   ("516110", "汽车ETF"),
    "轻工":   ("512430", "家居家电ETF"),      # 无轻工ETF, 以家居(轻工最大子行业)代理
    "美容":   ("515650", "消费50ETF"),        # 无美容ETF, 消费50含珀莱雅/贝泰妮等美容护理龙头
    "纺织":   ("159936", "可选消费ETF(广发)"), # 无纺织ETF, 全指可选消费含海澜之家/华利集团等
    "商贸":   ("562580", "可选消费ETF(华夏)"), # 无商贸ETF, 全指可选消费含中国中免/王府井等零售
    "社服":   ("159766", "旅游ETF"),
    "传媒":   ("512980", "传媒ETF"),
    "计算机": ("512720", "计算机ETF"),
    "通信":   ("515880", "通信ETF"),
    "电力":   ("561560", "电力ETF"),
    "军工":   ("512660", "军工ETF"),
    "机械":   ("159886", "机械ETF"),
    "房地产": ("512200", "房地产ETF"),
    "建材":   ("159745", "建材ETF"),
    "建筑":   ("516950", "基建ETF"),
    "交通":   ("159666", "交通运输ETF"),
    "公用":   ("159611", "电力ETF(公用事业)"),
    "环保":   ("512580", "环保ETF"),
    "有色":   ("512400", "有色金属ETF"),
    "化工":   ("159870", "化工ETF"),
    "钢铁":   ("515210", "钢铁ETF"),
    "煤炭":   ("515220", "煤炭ETF"),
    "石油":   ("159930", "能源ETF"),
}


# 细分赛道ETF映射表（独立于28宽行业，并行评估层）
# 格式: 赛道名 -> (ETF代码, ETF名称, 所属宽行业, 赛道类型)
SUB_SECTOR_ETF_MAP = {
    "创新药":   ("159992", "创新药ETF",   "医药",   "医药健康"),
    "生物医药": ("512290", "生物医药ETF", "医药",   "医药健康"),
    "医疗器械": ("159883", "医疗器械ETF", "医药",   "医药健康"),
    "半导体":   ("512480", "半导体ETF",   "电子",   "科技成长"),
    "科创芯片": ("588200", "科创芯片ETF", "电子",   "科技成长"),
    "人工智能": ("159819", "人工智能ETF", "计算机", "科技成长"),
    "云计算":   ("516510", "云计算ETF",   "计算机", "科技成长"),
    "软件信创": ("515230", "软件ETF",     "计算机", "科技成长"),
    "5G通信":   ("515050", "5G通信ETF",   "通信",   "科技成长"),
    "游戏":     ("159869", "游戏ETF",     "传媒",   "科技成长"),
    "机器人":   ("562500", "机器人ETF",   "机械",   "科技成长"),
    "新能源车": ("515030", "新能源车ETF", "汽车",   "新能源"),
    "光伏":     ("515790", "光伏ETF",     "—",      "新能源"),
    "电池":     ("159755", "电池ETF",     "—",      "新能源"),
    "券商":     ("512000", "券商ETF",     "非银",   "金融"),
    "军工龙头": ("512710", "军工龙头ETF", "军工",   "军工"),
    "白酒":     ("512690", "酒ETF",       "食品",   "消费"),
    "黄金股":   ("517520", "黄金股ETF",   "有色",   "周期资源"),
}
SUB_SECTOR_ORDER = list(SUB_SECTOR_ETF_MAP.keys())

# 行业指数ETF补齐状态: {行业名: 补齐到的日期}, 供看板页头透明标注
SECTOR_ETF_FILL_INFO = {}


def _fetch_etf_hist_sina(code):
    """备用: 用新浪接口获取ETF日线(当日收盘后即更新)"""
    try:
        df = ak.fund_etf_hist_sina(symbol=f"sh{code}" if code.startswith('5') else f"sz{code}")
        if df is None or len(df) == 0:
            return None
        df['date'] = pd.to_datetime(df['date'])
        # 新浪接口无成交额, 用成交量*收盘价估算
        if 'amount' not in df.columns:
            df['amount'] = df['volume'] * df['close']
        return df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
    except Exception:
        return None


def _fill_sector_with_etf(sector_name, sw_df):
    """
    申万行业指数为T+1更新, 当日缺失时用对应主流ETF(新浪源, 当日更新)推算补齐:
    - 价格: 行业指数前收 × ETF当日相对涨跌幅(OHLC分别推算)
    - 成交量: 申万5日均量 × ETF当日量比(保持申万量纲, 量能形态跟随ETF)
    仅补齐申万缺失的最新交易日; 申万数据源更新后自动被真实数据覆盖
    """
    etf_info = SECTOR_ETF_MAP.get(sector_name)
    if not etf_info or sw_df is None or len(sw_df) < 10:
        return sw_df
    etf_code = etf_info[0]
    try:
        etf = _fetch_etf_hist_sina(etf_code)
        if etf is None or len(etf) < 20:
            return sw_df
        etf = etf.sort_values('date').reset_index(drop=True)
        sw_last_date = sw_df.iloc[-1]['date']
        missing = etf[etf['date'] > sw_last_date]
        if len(missing) == 0:
            return sw_df
        sw_vol_ma5 = float(sw_df['volume'].tail(5).mean())
        filled = 0
        cur = sw_df.copy()
        for _, e in missing.iterrows():
            prev_date = cur.iloc[-1]['date']
            e_prev = etf[etf['date'] == prev_date]
            if len(e_prev) == 0 or float(e_prev.iloc[0]['close']) <= 0:
                continue
            e_prev_close = float(e_prev.iloc[0]['close'])
            base_close = float(cur.iloc[-1]['close'])
            scale = float(e['close']) / e_prev_close
            scale_o = float(e['open']) / e_prev_close
            scale_h = float(e['high']) / e_prev_close
            scale_l = float(e['low']) / e_prev_close
            # ETF量比(当日量/前5日均量) -> 映射到申万量纲
            e_idx = etf.index[etf['date'] == e['date']]
            if len(e_idx) > 0 and e_idx[0] >= 5:
                e_vol_ma5 = float(etf['volume'].iloc[e_idx[0] - 5:e_idx[0]].mean())
                vol_ratio = float(e['volume']) / e_vol_ma5 if e_vol_ma5 > 0 else 1.0
            else:
                vol_ratio = 1.0
            new_close = base_close * scale
            new_row = pd.DataFrame([{
                'date': e['date'],
                'open': base_close * scale_o,
                'high': base_close * scale_h,
                'low': base_close * scale_l,
                'close': new_close,
                'volume': sw_vol_ma5 * vol_ratio,
                'amount': sw_vol_ma5 * vol_ratio * new_close,
            }])
            cur = pd.concat([cur, new_row], ignore_index=True)
            filled += 1
        if filled > 0:
            d0 = missing.iloc[0]['date'].strftime('%m-%d')
            d1 = missing.iloc[-1]['date'].strftime('%m-%d')
            print(f" [ETF补齐至{d1}]" if filled == 1 else f" [ETF补齐{d0}~{d1}]", end="")
            # 记录补齐状态(供看板页头标注)
            SECTOR_ETF_FILL_INFO[sector_name] = missing.iloc[-1]['date'].strftime('%Y-%m-%d')
        return cur
    except Exception:
        return sw_df


def get_sub_sector_data(days=150):
    """
    获取细分赛道ETF日线数据(前复权)

    Args:
        days: 获取最近多少个交易日的数据

    Returns:
        dict: {赛道名: DataFrame(date, open, high, low, close, volume, amount)}
    """
    import time
    result = {}
    for name, (code, etf_name, _, _) in SUB_SECTOR_ETF_MAP.items():
        df = None
        # 主接口: 东方财富
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily", adjust="qfq")
            if df is not None and len(df) > 0:
                df = df.rename(columns={
                    '日期': 'date', '开盘': 'open', '收盘': 'close',
                    '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount'
                })
        except Exception as e:
            err_msg = str(e)
            if 'Connection' in err_msg or 'Remote' in err_msg:
                print(f"  {name}({code}): 东财接口受限, 尝试备用接口...")
                df = _fetch_etf_hist_sina(code)
            else:
                print(f"  {name}({code}): 获取失败 - {e}")
                continue

        if df is None or len(df) == 0:
            print(f"  {name}({code}): 无数据")
            continue

        try:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            result[name] = df.tail(days).reset_index(drop=True)
            print(f"  {name}({code}): OK ({len(result[name])}条)")
            time.sleep(0.3)  # 限速, 避免请求过快
        except Exception as e:
            print(f"  {name}({code}): 数据处理失败 - {e}")
    return result


def get_sector_hist_data(sector_name, days=120, etf_fill=True):
    """
    获取单个行业板块的历史日线数据

    申万指数源为T+1更新; etf_fill=True时, 当日缺失部分用对应主流ETF
    (新浪源, 当日收盘后即更新)推算补齐, 保证行业数据与大盘指数同步

    Args:
        sector_name: 行业显示名称（如"银行"）
        days: 获取最近多少个交易日的数据
        etf_fill: 是否启用ETF当日补齐

    Returns:
        DataFrame: 包含 date, open, high, low, close, volume, amount 的日线数据
    """
    code = SECTOR_MAP.get(sector_name)
    if not code:
        raise ValueError(f"未知行业: {sector_name}")

    try:
        df = ak.index_hist_sw(symbol=code, period="day")
        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 统一列名
        df = df.rename(columns={
            '日期': 'date', '开盘': 'open', '最高': 'high',
            '最低': 'low', '收盘': 'close', '成交量': 'volume',
            '成交额': 'amount'
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # 申万T+1: 当日缺失用对应ETF行情推算补齐
        if etf_fill:
            df = _fill_sector_with_etf(sector_name, df)

        # 取最近N天
        df = df.tail(days).reset_index(drop=True)
        return df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]

    except Exception as e:
        print(f"  获取{sector_name}数据失败: {e}")
        return pd.DataFrame()


def get_all_sectors_data(days=120):
    """
    获取所有28个行业板块的历史数据

    Returns:
        dict: {行业名称: DataFrame}
    """
    result = {}
    total = len(SECTOR_ORDER)
    for i, name in enumerate(SECTOR_ORDER):
        print(f"  [{i+1}/{total}] 获取 {name} 行业数据...", end="")
        df = get_sector_hist_data(name, days)
        if len(df) > 0:
            result[name] = df
            print(f" OK ({len(df)}条)")
        else:
            print(f" 失败")
    return result


def get_index_kline(symbol="sh000001", days=400):
    """
    获取市场指数K线数据

    Args:
        symbol: 指数代码 sh000001=上证指数, sz399006=创业板指, sh000300=沪深300
        days: 获取天数

    Returns:
        DataFrame: date, open, high, low, close, volume
    """
    try:
        df = ak.stock_zh_index_daily(symbol=symbol)
        if df is None or len(df) == 0:
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df.tail(days).reset_index(drop=True)
        return df[['date', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"  获取指数K线失败: {e}")
        return pd.DataFrame()


def _load_cache(name):
    """读取CSV历史缓存"""
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=['date'])
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _save_cache(name, df):
    """写入CSV历史缓存(按date去重)"""
    if len(df) == 0:
        return
    path = os.path.join(CACHE_DIR, name)
    df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')


def get_margin_data(days=60):
    """
    获取沪深两融数据（融资净买入）
    带CSV历史积累: 每次运行增量更新缓存,长期运行形成完整历史库

    Returns:
        DataFrame: date, margin_net (融资净买入额，单位亿元), margin_balance
    """
    cache = _load_cache("margin_history.csv")
    new_parts = []

    # 沪市两融: 逐日汇总 stock_margin_detail_sse (批量接口 stock_margin_sse 已多次返回不全)
    # 策略: 每次从 15 天前开始逐日拉, 覆盖缓存可能遗漏的数据
    # 若逐日接口全部失败, 回退到 stock_margin_sse 批量接口 (start_date 必须近)
    sh_rows = []
    today = datetime.now().date()
    start_try = today - timedelta(days=15)
    cur = start_try
    trade_days_checked = 0
    while cur <= today and trade_days_checked < 20:
        if cur.weekday() < 5:  # 跳过周末
            trade_days_checked += 1
            try:
                ds = cur.strftime('%Y%m%d')
                ddf = ak.stock_margin_detail_sse(date=ds)
                if ddf is not None and len(ddf) > 0:
                    bal = pd.to_numeric(ddf['融资余额'], errors='coerce').sum()
                    if bal > 0:
                        sh_rows.append({'date': pd.Timestamp(cur), 'balance': bal})
            except Exception:
                pass  # 节假日或当天未出数据, 跳过
        cur += timedelta(days=1)

    if sh_rows:
        df_sh = pd.DataFrame(sh_rows)
        new_parts.append(df_sh)
        print(f"  沪市两融(逐日): {len(df_sh)}个交易日, 最新={df_sh['date'].max().strftime('%Y-%m-%d')}")
    else:
        # 回退: stock_margin_sse 批量接口 (start_date 必须靠近今天, 否则接口只返回旧数据)
        try:
            end_date = datetime.now().strftime("%Y%m%d")
            # 只往前查 30 天, 避免接口因起始日过远而返回陈旧数据
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
            df_sh = ak.stock_margin_sse(start_date=start_date, end_date=end_date)
            if df_sh is not None and len(df_sh) > 0:
                df_sh['date'] = pd.to_datetime(df_sh['信用交易日期'], format='%Y%m%d')
                df_sh['balance'] = pd.to_numeric(df_sh['融资余额'], errors='coerce')
                new_parts.append(df_sh[['date', 'balance']].copy())
                print(f"  沪市两融(回退批量): {len(df_sh)}条, 最新={df_sh['date'].max().strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"  ⚠ 沪市两融全部接口失败: {e}")

    # 深市两融(改用macro_china_market_margin_sz批量接口, 旧逐日接口已失效)
    try:
        df_sz = ak.macro_china_market_margin_sz()
        if df_sz is not None and len(df_sz) > 0:
            df_sz['date'] = pd.to_datetime(df_sz['日期'])
            df_sz['balance'] = pd.to_numeric(df_sz['融资余额'], errors='coerce')
            df_sz = df_sz[['date', 'balance']].dropna(subset=['balance'])
            # 只取最近 days*2 天(与沪市对齐)
            df_sz = df_sz.tail(days * 2)
            new_parts.append(df_sz)
            print(f"  深市两融: {len(df_sz)}条")
    except Exception as e:
        print(f"  深市两融获取失败: {e}")

    # 合并沪深当日新增
    fresh = pd.DataFrame()
    if new_parts:
        fresh = pd.concat(new_parts, ignore_index=True)
        fresh = fresh.groupby('date')['balance'].sum().reset_index()

    # 与历史缓存合并(积累)
    combined = pd.concat([cache, fresh], ignore_index=True)
    if len(combined) == 0:
        return pd.DataFrame()
    combined = combined.dropna(subset=['balance'])
    combined = combined.drop_duplicates(subset=['date'], keep='last')
    combined = combined.sort_values('date').reset_index(drop=True)

    # 净买入 = 余额差分(在长期历史上计算,避免缓存边界断点)
    combined['margin_net'] = combined['balance'].diff()
    # 转亿元
    combined['margin_balance'] = combined['balance'] / 1e8
    combined['margin_net'] = combined['margin_net'] / 1e8

    _save_cache("margin_history.csv",
                combined[['date', 'balance', 'margin_balance', 'margin_net']])

    return combined[['date', 'margin_net', 'margin_balance']].tail(days).reset_index(drop=True)


def get_us_treasury_data(days=250):
    """
    获取美债收益率曲线数据(全球资产定价之锚)
    数据源: 东财中美国债收益率对比接口 bond_zh_us_rate
    Returns:
        DataFrame: date, y2, y5, y10, y30, spread_10y2y (单位: %)
        失败时返回 None
    """
    try:
        df = ak.bond_zh_us_rate(start_date="20200101")
        if df is None or len(df) == 0:
            print("  美债收益率: 无数据")
            return None
        df = df.rename(columns={
            '日期': 'date',
            '美国国债收益率2年': 'y2',
            '美国国债收益率5年': 'y5',
            '美国国债收益率10年': 'y10',
            '美国国债收益率30年': 'y30',
            '美国国债收益率10年-2年': 'spread_10y2y',
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        # 前向填充缺失值(个别日期缺失)
        for col in ['y2', 'y5', 'y10', 'y30', 'spread_10y2y']:
            df[col] = df[col].ffill()
        df['spread_10y2y'] = df['y10'] - df['y2']
        df = df.dropna(subset=['y10']).tail(days).reset_index(drop=True)
        print(f"  美债收益率: {len(df)}条, 最新={df.iloc[-1]['date'].strftime('%Y-%m-%d')}, "
              f"10Y={df.iloc[-1]['y10']:.2f}% 30Y={df.iloc[-1]['y30']:.2f}%")
        return df[['date', 'y2', 'y5', 'y10', 'y30', 'spread_10y2y']]
    except Exception as e:
        print(f"  美债收益率获取失败: {e}")
        return None


def get_global_indices(days=300):
    """
    获取全球主要指数(用于全球联动因果分析)
    数据源: 新浪美股指数接口 stock_us_daily
    Returns:
        dict: {名称: DataFrame(date, close)}
    """
    result = {}
    targets = {
        "纳斯达克": ".IXIC",
        "标普500": ".INX",
        "道琼斯": ".DJI",
    }
    for name, code in targets.items():
        try:
            df = ak.stock_us_daily(symbol=code)
            if df is not None and len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').tail(days).reset_index(drop=True)
                result[name] = df[['date', 'close']]
                print(f"  全球指数 {name}: {len(df)}条")
        except Exception as e:
            print(f"  全球指数 {name} 获取失败: {e}")
    return result


if __name__ == "__main__":
    print("=== 测试数据获取 ===\n")

    print("1. 获取行业板块数据（测试前3个）")
    for name in SECTOR_ORDER[:3]:
        df = get_sector_hist_data(name, 30)
        if len(df) > 0:
            print(f"  {name}: {len(df)}条, 最新={df.iloc[-1]['date'].strftime('%Y-%m-%d')}, 收盘={df.iloc[-1]['close']}")

    print("\n2. 获取上证指数K线")
    kline = get_index_kline("sh000001", 30)
    if len(kline) > 0:
        print(f"  {len(kline)}条, 最新={kline.iloc[-1]['date'].strftime('%Y-%m-%d')}, 收盘={kline.iloc[-1]['close']}")

    print("\n3. 获取两融数据")
    margin = get_margin_data(20)
    if len(margin) > 0:
        print(f"  {len(margin)}条, 最新={margin.iloc[-1]['date'].strftime('%Y-%m-%d')}, 净买入={margin.iloc[-1]['margin_net']:.2f}亿")

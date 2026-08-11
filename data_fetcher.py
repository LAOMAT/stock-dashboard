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
    "轻工":   ("", "无主流ETF,用个股替代"),
    "美容":   ("", "无主流ETF,用个股替代"),
    "纺织":   ("", "无主流ETF,用个股替代"),
    "商贸":   ("", "无主流ETF,用个股替代"),
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


def get_sector_hist_data(sector_name, days=120):
    """
    获取单个行业板块的历史日线数据

    Args:
        sector_name: 行业显示名称（如"银行"）
        days: 获取最近多少个交易日的数据

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

    # 沪市两融
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
        df_sh = ak.stock_margin_sse(start_date=start_date, end_date=end_date)
        if df_sh is not None and len(df_sh) > 0:
            df_sh['date'] = pd.to_datetime(df_sh['信用交易日期'], format='%Y%m%d')
            df_sh['balance'] = pd.to_numeric(df_sh['融资余额'], errors='coerce')
            new_parts.append(df_sh[['date', 'balance']].copy())
            print(f"  沪市两融: {len(df_sh)}条")
    except Exception as e:
        print(f"  沪市两融获取失败: {e}")

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

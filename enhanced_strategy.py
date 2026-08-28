# -*- coding: utf-8 -*-
"""
多维共振增强版行业ETF交易策略

=== 设计理念(融合工作空间交易理念) ===
1. 情绪周期理论(五柳问道): 冰点期重仓、高潮期减仓、退潮期轻仓
2. 国学理论(易道股法): 乾卦六爻定位、阴阳转化、先胜后战
   —— 评分即"量爻", 多因子共振即"阴阳转化"的确认, 高分才开仓即"先胜后战"
3. 缠论: 中枢位置、买卖点、背驰
4. 量学: 黄金柱、将军柱、量价关系
5. 行业政策 + 产业链分析

=== 核心架构: 三层底仓结构 ===
  正仓(道-战略层): 政策风口+产业链核心+生命周期启动/发酵, 中期持有(50-70%)
  机动仓(法-战术层): 情绪冰点/修复+缠论1买/3买+黄金柱支撑, 波段操作(20-40%)
  观察仓(术-验证层): 板块异动但多因子未共振, 小仓试探(5-10%), 验证成功升级机动仓

=== 多因子共振评分(0-100) ===
  lifecycle 0-25 / mrs 0-20 / emotion_cycle 0-20 /
  chan_structure 0-20 / volume_pillar 0-10 / policy_industry 0-5

=== 集成方式 ===
  - 与 backtest.py / generate_dashboard.py 数据结构兼容:
    generate_plan_from_dashboard() 直接消费 compute_sector_heatmap /
    compute_market_trend 的输出, 可选传入原始板块K线自动计算缠论/量柱信号
  - params['use_enhanced']=False 或数据缺失时自动退回 MRS+生命周期 基础逻辑(fallback)

独立测试: python enhanced_strategy.py
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- 工作空间已有引擎(可选依赖, 缺失时自动fallback) ---
try:
    import trend_engine
    _HAS_TREND = True
except Exception:
    _HAS_TREND = False

try:
    import chan_engine
    _HAS_CHAN = True
except Exception:
    _HAS_CHAN = False

try:
    import volpillar
    _HAS_VOL = True
except Exception:
    _HAS_VOL = False

try:
    from backtest import _mrs_to_position as _bt_mrs_to_position
    _HAS_BACKTEST = True
except Exception:
    _HAS_BACKTEST = False


# ======================================================================
# 默认参数
# ======================================================================
DEFAULT_PARAMS = {
    # 仓位层级分数线
    'zheng_score_min': 72,      # 正仓最低总分
    'jidong_score_min': 55,     # 机动仓最低总分
    'guancha_score_min': 42,    # 观察仓最低总分
    # 各层级仓位占总资金的比例区间
    'zheng_band': (0.50, 0.70),
    'jidong_band': (0.20, 0.40),
    'guancha_band': (0.05, 0.10),
    # 信号参数
    'recent_days': 10,          # 缠论买卖点有效窗口(自然日)
    'anomaly_delta': 8.0,       # |ΔScore|超过此值视为板块异动(观察仓候选)
    # 总仓位限制
    'max_total_position': 0.95,
    'min_position': 0.05,
    # 是否自动用板块K线计算缠论/量柱信号(需要传入sectors_kline)
    'compute_chan_vol': True,
    # 政策库(可注入自定义), None用内置DEFAULT_POLICY_DB
    'policy_db': None,
    # False时退回基础MRS+生命周期逻辑(现有策略fallback)
    'use_enhanced': True,
}


# ======================================================================
# 仓位动态调整算法(规格原文实现)
# ======================================================================
def calc_dynamic_position(base_position, emotion_pos, chan_signal, vol_signal):
    """
    基础仓位: MRS映射(现有逻辑)
    情绪系数: 冰点+0.2, 修复+0.1, 分歧0, 高潮-0.2, 退潮-0.3
    缠论系数: 1买+0.15, 3买+0.1, 2买+0.05, 无信号0, 1卖-0.15, 3卖-0.2
    量学系数: 黄金柱支撑+0.05, 将军柱压力-0.05
    扩展: 2卖-0.10(2买镜像), 跌破黄金柱-0.05(视同将军柱压力)
    """
    emotion_coef = {'冰点': 0.2, '修复': 0.1, '分歧': 0, '高潮': -0.2, '退潮': -0.3}
    chan_coef = {'1买': 0.15, '3买': 0.10, '2买': 0.05, '无': 0,
                 '1卖': -0.15, '2卖': -0.10, '3卖': -0.20}
    vol_coef = {'黄金柱': 0.05, '普通': 0, '将军柱': -0.05, '跌破': -0.05}

    e = str(emotion_pos).replace('期', '')
    c = chan_signal if chan_signal in chan_coef else '无'
    v = vol_signal if vol_signal in vol_coef else '普通'

    final_position = base_position * (1 + emotion_coef.get(e, 0)
                                      + chan_coef[c] + vol_coef[v])
    return max(0.05, min(0.95, final_position))  # 限制在5%-95%


def mrs_to_position(mrs):
    """MRS -> 总仓位上限(沿用 backtest.py 现有映射, 不可用时本地复制)"""
    if _HAS_BACKTEST:
        return _bt_mrs_to_position(mrs)
    if mrs >= 75: return 1.0
    if mrs >= 60: return 0.8
    if mrs >= 45: return 0.6
    if mrs >= 30: return 0.4
    return 0.15


def mrs_to_position_band(mrs):
    """MRS -> 建议总仓位区间(与看板MRS档位严格一致)

    区间划分: ≥75进攻8~10成 | 60-75偏多6~8成 | 45-60震荡4~6成 |
              30-45防守2~4成 | <30空仓(≤1.5成)
    Returns:
        (下限, 上限), 小数形式
    """
    if mrs >= 75: return (0.80, 1.00)
    if mrs >= 60: return (0.60, 0.80)
    if mrs >= 45: return (0.40, 0.60)
    if mrs >= 30: return (0.20, 0.40)
    return (0.00, 0.15)


def align_total_position_to_mrs(wide_result, sub_result=None):
    """
    总仓位预算协调器: 宽行业池+细分池合计仓位对齐MRS建议仓位区间

    规则:
      1. MRS建议仓位区间上限 = 两池合计总预算(硬约束)
      2. 合计超上限 -> 两池按现有仓位比例同步缩放(保持相对强弱)
      3. 合计低于下限 -> 不强制加仓(预算是上限约束, 信号不足允许低配)
      4. 细分池内部40%上限、宽行业池95%上限仍然生效(先池内归一再跨池协调)

    Args:
        wide_result: generate_plan_from_dashboard输出
        sub_result:  generate_sub_sector_plan输出(可选)
    Returns:
        dict: mrs/zone/band/cap_pct/wide_pct/sub_pct/total_pct/scaled/band_txt
        None: 两池均无有效计划
    """
    results = [r for r in (wide_result, sub_result)
               if r and not r.get('fallback') and r.get('plans')]
    if not results:
        return None

    src = wide_result if wide_result else sub_result
    mrs = src.get('mrs', 50.0)
    lo, hi = mrs_to_position_band(mrs)
    cap = hi * 100

    def _pool_pct(r):
        return r['total_position_pct'] if r else 0.0

    wide_pct, sub_pct = _pool_pct(wide_result), _pool_pct(sub_result)
    total = round(wide_pct + sub_pct, 1)

    scaled = False
    if total > cap + 0.05:  # 超预算 -> 两池等比缩放
        scale = cap / total
        for r in results:
            for p in r['plans']:
                if p['position_pct'] > 0:
                    p['position_pct'] = round(p['position_pct'] * scale, 1)
            r['total_position_pct'] = round(
                sum(p['position_pct'] for p in r['plans']), 1)
        wide_pct, sub_pct = _pool_pct(wide_result), _pool_pct(sub_result)
        total = round(wide_pct + sub_pct, 1)
        scaled = True

    zone = ('进攻' if mrs >= 75 else '偏多' if mrs >= 60 else
            '震荡' if mrs >= 45 else '防守' if mrs >= 30 else '空仓')
    if lo > 0:
        band_txt = f"{lo*10:.0f}~{hi*10:.0f}成"
    else:
        band_txt = f"空仓(≤{hi*10:g}成)"

    info = {
        'mrs': mrs, 'zone': zone,
        'band': (lo, hi), 'cap_pct': round(cap, 1),
        'wide_pct': wide_pct, 'sub_pct': sub_pct,
        'total_pct': total, 'scaled': scaled, 'band_txt': band_txt,
    }
    print(f"  [总仓位预算] MRS={mrs:.0f}({zone}) 建议{band_txt}(≤{cap:.0f}%) | "
          f"宽行业池{wide_pct:.1f}% + 细分池{sub_pct:.1f}% = 合计{total:.1f}%"
          f"{' [已按预算等比缩放]' if scaled else ' [符合预算]'}")
    return info


# ======================================================================
# 辅助: 缠论/量学信号合并
# ======================================================================
def merge_chan_signals(chan_data, current_price=None, recent_days=10, ref_date=None):
    """
    合并缠论信号(chan_engine.analyze输出 -> 单一操作信号)

    优先级: 近recent_days日买卖点 > 背驰先兆 > 仅中枢位置
    Returns:
        dict: signal(1买/2买/3买/1卖/2卖/3卖/无),
              zs_pos(中枢上/中枢内/中枢下/无中枢),
              divergence(顶背驰/底背驰/无), detail
    """
    out = {'signal': '无', 'zs_pos': '无中枢', 'divergence': '无', 'detail': '无缠论数据'}
    if not chan_data:
        return out

    tps = chan_data.get('trade_points') or []
    zs = chan_data.get('zhongshu') or []
    bottom_div = bool(chan_data.get('bottom_div'))
    top_div = bool(chan_data.get('top_div'))

    # --- 最近买卖点(在有效窗口内) ---
    ref = pd.Timestamp(ref_date) if ref_date is not None else pd.Timestamp.now()
    recent_tps = []
    for t in tps:
        try:
            d = pd.Timestamp(t['date'])
        except Exception:
            continue
        if 0 <= (ref - d).days <= recent_days:
            recent_tps.append((d, t.get('label', '')))
    if recent_tps:
        recent_tps.sort(key=lambda x: x[0])
        out['signal'] = recent_tps[-1][1]
    elif bottom_div:
        out['signal'] = '1买'   # 底背驰=1买先兆
    elif top_div:
        out['signal'] = '1卖'   # 顶背驰=1卖先兆

    # --- 中枢位置 ---
    if zs and current_price is not None:
        zg, zd = float(zs[-1]['zg']), float(zs[-1]['zd'])
        p = float(current_price)
        out['zs_pos'] = '中枢上' if p > zg else ('中枢下' if p < zd else '中枢内')
    elif zs:
        out['zs_pos'] = '有中枢'

    out['divergence'] = '底背驰' if bottom_div else ('顶背驰' if top_div else '无')
    sig_txt = out['signal'] if out['signal'] != '无' else f"仅{out['zs_pos']}"
    out['detail'] = f"{sig_txt} | {out['divergence']}"
    if out['signal'] in ('1买', '1卖') and not recent_tps and out['divergence'] != '无':
        out['detail'] += '(背驰先兆,非确认买卖点)'
    return out


def merge_vol_signals(vol_data, current_price=None):
    """
    合并量学信号(volpillar.analyze_volume输出 -> 单一操作信号)

    规则:
      现价 >= 最近黄金柱柱底 -> '黄金柱'(支撑有效)
      最近为将军柱且现价 < 柱顶 -> '将军柱'(柱顶压力)
      现价跌破最近黄金柱柱底 -> '跌破'(支撑失效, 危险信号)
      无黄金柱 -> '普通'
    Returns:
        dict: signal(黄金柱/将军柱/普通/跌破), broke_support(bool), detail
    """
    out = {'signal': '普通', 'broke_support': False, 'detail': '无量柱数据'}
    if not vol_data:
        return out

    golden = vol_data.get('golden') or []
    support_txt = (vol_data.get('summary') or {}).get('support_txt', '')
    if not golden:
        out['detail'] = support_txt or '近期无黄金柱'
        return out

    g = golden[-1]
    top, bottom = float(g['top']), float(g['bottom'])
    label = g.get('label', '黄金柱')

    if current_price is None:
        out['signal'] = '黄金柱' if label == '黄金柱' else '将军柱'
        out['detail'] = support_txt or f"最近{label}({g.get('date','')})"
        return out

    p = float(current_price)
    if p < bottom:
        out['signal'] = '跌破'
        out['broke_support'] = True
        out['detail'] = f"现价跌破{label}支撑[{bottom}-{top}], 支撑失效"
    elif label == '将军柱' and p < top:
        out['signal'] = '将军柱'
        out['detail'] = f"将军柱({g.get('date','')})柱顶{top}构成压力"
    else:
        out['signal'] = '黄金柱'
        out['detail'] = f"{label}({g.get('date','')})支撑区间[{bottom}-{top}]有效"
    return out


# ======================================================================
# 情绪周期分析器(五柳问道: 冰点->修复->分歧->高潮->退潮)
# ======================================================================
class EmotionCycleAnalyzer:
    """
    以MRS分值+斜率(可选市场宽度)判定情绪周期阶段
      冰点期: MRS<30 极低迷(周期必涨, 重仓布局窗口)
      修复期: 低位回升(30-45上行, 或45-60强修复)
      分歧期: 45-75震荡(多空拉扯, 做T)
      高潮期: MRS>=75, 或高位(>=60)动能衰竭(周期必跌, 减仓)
      退潮期: 高位快速回落(斜率<=-1.5)或中位继续下行(轻仓/空仓)
    """

    PHASES = ('冰点期', '修复期', '分歧期', '高潮期', '退潮期')

    # 阶段 -> 评分(0-20): 冰点期20, 修复期16, 分歧期12, 高潮期4, 退潮期0
    PHASE_SCORE = {'冰点期': 20, '修复期': 16, '分歧期': 12, '高潮期': 4, '退潮期': 0}
    # 阶段 -> 仓位调整系数
    PHASE_COEF = {'冰点期': 0.2, '修复期': 0.1, '分歧期': 0.0, '高潮期': -0.2, '退潮期': -0.3}

    def _extract_mrs(self, market_data):
        """从market_data提取(最新MRS, MRS斜率)"""
        if market_data is None:
            return 50.0, 0.0
        if isinstance(market_data, (int, float)):
            return float(market_data), 0.0
        if isinstance(market_data, pd.DataFrame) and 'mrs' in market_data.columns:
            m = market_data['mrs'].astype(float)
            slope = float(m.iloc[-3:].mean() - m.iloc[-6:-3].mean()) if len(m) >= 6 else 0.0
            return float(m.iloc[-1]), slope
        mrs = market_data.get('latest_mrs')
        series = market_data.get('mrs_smooth') or market_data.get('mrs') or []
        if mrs is None and len(series) > 0:
            mrs = series[-1]
        mrs = 50.0 if mrs is None else float(mrs)
        slope = 0.0
        if len(series) >= 6:
            s = pd.Series(series, dtype=float)
            slope = float(s.iloc[-3:].mean() - s.iloc[-6:-3].mean())
        return mrs, slope

    def determine_phase(self, market_data):
        """判断当前情绪周期阶段, 返回 冰点期/修复期/分歧期/高潮期/退潮期"""
        mrs, slope = self._extract_mrs(market_data)
        if mrs < 30:
            return '冰点期'
        if slope <= -1.5:
            return '退潮期'                      # 高位快速回落, 最优先回避
        if mrs >= 75 or (mrs >= 60 and slope <= 0):
            return '高潮期'
        if mrs < 45:
            return '修复期' if slope > 0 else '冰点期'
        if 45 <= mrs < 60 and slope > 1.5:
            return '修复期'                      # 中位强修复
        return '分歧期'

    def get_position_adjustment(self, phase):
        """获取仓位调整系数: 冰点+0.2, 修复+0.1, 分歧0, 高潮-0.2, 退潮-0.3"""
        return self.PHASE_COEF.get(str(phase).replace('期', '') + '期', 0.0)

    def get_phase_score(self, phase):
        """情绪周期因子得分(0-20)"""
        return self.PHASE_SCORE.get(phase, 12)


# ======================================================================
# 政策面 + 产业链分析器(易道股法: 政策风口=天时, 产业链核心=地利)
# ======================================================================
# 内置政策库: 板块 -> (政策热度0-3, 产业链地位0-2, 主题备注)
# 政策热度: 3=国家级战略风口 2=产业政策支持 1=常规 0=无
# 产业链地位: 2=核心卡位 1=重要环节 0=边缘
DEFAULT_POLICY_DB = {
    '计算机': (3, 2, 'AI+信创+数据要素国家战略'),
    '通信':   (3, 2, '算力基础设施/光模块核心卡位'),
    '军工':   (3, 1, '国防现代化, 订单驱动'),
    '电力':   (2, 2, '新型电力系统核心环节'),
    '机械':   (2, 1, '大规模设备更新+机器人'),
    '环保':   (2, 1, '双碳政策'),
    '医药':   (2, 1, '创新药支持政策'),
    '汽车':   (2, 1, '以旧换新+智能驾驶'),
    '家电':   (2, 1, '消费品以旧换新'),
    '非银':   (2, 1, '资本市场改革'),
    '有色':   (1, 2, '战略性矿产资源卡位'),
    '煤炭':   (1, 2, '能源保供压舱石'),
    '石油':   (1, 2, '国家能源安全'),
    '银行':   (1, 2, '金融体系核心, 高股息底仓'),
    '食品':   (1, 1, '消费提振'),
    '美容':   (1, 1, '颜值经济'),
    '社服':   (1, 1, '文旅消费'),
    '传媒':   (2, 1, 'AI应用+游戏版号常态化'),
    '电子':   (3, 2, '半导体自主可控'),
    '房地产': (2, 0, '政策托底但产业链地位弱化'),
}


# 细分赛道政策库: 赛道名 -> (政策热度0-3, 产业链地位0-2, 主题备注)
SUB_SECTOR_POLICY_DB = {
    '创新药':   (3, 2, '创新药出海+商保目录扩容'),
    '生物医药': (2, 1, '生物医药产业政策'),
    '医疗器械': (2, 1, '国产替代+设备更新'),
    '半导体':   (3, 2, '自主可控+大基金三期'),
    '科创芯片': (3, 2, '硬科技+国产替代'),
    '人工智能': (3, 2, 'AI+行动/国产算力'),
    '云计算':   (2, 1, '数字经济基础设施'),
    '软件信创': (2, 1, '信创替代+数据要素'),
    '5G通信':   (2, 1, '5G应用+算力网络'),
    '游戏':     (1, 1, '版号常态化+AIGC应用'),
    '机器人':   (2, 1, '智能制造+人形机器人'),
    '新能源车': (2, 1, '以旧换新+智能驾驶'),
    '光伏':     (2, 1, '反内卷供给侧改革'),
    '电池':     (2, 1, '固态电池+储能'),
    '券商':     (2, 1, '资本市场改革+并购重组'),
    '军工龙头': (3, 1, '国防现代化, 订单驱动'),
    '白酒':     (1, 1, '消费提振+高端白酒'),
    '黄金股':   (1, 2, '避险资产+资源卡位'),
}

# 细分赛道策略参数预设(单只仓位更小, 信号更敏感)
SUB_SECTOR_PARAMS = {
    'zheng_band': (0.10, 0.15),      # 正仓10-15%
    'jidong_band': (0.06, 0.10),     # 机动仓6-10%
    'guancha_band': (0.03, 0.05),    # 观察仓3-5%
    'max_total_position': 0.40,      # 细分池总仓位上限40%
    'recent_days': 7,                # 缠论窗口10→7天(更敏感)
    'anomaly_delta': 6.0,            # 异动阈值8→6
    'score_weights': {               # 评分权重: 情绪周期×1.2, 缠论×1.2, MRS×0.8, 政策×0.8
        'lifecycle': 1.0, 'mrs': 0.8, 'emotion_cycle': 1.2,
        'chan_structure': 1.2, 'volume_pillar': 1.0, 'policy_industry': 0.8,
    },
    'policy_db': SUB_SECTOR_POLICY_DB,
}


class PolicyIndustryAnalyzer:
    """政策热度 + 产业链地位评分(合计映射到0-5因子分)"""

    def __init__(self, policy_db=None):
        self.db = policy_db if policy_db is not None else DEFAULT_POLICY_DB

    def score_policy_heat(self, sector_name):
        """政策热度评分(0-3)"""
        return int(self.db.get(sector_name, (1, 0, ''))[0])

    def score_industry_chain(self, sector_name):
        """产业链地位评分(0-2)"""
        return int(self.db.get(sector_name, (1, 0, ''))[1])

    def get_theme(self, sector_name):
        return self.db.get(sector_name, (1, 0, ''))[2]

    def score_combined(self, sector_name):
        """政策+产业链因子分(0-5): 风口+核心=5, 一般=3, 无=0"""
        s = self.score_policy_heat(sector_name) + self.score_industry_chain(sector_name)
        if s >= 4:
            return 5
        if s >= 2:
            return 3
        return 0


# ======================================================================
# 主策略类
# ======================================================================
class EnhancedETFStrategy:
    """
    多维共振增强版行业ETF交易策略

    用法:
        st = EnhancedETFStrategy()
        plan = st.generate_trading_plan(scored, market_data, sectors_kline=None)
    集成(generate_dashboard.py):
        plan = generate_plan_from_dashboard(heatmap_data, market_data, sectors_data)
    """

    # 生命周期因子分(0-25): 启动25/发酵20/主升15/高潮5/退潮0, 震荡12/冰点8(扩展)
    LIFECYCLE_SCORE = {'启动期': 25, '发酵期': 20, '主升期': 15, '高潮期': 5,
                       '退潮期': 0, '震荡期': 12, '冰点期': 8}
    # 缠论结构因子分(0-20): 1买20/3买16/中枢下12/中枢内8/中枢上4/1卖0, 2买14/2卖2/3卖0(扩展)
    CHAN_SIGNAL_SCORE = {'1买': 20, '3买': 16, '2买': 14, '2卖': 2, '1卖': 0, '3卖': 0}
    CHAN_ZS_SCORE = {'中枢下': 12, '中枢内': 8, '有中枢': 8, '中枢上': 4, '无中枢': 8}
    # 量学因子分(0-10): 黄金柱支撑10, 将军柱压力2, 普通6, 跌破0(扩展)
    VOL_SCORE = {'黄金柱': 10, '普通': 6, '将军柱': 2, '跌破': 0}

    BUY_SIGNALS = ('1买', '2买', '3买')
    SELL_SIGNALS = ('1卖', '2卖', '3卖')

    def __init__(self, params=None):
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.emotion = EmotionCycleAnalyzer()
        self.policy = PolicyIndustryAnalyzer(self.params.get('policy_db'))
        # use_enhanced=False 或核心引擎缺失 -> fallback基础逻辑
        self.enhanced = bool(self.params.get('use_enhanced', True)) and _HAS_TREND

    # ---------------- 单因子打分 ----------------
    def _score_lifecycle(self, stage):
        return self.LIFECYCLE_SCORE.get(stage, 8)

    @staticmethod
    def _score_mrs(mrs):
        """MRS因子分(0-20): >75=20, 60-75=16, 45-60=12, 30-45=6, <30=2"""
        if mrs > 75: return 20
        if mrs >= 60: return 16
        if mrs >= 45: return 12
        if mrs >= 30: return 6
        return 2

    def _score_chan(self, chan_merged):
        sig = chan_merged['signal']
        if sig in self.CHAN_SIGNAL_SCORE:
            return self.CHAN_SIGNAL_SCORE[sig]
        return self.CHAN_ZS_SCORE.get(chan_merged['zs_pos'], 8)

    def _score_vol(self, vol_merged):
        return self.VOL_SCORE.get(vol_merged['signal'], 6)

    # ---------------- 板块数据标准化 ----------------
    @staticmethod
    def _latest_sector_row(sector_data):
        """接受 scored DataFrame 或 最新值dict, 返回 dict(score, delta5, close, date)"""
        if isinstance(sector_data, pd.DataFrame):
            if len(sector_data) == 0:
                return None
            row = sector_data.iloc[-1]
            return {'score': float(row.get('score', 50)),
                    'delta5': float(row.get('delta5', 0)) if pd.notna(row.get('delta5')) else 0.0,
                    'close': float(row.get('close', np.nan)),
                    'date': row.get('date')}
        if isinstance(sector_data, dict):
            return {'score': float(sector_data.get('score', 50)),
                    'delta5': float(sector_data.get('delta5', sector_data.get('delta', 0)) or 0),
                    'close': float(sector_data.get('close', np.nan))
                    if sector_data.get('close') is not None else np.nan,
                    'date': sector_data.get('date')}
        return None

    def _calc_lifecycle(self, score, delta5):
        if _HAS_TREND:
            return trend_engine.calc_lifecycle(score, delta5)
        # fallback: 简化生命周期
        if score >= 70: return '主升期' if delta5 >= 0 else '高潮期'
        if score >= 55: return '发酵期' if delta5 >= 0 else '退潮期'
        if score >= 40: return '启动期' if delta5 >= 5 else '震荡期'
        return '启动期' if delta5 >= 5 else '冰点期'

    # ---------------- 核心: 评估单个板块 ----------------
    def evaluate_sector(self, sector_name, sector_data, market_data,
                        chan_data=None, vol_data=None):
        """
        评估单个板块
        Args:
            sector_data: scored DataFrame(date/score/delta5/close) 或最新值dict
            market_data: compute_market_trend输出dict(含latest_mrs/mrs序列) 或MRS数值
            chan_data: chan_engine.analyze输出(可选)
            vol_data:  volpillar.analyze_volume输出(可选)
        Returns:
            (total_score, factor_scores, position_pct, action)
            完整明细见 evaluate_sector_detail()
        """
        d = self.evaluate_sector_detail(sector_name, sector_data, market_data,
                                        chan_data, vol_data)
        return d['total_score'], d['factor_scores'], d['position_pct'], d['action']

    def evaluate_sector_detail(self, sector_name, sector_data, market_data,
                               chan_data=None, vol_data=None):
        """评估单个板块, 返回完整交易计划明细dict"""
        row = self._latest_sector_row(sector_data)
        if row is None:
            return self._empty_plan(sector_name)

        mrs, _ = self.emotion._extract_mrs(market_data)
        emotion_phase = self.emotion.determine_phase(market_data)
        stage = self._calc_lifecycle(row['score'], row['delta5'])
        cur_price = row['close'] if not np.isnan(row['close']) else None

        # --- 信号合并(增强模式才用缠论/量学/政策) ---
        if self.enhanced:
            chan_m = merge_chan_signals(chan_data, cur_price,
                                        recent_days=self.params['recent_days'],
                                        ref_date=row.get('date'))
            vol_m = merge_vol_signals(vol_data, cur_price)
            policy_score = self.policy.score_combined(sector_name)
        else:
            chan_m = {'signal': '无', 'zs_pos': '无中枢', 'divergence': '无', 'detail': 'fallback模式'}
            vol_m = {'signal': '普通', 'broke_support': False, 'detail': 'fallback模式'}
            policy_score = 3

        # --- 多因子共振评分(支持score_weights加权) ---
        raw_scores = {
            'lifecycle': self._score_lifecycle(stage),
            'mrs': self._score_mrs(mrs),
            'emotion_cycle': self.emotion.get_phase_score(emotion_phase),
            'chan_structure': self._score_chan(chan_m) if self.enhanced else 8,
            'volume_pillar': self._score_vol(vol_m) if self.enhanced else 6,
            'policy_industry': policy_score if self.enhanced else 3,
        }
        weights = self.params.get('score_weights')
        if weights:
            # 加权归一: total = Σ(wᵢ×sᵢ) / Σ(wᵢ×maxᵢ) × 100
            max_scores = {'lifecycle': 25, 'mrs': 20, 'emotion_cycle': 20,
                          'chan_structure': 20, 'volume_pillar': 10, 'policy_industry': 5}
            weighted_sum = sum(raw_scores[k] * weights.get(k, 1.0) for k in raw_scores)
            weighted_max = sum(max_scores[k] * weights.get(k, 1.0) for k in max_scores)
            total_score = int(round(weighted_sum / weighted_max * 100, 0))
        else:
            total_score = int(sum(raw_scores.values()))
        factor_scores = raw_scores

        signals = {
            'stage': stage, 'emotion_phase': emotion_phase,
            'chan_signal': chan_m['signal'], 'zs_pos': chan_m['zs_pos'],
            'divergence': chan_m['divergence'],
            'vol_signal': vol_m['signal'], 'broke_support': vol_m['broke_support'],
            'policy_score': policy_score, 'delta5': row['delta5'],
        }

        # --- 仓位层级 ---
        level = self.determine_warehouse_level(total_score, signals)

        # --- 仓位百分比: 层级带宽 × MRS总仓位上限 × 动态系数 ---
        weight = self._level_weight(level, total_score)
        base_position = weight * mrs_to_position(mrs)
        if level == '空仓':
            position_pct = 0.0
        else:
            pos = calc_dynamic_position(base_position, emotion_phase,
                                        chan_m['signal'], vol_m['signal'])
            position_pct = round(pos * 100, 1)

        # --- 操作信号 / 出入场理由 / 风险等级 ---
        action = self._determine_action(level, signals)
        entry_reason = self._build_entry_reason(level, signals)
        exit_plan = self._build_exit_plan(level)
        risk_level = self._assess_risk(total_score, level, signals)

        return {
            'sector': sector_name,
            'etf': self._etf_label(sector_name),
            'total_score': total_score,
            'factor_scores': factor_scores,
            'warehouse_level': level,
            'position_pct': position_pct,
            'action': action,
            'entry_reason': entry_reason,
            'exit_plan': exit_plan,
            'risk_level': risk_level,
            'stage': stage,
            'emotion_phase': emotion_phase,
            'chan_detail': chan_m['detail'],
            'vol_detail': vol_m['detail'],
            'policy_theme': self.policy.get_theme(sector_name),
        }

    def _empty_plan(self, sector_name):
        return {'sector': sector_name, 'etf': self._etf_label(sector_name),
                'total_score': 0, 'factor_scores': {}, 'warehouse_level': '空仓',
                'position_pct': 0.0, 'action': '观望', 'entry_reason': '',
                'exit_plan': '', 'risk_level': '高', 'stage': '未知',
                'emotion_phase': '分歧期', 'chan_detail': '', 'vol_detail': '',
                'policy_theme': ''}

    @staticmethod
    def _etf_label(sector_name):
        """板块/赛道 -> '代码 名称'(data_fetcher可用时精确, 否则空)

        先查宽行业SECTOR_ETF_MAP, 未命中再查细分SUB_SECTOR_ETF_MAP
        """
        try:
            import data_fetcher
            # 先查宽行业
            code, name = data_fetcher.SECTOR_ETF_MAP.get(sector_name, ("", ""))
            if code:
                return f"{code} {name}".strip()
            # 再查细分赛道
            sub = data_fetcher.SUB_SECTOR_ETF_MAP.get(sector_name)
            if sub:
                return f"{sub[0]} {sub[1]}".strip()
            return ""
        except Exception:
            return ""

    # ---------------- 仓位层级判定 ----------------
    def determine_warehouse_level(self, score, signals):
        """
        确定仓位层级(正仓/机动仓/观察仓/空仓)
        Args:
            score: 多因子总分
            signals: dict(stage/emotion_phase/chan_signal/vol_signal/
                          broke_support/policy_score/delta5/divergence)
        """
        p = self.params
        stage = signals.get('stage', '震荡期')
        emotion = signals.get('emotion_phase', '分歧期')
        chan_sig = signals.get('chan_signal', '无')
        broke = signals.get('broke_support', False)
        policy = signals.get('policy_score', 0)
        delta = signals.get('delta5', 0)

        # 硬性回避: 退潮期/情绪退潮/卖点信号/跌破黄金柱 -> 不开新仓
        avoid = (stage == '退潮期' or emotion == '退潮期'
                 or chan_sig in self.SELL_SIGNALS or broke)
        if avoid:
            return '空仓'

        # 正仓(道-战略层): 高分 + 生命周期早段 + 政策风口 + 非情绪高潮
        if (score >= p['zheng_score_min']
                and stage in ('启动期', '发酵期', '主升期')
                and policy >= 4
                and emotion not in ('高潮期', '退潮期')
                and chan_sig not in self.SELL_SIGNALS):
            return '正仓'

        # 机动仓(法-战术层): 中高分 + (情绪冰点/修复 或 缠论买点)
        if (score >= p['jidong_score_min']
                and (emotion in ('冰点期', '修复期') or chan_sig in self.BUY_SIGNALS)
                and stage != '高潮期'):
            return '机动仓'

        # 观察仓(术-验证层): 分数一般但板块异动, 或分数达标但因子未共振
        if score >= p['guancha_score_min'] or abs(delta) >= p['anomaly_delta']:
            if stage != '高潮期' and emotion != '高潮期':
                return '观察仓'

        return '空仓'

    def _level_weight(self, level, score):
        """层级内按分数线性映射到仓位带宽"""
        p = self.params
        if level == '正仓':
            lo, hi = p['zheng_band']
            t = min(1.0, max(0.0, (score - p['zheng_score_min']) / (100 - p['zheng_score_min'])))
            return lo + t * (hi - lo)
        if level == '机动仓':
            lo, hi = p['jidong_band']
            span = max(1, p['zheng_score_min'] - p['jidong_score_min'])
            t = min(1.0, max(0.0, (score - p['jidong_score_min']) / span))
            return lo + t * (hi - lo)
        if level == '观察仓':
            lo, hi = p['guancha_band']
            span = max(1, p['jidong_score_min'] - p['guancha_score_min'])
            t = min(1.0, max(0.0, (score - p['guancha_score_min']) / span))
            return lo + t * (hi - lo)
        return 0.0

    # ---------------- 操作信号/理由/风险 ----------------
    def _determine_action(self, level, s):
        if level == '空仓':
            return '观望'
        stage, emotion = s['stage'], s['emotion_phase']
        chan_sig, broke = s['chan_signal'], s['broke_support']
        top_div = s.get('divergence') == '顶背驰'

        if level == '正仓':
            # 退出需三者共振: 生命周期高潮 + 情绪高潮 + 缠论顶背驰/卖点
            votes = int(stage == '高潮期') + int(emotion == '高潮期') \
                + int(top_div or chan_sig in self.SELL_SIGNALS)
            if votes >= 3:
                return '卖出'
            if votes == 2:
                return '止盈'
            return '买入' if chan_sig in self.BUY_SIGNALS else '持有'

        if level == '机动仓':
            # 退出: 情绪高潮 或 缠论1卖/3卖 或 跌破黄金柱
            if chan_sig in self.SELL_SIGNALS or broke or stage == '退潮期':
                return '卖出'
            if emotion == '高潮期' or stage == '高潮期':
                return '止盈'
            return '买入' if (chan_sig in self.BUY_SIGNALS
                              or emotion in ('冰点期', '修复期')) else '持有'

        # 观察仓: 验证失败立即退出, 验证成功升级
        if chan_sig in self.SELL_SIGNALS or broke:
            return '卖出'          # 验证失败
        return '买入'              # 试探性建仓

    @staticmethod
    def _build_entry_reason(level, s):
        if level == '空仓':
            return ''
        parts = []
        if s['stage'] in ('启动期', '发酵期', '主升期'):
            parts.append(s['stage'])
        if s['emotion_phase'] in ('冰点期', '修复期'):
            parts.append('情绪' + s['emotion_phase'].replace('期', ''))
        if s['chan_signal'] in EnhancedETFStrategy.BUY_SIGNALS:
            parts.append(s['chan_signal'] + '信号')
        if s['vol_signal'] == '黄金柱':
            parts.append('黄金柱支撑')
        if s['policy_score'] >= 5:
            parts.append('政策风口+产业链核心')
        elif s['policy_score'] >= 3:
            parts.append('政策面支持')
        if level == '观察仓' and not parts:
            parts.append('板块异动试探')
        return '+'.join(parts) if parts else '多因子部分共振'

    @staticmethod
    def _build_exit_plan(level):
        return {
            '正仓': '高潮期+情绪高潮+缠论顶背驰三者共振退出',
            '机动仓': '情绪高潮或缠论1卖/3卖或跌破黄金柱支撑退出',
            '观察仓': '验证失败立即退出; 验证成功升级为机动仓',
            '空仓': '',
        }[level]

    @staticmethod
    def _assess_risk(score, level, s):
        if (s['stage'] in ('高潮期', '退潮期') or s['emotion_phase'] in ('高潮期', '退潮期')
                or s.get('divergence') == '顶背驰' or s['broke_support']
                or s['chan_signal'] in EnhancedETFStrategy.SELL_SIGNALS):
            return '高'
        if level == '正仓' and score >= 80:
            return '低'
        if score >= 55:
            return '中'
        return '高'

    # ---------------- 完整交易计划 ----------------
    def generate_trading_plan(self, all_sectors, market_data, sectors_kline=None):
        """
        生成完整交易计划
        Args:
            all_sectors: {板块名: scored DataFrame 或 最新值dict}
            market_data: compute_market_trend输出dict 或 MRS数值
            sectors_kline: {板块名: 原始日线DataFrame}(可选, 传入则自动算缠论/量柱)
        Returns:
            dict: plans(按总分降序)/emotion_phase/mrs/total_position_pct/fallback
        """
        mrs, _ = self.emotion._extract_mrs(market_data)
        emotion_phase = self.emotion.determine_phase(market_data)

        plans = []
        for name, sdata in all_sectors.items():
            try:
                chan_data = vol_data = None
                if (self.enhanced and self.params['compute_chan_vol']
                        and sectors_kline is not None and name in sectors_kline):
                    kdf = sectors_kline[name]
                    if _HAS_CHAN and len(kdf) >= 60:
                        chan_data = chan_engine.analyze(
                            kdf.tail(120).reset_index(drop=True), min_gap=4)
                    if _HAS_VOL and len(kdf) >= 30:
                        vol_data = volpillar.analyze_volume(kdf)
                plans.append(self.evaluate_sector_detail(name, sdata, market_data,
                                                         chan_data, vol_data))
            except Exception as e:
                # 单板块异常不拖垮整体, 退回空仓计划
                d = self._empty_plan(name)
                d['chan_detail'] = f'评估异常: {e}'
                plans.append(d)

        plans.sort(key=lambda x: x['total_score'], reverse=True)

        # 总仓位归一: 合计不超过max_total_position
        cap = self.params['max_total_position'] * 100
        total = sum(p['position_pct'] for p in plans)
        if total > cap and total > 0:
            scale = cap / total
            for p in plans:
                p['position_pct'] = round(p['position_pct'] * scale, 1)
            total = cap

        return {
            'plans': plans,
            'emotion_phase': emotion_phase,
            'mrs': round(mrs, 1),
            'total_position_pct': round(sum(p['position_pct'] for p in plans), 1),
            'fallback': not self.enhanced,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        }


# ======================================================================
# 细分赛道专用: 宽行业闸门函数
# ======================================================================
def apply_wide_gate(sub_plans, wide_plans):
    """
    细分赛道信号受宽行业闸门调节

    规则:
      1. 父行业为"空仓/卖出"信号 -> 细分买入降级为观察仓, 仓位×0.5, 标注"父行业退潮压制"
      2. 父行业为"正仓/机动仓持有" -> 细分正常生效, 标注"与宽行业共振"
      3. 父行业正仓+细分也买入 -> 标注"主线共振↑", 看板高亮
      4. 父行业为"—"(光伏/电池等无宽行业) -> 不设闸门, 只受MRS+情绪约束

    Args:
        sub_plans: 细分赛道交易计划列表(EnhancedETFStrategy.generate_trading_plan输出['plans'])
        wide_plans: 宽行业交易计划列表(同上)

    Returns:
        list: 处理后的sub_plans(添加gate_status/gate_note字段)
    """
    # 构建宽行业映射: {板块名: plan}
    wide_map = {p['sector']: p for p in wide_plans}

    for p in sub_plans:
        sector = p['sector']
        # 从data_fetcher获取父行业
        try:
            import data_fetcher
            wide_sector = data_fetcher.SUB_SECTOR_ETF_MAP.get(sector, ('', '', '—', ''))[2]
        except Exception:
            wide_sector = '—'

        p['wide_sector'] = wide_sector

        # 无宽行业 -> 独立运行
        if wide_sector == '—':
            p['gate_status'] = 'independent'
            p['gate_note'] = '独立赛道(无宽行业)'
            continue

        wide_plan = wide_map.get(wide_sector)
        if wide_plan is None:
            p['gate_status'] = 'no_parent'
            p['gate_note'] = f'父行业[{wide_sector}]未评估'
            continue

        wide_action = wide_plan['action']
        wide_level = wide_plan['warehouse_level']

        # 规则1: 父行业空仓/卖出 -> 压制
        if wide_level == '空仓' or wide_action in ('卖出', '观望'):
            if p['action'] == '买入':
                p['action'] = '观望'
                p['position_pct'] = round(p['position_pct'] * 0.5, 1)
                p['gate_status'] = 'suppressed'
                p['gate_note'] = f'父行业[{wide_sector}]退潮压制, 买入降级'
            else:
                p['gate_status'] = 'suppressed'
                p['gate_note'] = f'父行业[{wide_sector}]退潮, 同步回避'

        # 规则2/3: 父行业持有/买入 -> 共振
        elif wide_action in ('买入', '持有', '止盈'):
            if p['action'] == '买入' and wide_action == '买入':
                p['gate_status'] = 'resonance_up'
                p['gate_note'] = f'主线共振↑: 父行业[{wide_sector}]同步买入'
            else:
                p['gate_status'] = 'resonance'
                p['gate_note'] = f'与宽行业[{wide_sector}]共振'

        else:
            p['gate_status'] = 'neutral'
            p['gate_note'] = f'父行业[{wide_sector}]中性'

    return sub_plans


# ======================================================================
# 与 generate_dashboard.py 集成的入口
# ======================================================================
def generate_plan_from_dashboard(heatmap_data, market_data, sectors_data=None, params=None):
    """
    直接消费 generate_dashboard.py 的中间产物生成增强版交易计划
    Args:
        heatmap_data: compute_sector_heatmap()输出(用其 'scored' 字段)
        market_data:  compute_market_trend()输出
        sectors_data: data_fetcher.get_all_sectors_data()原始K线(可选, 用于缠论/量柱)
        params:       策略参数覆盖(可选); params['use_enhanced']=False 退回基础策略
    """
    if not heatmap_data or 'scored' not in heatmap_data:
        return {'plans': [], 'emotion_phase': '分歧期', 'mrs': 50.0,
                'total_position_pct': 0.0, 'fallback': True,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M')}
    st = EnhancedETFStrategy(params)
    return st.generate_trading_plan(heatmap_data['scored'], market_data,
                                    sectors_kline=sectors_data)


def generate_sub_sector_plan(sub_heatmap_data, market_data, sub_sectors_data=None,
                             wide_plans=None, params=None):
    """
    生成细分赛道交易计划(带宽行业闸门)

    Args:
        sub_heatmap_data: 细分赛道compute_sector_heatmap()输出
        market_data:      compute_market_trend()输出
        sub_sectors_data: data_fetcher.get_sub_sector_data()原始K线(可选)
        wide_plans:       宽行业交易计划列表(用于闸门调节, 可选)
        params:           策略参数覆盖(默认用SUB_SECTOR_PARAMS)

    Returns:
        dict: plans(带gate_status)/emotion_phase/mrs/total_position_pct/fallback
    """
    if not sub_heatmap_data or 'scored' not in sub_heatmap_data:
        return {'plans': [], 'emotion_phase': '分歧期', 'mrs': 50.0,
                'total_position_pct': 0.0, 'fallback': True,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M')}

    # 使用细分赛道专用参数
    sub_params = dict(SUB_SECTOR_PARAMS)
    if params:
        sub_params.update(params)

    st = EnhancedETFStrategy(sub_params)
    result = st.generate_trading_plan(sub_heatmap_data['scored'], market_data,
                                      sectors_kline=sub_sectors_data)

    # 应用宽行业闸门
    if wide_plans and not result.get('fallback'):
        result['plans'] = apply_wide_gate(result['plans'], wide_plans)
        # 闸门可能改变了仓位, 重新归一
        cap = st.params['max_total_position'] * 100
        total = sum(p['position_pct'] for p in result['plans'])
        if total > cap and total > 0:
            scale = cap / total
            for p in result['plans']:
                p['position_pct'] = round(p['position_pct'] * scale, 1)
        result['total_position_pct'] = round(sum(p['position_pct'] for p in result['plans']), 1)

    return result


# ======================================================================
# 独立运行测试(合成数据, 无需网络)
# ======================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("  多维共振增强版ETF策略 - 独立测试(合成数据)")
    print("=" * 70)

    # --- 1. 构造合成市场数据(MRS修复期: 低位回升) ---
    mrs_list = [28, 29, 30, 31, 33, 34, 36, 37, 38, 40]
    market_data = {'latest_mrs': 40.0, 'mrs': mrs_list,
                   'mrs_smooth': pd.Series(mrs_list).rolling(3, min_periods=1).mean().round(1).tolist()}

    # --- 2. 构造合成板块得分数据 ---
    def make_scored(score, delta5, close):
        dates = pd.date_range('2026-07-20', periods=20, freq='B')
        scores = np.linspace(score - delta5, score, 20)
        return pd.DataFrame({'date': dates, 'score': scores,
                             'delta5': [np.nan] * 15 + [delta5] * 5,
                             'close': np.linspace(close * 0.95, close, 20)})

    sectors = {
        '银行':   make_scored(52, 8, 1000),    # 启动期
        '军工':   make_scored(60, 6, 2000),    # 发酵期
        '计算机': make_scored(80, -4, 3000),   # 高潮期
        '医药':   make_scored(35, 1, 1500),    # 冰点期
    }

    # --- 3. 构造合成缠论/量柱信号(模拟chan_engine/volpillar输出结构) ---
    today = pd.Timestamp('2026-08-14')
    chan_map = {
        '银行': {'trade_points': [{'date': (today - pd.Timedelta(days=3)).strftime('%Y-%m-%d'),
                                   'price': 980, 'label': '1买', 'type': 'bottom'}],
                 'zhongshu': [{'zg': 1050, 'zd': 990}], 'bottom_div': True, 'top_div': False},
        '军工': {'trade_points': [{'date': (today - pd.Timedelta(days=2)).strftime('%Y-%m-%d'),
                                   'price': 1980, 'label': '3买', 'type': 'bottom'}],
                 'zhongshu': [{'zg': 1950, 'zd': 1900}], 'bottom_div': False, 'top_div': False},
        '计算机': {'trade_points': [{'date': (today - pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                                     'price': 3050, 'label': '1卖', 'type': 'top'}],
                   'zhongshu': [{'zg': 2900, 'zd': 2800}], 'bottom_div': False, 'top_div': True},
        '医药': {'trade_points': [], 'zhongshu': [], 'bottom_div': False, 'top_div': False},
    }
    vol_map = {
        '银行': {'golden': [{'date': (today - pd.Timedelta(days=5)).strftime('%Y-%m-%d'),
                             'top': 990, 'bottom': 960, 'label': '黄金柱'}],
                 'summary': {'support_txt': '黄金柱支撑区间[960-990]'}},
        '军工': {'golden': [{'date': (today - pd.Timedelta(days=6)).strftime('%Y-%m-%d'),
                             'top': 1960, 'bottom': 1920, 'label': '黄金柱'}],
                 'summary': {'support_txt': '黄金柱支撑区间[1920-1960]'}},
        '计算机': {'golden': [{'date': (today - pd.Timedelta(days=4)).strftime('%Y-%m-%d'),
                               'top': 3100, 'bottom': 3000, 'label': '将军柱'}],
                   'summary': {'support_txt': '将军柱柱顶3100压力'}},
        '医药': {'golden': [], 'summary': {'support_txt': '近期无黄金柱支撑'}},
    }

    # --- 4. 逐板块评估 + 生成交易计划 ---
    st = EnhancedETFStrategy({'compute_chan_vol': False})
    print(f"\n情绪周期阶段: {st.emotion.determine_phase(market_data)} "
          f"(MRS={market_data['latest_mrs']})")
    print(f"策略模式: {'增强版(多维共振)' if st.enhanced else 'fallback(基础MRS逻辑)'}\n")

    header = f"{'板块':<5}{'层级':<5}{'总分':<5}{'仓位%':<7}{'操作':<5}{'风险':<4}入场理由"
    print(header)
    print("-" * 70)
    results = []
    for name, sdf in sectors.items():
        detail = st.evaluate_sector_detail(name, sdf, market_data,
                                           chan_map.get(name), vol_map.get(name))
        results.append(detail)
        print(f"{name:<6}{detail['warehouse_level']:<6}{detail['total_score']:<6}"
              f"{detail['position_pct']:<8}{detail['action']:<6}{detail['risk_level']:<5}"
              f"{detail['entry_reason']}")
        print(f"       因子: {detail['factor_scores']}")
        print(f"       退出: {detail['exit_plan']} | 缠论: {detail['chan_detail']} | 量柱: {detail['vol_detail']}")

    # --- 5. generate_trading_plan 整体测试 ---
    print("\n" + "=" * 70)
    print("  generate_trading_plan 输出(计划汇总)")
    print("=" * 70)
    # 把chan/vol挂到scored上行不通, 这里直接用evaluate覆盖; 整体接口用无缠论数据演示
    plan = st.generate_trading_plan(sectors, market_data)
    print(f"情绪阶段={plan['emotion_phase']} MRS={plan['mrs']} "
          f"建议总仓位={plan['total_position_pct']}% fallback={plan['fallback']}")
    for p in plan['plans']:
        print(f"  {p['sector']:<5} {p['warehouse_level']:<4} {p['total_score']:>3}分 "
              f"{p['position_pct']:>5}% {p['action']:<3} {p['entry_reason']}")

    # --- 6. 单元验证: 仓位动态调整 & 信号合并 ---
    print("\n" + "=" * 70)
    print("  单元验证")
    print("=" * 70)
    pos = calc_dynamic_position(0.30, '修复', '3买', '黄金柱')
    print(f"calc_dynamic_position(0.30, 修复, 3买, 黄金柱) = {pos:.3f} (期望≈0.375)")
    pos2 = calc_dynamic_position(0.60, '退潮', '3卖', '将军柱')
    print(f"calc_dynamic_position(0.60, 退潮, 3卖, 将军柱) = {pos2:.3f} (期望≈0.270)")
    cm = merge_chan_signals(chan_map['银行'], current_price=1000, ref_date=today)
    print(f"merge_chan_signals(银行) = {cm['signal']}/{cm['zs_pos']} (期望 1买/中枢内)")
    vm = merge_vol_signals(vol_map['银行'], current_price=1000)
    print(f"merge_vol_signals(银行, 1000) = {vm['signal']} (期望 黄金柱)")
    vm2 = merge_vol_signals(vol_map['银行'], current_price=950)
    print(f"merge_vol_signals(银行, 950) = {vm2['signal']}/broke={vm2['broke_support']} (期望 跌破/True)")

    # --- 7. fallback模式测试 ---
    st_fb = EnhancedETFStrategy({'use_enhanced': False})
    s, fs, pct, act = st_fb.evaluate_sector('银行', sectors['银行'], market_data)
    print(f"\nfallback模式: 银行 score={s} pos={pct}% action={act} (仅用MRS+生命周期)")
    print("\n测试完成。")

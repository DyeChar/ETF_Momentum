#!/usr/bin/env python3
"""
每日监控 — 多策略信号推送
==========================
策略1: 大类资产ETF动量轮动（4 ETF选最强，全部≤0 → 511880避险）
策略2: 中证2000ETF择时（慢/快动量三条件出场，出场→511880）
策略3: 高股息率跟踪（分红历史 + TTM股息率）

数据源：新浪财经API
同步自:
  - 1.中证2000ETF择时/backtest.py（择时出场逻辑）
  - 2.大类资产ETF轮动策略/backtest.py（轮动选股逻辑）
  - 3.高股息率跟踪/（分红解析 + 股息率计算）
"""

import json
import math
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ============================================================
# 策略1: 大类资产ETF动量轮动
# 同步自: 2.大类资产ETF轮动策略/config.py + backtest.py
# ============================================================
ETF_POOL = {
    'sh518880': '黄金ETF',      # 华安黄金ETF, 2013-07-29上市
    'sh513100': '纳指ETF',      # 国泰纳斯达克100ETF, 2013-05-28上市
    'sh510300': '沪深300ETF',   # 华泰柏瑞沪深300ETF, 2012-05-28上市
    'sz159915': '创业板ETF',    # 易方达创业板ETF, 2011-09-20上市
}
LOOKBACK_DAYS = 25              # 动量窗口（交易日）
CASH_THRESHOLD = 0.0            # 全部ETF动量 ≤ 此值 → 持有银华日利避险
CASH_ETF_CODE = 'sh511880'      # 银华日利（货币ETF）
CASH_ETF_NAME = '银华日利'

# ============================================================
# 策略2: 中证2000ETF择时
# 同步自: 1.中证2000ETF择时/config.py + backtest.py
# ============================================================
STOP_LOSS_TARGET = 'sh563300'       # 中证2000ETF（华泰柏瑞，563300）
STOP_LOSS_TARGET_NAME = '中证2000ETF'
STOP_LOSS_BENCHMARKS = {
    'sz159919': '沪深300ETF',       # 嘉实沪深300ETF
    'sh000015': '红利指数',          # 上证红利指数
    'sh510050': '上证50ETF',        # 华夏上证50ETF
}
STOP_LOSS_LOOKBACK_DAYS = 20        # 慢动量窗口（交易日）
FAST_LOOKBACK_DAYS = 10             # 快动量窗口（交易日）
FAST_MOMENTUM_THRESHOLD = 0.4       # 快动量阈值（<此值满足清仓条件）
MOMENTUM_THRESHOLD = 2.0            # 慢动量高位阈值（>此值不触发止损）

# ============================================================
# 通用配置
# ============================================================
HISTORY_FILE = 'history.json'       # 历史信号记录


def get_sina_kline(code: str, days: int = 30) -> list:
    """
    从新浪财经API获取ETF日K线数据
    code格式: sh510300 或 sz159915
    返回收盘价列表
    """
    url = 'https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData'
    params = {
        'symbol': code,
        'scale': '240',  # 日K线
        'datalen': days,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        if not data or len(data) < LOOKBACK_DAYS:
            return None

        # 提取收盘价
        prices = [float(item['close']) for item in data]
        return prices

    except Exception as e:
        print(f"获取 {code} 失败: {e}")
        return None


def calculate_score(prices: list, window: int = None) -> tuple:
    """计算动量得分"""
    if window is None:
        window = LOOKBACK_DAYS
    prices = prices[-window:]
    if len(prices) < window:
        return None, None, None

    y = np.log(prices)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)

    annualized_returns = math.pow(math.exp(slope), 250) - 1

    y_pred = slope * x + intercept
    ss_res = sum((y - y_pred) ** 2)
    ss_tot = (len(y) - 1) * np.var(y, ddof=1)
    r_squared = 1 - (ss_res / ss_tot)

    score = annualized_returns * r_squared
    return score, annualized_returns, r_squared


def load_history() -> dict:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_history(data: dict):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compare_with_last(current_ranking: list, history: dict) -> dict:
    today = datetime.now().strftime('%Y-%m-%d')
    last_record = history.get('last_result', {})
    last_date = last_record.get('date', '')
    last_ranking = last_record.get('ranking', [])

    changes = {
        'is_new': len(last_ranking) == 0,
        'top_changed': False,
        'ranking_changed': False,
        'changes_detail': []
    }

    if not changes['is_new'] and last_ranking:
        if current_ranking[0]['code'] != last_ranking[0]['code']:
            changes['top_changed'] = True
            changes['changes_detail'].append(
                f"首位变动: {last_ranking[0]['name']} → {current_ranking[0]['name']}"
            )

        current_order = [r['code'] for r in current_ranking]
        last_order = [r['code'] for r in last_ranking]
        if current_order != last_order:
            changes['ranking_changed'] = True
            for i, (curr, last) in enumerate(zip(current_ranking, last_ranking)):
                if curr['code'] != last['code']:
                    changes['changes_detail'].append(
                        f"第{i+1}位: {last['name']} → {curr['name']}"
                    )

    return changes


def format_output(ranking: list, changes: dict, recommend_cash: bool = False) -> str:
    """格式化策略1输出（大类资产ETF轮动）。"""
    # 变动文本（仅在首次或有变动时显示）
    change_tag = ""
    if changes['is_new']:
        change_tag = "，首次运行"
    elif changes['top_changed'] or changes['ranking_changed']:
        detail_str = "；".join(changes['changes_detail'])
        change_tag = f"，⚠️ {detail_str}"

    if recommend_cash:
        title = f"**📊 大类资产ETF动量轮动 👉 {CASH_ETF_NAME} ({CASH_ETF_CODE[2:]})，🛡️ 避险{change_tag}**"
    else:
        top_etf = ranking[0]
        pure_code = top_etf['code'][2:]
        title = f"**📊 大类资产ETF动量轮动 👉 {top_etf['name']} ({pure_code}){change_tag}**"

    lines = [title]

    # 当前排序（单行紧凑）
    ranking_parts = []
    for i, r in enumerate(ranking):
        symbol = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else ""))
        ranking_parts.append(f"{symbol} {r['name']}: {r['score']:.4f}")
    lines.append(f"【当前排序】 {' '.join(ranking_parts)}")

    return "\n\n".join(lines)


def check_stop_loss_signal() -> dict:
    """
    中证2000ETF择时 — 慢/快动量三条件出场监控。
    同步自: 1.中证2000ETF择时/backtest.py (line 229-510)

    对 sh563300(中证2000ETF) + 3个基准指数计算慢动量(20日)和快动量(10日)。

    出场条件（全部满足 → 卖出563300，买入511880避险）：
      1. 563300 慢动量(20日) 在4个指数中不排第一
      2. 563300 快动量(10日) < FAST_MOMENTUM_THRESHOLD(0.4)
      3. 563300 慢动量(20日) < MOMENTUM_THRESHOLD(2.0)（未处于高位）

    返回: {target, benchmarks, is_rank1, trigger}
    """
    W_SLOW = STOP_LOSS_LOOKBACK_DAYS
    W_FAST = FAST_LOOKBACK_DAYS
    # 数据天数取快/慢窗口最大值 + 缓冲
    fetch_days = max(W_SLOW, W_FAST) + 5

    # 1. 获取中证2000ETF数据
    target_prices = get_sina_kline(STOP_LOSS_TARGET, fetch_days)

    # 2. 获取各基准指数数据（慢动量排名用）
    benchmark_results = {}
    for code, name in STOP_LOSS_BENCHMARKS.items():
        prices = get_sina_kline(code, fetch_days)
        if prices and len(prices) >= W_SLOW:
            score, ann_ret, r2 = calculate_score(prices[-W_SLOW:], W_SLOW)
            benchmark_results[code] = {
                'name': name,
                'score': score,
                'annualized_return': ann_ret,
                'r_squared': r2,
            }
        else:
            benchmark_results[code] = {
                'name': name,
                'score': None,
                'annualized_return': None,
                'r_squared': None,
            }

    # 3. 计算目标ETF慢动量(20日)和快动量(10日)
    target_slow_score = None
    target_slow_ann = None
    target_slow_r2 = None
    target_fast_score = None

    if target_prices and len(target_prices) >= W_SLOW:
        target_slow_score, target_slow_ann, target_slow_r2 = calculate_score(
            target_prices[-W_SLOW:], W_SLOW
        )
    if target_prices and len(target_prices) >= W_FAST:
        target_fast_score, _, _ = calculate_score(
            target_prices[-W_FAST:], W_FAST
        )

    # 4. 判断目标ETF慢动量是否排第一
    all_scores = []
    for info in benchmark_results.values():
        if info['score'] is not None:
            all_scores.append(info['score'])
    if target_slow_score is not None:
        all_scores.append(target_slow_score)

    max_score = max(all_scores) if all_scores else 0
    is_rank1 = (target_slow_score is not None and target_slow_score >= max_score)

    # 5. 止损信号判断：慢动量非第一 & 快动量<0.4 & 慢动量<2.0
    trigger = False
    if target_slow_score is not None and target_fast_score is not None:
        trigger = (
            not is_rank1
            and target_fast_score < FAST_MOMENTUM_THRESHOLD
            and target_slow_score < MOMENTUM_THRESHOLD
        )

    return {
        'target': {
            'code': STOP_LOSS_TARGET,
            'name': STOP_LOSS_TARGET_NAME,
            'slow_score': target_slow_score,
            'fast_score': target_fast_score,
            'slow_annualized_return': target_slow_ann,
            'slow_r_squared': target_slow_r2,
        },
        'benchmarks': benchmark_results,
        'is_rank1': is_rank1,
        'trigger': trigger,
    }


def format_stop_loss_section(result: dict, last_trigger: bool = None) -> str:
    """格式化策略2输出（中证2000ETF择时）。"""
    target = result['target']

    # 标题 + 建议持仓合并一行
    target_pure = STOP_LOSS_TARGET[2:]
    changed = (last_trigger is not None and result['trigger'] != last_trigger)
    change_tag = "，⚠️ 信号变动！" if changed else ""
    if result['trigger']:
        title = f"**🛡️ 中证2000ETF择时 👉 空仓（{CASH_ETF_NAME} {CASH_ETF_CODE[2:]}）{change_tag}**"
    else:
        title = f"**🛡️ 中证2000ETF择时 👉 {STOP_LOSS_TARGET_NAME}（{target_pure}）{change_tag}**"

    lines = [title]

    # 慢动量排序
    all_items = []
    if target['slow_score'] is not None:
        all_items.append((target['code'], target['name'], target['slow_score']))
    for code, info in result['benchmarks'].items():
        if info['score'] is not None:
            all_items.append((code, info['name'], info['score']))
    all_items.sort(key=lambda x: x[2], reverse=True)

    ranking_parts = []
    for i, (code, name, score) in enumerate(all_items):
        prefix = ["🥇","🥈","🥉"][i] if i < 3 else ""
        ranking_parts.append(f"{prefix} {name}: {score:.4f}")
    lines.append(f"【慢动量排序】 {' '.join(ranking_parts)}")

    # 风控条件
    cond1 = not result['is_rank1']
    c1_detail = f"排第{[i+1 for i,(c,_,_) in enumerate(all_items) if c==target['code']][0] if not result['is_rank1'] else 1}名" if all_items else ""
    c1 = f"① 慢动量不排第一: 🔴 ({c1_detail})" if cond1 else f"① 慢动量不排第一: 🟢 (排第1名)"

    fast_score = target['fast_score']
    cond2 = (fast_score is not None and fast_score < FAST_MOMENTUM_THRESHOLD)
    c2 = f"② 快动量 < {FAST_MOMENTUM_THRESHOLD}: {'🔴' if cond2 else '🟢'} (快={fast_score:.4f})" if fast_score is not None else f"② 快动量 < {FAST_MOMENTUM_THRESHOLD}: -"

    slow_score = target['slow_score']
    cond3 = (slow_score is not None and slow_score < MOMENTUM_THRESHOLD)
    c3 = f"③ 慢动量 < {MOMENTUM_THRESHOLD}: {'🔴' if cond3 else '🟢'} (慢={slow_score:.4f})" if slow_score is not None else f"③ 慢动量 < {MOMENTUM_THRESHOLD}: -"

    all_triggered = cond1 and cond2 and cond3
    status = "🔴 止损" if all_triggered else "🟢 正常"
    lines.append(f"【风控】{status}，{c1} {c2} {c3}")

    return "\n\n".join(lines)


# ============================================================
# 策略3: 高股息率跟踪 — TTM股息率
# 同步自: 3.高股息率跟踪/
# ============================================================
FETCH_RETRY = 3
FETCH_RETRY_DELAY = 1.0
DIVIDEND_STOCKS = [
    # 银行
    ("601398","工商银行"),("601939","建设银行"),("600036","招商银行"),
    # 能源/公用/上市一直分红
    ("601088","中国神华"),("600900","长江电力"),("600941","中国移动"),("600025","华能水电"),
    # 交通运输
    ("601006","大秦铁路"),("600377","宁沪高速"),("000429","粤高速A"),("601816","京沪高铁"),
    # 白酒
    ("600519","贵州茅台"),("000858","五粮液"),("000568","泸州老窖"),
    # 消费
    ("600887","伊利股份"),
    # 制造
    ("000333","美的集团"),("600690","海尔智家"),
    # 工业/材料/基建
    ("601668","中国建筑"),("601390","中国中铁"),("601186","中国铁建"),
    # 金融/保险
    ("601318","中国平安"),
    # ETF
    ("563020","易方达红利低波"),
]
DIV_BATCH_DELAY = 0.3
DIV_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dividend_cache.parquet")

# 重点关注标的（微信推送中加粗显示）
HIGHLIGHT_CODES = {"600036", "000333", "600941", "601318", "601668", "601088", "600900"}
# 招商银行  美的集团  中国移动  中国平安  中国建筑  中国神华  长江电力

# Sina HTML 列名（MultiIndex）
COL_ANNOUNCE = ("分红", "公告日期", "公告日期")
COL_DIV10 = ("分红", "分红方案(每10股)", "派息(税前)(元)")
COL_PROGRESS = ("分红", "进度", "进度")
COL_EX = ("分红", "除权除息日", "除权除息日")

# ── 期数推断 + 状态枚举 ─────────────────────────────
# Sina 分红页无「期数」字段，用公告月份推断。
# 同一笔分红的预案→实施公告日差1-3个月，自然落入相邻月份区间 → 同一 period；
# 真实的多笔分红（中期/末期/特别）公告日差>4个月，落入不同区间。
def infer_period(month: int) -> str:
    """按公告月份推断分红期数。"""
    if month in (3, 4):        return "final"      # 年报季 → 年度末期预案
    elif month in (5, 6, 7):   return "final"      # 末期实施（预案发布后1-3月）
    elif month in (8, 9, 10):  return "interim"    # 中期（半年报后）
    else:                      return "special"    # 11-2月 跨年/特别分红

STATUS_RANK = {"cancelled": 1, "proposal": 2, "implemented": 3}
STATUS_MAP = {
    "实施": "implemented",
    "预案": "proposal",
    "取消": "cancelled",
    "停止": "cancelled",
}


def _aggregate_events(rows: list) -> list:
    """按 event_id 状态机聚合：保留最高状态、同状态保留最新公告。

    rows: 含 event_id / status / announce_date 的 dict 列表。
    返回每个 event_id 仅一条的最终列表。
    """
    best = {}
    for r in rows:
        eid = r["event_id"]
        if eid not in best:
            best[eid] = r
        elif STATUS_RANK[r["status"]] > STATUS_RANK[best[eid]["status"]]:
            best[eid] = r      # 实施覆盖预案
        elif STATUS_RANK[r["status"]] == STATUS_RANK[best[eid]["status"]]:
            if r["announce_date"] >= best[eid]["announce_date"]:
                best[eid] = r  # 同状态保留公告日期更新者
        # 状态优先级更低 → 保留缓存中的高优先级记录
    return list(best.values())

# ── FY 覆写表 ─────────────────────────────────────────
# 对 ≤8 规则无法正确推断的个别分红手动指定财年。
# 支持两种 key：
#   精确匹配: (code, "YYYY-MM-DD", dividend_per_10_share) → correct_fy
#   金额匹配: (code, dividend_per_share) → correct_fy  （适用于重复出现的模式）
# 8月是歧义月：中期应为FY=当年，末期应为FY=上年，≤8统一判FY=上年。
# 若某股票有规律8月中期分红（如同金额多年出现），用金额匹配一劳永逸。
# ──────────────────────────────────────────────────────
FY_OVERRIDE = {
    # -- 金额匹配（8月中期分红，每年同金额）--
    ("600941", 2.5025): 2025,  # 中国移动 FY2025中期
    ("600941", 2.3789): 2024,  # 中国移动 FY2024中期
    ("600941", 2.2247): 2023,  # 中国移动 FY2023中期
    ("600941", 1.8942): 2022,  # 中国移动 FY2022中期
    ("601728", 0.1812): 2025,  # 中国电信 FY2025中期
    ("601728", 0.1432): 2023,  # 中国电信 FY2023中期
    # -- 精确匹配（一次性修正）--
}


# ── 分红缓存（单文件 parquet）──────────────────────────
#
# 分红是低频事件，所有股票共用一个缓存文件。
# 列：code, ex_date, fiscal_year, dividend_per_share, announce_date, status
#
# 日常运行：
#   1. 从 Sina 抓取 HTML（1 次请求/股，快）
#   2. 对比缓存中的 announce_date，新公告按 ≤8 规则推断 FY
#   3. 检查 FY_OVERRIDE 覆写，追加到缓存
#   4. 缓存中的 FY 一旦写入不再变动
# ──────────────────────────────────────────────────────


def _load_dividend_cache():
    """加载全量分红缓存。"""
    if os.path.exists(DIV_CACHE_FILE):
        try:
            df = pd.read_parquet(DIV_CACHE_FILE)
            for col in ["ex_date", "announce_date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["code", "ex_date", "fiscal_year",
                                  "dividend_per_share", "announce_date", "status"])


def _save_dividend_cache(df):
    """保存全量分红缓存。"""
    df.to_parquet(DIV_CACHE_FILE, index=False)


def _fetch_dividend_html(code):
    """从 Sina 获取分红配股页面 HTML 文本。"""
    url = f"https://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/{code}.phtml"
    h = {"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"}
    for attempt in range(FETCH_RETRY):
        try:
            r = requests.get(url, headers=h, timeout=15)
            for enc in ["gb2312", "gbk", "gb18030", "utf-8"]:
                try:
                    t = r.content.decode(enc, errors="replace")
                    if "分红" in t or "派息" in t:
                        return t
                except Exception:
                    continue
            return r.content.decode("gbk", errors="replace")
        except Exception:
            if attempt < FETCH_RETRY - 1:
                time.sleep(FETCH_RETRY_DELAY * (2 ** attempt))
    return None


def _parse_dividend_html(html_text):
    """从 Sina HTML 解析分红记录列表。返回 list[dict]"""
    from io import StringIO
    try:
        tables = pd.read_html(StringIO(html_text))
    except Exception:
        return []

    target = None
    for t in tables:
        if any("派息" in str(c) for c in t.columns.tolist()):
            target = t
            break
    if target is None or target.empty:
        return []

    try:
        a_raw = target[COL_ANNOUNCE]
        d_raw = target[COL_DIV10]
        prog = target[COL_PROGRESS]
        e_raw = target[COL_EX]
    except KeyError:
        return []

    records = []
    for i in range(len(target)):
        try:
            status = STATUS_MAP.get(str(prog.iloc[i]).strip())
            if status is None:
                continue
            dv = float(d_raw.iloc[i])
            if dv <= 0:
                continue
            an_s = str(a_raw.iloc[i]).strip()
            ad = pd.Timestamp(an_s) if an_s not in ("--", "", "nan", "NaT") else pd.NaT
            if pd.isna(ad):
                continue
            es = str(e_raw.iloc[i]).strip()
            ex = pd.Timestamp(es) if es not in ("--", "", "nan", "NaT") else pd.NaT
            records.append({
                "announce_date": ad,
                "ex_date": ex if pd.notna(ex) else pd.NaT,
                "dividend_per_10": dv,  # 每10股
                "status": status,       # implemented / proposal / cancelled
                "period": infer_period(ad.month),  # final / interim / special
            })
        except (ValueError, TypeError):
            continue
    return records


def _infer_fy(code, ad, dv10, amount):
    """FY推断：覆写优先（精确匹配 > 金额匹配），否则≤8规则。"""
    exact_key = (code, ad.strftime("%Y-%m-%d"), dv10)
    amount_key = (code, amount)
    if exact_key in FY_OVERRIDE:
        return FY_OVERRIDE[exact_key]
    if amount_key in FY_OVERRIDE:
        return FY_OVERRIDE[amount_key]
    return ad.year - 1 if ad.month <= 8 else ad.year


def _get_dividend_data(code):
    """
    获取单只股票完整分红历史（缓存 + Sina 全量，按 event_id 状态机合并）。

    返回 DataFrame: ex_date, fiscal_year, dividend_per_share, announce_date,
                    status, event_id, period
    """
    full_cache = _load_dividend_cache()
    stock_cache = full_cache[full_cache["code"] == code] if not full_cache.empty else full_cache

    # 从 Sina 抓取全量（页面一次返回全部历史，无分页）
    html = _fetch_dividend_html(code)
    sina_records = _parse_dividend_html(html) if html else []

    # ── 合并候选：缓存旧记录（自带 event_id）+ Sina 新记录（现算）──
    combined = []
    if not stock_cache.empty:
        for _, r in stock_cache.iterrows():
            ad = pd.Timestamp(r["announce_date"])
            period = r.get("period", "") or infer_period(ad.month)
            event_id = r.get("event_id", "") or f"{code}_{int(r['fiscal_year'])}_{period}"
            combined.append({
                "event_id": event_id,
                "period": period,
                "fiscal_year": int(r["fiscal_year"]) if pd.notna(r["fiscal_year"]) else 0,
                "dividend_per_share": round(float(r["dividend_per_share"]), 4),
                "announce_date": ad,
                "ex_date": r["ex_date"],
                "status": r["status"],
            })

    for rec in sina_records:
        ad = rec["announce_date"]
        dv10 = rec["dividend_per_10"]
        amount = round(dv10 / 10.0, 4)
        fy = _infer_fy(code, ad, dv10, amount)
        combined.append({
            "event_id": f"{code}_{fy}_{rec['period']}",
            "period": rec["period"],
            "fiscal_year": fy,
            "dividend_per_share": amount,
            "announce_date": ad,
            "ex_date": rec["ex_date"],
            "status": rec["status"],
        })

    # ── 按 event_id 状态机聚合（预案→实施覆盖 / 同金额不同期数共存）──
    aggregated = _aggregate_events(combined)

    if aggregated:
        new_df = pd.DataFrame(aggregated)
        new_df["code"] = code
        for col in ["ex_date", "announce_date"]:
            if col in new_df.columns:
                new_df[col] = pd.to_datetime(new_df[col], errors="coerce")
        # 从 full_cache 移除该股票旧行，写入聚合结果
        full_cache = full_cache[full_cache["code"] != code] if not full_cache.empty else full_cache
        full_cache = pd.concat([full_cache, new_df], ignore_index=True)
        full_cache = full_cache.sort_values(
            ["code", "fiscal_year", "announce_date"], ascending=[True, False, False]
        ).reset_index(drop=True)
        _save_dividend_cache(full_cache)

    # 返回该股票数据
    stock = full_cache[full_cache["code"] == code] if not full_cache.empty else full_cache
    cols = ["ex_date", "fiscal_year", "dividend_per_share", "announce_date",
            "status", "event_id", "period"]
    if stock.empty:
        return pd.DataFrame(columns=cols)
    return stock[cols].copy()


def _consecutive_dividend_years(df):
    if df.empty: return 0
    years = set()
    for _,row in df.iterrows():
        try:
            if float(row["dividend_per_share"])>0:
                years.add(pd.Timestamp(row["ex_date"]).year)
        except: continue
    if not years: return 0
    mx = max(years); c = 0
    for y in range(mx,mx-50,-1):
        if y in years: c+=1
        else: break
    return c


def _compute_one(df, price, as_of_ts):
    nan = float("nan")
    empty = {"consecutive_years":0,"fy_dividend":0.0,"fy_detail":"","fy_yield":nan,
             "div_5y":0.0,"div_3y":0.0,"div_1y":0.0,"ttm_yield":nan,"fiscal_year":None}
    if df.empty: return empty
    df = df.copy(); df["ex_date"] = pd.to_datetime(df["ex_date"],errors="coerce")
    df["announce_date"] = pd.to_datetime(df["announce_date"],errors="coerce")
    # 1. 过滤取消记录
    if "status" in df.columns:
        df = df[df["status"] != "cancelled"]
    # 2. 按 event_id 去重，保留最高状态（implemented > proposal）
    if "event_id" in df.columns:
        df["_status_rank"] = df["status"].map(STATUS_RANK).fillna(0)
        df = (df.sort_values(["_status_rank", "announce_date"])
              .drop_duplicates("event_id", keep="last")
              .drop(columns=["_status_rank"]))
    if df.empty: return empty
    # 预案无除权日→排除出TTM但保留财年统计
    df_paid = df.dropna(subset=["ex_date"])

    cy = _consecutive_dividend_years(df_paid if not df_paid.empty else df)
    c1 = as_of_ts-pd.DateOffset(months=12)
    c3 = as_of_ts-pd.DateOffset(months=36)
    c5 = as_of_ts-pd.DateOffset(months=60)
    d1 = round(float(df_paid.loc[df_paid["ex_date"]>=c1,"dividend_per_share"].sum()) if not df_paid.empty else 0,4)
    d3 = round(float(df_paid.loc[df_paid["ex_date"]>=c3,"dividend_per_share"].sum()) if not df_paid.empty else 0,4)
    d5 = round(float(df_paid.loc[df_paid["ex_date"]>=c5,"dividend_per_share"].sum()) if not df_paid.empty else 0,4)

    curt = as_of_ts.year
    all_fy = sorted([fy for fy in set(df["fiscal_year"].dropna().unique()) if fy < curt], reverse=True)
    if all_fy:
        lfy = all_fy[0]  # 最新财年（含仅有预案的）
        fy_rows = df[df["fiscal_year"]==lfy].sort_values("dividend_per_share")
        fyd = round(float(fy_rows["dividend_per_share"].sum()),4)
        # 保护：选中FY分红不足上一FY的60% 且 ≤7月 → 末期可能未付，回退
        prev = float(df[df["fiscal_year"]==lfy-1]["dividend_per_share"].sum()) if not df[df["fiscal_year"]==lfy-1].empty else 0
        if prev > 0 and fyd < prev*0.6 and as_of_ts.month <= 7:
            if lfy-1 in all_fy:
                lfy = lfy-1; fy_rows = df[df["fiscal_year"]==lfy].sort_values("dividend_per_share")
                fyd = round(float(fy_rows["dividend_per_share"].sum()),4)
        parts = []
        for _,r in fy_rows.iterrows():
            tag = "(预)" if r.get("status")=="proposal" else ""
            parts.append(f"{r['dividend_per_share']:.4f}{tag}")
        fy_detail = "+".join(parts) if parts else f"{fyd:.4f}"
        fy_yld = round(fyd/price*100,2) if price>0 and fyd>0 else nan
    else: lfy=None; fyd=0.0; fy_detail=""; fy_yld=nan

    ttm = round(d1/price*100,2) if price>0 and d1>0 else nan
    return {"consecutive_years":cy,"fiscal_year":lfy,"fy_dividend":fyd,"fy_detail":fy_detail,
            "fy_yield":fy_yld,"div_5y":d5,"div_3y":d3,"div_1y":d1,"ttm_yield":ttm}


def _fetch_dividend_prices(codes):
    sina = [f"sh{c}" if c.startswith(("5","6","9")) else f"sz{c}" for c in codes]
    url = "https://hq.sinajs.cn/list="+",".join(sina)
    h = {"Referer":"https://finance.sina.com.cn"}
    result = {}
    for attempt in range(FETCH_RETRY):
        try:
            r = requests.get(url, headers=h, timeout=15); r.encoding="gb2312"; break
        except:
            if attempt<FETCH_RETRY-1: time.sleep(FETCH_RETRY_DELAY*(2**attempt))
            else: return {c:float("nan") for c in codes}
    for line in r.text.strip().split("\n"):
        if not line.strip(): continue
        try:
            parts = line.split('"')
            if len(parts)<2: continue
            sc = parts[0].replace("var hq_str_","").rstrip("=")
            cc = sc[2:]; fields = parts[1].split(",")
            if len(fields)<4: continue
            result[cc] = float(fields[3]) if fields[3] else 0.0
        except: continue
    for c in codes:
        if c not in result: result[c] = float("nan")
    return result


def format_dividend_section():
    """策略3：高股息率跟踪。Markdown 格式，与策略1/2风格一致。"""
    as_of = pd.Timestamp(datetime.now())
    codes = [c for c,_ in DIVIDEND_STOCKS]
    names = {c:n for c,n in DIVIDEND_STOCKS}

    prices = _fetch_dividend_prices(codes)

    rows = []
    for i,code in enumerate(codes):
        if i>0: time.sleep(DIV_BATCH_DELAY)
        df = _get_dividend_data(code)
        m = _compute_one(df, prices.get(code,float("nan")), as_of)
        rows.append({
            "code":code,"name":names.get(code,code),
            "price":prices.get(code,float("nan")),
            "years":m["consecutive_years"],
            "fy":m["fiscal_year"],
            "fy_yld":m["fy_yield"],
            "fy_detail":m["fy_detail"],
            "d1":m["div_1y"],"d3":m["div_3y"],"d5":m["div_5y"],
        })

    rows.sort(key=lambda r: r["fy_yld"] if not np.isnan(r["fy_yld"]) else -1, reverse=True)
    valid = [r for r in rows if not np.isnan(r["fy_yld"])]

    lines = ["**💰 高股息率跟踪**", ""]

    # 摘要
    if valid:
        lines.append(f"> 均{np.mean([r['fy_yld'] for r in valid]):.2f}%  中位{np.median([r['fy_yld'] for r in valid]):.2f}%  共{len(rows)}只")
        lines.append("")

    # Markdown 表格
    lines.append("| # | 名称(代码) | 现价 | 股息率 | 财年分红 | 连续 |")
    lines.append("|---|-----------|-----|--------|-----|---------|------|")
    medals = {0:"🥇",1:"🥈",2:"🥉"}
    for i,r in enumerate(rows):
        rank = medals.get(i, str(i+1))
        label = f"{r['name']}({r['code']})"
        if r['code'] in HIGHLIGHT_CODES:
            label = f"🔴{label}"
        ps = f"{r['price']:.2f}" if not np.isnan(r['price']) else "N/A"
        yd = f"{r['fy_yld']:.2f}%" if not np.isnan(r['fy_yld']) else "N/A"
        fd = r['fy_detail'] if r['fy_detail'] else "-"
        yrs = f"{int(r['years'])}年" if r['years']>0 else "-"
        lines.append(f"| {rank} | {label} | {ps} | {yd} | {fd} | {yrs} |")

    nodata = [r for r in rows if np.isnan(r["fy_yld"])]
    if nodata:
        lines.append("")
        lines.append("无分红数据: " + "  ".join(f"{r['name']}({r['code']})" for r in nodata))

    return "\n".join(lines)


def main():
    """主函数：运行两个策略并推送信号。"""
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"开始计算ETF动量得分... [{now_str}]")

    # ═══════════════════════════════════════════════════════
    # 策略1: 大类资产ETF动量轮动
    # ═══════════════════════════════════════════════════════
    results = []
    for code, name in ETF_POOL.items():
        prices = get_sina_kline(code, LOOKBACK_DAYS + 5)
        if prices is None or len(prices) < LOOKBACK_DAYS:
            print(f"  ⚠️ {code} ({name}) 数据不足")
            continue
        score, ann_ret, r2 = calculate_score(prices)
        if score is None:
            print(f"  ⚠️ {code} ({name}) 计算失败")
            continue
        results.append({
            'code': code, 'name': name, 'score': score,
            'annualized_return': ann_ret, 'r_squared': r2,
        })

    if not results:
        error_msg = (f"ETF动量轮动信号 ({datetime.now().strftime('%Y-%m-%d')})\n\n"
                     f"❌ 数据获取失败，请检查新浪API")
        print(error_msg)
        with open('output.txt', 'w', encoding='utf-8') as f:
            f.write(error_msg)
        return

    results.sort(key=lambda x: x['score'], reverse=True)

    # 现金阈值判断（同步自大类资产轮动 backtest）
    top_score = results[0]['score']
    recommend_cash = bool(top_score <= CASH_THRESHOLD)

    history = load_history()
    changes = compare_with_last(results, history)
    output = format_output(results, changes, recommend_cash=recommend_cash)
    print("\n" + output)

    # ═══════════════════════════════════════════════════════
    # 策略2: 中证2000ETF择时（慢/快动量三条件出场）
    # ═══════════════════════════════════════════════════════
    stop_loss_result = check_stop_loss_signal()
    last_trigger = history.get('stop_loss_history', {}).get('trigger')
    stop_loss_section = format_stop_loss_section(stop_loss_result, last_trigger)
    print("\n" + stop_loss_section)
    output += "\n\n" + stop_loss_section

    # ═══════════════════════════════════════════════════════
    # 策略3: 高股息率跟踪
    # ═══════════════════════════════════════════════════════
    print("\n[策略3] 正在计算高股息率跟踪...")
    dividend_section = format_dividend_section()
    print(dividend_section)
    output += "\n\n\n" + dividend_section

    # 保存历史
    today = datetime.now().strftime('%Y-%m-%d')
    history['last_result'] = {
        'date': today,
        'ranking': results,
        'recommend_cash': recommend_cash,
    }
    history['stop_loss_history'] = {
        'date': today,
        'target': STOP_LOSS_TARGET,
        'target_slow_score': float(stop_loss_result['target']['slow_score']) if stop_loss_result['target']['slow_score'] is not None else None,
        'target_fast_score': float(stop_loss_result['target']['fast_score']) if stop_loss_result['target']['fast_score'] is not None else None,
        'benchmark_scores': {
            code: float(info['score']) if info['score'] is not None else None
            for code, info in stop_loss_result['benchmarks'].items()
        },
        'trigger': bool(stop_loss_result['trigger']),
    }
    save_history(history)

    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"\n✅ 信号已保存至 output.txt [{datetime.now().strftime('%H:%M:%S')}]")
    return output


if __name__ == '__main__':
    main()
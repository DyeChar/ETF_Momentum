#!/usr/bin/env python3
"""
每日监控 — 多策略信号推送
==========================
策略1: 大类资产ETF动量轮动（4 ETF选最强，全部≤0 → 511880避险）
策略2: 中证2000ETF择时（慢/快动量三条件出场，出场→511880）
策略3: 高股息跟踪（分红历史 + TTM股息率）

数据源：新浪财经API
同步自:
  - 1.中证2000ETF择时/backtest.py（择时出场逻辑）
  - 2.大类资产ETF轮动策略/backtest.py（轮动选股逻辑）
  - 3.高股息跟踪/（分红解析 + 股息率计算）
"""

import json
import math
import os
import time
import warnings
from datetime import datetime
from io import StringIO as _StringIO

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
    # 变动状态
    if changes['is_new']:
        change_text = "首次运行，无历史对比"
    elif changes['top_changed'] or changes['ranking_changed']:
        detail_str = "；".join(changes['changes_detail'])
        change_text = f"⚠️ 有变动！{detail_str}"
    else:
        change_text = "✅ 与上次一致，无变动"

    lines = [f"**📊 大类资产ETF动量轮动信号**"]

    if recommend_cash:
        # 全部ETF动量 ≤ 0 → 避险
        lines.append(f"**【建议持仓】 👉 {CASH_ETF_NAME} ({CASH_ETF_CODE[2:]})，🛡️ 全部ETF动量≤0，避险**")
    else:
        top_etf = ranking[0]
        pure_code = top_etf['code'][2:]  # 去掉sh/sz前缀
        lines.append(f"**【建议持仓】 👉 {top_etf['name']} ({pure_code})，{change_text}**")

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


def format_stop_loss_section(result: dict) -> str:
    """格式化策略2输出（中证2000ETF择时）。"""
    target = result['target']

    lines = [f"**🛡️ 中证2000ETF择时信号**"]

    # 建议持仓
    target_pure = STOP_LOSS_TARGET[2:]
    if result['trigger']:
        lines.append(f"**【建议持仓】 👉 空仓（{CASH_ETF_NAME} {CASH_ETF_CODE[2:]}），🔴 动量止损信号触发！**")
    else:
        lines.append(f"**【建议持仓】 👉 {STOP_LOSS_TARGET_NAME}（{target_pure}），🟢 动量正常**")

    # 慢动量排序
    all_items = []
    if target['slow_score'] is not None:
        all_items.append((target['code'], target['name'], target['slow_score']))
    for code, info in result['benchmarks'].items():
        if info['score'] is not None:
            all_items.append((code, info['name'], info['score']))
    all_items.sort(key=lambda x: x[2], reverse=True)

    ranking_parts = []
    medals = ["🥇", "🥈", "🥉"]
    for i, (code, name, score) in enumerate(all_items):
        prefix = medals[i] if i < 3 else ""
        ranking_parts.append(f"{prefix} {name}: {score:.4f}")
    lines.append(f"【{STOP_LOSS_LOOKBACK_DAYS}日慢动量排序】 {' '.join(ranking_parts)}")

    # 风控详情
    rank_text = "✅ 是" if result['is_rank1'] else "❌ 否"

    fast_score = target['fast_score']
    slow_score = target['slow_score']

    fast_ok = (fast_score is not None and fast_score >= FAST_MOMENTUM_THRESHOLD)
    fast_text = "✅ 是" if fast_ok else "❌ 否"
    fast_detail = f" (快= {fast_score:.4f})" if fast_score is not None else ""

    slow_ok = (slow_score is not None and slow_score >= MOMENTUM_THRESHOLD)
    slow_text = "✅ 是" if slow_ok else "❌ 否"
    slow_detail = f" (慢= {slow_score:.4f})" if slow_score is not None else ""

    lines.append(
        f"【风控】"
        f" 排名第一: {rank_text}"
        f" 快动量<{FAST_MOMENTUM_THRESHOLD}: {fast_text}{fast_detail}"
        f" 慢动量>{MOMENTUM_THRESHOLD}: {slow_text}{slow_detail}"
    )

    return "\n\n".join(lines)


# ============================================================
# 策略3: 高股息跟踪 — TTM股息率
# 同步自: 3.高股息跟踪/
# ============================================================
FETCH_RETRY = 3
FETCH_RETRY_DELAY = 1.0
DIVIDEND_STOCKS = [
    ("601398","工商银行"),("601939","建设银行"),("601988","中国银行"),("601288","农业银行"),
    ("601328","交通银行"),("600036","招商银行"),("601166","兴业银行"),("601998","中信银行"),
    ("600000","浦发银行"),("601169","北京银行"),
    ("601088","中国神华"),("601857","中国石油"),("600028","中国石化"),("600900","长江电力"),
    ("600025","华能水电"),
    ("601006","大秦铁路"),("600377","宁沪高速"),("600350","山东高速"),("000429","粤高速A"),
    ("600012","皖通高速"),("600548","深高速"),
    ("600519","贵州茅台"),("000858","五粮液"),("600887","伊利股份"),("000895","双汇发展"),
    ("000651","格力电器"),("000333","美的集团"),("000568","泸州老窖"),("000538","云南白药"),
    ("600019","宝钢股份"),("600585","海螺水泥"),("600309","万华化学"),("601668","中国建筑"),
    ("601390","中国中铁"),("601186","中国铁建"),("600660","福耀玻璃"),("600104","上汽集团"),
    ("601318","中国平安"),("601601","中国太保"),("601336","新华保险"),
    ("600066","宇通客车"),("600177","雅戈尔"),("002003","伟星股份"),("000726","鲁泰A"),
    ("601566","九牧王"),
    ("600941","中国移动"),("601728","中国电信"),("601816","京沪高铁"),("601658","邮储银行"),
    ("688981","中芯国际"),("600938","中国海油"),("601919","中远海控"),("600905","三峡能源"),
    ("563020","易方达红利低波"),
]
DIV_BATCH_DELAY = 0.3
COL_ANNOUNCE = ("分红","公告日期","公告日期")
COL_DIV10 = ("分红","分红方案(每10股)","派息(税前)(元)")
COL_PROGRESS = ("分红","进度","进度")
COL_EX = ("分红","除权除息日","除权除息日")


def _fetch_dividend_html(code):
    url = f"https://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/{code}.phtml"
    h = {"Referer":"https://finance.sina.com.cn","User-Agent":"Mozilla/5.0"}
    for attempt in range(FETCH_RETRY):
        try:
            r = requests.get(url, headers=h, timeout=15)
            for enc in ["gb2312","gbk","gb18030","utf-8"]:
                try:
                    t = r.content.decode(enc, errors="replace")
                    if "分红" in t or "派息" in t: return t
                except: continue
            return r.content.decode("gbk", errors="replace")
        except Exception as e:
            if attempt < FETCH_RETRY-1:
                time.sleep(FETCH_RETRY_DELAY*(2**attempt))
    return None


def _parse_dividend_html(html_text):
    try: tables = pd.read_html(_StringIO(html_text))
    except Exception: return None
    target = None
    for t in tables:
        if any("派息" in str(c) for c in t.columns.tolist()): target = t; break
    if target is None or target.empty: return None
    try:
        a_raw = target[COL_ANNOUNCE]; d_raw = target[COL_DIV10]
        prog = target[COL_PROGRESS]; e_raw = target[COL_EX]
    except KeyError: return None
    # 第一遍：收集所有公告日期
    all_ads = []
    rows_data = []
    for i in range(len(target)):
        try:
            status = str(prog.iloc[i]).strip()
            if status not in ("实施","预案"): continue
            dv = float(d_raw.iloc[i])
            if dv <= 0: continue
            an_s = str(a_raw.iloc[i]).strip()
            ad = pd.Timestamp(an_s) if an_s not in ("--","","nan","NaT") else pd.NaT
            if pd.isna(ad): continue
            es = str(e_raw.iloc[i]).strip()
            ex = pd.Timestamp(es) if es not in ("--","","nan","NaT") else pd.NaT
            all_ads.append(ad)
            rows_data.append((ad, ex, dv, status))
        except (ValueError,TypeError): continue

    # 第二遍：根据公告月份推断财年
    # 规则：1-7月→上财年(末期)，9-12月→本财年(中期)
    # 8月歧义：同年有3-6月公告→中期(本财年)，否则末期(上财年，晚开会)
    records = []; seen = set()
    for ad, ex, dv, status in rows_data:
        if ad.month <= 7:
            fy = ad.year - 1
        elif ad.month >= 9:
            fy = ad.year
        else:  # month == 8
            # 同年有3-6月公告→中期；否则末期(晚开会)
            has_spring = any(a.year == ad.year and 3 <= a.month <= 6 for a in all_ads)
            fy = ad.year if has_spring else ad.year - 1

        key = (fy, round(dv/10.0, 4))
        if key in seen: continue
        seen.add(key)
        records.append({"ex_date":ex if pd.notna(ex) else pd.NaT,
                       "fiscal_year":fy,"dividend_per_share":dv/10.0,"status":status})

    if not records: return pd.DataFrame(columns=["ex_date","fiscal_year","dividend_per_share","status"])
    return pd.DataFrame(records).sort_values("ex_date",ascending=False).reset_index(drop=True)


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
    # 预案无除权日→排除出TTM但保留财年统计
    df_paid = df.dropna(subset=["ex_date"])
    if df.empty: return empty

    cy = _consecutive_dividend_years(df_paid if not df_paid.empty else df)
    c1 = as_of_ts-pd.DateOffset(months=12)
    c3 = as_of_ts-pd.DateOffset(months=36)
    c5 = as_of_ts-pd.DateOffset(months=60)
    d1 = round(float(df_paid.loc[df_paid["ex_date"]>=c1,"dividend_per_share"].sum()) if not df_paid.empty else 0,4)
    d3 = round(float(df_paid.loc[df_paid["ex_date"]>=c3,"dividend_per_share"].sum()) if not df_paid.empty else 0,4)
    d5 = round(float(df_paid.loc[df_paid["ex_date"]>=c5,"dividend_per_share"].sum()) if not df_paid.empty else 0,4)

    curt = as_of_ts.year
    # 完整财年：fiscal_year < 当前年 且至少有一笔分红（含预案）
    # 末期通常在7-8月除权，即使未来也算完整（方案已定）
    all_fy = set(df["fiscal_year"].dropna().unique())
    candidates = sorted([fy for fy in all_fy if fy < curt], reverse=True)
    # 数据驱动：选最新完整财年，但若分红不足上财年60%且≤7月→回退
    fy_max_ex = df_paid.groupby("fiscal_year")["ex_date"].max() if not df_paid.empty else pd.Series(dtype=float)
    complete = fy_max_ex[(fy_max_ex.index < curt)] if not fy_max_ex.empty else pd.Series(dtype=float)
    if not complete.empty:
        lfy = int(complete.index.max())
        fy_rows = df[df["fiscal_year"]==lfy].sort_values("dividend_per_share")
        fyd = round(float(fy_rows["dividend_per_share"].sum()),4)
        # 保护：选中FY分红不足上一FY的60% 且 ≤7月 → 末期未付，回退
        prev = float(df[df["fiscal_year"]==lfy-1]["dividend_per_share"].sum()) if not df[df["fiscal_year"]==lfy-1].empty else 0
        if prev > 0 and fyd < prev*0.6 and as_of_ts.month <= 7:
            if lfy-1 in candidates:
                lfy = lfy-1; fy_rows = df[df["fiscal_year"]==lfy].sort_values("dividend_per_share")
                fyd = round(float(fy_rows["dividend_per_share"].sum()),4)
        parts = []
        for _,r in fy_rows.iterrows():
            tag = "(预)" if r.get("status")=="预案" else ""
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
    """策略3：高股息跟踪。Markdown 格式，与策略1/2风格一致。"""
    as_of = pd.Timestamp(datetime.now())
    codes = [c for c,_ in DIVIDEND_STOCKS]
    names = {c:n for c,n in DIVIDEND_STOCKS}

    prices = _fetch_dividend_prices(codes)

    rows = []
    for i,code in enumerate(codes):
        if i>0: time.sleep(DIV_BATCH_DELAY)
        html = _fetch_dividend_html(code)
        df = _parse_dividend_html(html) if html else pd.DataFrame()
        m = _compute_one(df, prices.get(code,float("nan")), as_of)
        rows.append({
            "code":code,"name":names.get(code,code),
            "price":prices.get(code,float("nan")),
            "years":m["consecutive_years"],
            "fy":m["fiscal_year"],
            "fy_yld":m["fy_yield"],"ttm":m["ttm_yield"],
            "fy_detail":m["fy_detail"],
            "d1":m["div_1y"],"d3":m["div_3y"],"d5":m["div_5y"],
        })

    rows.sort(key=lambda r: r["fy_yld"] if not np.isnan(r["fy_yld"]) else -1, reverse=True)
    valid = [r for r in rows if not np.isnan(r["fy_yld"])]

    lines = ["**💰 高股息跟踪 — TTM股息率**"]

    # 摘要行
    if valid:
        top3 = "  ".join(f"`{r['name']}`{r['fy_yld']:.1f}%" for r in valid[:3])
        lines.append(f"均{np.mean([r['fy_yld'] for r in valid]):.2f}%  中位{np.median([r['fy_yld'] for r in valid]):.2f}%  Top3: {top3}")
        lines.append("")

    # 完整榜单：紧凑单行格式
    medals = {0:"🥇",1:"🥈",2:"🥉"}
    for i,r in enumerate(rows):
        pre = medals.get(i, f"{i+1:2d}.")
        ps = f"{r['price']:.2f}" if not np.isnan(r['price']) else "N/A"
        yd = f"{r['fy_yld']:.2f}%" if not np.isnan(r['fy_yld']) else "N/A"
        fd = r['fy_detail'] if r['fy_detail'] else "-"
        ttm = f"TTM{r['ttm']:.1f}%" if not np.isnan(r['ttm']) else ""
        fy_label = f"FY{int(r['fy'])}" if r['fy'] and not np.isnan(r['fy']) else ""
        yrs = f"连续{int(r['years'])}年" if r['years']>0 else ""

        parts = [
            f"{pre} {r['name']}({r['code']})",
            f"现{ps}",
            f"息{yd}",
            f"{fy_label}:{fd}",
        ]
        if ttm: parts.append(ttm)
        if yrs: parts.append(yrs)
        lines.append("  ".join(parts))

    # 无数据
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
    stop_loss_section = format_stop_loss_section(stop_loss_result)
    print("\n" + stop_loss_section)
    output += "\n\n" + stop_loss_section

    # ═══════════════════════════════════════════════════════
    # 策略3: 高股息跟踪
    # ═══════════════════════════════════════════════════════
    print("\n[策略3] 正在计算高股息跟踪...")
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
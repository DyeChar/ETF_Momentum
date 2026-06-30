#!/usr/bin/env python3
"""
大类资产ETF + 中证2000ETF 盘中实时监控
结合动量信号（日K线）与实时价格（新浪实时行情）

同步自:
  - 1.中证2000ETF择时/backtest.py（择时出场逻辑）
  - 2.大类资产ETF轮动策略/backtest.py（轮动选股逻辑）
"""

import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests

# ============================================================
# 策略1: 大类资产ETF动量轮动
# 同步自: 2.大类资产ETF轮动策略/config.py + backtest.py
# ============================================================
ETF_POOL = {
    'sh518880': '黄金ETF',
    'sh513100': '纳指ETF',
    'sz159915': '创业板ETF',
    'sh510300': '沪深300ETF',
}
LOOKBACK_DAYS = 25
CASH_THRESHOLD = 0.0            # 全部ETF动量 ≤ 此值 → 持有银华日利避险
CASH_ETF_CODE = 'sh511880'
CASH_ETF_NAME = '银华日利'

# ============================================================
# 策略2: 中证2000ETF择时
# 同步自: 1.中证2000ETF择时/config.py + backtest.py
# ============================================================
STOP_LOSS_TARGET = 'sh563300'       # 中证2000ETF（华泰柏瑞，563300）
STOP_LOSS_TARGET_NAME = '中证2000ETF'
STOP_LOSS_BENCHMARKS = {
    'sz159919': '沪深300ETF',
    'sh000015': '红利指数',
    'sh510050': '上证50ETF',
}
STOP_LOSS_LOOKBACK_DAYS = 20
FAST_LOOKBACK_DAYS = 10
FAST_MOMENTUM_THRESHOLD = 0.4
MOMENTUM_THRESHOLD = 2.0

# ============================================================
# 1. 实时行情（新浪实时接口）
# ============================================================
def get_realtime_quotes(codes: list) -> dict:
    """批量获取实时行情，返回 {code: {name, price, open, preclose, high, low, change_pct, volume, time}}"""
    url = "https://hq.sinajs.cn/list=" + ",".join(codes)
    headers = {"Referer": "https://finance.sina.com.cn"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = "gbk"
        raw = resp.text
    except Exception as e:
        print(f"实时行情获取失败: {e}")
        return {}

    result = {}
    for line in raw.strip().split("\n"):
        if not line.strip() or "=" not in line:
            continue
        # 解析 var hq_str_sh510050="字段1,字段2,...";
        code = line.split("=")[0].replace("var hq_str_", "").strip()
        data_str = line.split('"')[1] if '"' in line else ""
        if not data_str:
            continue
        fields = data_str.split(",")

        try:
            name = fields[0]
            open_price = float(fields[1]) if fields[1] else 0
            preclose = float(fields[2]) if fields[2] else 0
            price = float(fields[3]) if fields[3] else 0
            high = float(fields[4]) if fields[4] else 0
            low = float(fields[5]) if fields[5] else 0
            volume = float(fields[8]) if fields[8] else 0
            amount = float(fields[9]) if fields[9] else 0
            date_str = fields[30] if len(fields) > 30 else ""
            time_str = fields[31] if len(fields) > 31 else ""

            change_pct = (price / preclose - 1) * 100 if preclose > 0 else 0

            result[code] = {
                'name': name,
                'price': price,
                'open': open_price,
                'preclose': preclose,
                'high': high,
                'low': low,
                'change_pct': change_pct,
                'volume': volume,
                'amount': amount,
                'date': date_str,
                'time': time_str,
            }
        except (IndexError, ValueError):
            continue

    return result


# ============================================================
# 2. 日K线数据（新浪K线API）
# ============================================================
def get_sina_kline(code: str, days: int = 30) -> list:
    """获取日K线收盘价列表"""
    url = 'https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData'
    params = {'symbol': code, 'scale': '240', 'datalen': days}
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if not data or len(data) < days:
            return None
        return [float(item['close']) for item in data]
    except Exception as e:
        print(f"  K线获取 {code} 失败: {e}")
        return None


# ============================================================
# 3. 动量计算
# ============================================================
def calculate_score(prices: list, window: int = None) -> tuple:
    """动量得分 = 年化收益率 × R²"""
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


# ============================================================
# 4. 合并K线 + 实时价
# ============================================================
def get_intraday_prices(code: str, kline_days: int, quote: dict) -> list:
    """获取日K线收盘价，并追加当天实时价作为最新数据点"""
    prices = get_sina_kline(code, kline_days)
    if prices is None:
        return None
    rt = quote.get(code, {})
    rt_price = rt.get('price', 0)
    rt_date = rt.get('date', '')
    today = datetime.now().strftime('%Y-%m-%d')
    # 如果实时数据是今天的，且与K线最后一根不同（盘中新数据），追加
    if rt_price > 0 and rt_date == today:
        last_close = prices[-1]
        if abs(rt_price - last_close) > 0.0001:
            prices = prices + [rt_price]
    return prices


# ============================================================
# 5. 主函数
# ============================================================
def main():
    now = datetime.now()
    print(f"\n{'='*70}")
    print(f"  📡 大类资产ETF & 中证2000ETF 盘中监控")
    print(f"  ⏰ {now.strftime('%Y-%m-%d %H:%M:%S')}  {'🟢 交易中' if now.hour < 15 else '⏹ 已收盘'}")
    print(f"{'='*70}")

    # ---- 一、实时行情 ----
    all_codes = list(ETF_POOL.keys()) + [STOP_LOSS_TARGET] + list(STOP_LOSS_BENCHMARKS.keys())
    quotes = get_realtime_quotes(all_codes)

    print(f"\n{'─'*70}")
    print(f"  💹 实时行情")
    print(f"{'─'*70}")
    print(f"  {'标的':<14s} {'现价':>8s} {'涨跌':>8s} {'今开':>8s} {'最高':>8s} {'最低':>8s}")
    print(f"  {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for code in all_codes:
        q = quotes.get(code)
        if q:
            arrow = "🔴" if q['change_pct'] < 0 else ("🟢" if q['change_pct'] > 0 else "⚪")
            print(f"  {q['name']:<12s} {q['price']:>8.3f} {arrow}{q['change_pct']:>+6.2f}% {q['open']:>8.3f} {q['high']:>8.3f} {q['low']:>8.3f}")
        else:
            print(f"  {code:<14s} {'--':>8s}")

    # ---- 二、大类资产ETF动量轮动 ----
    print(f"\n{'─'*70}")
    print(f"  📊 大类资产ETF动量轮动（{LOOKBACK_DAYS}日窗口）")
    print(f"{'─'*70}")

    results = []
    for code, name in ETF_POOL.items():
        prices = get_intraday_prices(code, LOOKBACK_DAYS + 5, quotes)
        if prices and len(prices) >= LOOKBACK_DAYS:
            score, ann_ret, r2 = calculate_score(prices)
            if score is not None:
                q = quotes.get(code, {})
                chg = q.get('change_pct', 0)
                arrow = "🔴" if chg < 0 else "🟢"
                results.append({
                    'code': code, 'name': name, 'score': score,
                    'price': q.get('price', 0), 'change_pct': chg, 'arrow': arrow,
                })

    results.sort(key=lambda x: x['score'], reverse=True)
    medals = ["🥇", "🥈", "🥉"]
    for i, r in enumerate(results):
        prefix = medals[i] if i < 3 else "  "
        print(f"  {prefix} {r['name']:<10s} 动量{r['score']:>+9.4f}  {r['arrow']}{r['change_pct']:>+6.2f}%  ¥{r['price']:.3f}")

    top = results[0] if results else None
    if top:
        if top['score'] <= CASH_THRESHOLD:
            pure_cash = CASH_ETF_CODE[2:]
            print(f"\n  📌 建议持仓: {CASH_ETF_NAME} ({pure_cash}) 🛡️ 全部ETF动量≤0")
        else:
            pure = top['code'][2:]
            print(f"\n  📌 建议持仓: {top['name']} ({pure})")

    # ---- 三、中证2000ETF择时 ----
    print(f"\n{'─'*70}")
    print(f"  🛡️ 中证2000ETF择时（慢{STOP_LOSS_LOOKBACK_DAYS}日 / 快{FAST_LOOKBACK_DAYS}日）")
    print(f"{'─'*70}")

    W = STOP_LOSS_LOOKBACK_DAYS
    F = FAST_LOOKBACK_DAYS
    fetch_days = max(W, F) + 5

    # 获取各基准指数慢动量
    benchmark_results = {}
    bench_items = []
    for code, name in STOP_LOSS_BENCHMARKS.items():
        prices = get_intraday_prices(code, fetch_days, quotes)
        if prices and len(prices) >= W:
            score, _, _ = calculate_score(prices[-W:], W)
        else:
            score = None
        q = quotes.get(code, {})
        benchmark_results[code] = {'name': name, 'score': score, 'price': q.get('price', 0), 'change_pct': q.get('change_pct', 0)}
        bench_items.append((code, name, score, q.get('change_pct', 0), q.get('price', 0)))

    # 获取中证2000ETF快/慢动量
    target_prices = get_intraday_prices(STOP_LOSS_TARGET, fetch_days, quotes)
    target_slow_score = None
    target_fast_score = None
    if target_prices and len(target_prices) >= W:
        target_slow_score, _, _ = calculate_score(target_prices[-W:], W)
    if target_prices and len(target_prices) >= F:
        target_fast_score, _, _ = calculate_score(target_prices[-F:], F)

    tq = quotes.get(STOP_LOSS_TARGET, {})
    target_price = tq.get('price', 0)
    target_chg = tq.get('change_pct', 0)

    # 合并排名
    all_items = []
    if target_slow_score is not None:
        all_items.append((STOP_LOSS_TARGET, STOP_LOSS_TARGET_NAME, target_slow_score, target_chg, target_price))
    for code, name, score, chg, price in bench_items:
        if score is not None:
            all_items.append((code, name, score, chg, price))
    all_items.sort(key=lambda x: x[2], reverse=True)

    print(f"  {'标的':<14s} {'慢动量':>9s} {'涨跌':>8s} {'现价':>8s}")
    print(f"  {'─'*14} {'─'*9} {'─'*8} {'─'*8}")
    for i, (code, name, score, chg, price) in enumerate(all_items):
        prefix = medals[i] if i < 3 else "  "
        arrow = "🔴" if chg < 0 else "🟢"
        print(f"  {prefix} {name:<10s} {score:>+9.4f}  {arrow}{chg:>+6.2f}%  ¥{price:.3f}")

    # 止损判断
    all_scores = [info['score'] for info in benchmark_results.values() if info['score'] is not None]
    if target_slow_score is not None:
        all_scores.append(target_slow_score)
    max_score = max(all_scores) if all_scores else 0
    is_rank1 = (target_slow_score is not None and target_slow_score >= max_score)

    trigger = False
    if target_slow_score is not None and target_fast_score is not None:
        trigger = (not is_rank1 and target_fast_score < FAST_MOMENTUM_THRESHOLD and target_slow_score < MOMENTUM_THRESHOLD)

    print(f"\n  【风控】")
    print(f"  排名第一: {'✅ 是' if is_rank1 else '❌ 否'}")
    print(f"  快动量<{FAST_MOMENTUM_THRESHOLD}: {'✅ 是' if target_fast_score is not None and target_fast_score < FAST_MOMENTUM_THRESHOLD else '❌ 否'} (快= {target_fast_score:.4f})" if target_fast_score is not None else f"  快动量<{FAST_MOMENTUM_THRESHOLD}: 数据不足")
    print(f"  慢动量>{MOMENTUM_THRESHOLD}: {'✅ 是' if target_slow_score is not None and target_slow_score >= MOMENTUM_THRESHOLD else '❌ 否'} (慢= {target_slow_score:.4f})" if target_slow_score is not None else f"  慢动量>{MOMENTUM_THRESHOLD}: 数据不足")

    if trigger:
        print(f"\n  🔴 动量止损信号触发！当前建议: 空仓 → {CASH_ETF_NAME} ({CASH_ETF_CODE[2:]})")
    else:
        print(f"\n  🟢 动量正常，当前建议: 持仓 {STOP_LOSS_TARGET_NAME}（{STOP_LOSS_TARGET[2:]}）")

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()

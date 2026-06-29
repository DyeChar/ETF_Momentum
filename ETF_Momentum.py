#!/usr/bin/env python3
"""
A股ETF动量轮动 + 中证2000ETF择时 — 每日信号推送
==============================================
策略1: 大类资产ETF动量轮动（4 ETF选最强，全部≤0 → 511880避险）
策略2: 中证2000ETF择时（慢/快动量三条件出场，出场→511880）

核心逻辑：线性回归斜率 × R²拟合度
数据源：新浪财经API
同步自:
  - 1.中证2000ETF择时/backtest.py（择时出场逻辑）
  - 2.大类资产ETF轮动策略/backtest.py（轮动选股逻辑）
"""

import json
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
    if result['trigger']:
        lines.append(f"**【建议持仓】 👉 空仓（{CASH_ETF_NAME} {CASH_ETF_CODE[2:]}），🔴 动量止损信号触发！**")
    else:
        target_pure = STOP_LOSS_TARGET[2:]
        lines.append(f"**【建议持仓】 👉 {target_pure} {STOP_LOSS_TARGET_NAME}，🟢 动量正常**")

    # 指数动量排名（慢动量）
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
        ranking_parts.append(f"{prefix} {name}: {score:+.4f}")
    lines.append(f"【动量排名（{STOP_LOSS_LOOKBACK_DAYS}日慢动量）】 {' '.join(ranking_parts)}")

    # 风控详情
    rank_text = "✅ 是" if result['is_rank1'] else "❌ 否"

    fast_score = target['fast_score']
    slow_score = target['slow_score']

    fast_ok = (fast_score is not None and fast_score >= FAST_MOMENTUM_THRESHOLD)
    fast_text = "✅ 是" if fast_ok else "❌ 否"
    fast_detail = f" (快={fast_score:+.4f})" if fast_score is not None else ""

    slow_ok = (slow_score is not None and slow_score >= MOMENTUM_THRESHOLD)
    slow_text = "✅ 是" if slow_ok else "❌ 否"
    slow_detail = f" (慢={slow_score:+.4f})" if slow_score is not None else ""

    pure_target = STOP_LOSS_TARGET[2:]
    lines.append(
        f"【{pure_target} {STOP_LOSS_TARGET_NAME} 风控】"
        f" 排名第一: {rank_text}"
        f" 快动量<{FAST_MOMENTUM_THRESHOLD}: {fast_text}{fast_detail}"
        f" 慢动量>{MOMENTUM_THRESHOLD}: {slow_text}{slow_detail}"
    )

    return "\n\n".join(lines)


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
    recommend_cash = top_score <= CASH_THRESHOLD

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
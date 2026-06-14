#!/usr/bin/env python3
"""
A股ETF动量轮动策略 - 每日信号推送
核心逻辑：线性回归斜率 × R²拟合度
数据源：新浪财经API（海外访问稳定）
"""

import requests
import pandas as pd
import numpy as np
import math
import json
import os
from datetime import datetime

# 策略配置
ETF_POOL = {
    'sh518880': '黄金ETF',      # 上海
    'sh513100': '纳指ETF',      # 上海
    'sz159915': '创业板ETF',    # 深圳
    'sh510300': '沪深300ETF',   # 上海
}

LOOKBACK_DAYS = 25  # 回看窗口
HISTORY_FILE = 'history.json'  # 历史记录文件

# 动量止损监控配置
STOP_LOSS_TARGET = 'sz399101'    # 中小综指（止损监控对象）
STOP_LOSS_TARGET_NAME = '中小综指'
STOP_LOSS_BENCHMARKS = {
    'sz159919': '沪深300ETF',
    'sh000015': '红利指数',
    'sh510050': '上证50ETF',
}
MOMENTUM_THRESHOLD = 2.0         # 动量高位阈值（>2视为高位，不触发止损）


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


def calculate_score(prices: list) -> tuple:
    """计算动量得分"""
    prices = prices[-LOOKBACK_DAYS:]
    if len(prices) < LOOKBACK_DAYS:
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


def format_output(ranking: list, changes: dict) -> str:
    today = datetime.now().strftime('%Y-%m-%d')

    lines = [
        f"📊 ETF动量轮动信号 ({today})",
        "",
        "【当前排序】",
    ]

    for i, r in enumerate(ranking):
        symbol = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else "  "))
        lines.append(f"{symbol} {r['name']}: 得分 {r['score']:.4f}")

    if changes['is_new']:
        lines.extend(["", "【变动提示】", "首次运行，无历史对比"])
    elif changes['top_changed'] or changes['ranking_changed']:
        lines.extend(["", "【变动提示】⚠️ 有变动！"])
        for detail in changes['changes_detail']:
            lines.append(f"  • {detail}")
    else:
        lines.extend(["", "【变动提示】", "✅ 与上次一致，无变动"])

    top_etf = ranking[0]
    pure_code = top_etf['code'][2:]  # 去掉sh/sz前缀
    lines.extend(["", "【建议持仓】", f"👉 {top_etf['name']} ({pure_code})"])

    return "\n".join(lines)


def check_consecutive_lifting(prices: list) -> tuple:
    """
    检查399101中小综指是否连续3天动量回升
    需要 LOOKBACK_DAYS+3 天的数据才能计算今天、昨天、前天的动量

    返回: (is_lifting, today_mom, yesterday_mom, day_before_mom)
    """
    if len(prices) < LOOKBACK_DAYS + 3:
        return False, None, None, None

    today_mom, _, _ = calculate_score(prices[-LOOKBACK_DAYS:])
    yesterday_mom, _, _ = calculate_score(prices[-(LOOKBACK_DAYS + 1):-1])
    day_before_mom, _, _ = calculate_score(prices[-(LOOKBACK_DAYS + 2):-2])

    if today_mom is None or yesterday_mom is None or day_before_mom is None:
        return False, today_mom, yesterday_mom, day_before_mom

    is_lifting = today_mom > yesterday_mom > day_before_mom
    return is_lifting, today_mom, yesterday_mom, day_before_mom


def check_stop_loss_signal() -> dict:
    """
    动量止损监控
    对399101中小综指 + 3个基准指数进行动量计算与风控判断

    触发条件（全部满足时发出止损信号）：
    1. 399101动量在4个指数中不排第一
    2. 未出现连续3天动量回升
    3. 399101动量 < MOMENTUM_THRESHOLD（未处于高位）

    返回: 包含各指数动量、排名、连续回升状态、止损信号的dict
    """
    # 1. 获取399101数据（需要额外天数用于连续回升判断）
    target_prices = get_sina_kline(STOP_LOSS_TARGET, LOOKBACK_DAYS + 5)

    # 2. 获取各基准指数数据
    benchmark_results = {}
    for code, name in STOP_LOSS_BENCHMARKS.items():
        prices = get_sina_kline(code, LOOKBACK_DAYS + 5)
        if prices and len(prices) >= LOOKBACK_DAYS:
            score, ann_ret, r2 = calculate_score(prices[-LOOKBACK_DAYS:])
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

    # 3. 计算399101动量
    target_score = None
    target_ann = None
    target_r2 = None
    is_lifting = False
    today_mom = yesterday_mom = day_before_mom = None

    if target_prices and len(target_prices) >= LOOKBACK_DAYS:
        target_score, target_ann, target_r2 = calculate_score(target_prices[-LOOKBACK_DAYS:])
        is_lifting, today_mom, yesterday_mom, day_before_mom = check_consecutive_lifting(target_prices)

    # 4. 判断399101是否排第一
    all_scores = []
    for info in benchmark_results.values():
        if info['score'] is not None:
            all_scores.append(info['score'])
    if target_score is not None:
        all_scores.append(target_score)

    max_score = max(all_scores) if all_scores else 0
    is_rank1 = (target_score is not None and target_score >= max_score)

    # 5. 止损信号判断（三条件全部满足才触发）
    trigger = False
    if target_score is not None:
        trigger = (
            not is_rank1
            and not is_lifting
            and target_score < MOMENTUM_THRESHOLD
        )

    return {
        'target': {
            'code': STOP_LOSS_TARGET,
            'name': STOP_LOSS_TARGET_NAME,
            'score': target_score,
            'annualized_return': target_ann,
            'r_squared': target_r2,
        },
        'benchmarks': benchmark_results,
        'is_rank1': is_rank1,
        'is_lifting': is_lifting,
        'today_mom': today_mom,
        'yesterday_mom': yesterday_mom,
        'day_before_mom': day_before_mom,
        'trigger': trigger,
    }


def format_stop_loss_section(result: dict) -> str:
    """格式化动量止损监控段落，追加到 output 末尾"""
    lines = ["", "────────────────────────────", "🛡️ 小市值动量止损监控"]

    # 合并所有指数并按得分降序排列
    all_items = []
    target = result['target']
    if target['score'] is not None:
        all_items.append((target['code'], target['name'], target['score']))
    for code, info in result['benchmarks'].items():
        if info['score'] is not None:
            all_items.append((code, info['name'], info['score']))

    all_items.sort(key=lambda x: x[2], reverse=True)

    if all_items:
        lines.append("")
        lines.append("【指数动量排名】")
        medals = ["🥇", "🥈", "🥉"]
        for i, (code, name, score) in enumerate(all_items):
            prefix = medals[i] if i < 3 else "  "
            lines.append(f"  {prefix} {name}: {score:+.4f}")

    # 399101风控详情
    lines.append("")
    lines.append("【399101中小综指风控】")

    rank_text = "✅ 是" if result['is_rank1'] else "❌ 否"

    # 连续回升详情
    lift_detail = ""
    t, y, d = result['today_mom'], result['yesterday_mom'], result['day_before_mom']
    if all(v is not None for v in [t, y, d]):
        lift_detail = f" (今{t:+.4f} vs 昨{y:+.4f} vs 前{d:+.4f})"
    lift_text = "✅ 是" if result['is_lifting'] else "❌ 否"

    threshold_ok = (target['score'] is not None and target['score'] >= MOMENTUM_THRESHOLD)
    threshold_text = "✅ 是" if threshold_ok else "❌ 否"

    lines.append(f"  排名第一: {rank_text}")
    lines.append(f"  连续3日回升: {lift_text}{lift_detail}")
    lines.append(f"  动量>{MOMENTUM_THRESHOLD}: {threshold_text}")
    lines.append("  ─────────────")

    if result['trigger']:
        lines.append("  🔴 动量止损信号触发！")
    else:
        lines.append("  🟢 动量正常")

    lines.append("────────────────────────────")

    return "\n".join(lines)


def main():
    """主函数"""
    print("开始计算ETF动量得分...")

    results = []

    for code, name in ETF_POOL.items():
        prices = get_sina_kline(code, LOOKBACK_DAYS + 5)

        if prices is None or len(prices) < LOOKBACK_DAYS:
            print(f"警告: {code} 数据获取失败")
            continue

        score, ann_ret, r2 = calculate_score(prices)

        if score is None:
            print(f"警告: {code} 计算失败")
            continue

        results.append({
            'code': code,
            'name': name,
            'score': score,
            'annualized_return': ann_ret,
            'r_squared': r2
        })

    if not results:
        print("错误: 没有获取到任何ETF数据")
        with open('output.txt', 'w', encoding='utf-8') as f:
            f.write(f"ETF动量轮动信号 ({datetime.now().strftime('%Y-%m-%d')})\n\n数据获取失败，请检查API")
        return

    results.sort(key=lambda x: x['score'], reverse=True)

    history = load_history()
    changes = compare_with_last(results, history)

    output = format_output(results, changes)
    print("\n" + output)

    # ---- 动量止损监控（399101 + 3个基准指数） ----
    stop_loss_result = check_stop_loss_signal()
    stop_loss_section = format_stop_loss_section(stop_loss_result)
    print("\n" + stop_loss_section)
    output += "\n" + stop_loss_section

    today = datetime.now().strftime('%Y-%m-%d')
    history['last_result'] = {'date': today, 'ranking': results}
    history['stop_loss_history'] = {
        'date': today,
        'target_score': float(stop_loss_result['target']['score']) if stop_loss_result['target']['score'] is not None else None,
        'benchmark_scores': {
            code: float(info['score']) if info['score'] is not None else None
            for code, info in stop_loss_result['benchmarks'].items()
        },
        'is_lifting': bool(stop_loss_result['is_lifting']),
        'trigger': bool(stop_loss_result['trigger']),
    }
    save_history(history)

    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write(output)

    return output


if __name__ == '__main__':
    main()
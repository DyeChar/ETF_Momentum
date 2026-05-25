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

    today = datetime.now().strftime('%Y-%m-%d')
    history['last_result'] = {'date': today, 'ranking': results}
    save_history(history)

    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write(output)

    return output


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
A股ETF动量轮动策略 - 每日信号推送
核心逻辑：线性回归斜率 × R²拟合度
数据源：Yahoo Finance（海外服务器稳定访问）
"""

import yfinance as yf
import pandas as pd
import numpy as np
import math
import json
import os
from datetime import datetime

# 策略配置
# Yahoo Finance 代码格式：沪市.SS，深市.SZ
ETF_POOL = {
    '518880.SS': '黄金ETF',
    '513100.SS': '纳指ETF',
    '159915.SZ': '创业板ETF',
    '510300.SS': '沪深300ETF',
}

LOOKBACK_DAYS = 25  # 回看窗口
HISTORY_FILE = 'history.json'  # 历史记录文件


def get_etf_data(etf_codes: list) -> dict:
    """获取ETF日线数据"""
    # 计算开始日期（多取一些天数确保有足够数据）
    start_date = datetime.now() - pd.Timedelta(days=LOOKBACK_DAYS * 3)

    # yfinance 批量下载
    data = yf.download(etf_codes, start=start_date, progress=False)

    # 返回字典格式
    result = {}
    for code in etf_codes:
        if code in data.columns:
            # yfinance 返回 MultiIndex，需要提取单个ETF的数据
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    df = data['Close'][code]
                else:
                    df = data['Close']
                result[code] = df
            except:
                pass
    return result


def calculate_score(prices: pd.Series) -> tuple:
    """
    计算动量得分
    返回: (score, annualized_returns, r_squared)
    """
    # 取最近N日收盘价
    df = prices.tail(LOOKBACK_DAYS).dropna()

    if len(df) < LOOKBACK_DAYS:
        # 数据不足，返回 None
        return None, None, None

    # 对数价格线性回归
    y = np.log(df.values)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)

    # 年化收益率
    annualized_returns = math.pow(math.exp(slope), 250) - 1

    # R²拟合度
    y_pred = slope * x + intercept
    ss_res = sum((y - y_pred) ** 2)
    ss_tot = (len(y) - 1) * np.var(y, ddof=1)
    r_squared = 1 - (ss_res / ss_tot)

    # 综合得分
    score = annualized_returns * r_squared

    return score, annualized_returns, r_squared


def load_history() -> dict:
    """加载历史记录"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_history(data: dict):
    """保存历史记录"""
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compare_with_last(current_ranking: list, history: dict) -> dict:
    """对比上次排序结果"""
    today = datetime.now().strftime('%Y-%m-%d')

    last_record = history.get('last_result', {})
    last_date = last_record.get('date', '')
    last_ranking = last_record.get('ranking', [])

    changes = {
        'is_new': len(last_ranking) == 0,
        'date_changed': last_date != today,
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
    """格式化输出内容"""
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
        lines.extend([
            "",
            "【变动提示】",
            "首次运行，无历史对比"
        ])
    elif changes['top_changed'] or changes['ranking_changed']:
        lines.extend([
            "",
            "【变动提示】⚠️ 有变动！"
        ])
        for detail in changes['changes_detail']:
            lines.append(f"  • {detail}")
    else:
        lines.extend([
            "",
            "【变动提示】",
            "✅ 与上次一致，无变动"
        ])

    top_etf = ranking[0]
    lines.extend([
        "",
        "【建议持仓】",
        f"👉 {top_etf['name']} ({top_etf['code'].split('.')[0]})"
    ])

    return "\n".join(lines)


def main():
    """主函数"""
    print("开始计算ETF动量得分...")

    etf_codes = list(ETF_POOL.keys())

    # 获取数据
    data = get_etf_data(etf_codes)

    # 计算得分
    results = []
    for code in etf_codes:
        if code not in data or data[code] is None:
            print(f"警告: {code} 数据获取失败")
            continue

        prices = data[code]
        score, ann_ret, r2 = calculate_score(prices)

        if score is None:
            print(f"警告: {code} 数据不足")
            continue

        name = ETF_POOL.get(code, code)
        results.append({
            'code': code,
            'name': name,
            'score': score,
            'annualized_return': ann_ret,
            'r_squared': r2
        })

    if not results:
        print("错误: 没有获取到任何ETF数据")
        return

    # 按得分排序
    results.sort(key=lambda x: x['score'], reverse=True)

    # 加载历史并对比
    history = load_history()
    changes = compare_with_last(results, history)

    # 格式化输出
    output = format_output(results, changes)
    print("\n" + output)

    # 保存本次结果
    today = datetime.now().strftime('%Y-%m-%d')
    history['last_result'] = {
        'date': today,
        'ranking': results
    }
    save_history(history)

    # 输出到文件
    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write(output)

    return output


if __name__ == '__main__':
    main()
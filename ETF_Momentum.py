#!/usr/bin/env python3
"""
A股ETF动量轮动策略 - 每日信号推送
核心逻辑：线性回归斜率 × R²拟合度
"""

import efinance as ef
import pandas as pd
import numpy as np
import math
import json
import os
from datetime import datetime

# 策略配置
ETF_POOL = [
    '518880',  # 黄金ETF
    '513100',  # 纳指ETF
    '159915',  # 创业板ETF
    '510300',  # 沪深300ETF
]

LOOKBACK_DAYS = 25  # 回看窗口
HISTORY_FILE = 'history.json'  # 历史记录文件


def get_etf_data(etf_codes: list) -> dict:
    """获取ETF日线数据"""
    today = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - pd.Timedelta(days=LOOKBACK_DAYS * 2)).strftime('%Y%m%d')

    k_daily = ef.stock.get_quote_history(etf_codes, start_date)
    return k_daily


def calculate_score(prices: pd.Series) -> tuple:
    """
    计算动量得分
    返回: (score, annualized_returns, r_squared)
    """
    # 取最近N日收盘价
    df = prices.tail(LOOKBACK_DAYS)

    # 对数价格线性回归
    y = np.log(df)
    x = np.arange(df.size)
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


def get_etf_name(etf_code: str) -> str:
    """获取ETF名称"""
    try:
        quote = ef.utils.search_quote(etf_code)
        return quote.name if quote else etf_code
    except:
        return etf_code


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
    """
    对比上次排序结果
    返回变动信息
    """
    today = datetime.now().strftime('%Y-%m-%d')

    # 获取上次记录
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
        # 检查首位是否变动
        if current_ranking[0]['code'] != last_ranking[0]['code']:
            changes['top_changed'] = True
            changes['changes_detail'].append(
                f"首位变动: {last_ranking[0]['name']} → {current_ranking[0]['name']}"
            )

        # 检查排序是否变动
        current_order = [r['code'] for r in current_ranking]
        last_order = [r['code'] for r in last_ranking]
        if current_order != last_order:
            changes['ranking_changed'] = True

            # 找出具体变动
            for i, (curr, last) in enumerate(zip(current_ranking, last_ranking)):
                if curr['code'] != last['code']:
                    changes['changes_detail'].append(
                        f"第{i+1}位: {last['name']} → {curr['name']}"
                    )

    return changes


def format_output(ranking: list, changes: dict) -> str:
    """格式化输出内容（用于微信推送）"""
    today = datetime.now().strftime('%Y-%m-%d')

    lines = [
        f"📊 ETF动量轮动信号 ({today})",
        "",
        "【当前排序】",
    ]

    for i, r in enumerate(ranking):
        symbol = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else "  "))
        lines.append(f"{symbol} {r['name']}: 得分 {r['score']:.4f}")

    # 变动提示
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

    # 推荐操作
    top_etf = ranking[0]
    lines.extend([
        "",
        "【建议持仓】",
        f"👉 {top_etf['name']} ({top_etf['code']})"
    ])

    return "\n".join(lines)


def main():
    """主函数"""
    print("开始计算ETF动量得分...")

    # 获取数据
    k_daily = get_etf_data(ETF_POOL)

    # 计算得分
    results = []
    for etf in ETF_POOL:
        try:
            prices = k_daily[etf]['收盘']
            score, ann_ret, r2 = calculate_score(prices)
            name = get_etf_name(etf)
            results.append({
                'code': etf,
                'name': name,
                'score': score,
                'annualized_return': ann_ret,
                'r_squared': r2
            })
        except Exception as e:
            print(f"处理 {etf} 时出错: {e}")

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

    # 输出到文件（供GitHub Actions读取）
    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write(output)

    return output


if __name__ == '__main__':
    main()
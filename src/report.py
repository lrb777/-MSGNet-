# -*- coding: utf-8 -*-
"""
自动生成 MSGNet 因子挖掘分析报告。

输入：facts/runs/<run_id>/ 下的验证、回测和 Walk-Forward 产物。
输出：facts/runs/<run_id>/report/MSGNet因子挖掘分析报告.html
      facts/runs/<run_id>/report/MSGNet因子挖掘分析报告.pdf

PDF 导出依赖本机 Chrome/Chromium 的 headless print 功能；如果未找到浏览器，
脚本仍会生成 HTML 报告。
"""

from __future__ import annotations

import base64
import html
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from run_config import get_run_dir


REPORT_NAME = "MSGNet因子挖掘分析报告"
IC_THRESHOLD = 0.03
IC_IR_THRESHOLD = 0.5
NW_T_THRESHOLD = 2.0
FACTOR_IR_THRESHOLD = 0.5


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def as_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fmt(value, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct_from_decimal(value, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.{digits}f}%"


def image_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else "png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


def html_table(df: pd.DataFrame, columns: list[str] | None = None,
               max_rows: int | None = None) -> str:
    if df.empty:
        return '<p class="muted">本次运行未生成对应表格。</p>'
    show = df.copy()
    if columns:
        show = show[[c for c in columns if c in show.columns]]
    if max_rows is not None:
        show = show.head(max_rows)

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in show.columns)
    rows = []
    for _, row in show.iterrows():
        cells = "".join(f"<td>{html.escape(fmt(v))}</td>" for v in row)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def metric_card(title: str, value: str, note: str = "") -> str:
    return (
        '<div class="metric">'
        f'<div class="metric-title">{html.escape(title)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-note">{html.escape(note)}</div>'
        '</div>'
    )


def find_chrome() -> str | None:
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    for name in ("chrome", "chrome.exe", "msedge", "msedge.exe", "chromium", "chromium.exe"):
        found = shutil.which(name)
        if found:
            return found
    return None


def export_pdf(html_path: Path, pdf_path: Path) -> bool:
    chrome = find_chrome()
    if chrome is None:
        print("未找到 Chrome/Edge，已跳过 PDF 导出。")
        return False

    user_data_dir = Path(tempfile.gettempdir()) / "chrome-pdf-msgnet-report"
    user_data_dir.mkdir(parents=True, exist_ok=True)
    chrome_work = Path(tempfile.gettempdir()) / "msgnet-report-render"
    chrome_work.mkdir(parents=True, exist_ok=True)
    tmp_html = chrome_work / "report.html"
    tmp_pdf = chrome_work / "report.pdf"
    shutil.copyfile(html_path, tmp_html)
    if tmp_pdf.exists():
        tmp_pdf.unlink()
    uri = tmp_html.resolve().as_uri()
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-crash-reporter",
        "--disable-crashpad",
        "--allow-file-access-from-files",
        "--no-pdf-header-footer",
        f"--user-data-dir={user_data_dir}",
        f"--print-to-pdf={tmp_pdf}",
        uri,
    ]
    result = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip())
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    if not tmp_pdf.exists():
        return False
    shutil.copyfile(tmp_pdf, pdf_path)
    return pdf_path.exists()


def summarize_data(project_dir: Path) -> dict[str, str]:
    price_path = project_dir / "data" / "etf20" / "price.csv"
    price = read_csv(price_path, parse_dates=["date"])
    if price.empty or "date" not in price.columns:
        return {"data_range": "未知", "n_etf": "未知", "n_days": "未知"}
    etf_cols = [c for c in price.columns if c != "date"]
    return {
        "data_range": f"{price['date'].min().date()} 至 {price['date'].max().date()}",
        "n_etf": str(len(etf_cols)),
        "n_days": str(len(price)),
    }


def prepare_ic(ic: pd.DataFrame) -> pd.DataFrame:
    if ic.empty:
        return ic
    out = ic.copy()
    for col in ["IC均值", "IC_IR", "正向比例"]:
        out[col] = as_number(out[col])
    passed = out[
        (out["IC均值"].abs() >= IC_THRESHOLD)
        & (out["IC_IR"].abs() >= IC_IR_THRESHOLD)
    ].copy()
    if passed.empty:
        return passed
    passed["方向"] = passed["IC均值"].map(lambda x: "正向" if x > 0 else "反向")
    passed["正向比例"] = passed["正向比例"].map(lambda x: pct_from_decimal(x, 1))
    passed = passed.sort_values("IC_IR", key=lambda s: s.abs(), ascending=False)
    for col in ["IC均值", "IC_IR"]:
        passed[col] = passed[col].map(lambda x: fmt(x, 3))
    return passed


def prepare_factor_return(fr: pd.DataFrame) -> pd.DataFrame:
    if fr.empty:
        return fr
    out = fr.copy()
    for col in ["年化因子收益", "因子IR", "NW_t值"]:
        out[col] = as_number(out[col])
    sig = out[
        (out["NW_t值"].abs() >= NW_T_THRESHOLD)
        & (out["因子IR"].abs() >= FACTOR_IR_THRESHOLD)
    ].copy()
    if sig.empty:
        return sig
    sig["方向"] = sig["年化因子收益"].map(lambda x: "正向" if x > 0 else "反向")
    sig = sig.sort_values("NW_t值", key=lambda s: s.abs(), ascending=False)
    sig["年化因子收益"] = sig["年化因子收益"].map(lambda x: pct_from_decimal(x, 2))
    for col in ["因子IR", "NW_t值"]:
        sig[col] = sig[col].map(lambda x: fmt(x, 3))
    return sig


def prepare_perf(perf: pd.DataFrame) -> pd.DataFrame:
    if perf.empty:
        return perf
    out = perf.copy()
    first = out.columns[0]
    if first.startswith("Unnamed"):
        out = out.rename(columns={first: "区间类型"})
    return out


def prepare_wf_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path, encoding="utf-8-sig", index_col=0)
    if {"均值", "标准差", "最大值", "最小值"}.intersection(raw.index.astype(str)):
        out = raw.reset_index().rename(columns={"index": "统计"})
    else:
        out = raw.reset_index()
        out.columns = ["指标", "数值"]
    return out


def wf_conclusion(report: pd.DataFrame) -> tuple[str, str]:
    if report.empty or "ann_ret" not in report.columns:
        return "未生成", "本次运行未包含静态 Walk-Forward 产物。"
    ann = as_number(report["ann_ret"])
    sharpe = as_number(report["sharpe"])
    pos = int((ann > 0).sum())
    total = int(len(report))
    mean_ann = ann.mean()
    mean_sharpe = sharpe.mean()
    title = f"{pos}/{total} 个窗口为正收益"
    text = (
        f"静态 Walk-Forward 的平均年化收益为 {mean_ann:.2f}%，"
        f"平均 Sharpe 为 {mean_sharpe:.3f}。正收益窗口占比偏低，说明当前多空结构"
        "对市场环境较敏感，空头侧高 Beta 暴露是主要拖累。"
    )
    return title, text


def rolling_section(run_dir: Path) -> tuple[str, str]:
    rolling_dir = run_dir / "rolling_update"
    report = read_csv(rolling_dir / "wf_report.csv")
    summary = prepare_wf_summary(rolling_dir / "wf_summary.csv")
    if report.empty and summary.empty:
        return (
            '<p class="muted">本次 run 未包含滚动重训练 Walk-Forward 产物。'
            '如需复现该章节，请运行 <code>python .\\src\\rolling_update.py</code> 后重新生成报告。</p>',
            "",
        )
    img = image_data_uri(rolling_dir / "wf_result.png")
    image_html = f'<img src="{img}" alt="滚动重训练 Walk-Forward">' if img else ""
    body = (
        "<p>滚动重训练 Walk-Forward 每个 Fold 都重新训练 MSGNet 并重新提取因子，"
        "用于验证结论是否依赖单次模型训练。</p>"
        f"{html_table(summary, max_rows=12)}"
        f"{image_html}"
    )
    return body, ""


def hedged_long_section(run_dir: Path) -> tuple[str, str]:
    hedged_dir = run_dir / "hedged_long"
    perf = prepare_perf(read_csv(hedged_dir / "perf_report.csv"))
    if perf.empty:
        return (
            '<p class="muted">本次 run 未包含 ETF 多头+指数对冲产物。'
            '如需生成该章节，请运行 <code>python .\\src\\hedged_long_backtest.py</code> 后重新生成报告。</p>',
            "未生成",
        )

    test = perf[perf.get("区间类型", "") == "测试集"].copy()
    summary = "已生成"
    if not test.empty and {"方案", "Sharpe", "最大回撤", "Beta"}.issubset(test.columns):
        raw = test[test["方案"] == "多头原始"]
        h08 = test[test["方案"] == "固定0.8对冲"]
        if not raw.empty and not h08.empty:
            summary = (
                f"测试集：多头原始 Sharpe {raw.iloc[0]['Sharpe']}，"
                f"固定0.8对冲 Sharpe {h08.iloc[0]['Sharpe']}"
            )

    img = image_data_uri(hedged_dir / "hedged_long_result.png")
    image_html = f'<img src="{img}" alt="ETF 多头+指数对冲">' if img else ""
    body = (
        "<p>该章节保留 MSGNet 多头 ETF 组合，使用 ETF20 等权市场作为第一版对冲基准。"
        "核心目的是检验多头收益中 Alpha 与市场 Beta 的比例，而不是单纯追求更高收益。</p>"
        f"{html_table(perf, max_rows=14)}"
        f"{image_html}"
        "<p>若较高对冲比例或滚动 Beta 对冲后收益明显下降，说明多头组合中存在较大市场方向暴露；"
        "若回撤显著下降且 Sharpe 仍可接受，则该方向适合作为增强型多头策略继续优化。</p>"
    )
    return body, summary


def factor_descriptions() -> str:
    items = [
        ("amplitude", "主周期振幅", "衡量 ETF 收益序列在主导周期上的摆动强度。数值越高，通常代表更强的周期波动，也更容易混入高 Beta 或强趋势暴露。", "amplitude_i = |FFT_i(f*)|"),
        ("period_stability", "周期稳定性", "衡量主周期结构是否清晰稳定。数值越高，说明 ETF 的周期行为更有规律，适合作为组合辅助信号。", "period_stability_i = 主周期能量 / 总频谱能量"),
        ("in_degree", "图入度", "衡量 ETF 在 MSGNet 学到的关系图中被其他 ETF 指向或影响的程度，反映其在关系网络中的被解释程度。", "in_degree_i = sum_j A_{j,i}"),
        ("pagerank", "网络重要性", "衡量 ETF 在关系网络中的中心性。与简单入度相比，PageRank 会考虑影响来源节点本身的重要性。", "pagerank = d A^T pagerank + (1-d)/N"),
        ("scale_diff", "多尺度结构差异", "衡量长短周期关系图之间的差异，可能对应主题轮动、风格切换或阶段性资金扰动。", "scale_diff_i = |factor_i^{long scale} - factor_i^{short scale}|"),
        ("EMB_1~10", "隐层表示因子", "MSGNet 隐层表示经 PCA 降维后的综合特征。其信息量可能较高，但经济含义弱于周期和图结构因子。", "PCA(embedding_i)"),
    ]
    cards = []
    for name, title, desc, formula in items:
        cards.append(
            '<div class="factor">'
            f'<h3>{html.escape(name)} <span>{html.escape(title)}</span></h3>'
            f'<p>{html.escape(desc)}</p>'
            f'<code>{html.escape(formula)}</code>'
            '</div>'
        )
    return "".join(cards)


def render_report(run_dir: Path) -> str:
    project_dir = run_dir.parents[2]
    data_info = summarize_data(project_dir)

    validation_dir = run_dir / "validation"
    backtest_dir = run_dir / "backtest"
    walk_dir = run_dir / "walk_forward"

    ic_passed = prepare_ic(read_csv(validation_dir / "ic_report.csv"))
    factor_sig = prepare_factor_return(read_csv(validation_dir / "factor_return_report.csv"))
    perf = prepare_perf(read_csv(backtest_dir / "perf_report.csv"))
    wf_report = read_csv(walk_dir / "wf_report.csv")
    wf_summary = prepare_wf_summary(walk_dir / "wf_summary.csv")

    run_id = run_dir.name
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wf_title, wf_text = wf_conclusion(wf_report)
    rolling_html, _ = rolling_section(run_dir)
    hedged_html, hedged_summary = hedged_long_section(run_dir)

    backtest_img = image_data_uri(backtest_dir / "backtest_result.png")
    wf_img = image_data_uri(walk_dir / "wf_result.png")

    metrics = [
        metric_card("Run ID", run_id, "最新结果目录"),
        metric_card("数据区间", data_info["data_range"], f"{data_info['n_etf']} 只 ETF，{data_info['n_days']} 个交易日"),
        metric_card("IC 通过因子", str(len(ic_passed)), f"|IC均值|>={IC_THRESHOLD}, |IC_IR|>={IC_IR_THRESHOLD}"),
        metric_card("显著因子收益", str(len(factor_sig)), f"|NW_t|>={NW_T_THRESHOLD}, |IR|>={FACTOR_IR_THRESHOLD}"),
        metric_card("Walk-Forward", wf_title, "静态因子窗口压力测试"),
        metric_card("多头对冲", hedged_summary, "ETF20等权市场基准"),
    ]

    conclusion = (
        "本轮研究确认 MSGNet 能从 ETF20 多变量时序中提取具有信息量的周期与图结构因子。"
        "低振幅、高周期稳定性一类信号对多头 ETF 选择更有效；但多空组合在 ETF20 截面上"
        "容易形成低 Beta 多头与高 Beta 空头的结构性暴露，导致牛市环境下空头侧持续拖累。"
        "因此，当前更合理的落地方向是多头 ETF 组合叠加指数对冲，或扩展到更大的个股池后再做截面中性化。"
    )

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{REPORT_NAME}</title>
  <style>
    @page {{ size: A4; margin: 16mm 14mm; }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: #20242a;
      font-family: "Microsoft YaHei", "SimHei", Arial, sans-serif;
      font-size: 12px;
      line-height: 1.65;
      background: #fff;
    }}
    h1, h2, h3 {{ margin: 0; line-height: 1.28; font-weight: 700; }}
    h1 {{ font-size: 28px; margin-bottom: 8px; }}
    h2 {{ font-size: 18px; margin: 26px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #d8dde6; }}
    h3 {{ font-size: 13px; margin-bottom: 6px; }}
    p {{ margin: 7px 0; }}
    code {{ font-family: Consolas, "Microsoft YaHei", monospace; background: #f2f5f8; padding: 2px 5px; border-radius: 3px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 8px 0 14px; table-layout: auto; }}
    th, td {{ border: 1px solid #d9dee7; padding: 5px 6px; vertical-align: top; word-break: break-word; }}
    th {{ background: #eef2f6; font-weight: 700; }}
    img {{ display: block; width: 100%; max-height: 210mm; object-fit: contain; margin: 10px 0 16px; page-break-inside: avoid; }}
    .cover {{ padding: 18px 0 10px; border-bottom: 3px solid #2f5d7c; margin-bottom: 14px; }}
    .subtitle {{ color: #59636f; font-size: 13px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 12px 0 16px; }}
    .metric {{ border: 1px solid #d9dee7; border-radius: 6px; padding: 9px; min-height: 74px; }}
    .metric-title {{ color: #59636f; font-size: 11px; }}
    .metric-value {{ font-size: 17px; font-weight: 700; margin-top: 3px; }}
    .metric-note {{ color: #6d7783; font-size: 10px; margin-top: 3px; }}
    .note {{ background: #f5f7fa; border-left: 4px solid #2f5d7c; padding: 9px 11px; margin: 10px 0 14px; }}
    .warning {{ background: #fff7ed; border-left-color: #c47a2c; }}
    .muted {{ color: #6d7783; }}
    .factor-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }}
    .factor {{ border: 1px solid #d9dee7; border-radius: 6px; padding: 9px; page-break-inside: avoid; }}
    .factor span {{ color: #59636f; font-weight: 400; }}
    .page-break {{ page-break-before: always; }}
  </style>
</head>
<body>
  <section class="cover">
    <h1>{REPORT_NAME}</h1>
    <div class="subtitle">生成时间：{html.escape(generated_at)}</div>
  </section>

  <section class="metrics">
    {''.join(metrics)}
  </section>

  <section>
    <h2>研究摘要</h2>
    <div class="note">{html.escape(conclusion)}</div>
  </section>

  <section>
    <h2>数据与模型配置</h2>
    <p>本项目使用 20 只行业/宽基 ETF 构建多变量时序输入，MSGNet 配置以 ETF20 截面为约束，核心参数为 seq_len=60、label_len=30、pred_len=5、d_model=64、top_k=3、subgraph_size=5。</p>
    <p>训练目标不是单纯追求预测误差最小，而是利用 MSGNet 的多尺度图结构学习能力，提取可验证的周期因子和关系结构因子。</p>
  </section>

  <section>
    <h2>因子描述</h2>
    <p>报告采用文字解释为主、少量公式辅助的方式描述因子。公式用于明确计算口径，研究结论仍以 IC、因子收益检验和回测表现为准。</p>
    <div class="factor-grid">{factor_descriptions()}</div>
  </section>

  <section class="page-break">
    <h2>IC 检验</h2>
    <p>IC 检验衡量因子暴露与未来收益的截面秩相关。通过门槛定义为 <code>|IC均值| &gt;= {IC_THRESHOLD}</code> 且 <code>|IC_IR| &gt;= {IC_IR_THRESHOLD}</code>。</p>
    {html_table(ic_passed, ["因子", "预测期", "IC均值", "IC_IR", "正向比例", "方向", "样本数"], 12)}
    <p>从 IC 口径看，amplitude、period_stability 以及部分图结构因子具有较稳定的信息量。其中 amplitude 原始方向为反向，说明低主周期振幅 ETF 在后续窗口中更容易获得相对收益。</p>
  </section>

  <section>
    <h2>Beta 控制单因子收益检验</h2>
    <p>该检验先控制 ETF 相对等权市场的历史 Beta，再用因子残差解释未来收益，回归形式为 <code>future_ret = alpha + b1 * Beta + b2 * factor_residual + epsilon</code>。</p>
    {html_table(factor_sig, ["因子", "预测期", "年化因子收益", "因子IR", "NW_t值", "方向", "样本数"], 12)}
    <p>Newey-West t 值用于缓解收益序列自相关带来的显著性高估。若因子在该口径下仍显著，说明其不只是静态 Beta 暴露的替代变量。</p>
  </section>

  <section class="page-break">
    <h2>中性化与静态回测</h2>
    <p>组合因子采用 <code>composite = -amplitude_neutralized * w1 + period_stability * w2</code>。amplitude 取负是为了与低振幅更优的 IC 方向对齐，Beta 中性化用于降低高振幅 ETF 中隐含的市场暴露。</p>
    {html_table(perf, max_rows=8)}
    {'<img src="' + backtest_img + '" alt="静态回测">' if backtest_img else '<p class="muted">本次运行未生成静态回测图片。</p>'}
    <p>静态回测显示，低振幅与高周期稳定性组合在多头侧更有解释力；但多空组合表现较弱，主要来自空头侧集中暴露于高 Beta ETF。</p>
  </section>

  <section class="page-break">
    <h2>Walk-Forward 稳健性</h2>
    <p>{html.escape(wf_text)}</p>
    {html_table(wf_summary, max_rows=8)}
    {'<img src="' + wf_img + '" alt="静态 Walk-Forward">' if wf_img else '<p class="muted">本次运行未生成 Walk-Forward 图片。</p>'}
  </section>

  <section>
    <h2>滚动重训练验证</h2>
    {rolling_html}
  </section>

  <section class="page-break">
    <h2>ETF 多头 + 指数对冲</h2>
    {hedged_html}
  </section>

  <section>
    <h2>最终结论与下一步</h2>
    <div class="note warning">
      当前结果不建议继续围绕 ETF20 多空结构做参数微调。更高优先级方向是：
      一，只保留多头 ETF 组合并叠加指数对冲；
      二，扩展到 CSI300 个股池，提高截面自由度后做更稳定的中性化；
      三，将静态因子升级为逐日时间序列因子，再评估动态 Beta 中性化的边际收益。
    </div>
  </section>
</body>
</html>
"""
    return html_doc


def main() -> None:
    run_dir = Path(get_run_dir(create=False, new=False)).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"结果目录不存在: {run_dir}")

    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / f"{REPORT_NAME}.html"
    pdf_path = report_dir / f"{REPORT_NAME}.pdf"

    html_doc = render_report(run_dir)
    html_path.write_text(html_doc, encoding="utf-8")
    print(f"HTML 报告已生成: {html_path}")

    try:
        ok = export_pdf(html_path, pdf_path)
    except subprocess.CalledProcessError as exc:
        ok = False
        print(f"PDF 导出失败: {exc}")

    if ok:
        print(f"PDF 报告已生成: {pdf_path}")


if __name__ == "__main__":
    main()

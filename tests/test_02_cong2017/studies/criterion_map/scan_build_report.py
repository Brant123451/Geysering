# -*- coding: utf-8 -*-
"""Assemble report.html for the Series-B sweep + 63-config criterion map
(fully-synchronous frozen solver, same copy as caseA/caseB)."""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def fnum(v, nd=2):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def seriesB_table() -> tuple[str, int]:
    rows = load(OUT / "seriesB_fullsync.csv")
    rows.sort(key=lambda r: float(r["Dr_mm"]))
    trs, n_match = [], 0
    for r in rows:
        ok = r["match"] == "OK"
        n_match += ok
        badge = ("<span style='color:#16a34a;font-weight:700'>OK</span>" if ok
                 else "<span style='color:#dc2626;font-weight:700'>MISS</span>")
        gm = "喷发" if r["geyser_model"] == "1" else "不喷发"
        ge = "喷发" if r["geyser_meas"] == "1" else "不喷发"
        trs.append(
            f"<tr><td><b>{r['run']}</b></td><td>{r['Dr_mm']}</td>"
            f"<td>{r['Dr_over_D']}</td><td>{fnum(r['Vair_star'])}</td>"
            f"<td>{ge}</td><td>{gm} {badge}</td>"
            f"<td>{fnum(r['Ta_meas_s'])}</td><td>{fnum(r['Ta_model_s'])}</td>"
            f"<td>{fnum(r['Yfs_max_m'])}</td>"
            f"<td>{fnum(r['vfs_meas'], 3)} / {fnum(r['v_fs_model'], 3)}</td>"
            f"<td>{fnum(r['vint_meas'], 3)} / {fnum(r['v_int_model'], 3)}</td></tr>")
    return "".join(trs), n_match


def criterion_table() -> tuple[str, int, int, list[dict]]:
    rows = []
    for p in sorted(OUT.glob("criterion_scan_fullsync_*.csv")):
        rows += load(p)
    seen, uniq = set(), []
    for r in rows:
        k = (r["Dr_mm"], r["L0_m"], r["H0_m"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    rows = sorted(uniq, key=lambda r: (float(r["L0_m"]), float(r["Dr_mm"]),
                                       float(r["H0_m"])))
    trs, n_agree = [], 0
    for r in rows:
        same = r["geyser_model"] == r["criterion_geyser"]
        n_agree += same
        style = "" if same else " style='background:#fef2f2'"
        gm = "喷" if r["geyser_model"] == "1" else "无"
        gc = "喷" if r["criterion_geyser"] == "1" else "无"
        mark = "=" if same else "≠"
        trs.append(
            f"<tr{style}><td>{r['Dr_mm']}</td><td>{r['Dr_over_D']}</td>"
            f"<td>{r['L0_m']}</td><td>{r['H0_m']}</td>"
            f"<td>{fnum(r['Vair_star'])}</td><td>{gc}</td><td>{gm}</td>"
            f"<td>{mark}</td><td>{fnum(r['Yfs_max_m'])}</td></tr>")
    return "".join(trs), n_agree, len(rows), rows


def main() -> None:
    sb_trs, sb_match = seriesB_table()
    cr_trs, cr_agree, cr_n, cr_rows = criterion_table()
    near = sum(1 for r in cr_rows
               if r["geyser_model"] == "0" and r["criterion_geyser"] == "1"
               and float(r["Yfs_max_m"]) >= 1.35)
    n_diff = cr_n - cr_agree

    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Scan: Cong 2017 Series-B 系列对照 + 63 构型判据地图（全同步求解器）</title>
<style>
body{{font-family:-apple-system,Segoe UI,Arial,'Microsoft YaHei',sans-serif;margin:0;background:#f6f8fb;color:#1f2937}}
.wrap{{max-width:1180px;margin:24px auto;padding:0 18px}}
.panel{{background:#fff;border:1px solid #ddd;border-radius:12px;padding:16px;margin-bottom:16px}}
img{{width:100%;border:1px solid #ddd;border-radius:10px;background:#fff}}
table{{border-collapse:collapse;width:100%;font-size:12.5px}}
th,td{{text-align:left;padding:5px 7px;border-bottom:1px solid #eee}}
th{{background:#f3f4f6;position:sticky;top:0}}
p{{line-height:1.55;color:#374151}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start}}
.muted{{font-size:13px;color:#6b7280}}
.scroll{{max-height:520px;overflow-y:auto;border:1px solid #eee;border-radius:8px}}
.big{{font-size:15px}}
</style></head><body><div class="wrap">
<h1>Scan — Cong, Chan &amp; Lee (2017) Series-B 系列对照 + 63 构型判据地图</h1>
<p class="big">Campaign 2 的「分支判别器」证据包：<b>同一份冻结全同步求解器</b>（与 caseA/caseB
哈希一致、零参数改动）跑 ① Series-B 高速摄像 7 工况（只改竖管直径 Dr）、②
覆盖全试验参数范围的 63 构型合成扫描（Dr × L0 × H0），模型盲分类后与论文双参数判据
<code>Dr/D ≤ 0.62 且 V*air ≥ 3.42</code> 对照。判定标准：可见水面到达管顶
（Yfs ≥ 0.98 × 1.8 m）即喷发。</p>

<div class="panel">
  <h2 style="margin-top:0">论文中与本扫描相关的图表</h2>
  <table>
    <tr><th>论文图表</th><th>内容</th><th>与本扫描的关系</th></tr>
    <tr><td><b>Table 2</b>（p3）</td><td>Series B 全部 22 次实验汇总（video + high-speed）</td>
        <td>7 工况对照的实测列；63 扫描的参数网格按其范围构造</td></tr>
    <tr><td><b>Fig. 5</b>（p6）</td><td>vnet vs vTaylor（Series A）</td>
        <td>不喷发分支的泰勒气泡尺度参照</td></tr>
    <tr><td><b>Fig. 11</b>（p10）</td><td>不同初始气量的对照实验（h-t 曲线）</td>
        <td>「气量越大越接近喷发」的实验证据，对应扫描的 V*air 轴</td></tr>
    <tr><td><b>Fig. 12</b>（p11）</td><td>喷发形成过程观察（四阶段）</td>
        <td>模型闭合元素（前沿受阻、气囊压缩、口部进气）的现象学依据</td></tr>
  </table>
  <div class="grid2" style="margin-top:10px">
    <div><h3 style="margin:4px 0">Table 2（Series B 汇总，含判据两列）</h3>
      <img src="paper_scans/table2_seriesB.png"></div>
    <div><h3 style="margin:4px 0">Fig. 11 不同初始气量对照</h3>
      <img src="paper_scans/fig11_criterion.png"></div>
  </div>
</div>

<div class="panel">
  <h2 style="margin-top:0">① Series-B 高速摄像 7 工况（B-H1..B-H7）</h2>
  <p>固定 H0=0.66 m、L0=0.61 m，Dr/D = 0.32→0.92；实测分支边界在 0.52 与 0.62 之间。
  模型分类 <b>{sb_match}/7</b>：喷发端（B-H1）与不喷发端（B-H4..B-H7）全对，
  <b>B-H2/B-H3（21/26 mm）误判为不喷发</b>——见下方诚实结论。</p>
  <img src="outputs/seriesB_fullsync.png">
  <div class="scroll" style="margin-top:10px">
  <table>
    <tr><th>Run</th><th>Dr [mm]</th><th>Dr/D</th><th>V*air</th>
        <th>实测</th><th>模型</th><th>Ta 实测 [s]</th><th>Ta 模型 [s]</th>
        <th>Yfs,max 模型 [m]</th><th>v_fs 实/模 [m/s]</th><th>v_int 实/模 [m/s]</th></tr>
    {sb_trs}
  </table></div>
  <p class="muted">v_fs/v_int 模型值取到达→喷发（或气核峰）窗内 0.6 s 滚动最速爬升；
  B-H1 的 v_fs 1.71 m/s 是喷射尖峰（实测表值 0.924 为全程平均）。Ta 模型平均偏晚
  ~1.5 s（B-H7 最晚 11.4 s：宽竖管排气弱、前沿受阻闭合把最后 0.2 m 拖长）。</p>
</div>

<div class="panel">
  <h2 style="margin-top:0">② 63 构型判据地图</h2>
  <p>Dr = 16–46 mm（7 档）× L0 = 0.61/1.2/1.8 m × H0 = 0.66/0.77/0.88 m = 63 构型，
  每构型一次完整全同步模拟（t_end=13 s，~90–170 s/构型）。模型分类与论文判据一致
  <b>{cr_agree}/{cr_n}</b>；{n_diff} 处分歧<b>全部单向</b>（判据=喷发、模型=不喷发），
  其中 {near} 处为近失（Yfs,max ≥ 1.35 m，差 &lt;0.45 m 到顶）。</p>
  <img src="outputs/criterion_map_fullsync.png">
  <div class="scroll" style="margin-top:10px">
  <table>
    <tr><th>Dr [mm]</th><th>Dr/D</th><th>L0 [m]</th><th>H0 [m]</th>
        <th>V*air</th><th>判据</th><th>模型</th><th>一致</th><th>Yfs,max [m]</th></tr>
    {cr_trs}
  </table></div>
</div>

<div class="panel">
  <h2 style="margin-top:0">诚实结论（全同步 vs 旧分级耦合）</h2>
  <p><b>旧成绩的性质</b>：此前 README 记录的「7/7、62/63」出自分级耦合系统——其分类器
  接近静态气量/直径判据本身，与论文判据同构，属<b>循环验证</b>，不是动力学预测。</p>
  <p><b>全同步的真实成绩</b>：{sb_match}/7 与 {cr_agree}/{cr_n}。模型把喷发/不喷发的
  <b>物理分支端点</b>抓对（16 mm 端必喷、宽管端必不喷、Dr/D&gt;0.62 全部不喷——
  <b>无一例假阳性</b>），但对中等直径（21–31 mm）与大气量（L0 ≥ 1.2）组合系统性
  <b>低估喷发</b>：水面顶托到 1.0–1.6 m 后欠最后一段。</p>
  <p><b>机理定位</b>（与 caseB README 的已识别局限同源）：口部体积中性交换在水库
  持续供水布置下抵消进气的抬水位移（B-H6 顶托 0.78 vs 实测 1.21 m）；对水库布置
  关闭该交换可恢复顶托、但同时废掉 B-H1 的气囊压力重建（已试已回退，求解器内注记）。
  本折中把守恒与 16 mm 端的喷发力学放在优先级上，代价即此单向偏差。三个救援闭合
  变体（到达门控 / 蠕移下限 / 到达锁存 + 0.44√(gD) 前沿帽）均以 B-H1 失喷为代价，
  已全部回退——冻结求解器与 caseA/caseB 保持哈希一致。</p>
  <p class="muted">复跑：<code>python scan_run_seriesB.py</code>（7 × ~90 s）；
  <code>python scan_run_criterion_map.py [0.61|1.2|1.8]</code>（分片可并行、断点续跑）；
  <code>python scan_make_figures.py</code> 出图；<code>python scan_build_report.py</code>
  重建本页。明细 CSV 在 <code>outputs/seriesB_fullsync.csv</code> 与
  <code>outputs/criterion_scan_fullsync_L0p*.csv</code>；旧分级耦合产物保留于
  <code>outputs/cong2017_*</code> 供对照。</p>
</div>
</div></body></html>"""
    (HERE / "report.html").write_text(html, encoding="utf-8")
    print(f"-> {HERE / 'report.html'}  (seriesB {sb_match}/7, criterion {cr_agree}/{cr_n})")


if __name__ == "__main__":
    main()

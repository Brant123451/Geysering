from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PAPER_SOLVER = Path(r"D:\tests\Research\The lase case\paper_solver_copy")
if str(PAPER_SOLVER) not in sys.path:
    sys.path.insert(0, str(PAPER_SOLVER))

import run_own_model_slug_5_3 as base
import run_paper_model_slug_5_3 as paper


G = 9.81
RHO_L = 1000.0
P_ATM = 1.0e5


@dataclass(frozen=True)
class CongBH1:
    pipe_length: float = 6.0
    pipe_diameter: float = 0.05
    riser_diameter: float = 0.016
    riser_height: float = 1.8
    tee_x: float = 3.47
    head_h0: float = 0.66
    air_pocket_length: float = 0.61
    pocket_water_depth: float = 0.010


def beta_from_depth(depth: float, diameter: float) -> float:
    h = float(np.clip(depth, 0.0, diameter))
    gamma = math.acos(np.clip(1.0 - 2.0 * h / diameter, -1.0, 1.0))
    return float(np.clip((gamma - math.sin(gamma) * math.cos(gamma)) / math.pi, 1.0e-4, 0.999))


def depth_from_beta(beta: np.ndarray, diameter: float) -> np.ndarray:
    old_d = base.D_PIPE
    try:
        base.D_PIPE = diameter
        gamma = base.gamma_from_beta(np.asarray(beta, dtype=float))
        return 0.5 * diameter * (1.0 - np.cos(gamma))
    finally:
        base.D_PIPE = old_d


def configure_model(cfg: CongBH1, cfl: float) -> None:
    base.L_PIPE = cfg.pipe_length
    base.D_PIPE = cfg.pipe_diameter
    base.PIPE_AREA = 0.25 * math.pi * cfg.pipe_diameter * cfg.pipe_diameter
    base.PHI = 0.0
    base.CFL = cfl
    base.ALPHA_L0 = 0.999
    base.ALPHA_G0 = 0.001
    base.P_OUT = P_ATM
    base.RHO_L = RHO_L
    base.NU_BETA = 0.0
    base.SOURCE_MODE = "phase"
    base.FLUX_ORDER = 1
    base.INTERFACE_CONTROL = True
    base.PRESSURIZED_TRIGGER = 0.985
    base.PRESSURIZED_BETA = 0.999
    base.WATERHAMMER_SPEED = 1000.0
    base.FRONT_SPEED_LIMIT = 3.0
    base.P_CLAMP_LO_FRAC = 0.20
    base.P_CLAMP_HI_FRAC = 4.0
    base.UGS_LIMIT = 15.0
    base.INLET_BETA_OVERRIDE = 0.999
    base.THETA_FIELD = None
    paper.CUTCELL_BG_FLOOR = 1.0e-4
    paper.CONSERVATIVE_STORE = True


def closed_pipe_ghosts(
    beta: np.ndarray,
    mom: np.ndarray,
    rhog: np.ndarray,
    ugs: np.ndarray,
    inlet_superficial: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(beta)
    beta_g = np.empty(n + 2)
    mom_g = np.empty(n + 2)
    rhog_g = np.empty(n + 2)
    ugs_g = np.empty(n + 2)
    beta_g[1:-1] = beta
    mom_g[1:-1] = mom
    rhog_g[1:-1] = rhog
    ugs_g[1:-1] = ugs

    beta_g[0] = 0.999
    mom_g[0] = max(inlet_superficial, mom[0])
    rhog_g[0] = rhog[0]
    ugs_g[0] = 0.0

    beta_g[-1] = beta[-1]
    mom_g[-1] = -mom[-1]
    rhog_g[-1] = rhog[-1]
    ugs_g[-1] = -ugs[-1]
    return beta_g, mom_g, rhog_g, ugs_g


def pressure_momentum_step_closed(
    dt: float,
    dx: float,
    beta_old: np.ndarray,
    mom_old: np.ndarray,
    beta_new: np.ndarray,
    mom_new: np.ndarray,
    pres: np.ndarray,
    rhog: np.ndarray,
    ugs: np.ndarray,
    case: base.SlugCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(beta_new)
    gamma_eos = 1.0 / (base.R_GAS * base.T_GAS)
    p_lo = base.P_CLAMP_LO_FRAC * base.P_OUT
    p_hi = base.P_CLAMP_HI_FRAC * base.P_OUT
    rho0 = float(base.rho_gas_iso(base.P_OUT))
    alpha_b = base.gas_alpha_at_face(beta_new)
    alpha_old = base.gas_alpha_at_face(beta_old)
    v_g = np.maximum(alpha_b * dx * base.PIPE_AREA, base.EPS)
    v_old = np.maximum(alpha_old * dx * base.PIPE_AREA, base.EPS)

    beta_face = base.face_liquid_beta(beta_old)
    ul_face = base.face_average(mom_old / np.maximum(beta_old, base.EPS))
    ug_face = ugs / np.maximum(alpha_old, base.EPS)
    src = base.gas_momentum_source(beta_face, ul_face, ug_face, rhog)
    m_star = rhog * ugs + dt * src
    mn = rhog * ugs

    adv_left = np.zeros(n)
    adv_right = np.zeros(n)
    ug_face_left = np.empty(n)
    ug_face_left[0] = 0.0
    if n > 1:
        ug_face_left[1:] = 0.5 * (ugs[1:] + ugs[:-1]) / np.maximum(0.5 * (alpha_b[1:] + alpha_b[:-1]), base.EPS)
    adv_left[1:] = np.maximum(ug_face_left[1:], 0.0) * (mn[1:] - mn[:-1]) / dx
    if n > 1:
        ug_face_right = 0.5 * (ugs[1:] + ugs[:-1]) / np.maximum(0.5 * (alpha_b[1:] + alpha_b[:-1]), base.EPS)
        adv_right[:-1] = np.minimum(ug_face_right, 0.0) * (mn[1:] - mn[:-1]) / dx
    m_star = m_star - dt * (adv_left + adv_right)

    kap = dt * alpha_b / dx
    lam = dt * base.PIPE_AREA / np.maximum(v_g * gamma_eos, base.EPS)
    a = np.zeros(n)
    b = np.empty(n)
    c = np.zeros(n)
    d = np.empty(n)
    b[0] = 1.0 + lam[0] * kap[0]
    c[0] = -lam[0] * kap[0]
    d[0] = (v_old[0] / v_g[0]) * pres[0] - lam[0] * m_star[0]
    if n > 1:
        a[1:] = -lam[1:] * kap[:-1]
        c[:-1] = -lam[:-1] * kap[:-1]
        b[1:] = 1.0 + lam[1:] * (kap[:-1] + kap[1:])
        d[1:] = (v_old[1:] / v_g[1:]) * pres[1:] + lam[1:] * (m_star[:-1] - m_star[1:])
        b[-1] = 1.0 + lam[-1] * kap[-2]
        c[-1] = 0.0
    pres_new = np.clip(base.thomas_solve(a, b, c, d), p_lo, p_hi)
    rhog_new = base.rho_gas_iso(pres_new)
    ugs_new = np.empty(n)
    if n > 1:
        ugs_new[:-1] = (m_star[:-1] - kap[:-1] * (pres_new[1:] - pres_new[:-1])) / np.maximum(rhog_new[:-1], base.EPS)
    ugs_new[-1] = m_star[-1] / max(rhog_new[-1], base.EPS)
    ugs_new = np.clip(ugs_new, -base.UGS_LIMIT, base.UGS_LIMIT)

    rho_face = base.face_average(rhog_new)
    ul = mom_new / np.maximum(beta_new, base.EPS)
    ug = base.face_average(ugs_new) / np.maximum(1.0 - beta_new, base.EPS)
    force = base.liquid_momentum_source(beta_new, ul, ug, rho_face)
    u_new = np.clip(ul + dt * force / np.maximum(beta_new, base.EPS), -20.0, 20.0)
    return pres_new, rhog_new, ugs_new, beta_new * u_new


def write_profile(path: Path, x: np.ndarray, frames: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write('VARIABLES="x","alpha_l","depth","u_l","u_g","p_g","regime"\n')
        for frame in frames:
            beta = np.asarray(frame["beta"], dtype=float)
            mom = np.asarray(frame["mom"], dtype=float)
            pres = np.asarray(frame["pres"], dtype=float)
            ugs = np.asarray(frame["ugs"], dtype=float)
            regime = np.asarray(frame["regime"], dtype=float)
            depth = depth_from_beta(beta, base.D_PIPE)
            ul = mom / np.maximum(beta, base.EPS)
            ug = ugs / np.maximum(1.0 - beta, base.EPS)
            f.write(f'ZONE T="{frame["time"]:.8f}", I={len(x)}, F=POINT\n')
            for row in zip(x, beta, depth, ul, ug, pres, regime):
                f.write(" ".join(f"{float(value): .12e}" for value in row) + "\n")


def write_viewer(path: Path, x: np.ndarray, frames: list[dict[str, object]], cfg: CongBH1) -> None:
    data = []
    for frame in frames:
        beta = np.asarray(frame["beta"], dtype=float)
        data.append(
            {
                "t": float(frame["time"]),
                "depth": depth_from_beta(beta, cfg.pipe_diameter).round(5).tolist(),
                "regime": np.asarray(frame["regime"], dtype=float).round(3).tolist(),
            }
        )
    payload = {
        "x": x.round(5).tolist(),
        "D": cfg.pipe_diameter,
        "L": cfg.pipe_length,
        "tee": cfg.tee_x,
        "frames": data,
    }
    html = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Cong 2017 BH1 own-model simulation</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:20px;background:#f7f7f7;color:#222}
.wrap{max-width:1180px;margin:auto;background:white;border:1px solid #ddd;border-radius:10px;padding:16px}
canvas{width:100%;height:260px;border:1px solid #ccc;border-radius:8px;background:white}
.row{display:flex;gap:12px;align-items:center;margin-top:12px}
input[type=range]{flex:1}
.meta{font-size:13px;color:#555;line-height:1.5}
</style>
</head>
<body>
<div class="wrap">
<h2>Cong 2017 B-H1: own decoupled two-fluid model, horizontal pipe field</h2>
<canvas id="cv" width="1120" height="260"></canvas>
<div class="row"><button id="play">播放</button><input id="slider" type="range" min="0" max="0" value="0"><span id="label"></span></div>
<p class="meta">数据来自论文算法 driver：四变量分层支路 + IKH restoring coefficient + pressurized segment + RT-Riemann active-set closure + cut-cell remap。蓝色为水，浅色为空气；红线为 T 接头位置。这里没有拼接 ODE 或旧 Fortran 竖管场。</p>
</div>
<script>
const DATA = __DATA__;
const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
const slider = document.getElementById("slider");
const label = document.getElementById("label");
const play = document.getElementById("play");
slider.max = DATA.frames.length - 1;
let timer = null;
function draw(i){
  const fr = DATA.frames[i];
  ctx.clearRect(0,0,cv.width,cv.height);
  const left=46, top=90, W=cv.width-86, H=70;
  ctx.strokeStyle="#222"; ctx.lineWidth=2; ctx.strokeRect(left, top, W, H);
  for(let k=0;k<DATA.x.length;k++){
    const x = DATA.x[k], xp = left + W*x/DATA.L;
    const xp2 = left + W*(k+1)/DATA.x.length;
    const h = Math.max(0, Math.min(DATA.D, fr.depth[k]));
    const hw = H*h/DATA.D;
    ctx.fillStyle = fr.regime[k] > 0.5 ? "#4387c5" : "#76b7df";
    ctx.fillRect(xp, top + H - hw, Math.max(1, xp2-xp+1), hw);
    ctx.fillStyle = "#f0e5b8";
    ctx.fillRect(xp, top, Math.max(1, xp2-xp+1), H-hw);
  }
  const tee = left + W*DATA.tee/DATA.L;
  ctx.strokeStyle="#b22"; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(tee,top-35); ctx.lineTo(tee,top+H+35); ctx.stroke();
  ctx.fillStyle="#b22"; ctx.font="14px sans-serif"; ctx.fillText("T junction x="+DATA.tee+" m", tee+6, top-14);
  ctx.fillStyle="#222"; ctx.font="15px sans-serif"; ctx.fillText("t = "+fr.t.toFixed(3)+" s", left, 42);
  ctx.fillText("horizontal pipe L="+DATA.L+" m, D="+DATA.D+" m", left, 65);
  label.textContent = i + " / " + (DATA.frames.length-1) + "   t=" + fr.t.toFixed(3) + " s";
}
slider.oninput = () => draw(+slider.value);
play.onclick = () => {
  if(timer){ clearInterval(timer); timer=null; play.textContent="播放"; return; }
  play.textContent="暂停";
  timer = setInterval(() => {
    let i = (+slider.value + 1) % DATA.frames.length;
    slider.value = i; draw(i);
  }, 120);
};
draw(0);
</script>
</body>
</html>""".replace("__DATA__", json.dumps(payload))
    path.write_text(html, encoding="utf-8")


def run(cfg: CongBH1, n: int, t_end: float, output_dt: float, cfl: float, outdir: Path) -> Path:
    configure_model(cfg, cfl)
    outdir.mkdir(parents=True, exist_ok=True)
    dx = cfg.pipe_length / n
    x = (np.arange(n) + 0.5) * dx
    pocket_beta = beta_from_depth(cfg.pocket_water_depth, cfg.pipe_diameter)
    pocket_start = cfg.pipe_length - cfg.air_pocket_length
    inlet_u = 0.45 * math.sqrt(G * cfg.pipe_diameter)
    case = base.SlugCase(161, inlet_u, 0.0, 0.0)
    rho0 = float(base.rho_gas_iso(P_ATM))
    p0 = P_ATM + RHO_L * G * cfg.head_h0
    ap0 = base.PIPE_AREA * (1.0 + G * cfg.head_h0 / (base.WATERHAMMER_SPEED * base.WATERHAMMER_SPEED))

    beta = np.where(x < pocket_start, 0.999, pocket_beta)
    mom = np.zeros(n)
    pres = np.where(x < pocket_start, p0, P_ATM)
    rhog = base.rho_gas_iso(pres)
    ugs = np.zeros(n)
    segments = [paper.PressurizedSegment(0.0, pocket_start, ap0, 0.0, True)]

    frames: list[dict[str, object]] = []
    time = 0.0
    next_out = 0.0
    step = 0
    while time < t_end - 1.0e-12:
        beta_w, mom_w, rhog_w, ugs_w = paper.stable_segments_background_cut_cell(beta, mom, rhog, ugs, segments, base.PRESSURIZED_TRIGGER)
        ul = mom_w / np.maximum(beta_w, base.EPS)
        ug = ugs_w / np.maximum(1.0 - beta_w, base.EPS)
        kappa = base.ikh_kappa(beta_w, ul, ug, rhog_w)
        amax = float(np.max(np.abs(ul) + np.sqrt(np.maximum(kappa * beta_w, 1.0e-12))))
        dt = base.CFL * dx / max(amax, 1.0e-6)
        dt = min(dt, 0.01, t_end - time)

        beta_g, mom_g, rhog_g, ugs_g = closed_pipe_ghosts(beta_w, mom_w, rhog_w, ugs_w, inlet_u)
        fl1, fl2 = base.numerical_flux(beta_g, mom_g, rhog_g, ugs_g)
        beta_raw = beta - dt / dx * (fl1[1:] - fl1[:-1])
        mom_raw = mom - dt / dx * (fl2[1:] - fl2[:-1])
        beta_raw = np.clip(beta_raw, 1.0e-4, 0.999)
        pres_raw, rhog_raw, ugs_raw, mom_raw = pressure_momentum_step_closed(
            dt, dx, beta, mom, beta_raw, mom_raw, pres, rhog, ugs, case
        )

        old_segments = segments
        new_segments = []
        pockets = paper.build_closed_gas_pockets(beta_raw, pres_raw, rhog_raw, old_segments, dx, 0.04, 0.985)
        for segment in old_segments:
            updated, _, _, ok = paper.update_segment(
                segment,
                beta_raw,
                mom_raw,
                pres_raw,
                rhog_raw,
                ugs_raw,
                dt,
                dx,
                speed_cap=3.0,
                gas_pressure_scale=1.0,
                pockets=pockets,
                constraints=paper.SegmentBoundaryConstraints(nonnegative_left=True),
            )
            if ok and updated.length > 0.25 * dx:
                new_segments.append(updated)
        segments = paper.prepare_cut_cell_segments(
            new_segments,
            beta_raw,
            mom_raw,
            pres_raw,
            rhog_raw,
            ugs_raw,
            dx,
            head_cap=3.0,
            gas_pocket_min_length=0.04,
            gas_pocket_alpha_cap=0.985,
            split_max_length=10.0,
            split_gap_length=0.04,
        )
        beta, mom, pres, rhog, ugs = beta_raw, mom_raw, pres_raw, rhog_raw, ugs_raw
        paper.apply_cut_cell_event_remap(old_segments, segments, beta, mom, pres, rhog, ugs, beta_raw, mom_raw, pres_raw, rhog_raw, ugs_raw, dx, dt, base.PRESSURIZED_TRIGGER)
        regime = paper.segments_full_cell_mask(n, dx, segments).astype(float)
        time += dt
        step += 1
        if time >= next_out - 1.0e-12:
            frames.append({"time": time, "beta": beta.copy(), "mom": mom.copy(), "pres": pres.copy(), "ugs": ugs.copy(), "regime": regime.copy()})
            next_out += output_dt

    profile = outdir / "cong2017_bh1_own_model_profile.dat"
    viewer = outdir / "index.html"
    write_profile(profile, x, frames)
    write_viewer(viewer, x, frames, cfg)
    summary = {
        "model": "own decoupled two-fluid paper solver",
        "algorithm": "four-variable stratified branch + RT-Riemann active-set closure + cut-cell remap",
        "case": {
            "name": "Cong 2017 B-H1",
            "pipe_length_m": cfg.pipe_length,
            "pipe_diameter_m": cfg.pipe_diameter,
            "riser_diameter_m": cfg.riser_diameter,
            "riser_height_m": cfg.riser_height,
            "tee_x_m": cfg.tee_x,
            "upstream_head_h0_m": cfg.head_h0,
            "initial_air_pocket_length_m": cfg.air_pocket_length,
            "initial_pocket_water_depth_m": cfg.pocket_water_depth,
            "initial_air_pocket_pressure_pa": P_ATM,
        },
        "n": n,
        "steps": step,
        "t_end": time,
        "profile": str(profile),
        "viewer": str(viewer),
        "note": "This is the corrected horizontal B-H1 release driver using the paper solver. It replaces the deleted ODE/old-Fortran visualizations and does not fabricate a vertical-riser field.",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return viewer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=330)
    parser.add_argument("--t-end", type=float, default=8.2)
    parser.add_argument("--output-dt", type=float, default=0.08)
    parser.add_argument("--cfl", type=float, default=0.28)
    parser.add_argument("--out-dir", type=Path, default=HERE / "outputs" / "own_model_bh1")
    args = parser.parse_args()
    viewer = run(CongBH1(), args.n, args.t_end, args.output_dt, args.cfl, args.out_dir)
    print(f"written: {viewer}")


if __name__ == "__main__":
    main()

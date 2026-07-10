from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import run_cong2017_bh1_own_model as horiz

base = horiz.base
paper = horiz.paper


HERE = Path(__file__).resolve().parent
G = 9.81
RHO_L = 1000.0
P_ATM = 1.0e5


def read_horizontal_profile(path: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    x: list[float] = []
    frames: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("VARIABLES"):
                continue
            if line.startswith("ZONE"):
                if current is not None:
                    frames.append(current)
                t_str = line.split('"')[1]
                current = {"time": float(t_str), "depth": [], "regime": []}
                continue
            if current is None:
                continue
            values = [float(v) for v in line.split()]
            if not frames:
                x.append(values[0])
            current["depth"].append(values[2])
            current["regime"].append(values[6])
    if current is not None:
        frames.append(current)
    return np.asarray(x, dtype=float), frames


def configure_vertical(cfg: horiz.CongBH1, n: int, cfl: float) -> None:
    base.L_PIPE = cfg.riser_height
    base.D_PIPE = cfg.riser_diameter
    base.PIPE_AREA = 0.25 * math.pi * cfg.riser_diameter * cfg.riser_diameter
    base.PHI = 0.5 * math.pi
    base.THETA_FIELD = np.full(n, 0.5 * math.pi)
    base.CFL = cfl
    base.ALPHA_L0 = 0.999
    base.ALPHA_G0 = 0.001
    base.P_OUT = P_ATM
    base.RHO_L = RHO_L
    base.NU_BETA = 0.0
    base.SOURCE_MODE = "phase"
    base.FLUX_ORDER = 1
    base.INTERFACE_CONTROL = False
    base.PRESSURIZED_TRIGGER = 0.985
    base.P_CLAMP_LO_FRAC = 0.20
    base.P_CLAMP_HI_FRAC = 4.0
    base.UGS_LIMIT = 25.0
    base.INLET_BETA_OVERRIDE = 0.001
    paper.CUTCELL_BG_FLOOR = 1.0e-4
    paper.CONSERVATIVE_STORE = True


def run_vertical_riser(
    cfg: horiz.CongBH1,
    n: int,
    t_end: float,
    output_dt: float,
    cfl: float,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, float]]:
    configure_vertical(cfg, n, cfl)
    dy = cfg.riser_height / n
    y = (np.arange(n) + 0.5) * dy
    initial_water_height = cfg.head_h0
    bottom_gas_len = 0.05
    p_drive = P_ATM + RHO_L * G * 1.90 * cfg.head_h0

    beta = np.where(y <= initial_water_height, 0.999, 0.001)
    beta[y <= bottom_gas_len] = 0.001
    mom = np.zeros(n)
    pres = np.where(y <= initial_water_height, P_ATM + RHO_L * G * np.maximum(initial_water_height - y, 0.0), P_ATM)
    pres[y <= bottom_gas_len] = p_drive
    rhog = base.rho_gas_iso(pres)
    ugs = np.zeros(n)

    # Gas flux scale from B-H1 air volume divided by the riser area and arrival burst time.
    air_volume = 0.25 * math.pi * cfg.pipe_diameter * cfg.pipe_diameter * cfg.air_pocket_length
    burst_time = 1.25
    usg_in = min(3.0, air_volume / (base.PIPE_AREA * burst_time))
    case = base.SlugCase(171, 0.0, usg_in, 0.0)

    frames: list[dict[str, object]] = []
    time = 0.0
    next_out = 0.0
    step = 0
    while time < t_end - 1.0e-12:
        ul = mom / np.maximum(beta, base.EPS)
        ug = ugs / np.maximum(1.0 - beta, base.EPS)
        kappa = base.ikh_kappa(beta, ul, ug, rhog)
        amax = float(np.max(np.abs(ul) + np.sqrt(np.maximum(kappa * beta, 1.0e-12))))
        dt = base.CFL * dy / max(amax, 1.0e-6)
        dt = min(dt, 0.0025, t_end - time)

        beta_g, mom_g, rhog_g, ugs_g = base.build_ghosts(beta, mom, rhog, ugs, time, 0.0, case, 0.0)
        fl1, fl2 = base.numerical_flux(beta_g, mom_g, rhog_g, ugs_g)
        beta_new = np.clip(beta - dt / dy * (fl1[1:] - fl1[:-1]), 0.001, 0.999)
        mom_new = mom - dt / dy * (fl2[1:] - fl2[:-1])
        pres_new, rhog_new, ugs_new, mom_new = base.pressure_momentum_step(dt, dy, beta, mom, beta_new, mom_new, pres, rhog, ugs, case)

        # Keep a compressed air reservoir at the riser bottom during the arrival burst.
        if time < burst_time:
            bottom = y <= bottom_gas_len
            beta_new[bottom] = np.minimum(beta_new[bottom], 0.02)
            pres_new[bottom] = np.maximum(pres_new[bottom], p_drive)
            rhog_new[bottom] = base.rho_gas_iso(pres_new[bottom])
            ugs_new[bottom] = np.maximum(ugs_new[bottom], 0.5 * usg_in)

        beta, mom, pres, rhog, ugs = beta_new, mom_new, pres_new, rhog_new, ugs_new
        time += dt
        step += 1
        if time >= next_out - 1.0e-12:
            frames.append(
                {
                    "time": time,
                    "alpha_l": beta.copy(),
                    "pressure": pres.copy(),
                    "u_l": (mom / np.maximum(beta, base.EPS)).copy(),
                }
            )
            next_out += output_dt

    meta = {"steps": float(step), "usg_in": float(usg_in), "p_drive_pa": float(p_drive)}
    return y, frames, meta


def write_complete_profile(path: Path, y: np.ndarray, frames: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write('VARIABLES="y","alpha_l","u_l","p_g"\n')
        for frame in frames:
            f.write(f'ZONE T="{frame["time"]:.8f}", I={len(y)}, F=POINT\n')
            for row in zip(y, frame["alpha_l"], frame["u_l"], frame["pressure"]):
                f.write(" ".join(f"{float(value): .12e}" for value in row) + "\n")


def nearest_frame(frames: list[dict[str, object]], t: float) -> dict[str, object]:
    if t <= float(frames[0]["time"]):
        return frames[0]
    if t >= float(frames[-1]["time"]):
        return frames[-1]
    times = [float(frame["time"]) for frame in frames]
    idx = int(np.searchsorted(times, t, side="left"))
    if idx > 0 and abs(times[idx - 1] - t) < abs(times[idx] - t):
        idx -= 1
    return frames[idx]


def write_viewer(
    path: Path,
    cfg: horiz.CongBH1,
    hx: np.ndarray,
    hframes: list[dict[str, object]],
    vy: np.ndarray,
    vframes: list[dict[str, object]],
    arrival_time: float,
    output_dt: float,
) -> None:
    t_end = arrival_time + float(vframes[-1]["time"])
    times = np.arange(0.0, t_end + 0.5 * output_dt, output_dt)
    frames = []
    static_v = {
        "alpha_l": np.where(vy <= cfg.head_h0, 0.999, 0.001),
        "pressure": np.where(vy <= cfg.head_h0, P_ATM + RHO_L * G * np.maximum(cfg.head_h0 - vy, 0.0), P_ATM),
    }
    for t in times:
        hf = nearest_frame(hframes, min(t, float(hframes[-1]["time"])))
        if t < arrival_time:
            vf_alpha = static_v["alpha_l"]
        else:
            vf = nearest_frame(vframes, t - arrival_time)
            vf_alpha = np.asarray(vf["alpha_l"], dtype=float)
        frames.append(
            {
                "t": float(t),
                "hdepth": np.asarray(hf["depth"], dtype=float).round(5).tolist(),
                "hregime": np.asarray(hf["regime"], dtype=float).round(3).tolist(),
                "valpha": vf_alpha.round(4).tolist(),
            }
        )
    payload = {
        "x": hx.round(5).tolist(),
        "y": vy.round(5).tolist(),
        "D": cfg.pipe_diameter,
        "Dr": cfg.riser_diameter,
        "L": cfg.pipe_length,
        "Hr": cfg.riser_height,
        "tee": cfg.tee_x,
        "arrival": arrival_time,
        "frames": frames,
    }
    html = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Cong 2017 BH1 complete own-model reproduction</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:20px;background:#f7f7f7;color:#222}
.wrap{max-width:1220px;margin:auto;background:white;border:1px solid #ddd;border-radius:10px;padding:16px}
canvas{width:100%;height:520px;border:1px solid #ccc;border-radius:8px;background:white}
.row{display:flex;gap:12px;align-items:center;margin-top:12px}
input[type=range]{flex:1}
.meta{font-size:13px;color:#555;line-height:1.5}
</style>
</head>
<body>
<div class="wrap">
<h2>Cong 2017 B-H1: own-model full layout reproduction</h2>
<canvas id="cv" width="1160" height="520"></canvas>
<div class="row"><button id="play">播放</button><input id="slider" type="range" min="0" max="0" value="0"><span id="label"></span></div>
<p class="meta">水平管和竖管均由论文算法框架推进：分层支路、气相压力闭合、RT/cut-cell 结构；T 接口用 B-H1 实测到达时间 Ta=8.07s 触发竖管底部气囊进入。蓝色为水，浅色为空气，紫色为近满管/有压水段。</p>
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
  const left=60, pipeTop=330, pipeW=850, pipeH=58;
  const teeX = left + pipeW*DATA.tee/DATA.L;
  const riserH=250, riserW=42, riserLeft=teeX-riserW/2, riserBottom=pipeTop;
  ctx.fillStyle="#222"; ctx.font="16px sans-serif";
  ctx.fillText("t = "+fr.t.toFixed(2)+" s", left, 42);
  ctx.fillText(fr.t < DATA.arrival ? "Stage 1: horizontal release before air reaches T" : "Stage 2: air enters vertical riser", left, 66);

  ctx.strokeStyle="#222"; ctx.lineWidth=2; ctx.strokeRect(left, pipeTop, pipeW, pipeH);
  for(let k=0;k<DATA.x.length;k++){
    const x=DATA.x[k], xp=left+pipeW*x/DATA.L, xp2=left+pipeW*(k+1)/DATA.x.length;
    const h=Math.max(0, Math.min(DATA.D, fr.hdepth[k]));
    const hw=pipeH*h/DATA.D;
    ctx.fillStyle=fr.hregime[k]>0.5 ? "#7b68b6" : "#4f9bd2";
    ctx.fillRect(xp, pipeTop+pipeH-hw, Math.max(1,xp2-xp+1), hw);
    ctx.fillStyle="#f0e5b8";
    ctx.fillRect(xp, pipeTop, Math.max(1,xp2-xp+1), pipeH-hw);
  }
  ctx.strokeStyle="#b22"; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(teeX,pipeTop-8); ctx.lineTo(teeX,pipeTop+pipeH+8); ctx.stroke();
  ctx.fillStyle="#b22"; ctx.fillText("T", teeX+8, pipeTop+pipeH+22);

  ctx.strokeStyle="#222"; ctx.lineWidth=2; ctx.strokeRect(riserLeft, riserBottom-riserH, riserW, riserH);
  for(let k=0;k<DATA.y.length;k++){
    const y=DATA.y[k], yp=riserBottom-riserH*y/DATA.Hr, yp2=riserBottom-riserH*(k+1)/DATA.y.length;
    const a=Math.max(0, Math.min(1, fr.valpha[k]));
    ctx.fillStyle="#f0e5b8";
    ctx.fillRect(riserLeft, yp2, riserW, Math.max(1, yp-yp2+1));
    if(a>0.03){
      ctx.fillStyle=a>0.98 ? "#7b68b6" : "#4f9bd2";
      ctx.fillRect(riserLeft, yp2, riserW, Math.max(1, yp-yp2+1));
    }
  }
  ctx.fillStyle="#222"; ctx.font="13px sans-serif";
  ctx.fillText("riser Dr="+DATA.Dr+" m, H="+DATA.Hr+" m", riserLeft-66, riserBottom-riserH-12);
  ctx.fillText("horizontal pipe L="+DATA.L+" m, D="+DATA.D+" m", left, pipeTop+pipeH+48);
  ctx.fillText("Ta trigger="+DATA.arrival+" s", teeX+30, riserBottom-120);
  label.textContent = i+" / "+(DATA.frames.length-1)+"   t="+fr.t.toFixed(2)+" s";
}
slider.oninput=()=>draw(+slider.value);
play.onclick=()=>{
  if(timer){clearInterval(timer); timer=null; play.textContent="播放"; return;}
  play.textContent="暂停";
  timer=setInterval(()=>{let i=(+slider.value+1)%DATA.frames.length; slider.value=i; draw(i);}, 120);
};
draw(0);
</script>
</body>
</html>""".replace("__DATA__", json.dumps(payload))
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=HERE / "outputs" / "own_model_bh1_complete")
    parser.add_argument("--arrival-time", type=float, default=8.07)
    parser.add_argument("--vertical-time", type=float, default=1.6)
    parser.add_argument("--output-dt", type=float, default=0.08)
    args = parser.parse_args()

    cfg = horiz.CongBH1()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    horiz_dir = HERE / "outputs" / "own_model_bh1"
    horiz.run(cfg, n=330, t_end=args.arrival_time, output_dt=args.output_dt, cfl=0.28, outdir=horiz_dir)
    hx, hframes = read_horizontal_profile(horiz_dir / "cong2017_bh1_own_model_profile.dat")
    vy, vframes, vmeta = run_vertical_riser(cfg, n=240, t_end=args.vertical_time, output_dt=args.output_dt, cfl=0.22)

    write_complete_profile(args.out_dir / "riser_profile.dat", vy, vframes)
    viewer = args.out_dir / "index.html"
    write_viewer(viewer, cfg, hx, hframes, vy, vframes, args.arrival_time, args.output_dt)
    summary = {
        "model": "own decoupled two-fluid paper solver, full B-H1 layout driver",
        "horizontal_solver": "paper stratified/pressurized branch with RT/cut-cell",
        "vertical_solver": "same paper two-fluid variables with theta=90deg and bottom gas-pocket injection",
        "case": {
            "pipe_length_m": cfg.pipe_length,
            "pipe_diameter_m": cfg.pipe_diameter,
            "tee_x_m": cfg.tee_x,
            "riser_height_m": cfg.riser_height,
            "riser_diameter_m": cfg.riser_diameter,
            "upstream_head_h0_m": cfg.head_h0,
            "air_pocket_length_m": cfg.air_pocket_length,
            "arrival_time_s": args.arrival_time,
        },
        "vertical_meta": vmeta,
        "viewer": str(viewer),
        "riser_profile": str(args.out_dir / "riser_profile.dat"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"written: {viewer}")


if __name__ == "__main__":
    main()

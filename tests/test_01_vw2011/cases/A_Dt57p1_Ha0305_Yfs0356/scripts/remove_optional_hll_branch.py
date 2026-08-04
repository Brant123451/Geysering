"""Collapse the Case A tunnel solver to its production contact-preserving branch.

This one-time, guarded refactor removes the optional HLL/Roe initial-release and
reflection machinery.  It refuses to edit an unexpected source layout.
"""
from __future__ import annotations

import re
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "model" / "vw2011_network_twofluid.py"


def replace_once(text: str, pattern: str, replacement: str, label: str, flags: int = re.S) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"Expected one {label} block, found {count}")
    return updated


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    if "use_mixed_flow_hll" not in text:
        raise RuntimeError("The optional HLL branch is already absent")

    text = replace_once(
        text,
        r"    use_mixed_flow_hll: bool = False.*?    mixed_flow_roe_entropy_fix: float = 0\.10.*?\n",
        "",
        "optional mixed-flow configuration",
    )
    text = text.replace("Alt[capsule] = 0.0 if case.use_mixed_flow_hll else 0.02 * A", "Alt[capsule] = 0.02 * A")
    text = replace_once(
        text,
        r"    initial_bore_contact_seen = not case\.use_mixed_flow_hll\n"
        r"    initial_bore_done = not case\.use_mixed_flow_hll\n"
        r"    initial_bore_impact_time = float\(\"nan\"\)\n"
        r"    initial_bore_reflection_time = float\(\"nan\"\)\n"
        r"    reflected_bore_front_x = float\(\"nan\"\)\n",
        "",
        "initial-release state",
        flags=0,
    )
    text = replace_once(
        text,
        r"        reflection_handoff_this_step = False\n",
        "",
        "reflection-step flag",
        flags=0,
    )
    text = replace_once(
        text,
        r"        pressure_gas_min = 0\.01 if not initial_bore_done else 0\.05\n"
        r"        Pt, wett, _ = _pressure\(Alt, np\.full\(Nt, A\), Mgt, dx, a2, vent_top=False, p_floor=0\.0,\n"
        r"                                gas_min=pressure_gas_min, tension_head=TENSION_HEAD,\n"
        r"                                mass_consistent=not initial_bore_done\)",
        "        Pt, wett, _ = _pressure(Alt, np.full(Nt, A), Mgt, dx, a2, vent_top=False, p_floor=0.0,\n"
        "                                gas_min=0.05, tension_head=TENSION_HEAD, mass_consistent=False)",
        "pressure switch",
    )
    text = replace_once(
        text,
        r"        poc_mask = \(_pocket_mask\(Alt, A, Mgt, dx, gas_min=0\.01\)\n"
        r"                    if not initial_bore_done else \(Alt / A\) < 0\.95\)",
        "        poc_mask = (Alt / A) < 0.95",
        "pocket-mask switch",
    )
    text = replace_once(
        text,
        r"        F1_wet = 0\.5 \* \(f1\[:-1\] \+ f1\[1:\]\) - 0\.5 \* sf \* \(Alg\[1:\] - Alg\[:-1\]\)\n"
        r"        F2_wet = 0\.5 \* \(f2\[:-1\] \+ f2\[1:\]\) - 0\.5 \* sf \* \(Qlg\[1:\] - Qlg\[:-1\]\)\n"
        r"        initial_fill_active =.*?\n"
        r"            F1, F2 = F1_base, F2_base\n",
        "        F1_wet = 0.5 * (f1[:-1] + f1[1:]) - 0.5 * sf * (Alg[1:] - Alg[:-1])\n"
        "        F2_wet = 0.5 * (f2[:-1] + f2[1:]) - 0.5 * sf * (Qlg[1:] - Qlg[:-1])\n"
        "        s_mat = (np.maximum(np.abs(uLf[:-1]), np.abs(uLf[1:]))\n"
        "                 + math.sqrt(G * case.D))\n"
        "        uf_t = 0.5 * (uLf[:-1] + uLf[1:])\n"
        "        F1 = np.where(wet_face, F1_wet,\n"
        "                      np.where(uf_t >= 0.0, f1[:-1], f1[1:]))\n"
        "        F2 = np.where(wet_face, F2_wet,\n"
        "                      0.5 * (f2[:-1] + f2[1:])\n"
        "                      - 0.5 * s_mat * (Qlg[1:] - Qlg[:-1]))\n",
        "mixed-flow flux",
    )
    text = replace_once(
        text,
        r"        if initial_fill_active:.*?        # sources: pressure gradient \(theta=0, no gravity\) \+ friction\n"
        r"        if initial_fill_active:.*?        Pth = np\.empty\(Nt \+ 2\); Pth\[1:-1\] = Pt_momentum",
        "        # sources: pressure gradient (theta=0, no gravity) + friction\n"
        "        Pt_momentum = Pt\n"
        "        Pth = np.empty(Nt + 2); Pth[1:-1] = Pt_momentum",
        "wetting-front correction and pressure switch",
    )
    text = replace_once(
        text,
        r"        # The HLL branch resolves only the initial full-to-dry release\..*?"
        r"        # ---------- gas transport: pocket-front propagation \(crown gravity current\) ----------\n",
        "        # ---------- gas transport: pocket-front propagation (crown gravity current) ----------\n",
        "reflection hand-off",
    )
    text = replace_once(
        text,
        r"        if \(case\.use_mixed_flow_hll\n"
        r"                and \(not initial_bore_done or reflection_handoff_this_step\)\):.*?"
        r"        body_thr = 0\.05",
        "        body_thr = 0.05",
        "gas-mass hand-off",
    )
    text = replace_once(
        text,
        r"            elif not initial_bore_done:.*?                u_rel \*= phi_flow\n",
        "",
        "initial-release crown switch",
    )
    text = text.replace("if initial_bore_done and len(regs) > 1:", "if len(regs) > 1:")
    text = replace_once(
        text,
        r"    rec\[\"initial_bore_impact_time\"\] = float\(initial_bore_impact_time\)\n"
        r"    rec\[\"initial_bore_reflection_time\"\] = float\(initial_bore_reflection_time\)\n"
        r"    rec\[\"reflected_bore_front_x\"\] = float\(reflected_bore_front_x\)\n",
        "",
        "release diagnostics",
        flags=0,
    )
    if re.search(r"use_mixed_flow|initial_bore_|reflection_|mixed_flow_", text):
        raise RuntimeError("The refactor left optional-HLL controls behind")
    SOURCE.write_text(text, encoding="utf-8")
    print(f"Removed optional HLL/Roe release machinery from {SOURCE}")


if __name__ == "__main__":
    main()

"""Redraw the three experimental layouts used in the manuscript.

Campaign 1 is a close vector trace of Vasconcelos and Wright (2011, Fig. 2),
including the source component arrangement, leaders, dimensions, and labels.
Campaigns 2 and 3 follow Cong et al. (2017, Fig. 1) and Liu et al. (2020,
Fig. 2), respectively, and remain deliberately diagrammatic.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle


OUT = Path(__file__).resolve().parents[1] / "figures"
WATER = "#bfe8f7"
AIR = "#eeeeee"
INK = "#111111"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.bfit": "Times New Roman:bold:italic",
        "mathtext.fallback": "stix",
        "mathtext.default": "it",
        "text.usetex": False,
        "font.size": 9.2,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def setup_ax(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return fig, ax


def save(fig, stem, *, enlarge_text=True):
    OUT.mkdir(parents=True, exist_ok=True)
    # The vector drawings are reduced to manuscript text width.  Enlarge all
    # in-figure labels before export so the final PDF remains readable in print.
    if enlarge_text:
        for ax in fig.axes:
            for label in ax.texts:
                label.set_fontsize(label.get_fontsize() * 1.25)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUT / f"{stem}.png", dpi=450, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def dim(ax, x0, x1, y, label, text_offset=-0.12):
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops=dict(arrowstyle="|-|", lw=0.75, color=INK, shrinkA=0, shrinkB=0),
    )
    ax.text((x0 + x1) / 2, y + text_offset, label, ha="center", va="top", fontsize=8.3)


def valve(ax, x, y, h=0.34, label=None, label_y=None):
    ax.plot([x, x], [y - h / 2, y + h / 2], color=INK, lw=1.2)
    ax.plot([x - 0.10, x + 0.10], [y + 0.13, y + 0.13], color=INK, lw=0.8)
    if label:
        ax.text(x, label_y if label_y is not None else y + 0.28, label, ha="center", va="bottom", fontsize=8.0)


def pressure_tap(ax, x, y, label, side="below"):
    ax.add_patch(Circle((x, y), 0.025, fc=INK, ec=INK))
    dy = -0.25 if side == "below" else 0.25
    va = "top" if side == "below" else "bottom"
    ax.text(x, y + dy, label, ha="center", va=va, fontsize=8.0)


def flow_arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", lw=0.85, color=INK))


def campaign1():
    # Coordinates follow the relative positions in the published Fig. 2.
    # The source caption remains in LaTeX rather than inside this figure.
    fig, ax = setup_ax((11.2, 5.55), (0.0, 20.4), (0.25, 10.35))
    grey_water = "#bdbdbd"
    grey_coupling = "#8c8c8c"
    grey_valve = "#707070"
    source_font = dict(fontfamily="Times New Roman", color=INK)

    def source_text(x, y, value, *, size=9.1, ha="left", va="center", **kwargs):
        ax.text(
            x,
            y,
            value,
            fontsize=size,
            ha=ha,
            va=va,
            linespacing=1.10,
            **source_font,
            **kwargs,
        )

    def horizontal_dimension_end(x, y, *, cap=0.38, slash=0.26):
        """Source endpoint for a horizontal dimension: vertical cap plus one /."""
        ax.plot([x, x], [y - cap / 2, y + cap / 2], color=INK, lw=0.9)
        half = slash / 2
        ax.plot([x - half, x + half], [y - half, y + half], color=INK, lw=0.9)

    def vertical_dimension_end(x, y, *, cap=0.26, slash=0.22):
        """Source endpoint for a vertical dimension: horizontal cap plus one backslash."""
        ax.plot([x - cap / 2, x + cap / 2], [y, y], color=INK, lw=0.9)
        half = slash / 2
        ax.plot([x - half, x + half], [y + half, y - half], color=INK, lw=0.9)

    pipe_bottom, pipe_top = 4.00, 4.92
    upstream_left = 1.22
    valve_left, valve_right = 4.45, 5.18
    break_x = 11.30
    coupling_left, coupling_right = 14.64, 16.04
    downstream_right = 19.05

    # Differential manometer and air-pump connections.
    ax.add_patch(Rectangle((0.68, 7.65), 4.24, 1.18, fc="white", ec=INK, lw=1.0))
    source_text(0.84, 8.24, "Differential manometer,\nprecision 0.031 m", size=9.0)
    ax.plot([0.88, 0.88, upstream_left], [7.65, 4.18, 4.18], color=INK, lw=1.1)
    ax.add_patch(Rectangle((1.06, 6.38), 2.30, 0.82, fc="white", ec=INK, lw=1.0))
    source_text(2.21, 6.79, "Air pump", size=9.2, ha="center")
    # In the source, the pump is tapped from its left side into a vertical
    # branch that enters the upstream air-filled pipe at mid-depth.
    ax.plot([1.06, 1.06, upstream_left], [7.65, 4.55, 4.55], color=INK, lw=1.05)

    # Upstream air-filled reach, water-filled middle reach, and downstream reach.
    ax.add_patch(
        Rectangle(
            (upstream_left, pipe_bottom),
            valve_left - upstream_left,
            pipe_top - pipe_bottom,
            fc="white",
            ec=INK,
            lw=1.0,
        )
    )
    ax.add_patch(
        Rectangle(
            (valve_right, pipe_bottom),
            coupling_left - valve_right,
            pipe_top - pipe_bottom,
            fc=grey_water,
            ec=INK,
            lw=1.0,
        )
    )
    ax.add_patch(
        Rectangle(
            (coupling_right, pipe_bottom),
            downstream_right - coupling_right,
            pipe_top - pipe_bottom,
            fc=grey_water,
            ec=INK,
            lw=1.0,
        )
    )

    # Butterfly valve body, central leaf, stem, and hand wheel.
    ax.add_patch(
        Rectangle(
            (valve_left, pipe_bottom - 0.16),
            valve_right - valve_left,
            pipe_top - pipe_bottom + 0.32,
            fc=grey_valve,
            ec=INK,
            lw=1.0,
        )
    )
    valve_centre = 0.5 * (valve_left + valve_right)
    valve_scale_left = valve_centre - 0.155
    valve_scale_right = valve_centre + 0.155
    ax.add_patch(
        Rectangle(
            (valve_centre - 0.085, pipe_bottom - 0.25),
            0.17,
            pipe_top - pipe_bottom + 0.50,
            fc=INK,
            ec=INK,
            lw=0.8,
        )
    )
    ax.plot([valve_centre, valve_centre], [pipe_top + 0.25, 5.92], color=INK, lw=1.0)
    wheel = Ellipse(
        (valve_centre, 6.08), 0.88, 0.30, fc="white", ec=INK, lw=1.0
    )
    ax.add_patch(wheel)
    ax.plot([valve_centre - 0.36, valve_centre + 0.36], [6.01, 6.15], color=INK, lw=0.8)
    ax.plot([valve_centre - 0.36, valve_centre + 0.36], [6.15, 6.01], color=INK, lw=0.8)

    # Published 1.07-m sensor offset and pressure transducer.
    tap_x = 8.28
    dim_y = 3.72
    ax.plot([valve_scale_right, tap_x], [dim_y, dim_y], color=INK, lw=0.9)
    horizontal_dimension_end(valve_scale_right, dim_y)
    horizontal_dimension_end(tap_x, dim_y)
    source_text((valve_scale_right + tap_x) / 2, 3.45, "1.07 m", size=9.0, ha="center")
    ax.add_patch(Rectangle((tap_x - 0.14, pipe_bottom - 0.18), 0.28, 0.18, fc=grey_valve, ec=INK, lw=0.9))
    ax.plot([tap_x, tap_x + 0.34], [pipe_bottom - 0.18, 2.74], color=INK, lw=0.9)
    source_text(6.15, 2.45, "Piezo-resistive pressure transducer", size=8.9)

    # Source-style pipe break: two parallel vertical jagged cuts cross the
    # upper dimension line and both pipe walls.  The gap is physical white
    # space rather than four detached slash marks.
    cut_left = break_x - 0.16
    cut_right = break_x + 0.16
    cut_bottom = pipe_bottom - 0.34
    cut_top = 5.78
    ax.add_patch(
        Rectangle(
            (cut_left, cut_bottom),
            cut_right - cut_left,
            cut_top - cut_bottom,
            fc="white",
            ec="none",
            zorder=5,
        )
    )

    cut_mid = 0.5 * (pipe_bottom + pipe_top)

    def jagged_cut(x):
        ax.plot(
            [
                x,
                x,
                x + 0.18,
                x + 0.07,
                x + 0.22,
                x,
                x,
            ],
            [
                cut_top,
                cut_mid + 0.31,
                cut_mid + 0.15,
                cut_mid + 0.05,
                cut_mid - 0.13,
                cut_mid - 0.27,
                cut_bottom,
            ],
            color=INK,
            lw=1.0,
            zorder=6,
        )

    jagged_cut(cut_left)
    jagged_cut(cut_right)

    # Middle-pipe and downstream-pipe dimension lines and labels.
    upper_dim_y = 5.43
    ax.plot([upstream_left, valve_scale_left], [upper_dim_y, upper_dim_y], color=INK, lw=0.9)
    horizontal_dimension_end(upstream_left, upper_dim_y)
    horizontal_dimension_end(valve_scale_left, upper_dim_y)
    ax.plot([valve_scale_right, break_x - 0.17], [upper_dim_y, upper_dim_y], color=INK, lw=0.9)
    ax.plot([break_x + 0.17, coupling_left], [upper_dim_y, upper_dim_y], color=INK, lw=0.9)
    horizontal_dimension_end(valve_scale_right, upper_dim_y)
    horizontal_dimension_end(coupling_left, upper_dim_y)
    source_text(
        7.30,
        6.16,
        "Middle pipe, initially filled with water\n"
        r"$L$=2.97 m, $D$=0.094 m",
        size=9.0,
    )
    ax.plot([coupling_right, downstream_right], [upper_dim_y, upper_dim_y], color=INK, lw=0.9)
    horizontal_dimension_end(coupling_right, upper_dim_y)
    horizontal_dimension_end(downstream_right, upper_dim_y)
    source_text(
        16.15,
        6.25,
        "Downstream pipe\n"
        r"$L$=0.490 m, $D$=0.094 m",
        size=8.8,
    )

    # PVC coupling and ventilation tower, including the scale line shown in Fig. 2.
    ax.add_patch(
        Rectangle(
            (coupling_left, pipe_bottom - 0.17),
            coupling_right - coupling_left,
            pipe_top - pipe_bottom + 0.34,
            fc=grey_coupling,
            ec=INK,
            lw=1.0,
            zorder=4,
        )
    )
    tower_left, tower_right = 14.97, 15.43
    tower_bottom, tower_top = pipe_top + 0.17, 9.37
    water_top = 7.47
    ax.add_patch(Rectangle((tower_left, tower_bottom), tower_right - tower_left, water_top - tower_bottom, fc=grey_water, ec="none", zorder=3))
    ax.add_patch(Rectangle((tower_left, water_top), tower_right - tower_left, tower_top - water_top, fc="white", ec="none", zorder=3))
    ax.plot([tower_left, tower_left], [tower_bottom, tower_top], color=INK, lw=1.0, zorder=5)
    ax.plot([tower_right, tower_right], [tower_bottom, tower_top], color=INK, lw=1.0, zorder=5)
    ax.plot([tower_left, tower_right], [tower_top, tower_top], color=INK, lw=1.0, zorder=5)
    ax.plot([tower_left, tower_right], [water_top, water_top], color=INK, lw=0.7, zorder=5)
    scale_x = 15.58
    ax.plot([scale_x, scale_x], [tower_bottom, tower_top], color=INK, lw=0.8)
    vertical_dimension_end(scale_x, tower_bottom)
    vertical_dimension_end(scale_x, tower_top)

    source_text(10.55, 8.12, "Initial water level in\nthe ventilation tower", size=8.9)
    ax.plot([13.62, tower_left], [7.93, water_top], color=INK, lw=0.9)
    source_text(
        15.82,
        8.18,
        "Ventilation tower\n" r"$L$=0.610 m, variable $D$",
        size=8.9,
    )

    coupling_dim_y = 3.46
    ax.plot([coupling_left, coupling_right], [coupling_dim_y, coupling_dim_y], color=INK, lw=0.9)
    horizontal_dimension_end(coupling_left, coupling_dim_y)
    horizontal_dimension_end(coupling_right, coupling_dim_y)
    source_text(
        15.34,
        2.92,
        "PVC coupling\n" r"$L$=0.140 m",
        size=8.9,
        ha="center",
    )

    # Lower source labels and leaders.
    source_text(
        0.62,
        2.67,
        "Upstream pipe, initially\nfilled with air\n"
        r"$L$=0.546 m, $D$=0.094 m",
        size=8.9,
    )
    ax.plot([2.42, 2.08], [pipe_bottom, 3.18], color=INK, lw=0.9)
    source_text(
        4.72,
        1.20,
        "Butterfly valve - " r"$D$=0.102 m",
        size=8.9,
        ha="center",
    )
    ax.plot([valve_centre - 0.08, 5.36], [pipe_bottom - 0.25, 1.72], color=INK, lw=0.9)
    source_text(12.34, 1.24, "Near-horizontal slope", size=8.9)
    ax.plot([12.12, 13.02], [pipe_bottom, 1.73], color=INK, lw=0.9)

    save(fig, "campaign1_apparatus_redrawn", enlarge_text=False)


def campaign2():
    fig, ax = setup_ax((11.2, 4.15), (-0.2, 10.4), (-0.45, 3.45))
    y0, h = 0.80, 0.34
    tank_l, tank_r = 0.35, 1.85
    pipe_l, pipe_r = tank_r, 9.55
    tee = 5.55

    # Constant-head upstream tank and horizontal pipe (Series B).
    ax.add_patch(Rectangle((tank_l, y0 - 0.05), tank_r - tank_l, 1.85, fc="white", ec="none"))
    ax.add_patch(Rectangle((tank_l, y0), tank_r - tank_l, 1.00, fc=WATER, ec="none"))
    ax.plot([tank_l, tank_l], [y0 - 0.05, 2.75], color=INK, lw=1.0)
    ax.plot([tank_r, tank_r], [y0 - 0.05, 2.75], color=INK, lw=1.0)
    ax.plot([tank_l, tank_r], [y0, y0], color=INK, lw=1.0)
    ax.text((tank_l + tank_r) / 2, 2.90, "Constant-head tank", ha="center", va="bottom")
    ax.text((tank_l + tank_r) / 2, 1.30, "$H_0$", ha="center", va="center")

    # The water-filled portion ends at the selected valve; the remaining reach is the pocket.
    pocket_l = 8.32
    ax.add_patch(Rectangle((pipe_l, y0), pocket_l - pipe_l, h, fc=WATER, ec="none"))
    ax.add_patch(Rectangle((pocket_l, y0), pipe_r - pocket_l, h, fc=AIR, ec="none"))
    ax.plot([pipe_l, pipe_r], [y0, y0], color=INK, lw=1.0)
    ax.plot([pipe_l, tee - 0.10], [y0 + h, y0 + h], color=INK, lw=1.0)
    ax.plot([tee + 0.10, pipe_r], [y0 + h, y0 + h], color=INK, lw=1.0)
    ax.plot([pipe_r, pipe_r], [y0, y0 + h], color=INK, lw=1.0)

    # Vertical riser and initial water level.
    rw = 0.18
    ax.add_patch(Rectangle((tee - rw / 2, y0 + h), rw, 1.65, fc=WATER, ec="none"))
    ax.plot([tee - rw / 2, tee - rw / 2], [y0 + h, 3.05], color=INK, lw=1.0)
    ax.plot([tee + rw / 2, tee + rw / 2], [y0 + h, 3.05], color=INK, lw=1.0)
    ax.text(tee, 3.18, "Vertical riser\n$L_r=1.8$ m, variable $D_r$", ha="center", va="bottom")
    ax.annotate("Initial level", xy=(tee, 2.79), xytext=(4.45, 2.55),
                arrowprops=dict(arrowstyle="->", lw=0.75), ha="right", va="center", fontsize=8.1)

    valve(ax, pipe_l + 0.10, y0 + h / 2, label="Valve 1\n(open)", label_y=1.45)
    for x, lab in [(6.60, "Valve 2"), (7.35, "Valve 3"), (8.20, "Valve 4")]:
        valve(ax, x, y0 + h / 2, label=lab, label_y=1.40)
    ax.add_patch(Circle((tee, y0), 0.025, fc=INK, ec=INK))
    ax.text(tee - 0.10, y0 + 0.10, "PT2", ha="right", va="bottom", fontsize=7.8)
    pressure_tap(ax, 9.05, y0 + h, "PT1", side="above")
    ax.text((pocket_l + pipe_r) / 2, y0 + h / 2, "Air pocket, $L_0$", ha="center", va="center", fontsize=8.3)
    ax.text(4.05, y0 + h / 2, "Horizontal pipe, $D=0.05$ m", ha="center", va="center", fontsize=8.4)
    flow_arrow(ax, 8.00, 6.95, y0 + h / 2)

    dim(ax, pipe_l, tee, 0.52, "$3.47$ m")
    dim(ax, tee, pocket_l, 0.32, "$3.12-L_0$", text_offset=-0.10)
    dim(ax, pocket_l, pipe_r, 0.32, "$L_0$", text_offset=-0.10)

    # Recirculation loop and downstream tank.
    ax.add_patch(Rectangle((9.72, 0.25), 0.55, 1.18, fc="white", ec="none"))
    ax.plot([9.72, 10.27, 10.27], [0.25, 0.25, 1.43], color=INK, lw=0.9)
    ax.plot([9.72, 9.72], [0.25, 0.78], color=INK, lw=0.9)
    ax.add_patch(Rectangle((9.72, 0.25), 0.55, 0.34, fc=WATER, ec="none"))
    ax.text(9.995, 1.56, "Downstream\ntank", ha="center", va="bottom", fontsize=8.0)
    ax.plot([pipe_r, 9.72], [y0 + h / 2, y0 + h / 2], color=INK, lw=0.9)
    ax.text(9.62, 1.32, "cap", ha="center", va="bottom", fontsize=8.0)
    ax.plot([9.72, 9.72, 0.05, 0.05, tank_l], [0.25, -0.12, -0.12, 0.72, 0.72], color=INK, lw=0.8)
    ax.add_patch(Circle((2.55, -0.12), 0.13, fc="white", ec=INK, lw=0.8))
    ax.plot([2.46, 2.64], [-0.21, -0.03], color=INK, lw=0.8)
    ax.plot([2.46, 2.64], [-0.03, -0.21], color=INK, lw=0.8)
    ax.text(2.55, -0.32, "Pump", ha="center", va="top", fontsize=8.0)
    flow_arrow(ax, 3.70, 2.95, -0.12)
    ax.text(5.65, -0.30, "Recirculating flow", ha="center", va="top", fontsize=8.1)
    ax.text(0.15, 3.42, "Series B", ha="left", va="top", fontweight="bold")
    save(fig, "campaign2_apparatus_redrawn")


def _liu_panel(ax, y, series, show_pocket=False, open_channel=False):
    x_tank_l, x_tank_r = 0.25, 1.30
    x_ch_l, x_ch_r = 4.20, 4.92
    x_right = 9.70
    down_y = y + 0.05
    up_y = y + 0.48
    down_h, up_h = 0.34, 0.34

    # Downstream tank and pipe.
    tank_level = down_y + (0.22 if open_channel else 0.55)
    ax.add_patch(Rectangle((x_tank_l, y - 0.05), x_tank_r - x_tank_l, tank_level - (y - 0.05), fc=WATER, ec="none"))
    ax.plot([x_tank_l, x_tank_l], [y - 0.05, y + 1.20], color=INK, lw=1.0)
    ax.plot([x_tank_r, x_tank_r], [y - 0.05, y + 1.20], color=INK, lw=1.0)
    ax.plot([x_tank_l, x_tank_r], [y - 0.05, y - 0.05], color=INK, lw=1.0)

    if open_channel:
        ax.add_patch(Rectangle((x_tank_r, down_y), x_ch_l - x_tank_r, down_h * 0.55, fc=WATER, ec="none"))
    else:
        ax.add_patch(Rectangle((x_tank_r, down_y), x_ch_l - x_tank_r, down_h, fc=WATER, ec="none"))
    ax.plot([x_tank_r, x_ch_l], [down_y, down_y], color=INK, lw=1.0)
    ax.plot([x_tank_r, x_ch_l], [down_y + down_h, down_y + down_h], color=INK, lw=1.0)

    # Junction chamber, riser, and invert drop.
    if open_channel:
        chamber_water = Polygon(
            [(x_ch_l, down_y), (x_ch_r, down_y), (x_ch_r, up_y + 0.16), (x_ch_l, down_y + down_h * 0.55)],
            closed=True,
            fc=WATER,
            ec="none",
        )
        ax.add_patch(chamber_water)
    else:
        ax.add_patch(Rectangle((x_ch_l, down_y), x_ch_r - x_ch_l, up_y + up_h - down_y, fc=WATER, ec="none"))
    ax.add_patch(Rectangle((x_ch_l, down_y), x_ch_r - x_ch_l, up_y + up_h - down_y, fc="none", ec=INK, lw=1.0))
    riser_x, rw = (x_ch_l + x_ch_r) / 2, 0.18
    riser_top = y + 1.78
    riser_fill = up_y + up_h if open_channel else y + 1.03
    ax.add_patch(Rectangle((riser_x - rw / 2, up_y + up_h), rw, max(riser_fill - (up_y + up_h), 0), fc=WATER, ec="none"))
    ax.plot([riser_x - rw / 2, riser_x - rw / 2], [up_y + up_h, riser_top], color=INK, lw=1.0)
    ax.plot([riser_x + rw / 2, riser_x + rw / 2], [up_y + up_h, riser_top], color=INK, lw=1.0)

    # Upstream pipe with a slight 1:100 slope.
    slope = 0.10
    if open_channel:
        water_poly = Polygon(
            [(x_ch_r, up_y), (x_right, up_y + slope), (x_right, up_y + slope + 0.16), (x_ch_r, up_y + 0.16)],
            closed=True,
            fc=WATER,
            ec="none",
        )
        ax.add_patch(water_poly)
    else:
        poly = Polygon([(x_ch_r, up_y), (x_right, up_y + slope), (x_right, up_y + up_h + slope), (x_ch_r, up_y + up_h)], closed=True, fc=WATER, ec="none")
        ax.add_patch(poly)
    ax.plot([x_ch_r, x_right], [up_y, up_y + slope], color=INK, lw=1.0)
    ax.plot([x_ch_r, x_right], [up_y + up_h, up_y + up_h + slope], color=INK, lw=1.0)
    valve(ax, 9.30, up_y + up_h / 2 + 0.09, h=0.32, label="Ball valve", label_y=up_y + 0.72)
    flow_arrow(ax, 9.10, 8.25, up_y + up_h / 2 + 0.08)
    flow_arrow(ax, 4.00, 3.62, down_y + down_h / 2)

    if show_pocket:
        pocket = Polygon([(6.55, up_y + 0.20), (8.35, up_y + 0.24), (8.35, up_y + up_h + 0.067), (6.55, up_y + up_h * 0.72)], closed=True, fc=AIR, ec=INK, lw=0.7)
        ax.add_patch(pocket)
        ax.text(7.42, up_y + 0.29, "Air pocket (C9)", ha="center", va="center", fontsize=7.2)
        ax.text(7.35, up_y - 0.06, "B3: same full-pipe state without the pocket", ha="center", va="top", fontsize=7.4)

    # Tail control and instrument positions.
    if open_channel:
        ax.plot([0.95, 0.95], [y - 0.05, y + 0.48], color=INK, lw=1.3)
        ax.text(0.95, y + 0.56, "Overflow weir", ha="center", va="bottom", fontsize=7.7)
    else:
        ax.plot([0.95, 0.95], [y - 0.05, y + 0.72], color=INK, lw=1.3)
        ax.text(0.95, y + 0.80, "Tailgate", ha="center", va="bottom", fontsize=7.8)

    ax.add_patch(Circle((riser_x, y + 1.38), 0.025, fc=INK, ec=INK))
    ax.text(riser_x - 0.12, y + 1.52, "PT1", ha="right", va="center", fontsize=7.5)
    ax.add_patch(Circle((riser_x - 0.13, up_y + up_h), 0.025, fc=INK, ec=INK))
    ax.text(riser_x - 0.19, up_y + up_h + 0.18, "PT2", ha="right", va="center", fontsize=7.5)
    ax.add_patch(Circle((riser_x, down_y), 0.025, fc=INK, ec=INK))
    ax.text(riser_x, down_y - 0.12, "PT3", ha="center", va="top", fontsize=7.8)
    ax.add_patch(Circle((x_ch_r + 0.25, up_y + up_h), 0.025, fc=INK, ec=INK))
    ax.text(x_ch_r + 0.37, up_y + up_h + 0.16, "PT4", ha="left", va="center", fontsize=7.5)
    ax.text(2.75, down_y + down_h / 2, "Downstream pipe\n$L_d=5.95$ m, $D_d=0.28$ m", ha="center", va="center", fontsize=7.8)
    ax.text(7.35, up_y + up_h + 0.28, "$L_u=5.80$ m, $D_u=0.20$ m, slope 1:100", ha="center", va="bottom", fontsize=7.4)
    ax.text(riser_x + 0.20, y + 1.45, "Riser\n$L_r=1.22$ m\n$d_r=0.06$ m", ha="left", va="center", fontsize=7.5)
    ax.text(x_ch_l - 0.08, y + 0.63, "$0.18$ m\ndrop", ha="right", va="center", fontsize=7.6)
    ax.text(
        (x_ch_l + x_ch_r) / 2,
        y + 0.40,
        "Chamber\n$0.30\\times0.30$ m\n$H=0.45$ m",
        ha="center",
        va="center",
        fontsize=5.3,
        linespacing=0.92,
    )
    ax.text((x_tank_l + x_tank_r) / 2, y + 1.30, "Downstream tank", ha="center", va="bottom", fontsize=7.8)
    ax.text(0.03, y + 1.72, series, ha="left", va="top", fontweight="bold", fontsize=8.7)


def campaign3():
    fig, ax = setup_ax((11.2, 5.35), (-0.1, 10.15), (-0.35, 4.55))
    _liu_panel(ax, 2.38, "(a) Series A / Case A2", show_pocket=False, open_channel=True)
    _liu_panel(ax, 0.08, "(b) Series B/C / Cases B3 and C9", show_pocket=True, open_channel=False)
    save(fig, "campaign3_apparatus_redrawn")


if __name__ == "__main__":
    campaign1()
    campaign2()
    campaign3()

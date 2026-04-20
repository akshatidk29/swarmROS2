#!/usr/bin/env python3
"""Swarm navigation report generator."""

import json
import argparse
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as pe

BG        = "#0D0D0D"
PANEL     = "#141414"
GRID_COL  = "#222222"
BORDER    = "#333333"
TEXT_PRI  = "#E8E8E8"
TEXT_SEC  = "#888888"
ACCENT    = "#00C8D4"
DANGER    = "#CC3333"
SAFE      = "#2A7A5A"
BLUE_TRK  = "#2255AA"

ROBOT_COLS = {
    "robot_1": "#CC3333",
    "robot_2": "#55AA66",
    "robot_3": "#4488AA",
}
ITEM_COLS = {
    "red":   "#CC3333",
    "green": "#55AA66",
    "blue":  "#4488AA",
}

MONO  = "DejaVu Sans Mono"
SANS  = "DejaVu Sans"


def apply_global_style():
    plt.rcParams.update({
        "figure.facecolor":     BG,
        "axes.facecolor":       PANEL,
        "axes.edgecolor":       BORDER,
        "axes.labelcolor":      TEXT_SEC,
        "axes.titlecolor":      TEXT_PRI,
        "xtick.color":          TEXT_SEC,
        "ytick.color":          TEXT_SEC,
        "xtick.labelcolor":     TEXT_SEC,
        "ytick.labelcolor":     TEXT_SEC,
        "grid.color":           GRID_COL,
        "grid.linewidth":       0.6,
        "grid.linestyle":       "-",
        "text.color":           TEXT_PRI,
        "font.family":          SANS,
        "font.size":            9,
        "axes.titlesize":       10,
        "axes.labelsize":       8,
        "legend.facecolor":     PANEL,
        "legend.edgecolor":     BORDER,
        "legend.labelcolor":    TEXT_SEC,
        "legend.fontsize":      7.5,
        "figure.dpi":           150,
        "savefig.facecolor":    BG,
        "savefig.edgecolor":    BG,
    })


def draw_header_bar(fig, title_main, title_sub=""):
    """
    Draws a full-width header rule at the very top of the figure.
    Returns nothing – purely decorative chrome.
    """
    fig.add_artist(patches.FancyArrowPatch(
        posA=(0, 0.985), posB=(1, 0.985),
        arrowstyle="-", color=ACCENT, linewidth=2.5,
        transform=fig.transFigure, zorder=100
    ))
    fig.text(0.04, 0.965, title_main.upper(),
             fontsize=13, fontweight="bold", color=TEXT_PRI,
             fontfamily=MONO, va="top", zorder=100)
    if title_sub:
        fig.text(0.04, 0.947, title_sub,
                 fontsize=7.5, color=TEXT_SEC,
                 fontfamily=MONO, va="top", zorder=100)
    fig.add_artist(patches.FancyArrowPatch(
        posA=(0, 0.015), posB=(1, 0.015),
        arrowstyle="-", color=BORDER, linewidth=0.8,
        transform=fig.transFigure, zorder=100
    ))
    fig.text(0.97, 0.012, "SWARM-NAV // CONFIDENTIAL",
             fontsize=6, color=BORDER, ha="right",
             fontfamily=MONO, zorder=100)


def styled_axes(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)
    ax.grid(True, color=GRID_COL, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title.upper(), color=ACCENT, fontsize=8.5,
                     fontfamily=MONO, pad=6, loc="left", fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel.upper(), fontsize=7, color=TEXT_SEC, fontfamily=MONO)
    if ylabel:
        ax.set_ylabel(ylabel.upper(), fontsize=7, color=TEXT_SEC, fontfamily=MONO)


def label_box(ax, x, y, text, color=ACCENT):
    """Small mono annotation box."""
    ax.annotate(text, xy=(x, y), fontsize=6.5, fontfamily=MONO, color=color,
                bbox=dict(boxstyle="square,pad=0.3", fc=BG, ec=color, lw=0.8), zorder=20)


def draw_world(ax):
    walls = [
        (-5.075, -4.075, 10.15, 0.15),
        (-5.075,  3.925, 10.15, 0.15),
        (-5.075, -4.075, 0.15,  8.15),
        ( 4.925, -4.075, 0.15,  8.15),
    ]
    for w in walls:
        ax.add_patch(patches.Rectangle(
            (w[0], w[1]), w[2], w[3],
            facecolor=BORDER, edgecolor=TEXT_SEC,
            linewidth=0.6, zorder=2))

    shelves = [
        (-2.5,  2.0, 1.4, 0.4, 0),
        (-2.5,  0.0, 2.0, 0.4, 0),
        (-2.5, -2.0, 2.0, 0.4, 0),
        ( 0.5,  2.0, 2.0, 0.4, 0),
        ( 0.5,  0.0, 2.0, 0.4, 0),
        ( 0.5, -2.0, 2.0, 0.4, 0),
        ( 3.5,  2.0, 1.5, 0.4, 20),
        ( 3.5, -2.0, 1.5, 0.4, 0),
    ]
    for cx, cy, w, h, angle in shelves:
        import matplotlib.transforms as mtransforms
        t = mtransforms.Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData
        rect = patches.Rectangle(
            (cx - w/2, cy - h/2), w, h,
            facecolor="#141C1E", edgecolor="#1E3A3E",
            linewidth=0.8, zorder=3, transform=t)
        ax.add_patch(rect)

    bins_data = [
        (-3.3,  2.7, 0.6, 0.6, ITEM_COLS["red"],   "R"),
        (-0.3, -3.3, 0.6, 0.6, ITEM_COLS["green"],  "G"),
        ( 2.7, -3.3, 0.6, 0.6, ITEM_COLS["blue"],   "B"),
    ]
    for bx, by, bw, bh, bc, label in bins_data:
        ax.add_patch(patches.Rectangle(
            (bx, by), bw, bh,
            facecolor=bc, alpha=0.12,
            edgecolor=bc, linewidth=1.2, zorder=4))
        ax.text(bx + bw/2, by + bh/2, label,
                fontsize=8, fontfamily=MONO, fontweight="bold",
                color=bc, ha="center", va="center", zorder=5)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID_COL, linewidth=0.4, linestyle="-", zorder=0)
    ax.set_axisbelow(True)


def extract_stats(data):
    events = data.get("events", [])
    robot_tasks = {}
    discoveries = {}

    for ev in events:
        bot, col, etype = ev["robot"], ev["color"], ev["type"]
        if etype == "discovered":
            if col not in discoveries:
                discoveries[col] = bot
            continue
        robot_tasks.setdefault(bot, {}).setdefault(col, {})[etype] = ev["time"]

    explore_times, delivery_times = [], []
    for bot in robot_tasks:
        for col in robot_tasks[bot]:
            t = robot_tasks[bot][col]
            if "picked" in t:
                explore_times.append(t["picked"])
                if "placed" in t:
                    delivery_times.append(t["placed"] - t["picked"])

    total_time  = data.get("total_time", 0)
    placed      = sum(1 for e in events if e["type"] == "placed")

    return {
        "total_time":     total_time,
        "events":         events,
        "robot_tasks":    robot_tasks,
        "discoveries":    discoveries,
        "explore_times":  explore_times,
        "delivery_times": delivery_times,
        "placed":         placed,
    }


def page_executive_summary(pdf, s, log_name):
    fig = plt.figure(figsize=(8.5, 11))
    draw_header_bar(fig,
                    "Swarm Sorting Collaboration Report",
                    f"SOURCE  {log_name}    GENERATED  {__import__('datetime').datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    kpis = [
        ("MISSION DURATION",  f"{s['total_time']:.2f} s"),
        ("OBJECTS SORTED",    str(s["placed"])),
    ]
    strip_y  = 0.90
    strip_h  = 0.060
    col_w    = 0.22
    for i, (label, value) in enumerate(kpis):
        x0 = 0.04 + i * (col_w + 0.01)
        p = patches.FancyBboxPatch(
            (x0, strip_y - strip_h), col_w, strip_h,
            boxstyle="square,pad=0",
            facecolor=PANEL, edgecolor=ACCENT, linewidth=0.8,
            transform=fig.transFigure, zorder=5)
        fig.patches.append(p)
        fig.text(x0 + 0.011, strip_y - 0.010, label,
                 fontsize=6, color=TEXT_SEC, fontfamily=MONO,
                 va="top", transform=fig.transFigure, zorder=6)
        color = TEXT_PRI
        fig.text(x0 + 0.011, strip_y - 0.034, value,
                 fontsize=14, color=color, fontfamily=MONO, fontweight="bold",
                 va="top", transform=fig.transFigure, zorder=6)

    sec1_y = 0.820
    fig.text(0.04, sec1_y, "// COLLABORATION MATRIX",
             fontsize=8, color=ACCENT, fontfamily=MONO, fontweight="bold")
    fig.add_artist(patches.FancyArrowPatch(
        posA=(0.04, sec1_y - 0.008), posB=(0.96, sec1_y - 0.008),
        arrowstyle="-", color=BORDER, linewidth=0.6,
        transform=fig.transFigure))

    cols_x = [0.05, 0.20, 0.40, 0.62, 0.82]
    headers = ["OBJECT", "DISCOVERED BY", "PICKED BY", "STATUS", "COOP?"]
    row_y = sec1_y - 0.030
    for hx, hd in zip(cols_x, headers):
        fig.text(hx, row_y, hd, fontsize=7, color=TEXT_SEC,
                 fontfamily=MONO, transform=fig.transFigure)

    discoveries = s["discoveries"]
    robot_tasks = s["robot_tasks"]
    for i, col in enumerate(["red", "green", "blue"]):
        row_y -= 0.028
        founder = discoveries.get(col, "N/A")
        picker  = "N/A"
        for b in robot_tasks:
            if col in robot_tasks[b] and "picked" in robot_tasks[b][col]:
                picker = b; break
        coop     = founder != picker and founder != "N/A"
        status   = "SORTED" if col in [e["color"] for e in s["events"] if e["type"] == "placed"] else "PENDING"
        coop_str = "YES" if coop else "NO"

        item_col = ITEM_COLS.get(col, TEXT_PRI)
        vals = [col.upper(), founder, picker, status, coop_str]
        vcol = [item_col, TEXT_PRI, TEXT_PRI,
                SAFE if status == "SORTED" else DANGER,
                ACCENT if coop else TEXT_SEC]
        for vx, v, vc in zip(cols_x, vals, vcol):
            fig.text(vx, row_y, v, fontsize=8, color=vc,
                     fontfamily=MONO, transform=fig.transFigure)

    sec2_y = row_y - 0.055
    fig.text(0.04, sec2_y, "// PERFORMANCE AVERAGES",
             fontsize=8, color=ACCENT, fontfamily=MONO, fontweight="bold")
    fig.add_artist(patches.FancyArrowPatch(
        posA=(0.04, sec2_y - 0.008), posB=(0.96, sec2_y - 0.008),
        arrowstyle="-", color=BORDER, linewidth=0.6,
        transform=fig.transFigure))

    avg_exp = np.mean(s["explore_times"])  if s["explore_times"]  else 0
    avg_del = np.mean(s["delivery_times"]) if s["delivery_times"] else 0
    perf_rows = [
        ("AVG EXPLORATION TIME", f"{avg_exp:.2f} s"),
        ("AVG DELIVERY TIME",    f"{avg_del:.2f} s"),
        ("TOTAL EVENTS",         str(len(s["events"]))),
    ]
    r_y = sec2_y - 0.030
    for label, val in perf_rows:
        fig.text(0.05, r_y, label, fontsize=8, color=TEXT_SEC,
                 fontfamily=MONO, transform=fig.transFigure)
        fig.text(0.40, r_y, val,   fontsize=8, color=TEXT_PRI,
                 fontfamily=MONO,  fontweight="bold", transform=fig.transFigure)
        r_y -= 0.026

    sec3_y = r_y - 0.035
    fig.text(0.04, sec3_y, "// EVENT TIMELINE",
             fontsize=8, color=ACCENT, fontfamily=MONO, fontweight="bold")
    fig.add_artist(patches.FancyArrowPatch(
        posA=(0.04, sec3_y - 0.008), posB=(0.96, sec3_y - 0.008),
        arrowstyle="-", color=BORDER, linewidth=0.6,
        transform=fig.transFigure))

    log_y = sec3_y - 0.028
    for ev in sorted(s["events"], key=lambda x: x["time"]):
        if log_y < 0.04:
            break
        col_ev = ITEM_COLS.get(ev["color"], TEXT_PRI)
        type_col = ACCENT if ev["type"] == "picked" else \
                   SAFE   if ev["type"] == "placed" else TEXT_SEC
        line = (f"[{ev['time']:6.1f}s]  "
                f"{ev['robot']:<10}  "
                f"{ev['type']:<11}  "
                f"{ev['color']:<6}  "
                f"({ev['x']:+.2f}, {ev['y']:+.2f})")
        fig.text(0.05, log_y, line, fontsize=7.2, color=TEXT_SEC,
                 fontfamily=MONO, transform=fig.transFigure)
        p2 = patches.Rectangle(
            (0.045, log_y - 0.001), 0.004, 0.010,
            facecolor=col_ev, transform=fig.transFigure, zorder=5)
        fig.patches.append(p2)
        log_y -= 0.022

    pdf.savefig(fig, bbox_inches="tight")
    plt.close()


def page_spatial(pdf, data, s, log_name):
    fig = plt.figure(figsize=(11, 8.5))
    draw_header_bar(fig, "Spatial Analysis — Top View", f"SOURCE  {log_name}")

    ax = fig.add_axes([0.06, 0.08, 0.72, 0.82])
    draw_world(ax)
    styled_axes(ax, title="Robot Trajectories & Sorting Events",
                xlabel="X  (m)", ylabel="Y  (m)")
    ax.set_facecolor("#0F0F0F")

    trajectories = data.get("trajectories", {})
    for bot, traj in trajectories.items():
        if not traj:
            continue
        arr   = np.array(traj)
        color = ROBOT_COLS.get(bot, TEXT_SEC)
        ax.plot(arr[:, 1], arr[:, 2],
                color=color, alpha=0.45, linewidth=1.2, zorder=5,
                solid_capstyle="round")
        ax.scatter(arr[0, 1], arr[0, 2],
                   marker="s", s=40, color=color, edgecolors=BG, linewidths=0.8, zorder=10)

    for ev in s["events"]:
        if ev["type"] == "discovered":
            continue
        col_ev = ITEM_COLS.get(ev["color"], TEXT_PRI)
        mk = "^" if ev["type"] == "picked" else "v"
        ax.scatter(ev["x"], ev["y"], marker=mk, s=90,
                   color=col_ev, edgecolors=BG, linewidths=0.7, zorder=15)
        ax.annotate(
            f"{ev['type'][0].upper()}{ev['color'][0].upper()}",
            xy=(ev["x"], ev["y"]), xytext=(5, 5),
            textcoords="offset points",
            fontsize=6, fontfamily=MONO, color=col_ev,
            zorder=16)

    leg_ax = fig.add_axes([0.80, 0.08, 0.18, 0.82])
    leg_ax.set_facecolor(PANEL)
    for spine in leg_ax.spines.values():
        spine.set_edgecolor(BORDER); spine.set_linewidth(0.6)
    leg_ax.set_xticks([]); leg_ax.set_yticks([])

    leg_ax.text(0.05, 0.97, "LEGEND", fontsize=7.5, color=ACCENT,
                fontfamily=MONO, fontweight="bold", va="top", transform=leg_ax.transAxes)
    items = []
    for bot, bc in ROBOT_COLS.items():
        items.append(mlines.Line2D([], [], color=bc, linewidth=1.5, label=f"{bot.upper()} PATH"))
    for col, cc in ITEM_COLS.items():
        items.append(mlines.Line2D([], [], marker="^", color=cc, linestyle="none",
                                   markersize=6, label=f"PICK ({col.upper()})"))
        items.append(mlines.Line2D([], [], marker="v", color=cc, linestyle="none",
                                   markersize=6, label=f"PLACE ({col.upper()})"))
    leg_ax.legend(handles=items, loc="upper left", bbox_to_anchor=(0.02, 0.92),
                  frameon=False, fontsize=6.5,
                  labelcolor=TEXT_SEC)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close()



def page_task_dist(pdf, s, log_name):
    labels, explore_d, delivery_d = [], [], []
    for bot in sorted(s["robot_tasks"].keys()):
        for col in sorted(s["robot_tasks"][bot].keys()):
            t = s["robot_tasks"][bot][col]
            if "picked" in t:
                labels.append(f"{bot.upper()} / {col.upper()}")
                explore_d.append(t["picked"])
                placed_t = t.get("placed", s["total_time"])
                delivery_d.append(placed_t - t["picked"])

    fig = plt.figure(figsize=(11, 8.5))
    draw_header_bar(fig, "Task Distribution & Efficiency", f"SOURCE  {log_name}")

    if not labels:
        fig.text(0.5, 0.5, "NO TASK DATA", ha="center", va="center",
                 fontsize=18, color=BORDER, fontfamily=MONO)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()
        return

    ax = fig.add_axes([0.26, 0.10, 0.68, 0.78])
    styled_axes(ax, title="Gantt-Style Task Timeline",
                xlabel="Mission Time  (s)", ylabel="")

    y_pos = np.arange(len(labels))
    bar_h = 0.40

    ax.barh(y_pos, explore_d, height=bar_h,
            color=BLUE_TRK, edgecolor=BG, linewidth=0.4,
            label="EXPLORATION PHASE", alpha=0.9)
    ax.barh(y_pos, delivery_d, left=explore_d, height=bar_h,
            color=ACCENT, edgecolor=BG, linewidth=0.4,
            label="DELIVERY PHASE", alpha=0.9)

    for i, (e, d) in enumerate(zip(explore_d, delivery_d)):
        ax.text(e / 2, i, f"{e:.1f}s",
                ha="center", va="center",
                fontsize=6.5, fontfamily=MONO, color=BG, fontweight="bold")
        ax.text(e + d / 2, i, f"{d:.1f}s",
                ha="center", va="center",
                fontsize=6.5, fontfamily=MONO, color=BG, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.xaxis.grid(True, color=GRID_COL, linewidth=0.5)
    ax.yaxis.grid(False)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.2)

    for i, lbl in enumerate(labels):
        bot_part, col_part = lbl.split(" / ")
        item_col = ITEM_COLS.get(col_part.lower(), TEXT_PRI)
        fig.text(0.03, 0.10 + 0.78 * (i + 0.5) / len(labels),
                 lbl,
                 fontsize=7.5, fontfamily=MONO,
                 color=item_col, va="center",
                 transform=fig.transFigure)

    ax.axvline(s["total_time"], color=DANGER, linewidth=0.8,
               linestyle="--", zorder=10)
    ax.text(s["total_time"] + 0.5, len(labels) - 0.3,
            f"MISSION END\n{s['total_time']:.1f}s",
            fontsize=6.5, fontfamily=MONO, color=DANGER)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Swarm Navigation PDF Report")
    parser.add_argument("log_file",  type=str, help="Path to JSON log file")
    parser.add_argument("--pdf",     type=str, default="swarm_report.pdf",
                        help="Output PDF filename")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"[ERROR] Log file not found: {args.log_file}")
        sys.exit(1)

    with open(args.log_file, "r") as f:
        data = json.load(f)

    apply_global_style()
    stats    = extract_stats(data)
    log_name = os.path.basename(args.log_file)

    print(f"[SWARM-REPORT]  Generating {args.pdf} ...")

    with PdfPages(args.pdf) as pdf:
        page_executive_summary(pdf, stats, log_name)
        page_spatial(pdf, data, stats, log_name)
        page_task_dist(pdf, stats, log_name)

        d = pdf.infodict()
        d["Title"]    = "Swarm Navigation Report"
        d["Author"]   = "SWARM-NAV SYSTEM"
        d["Subject"]  = log_name
        d["Keywords"] = "swarm robotics navigation sorting"

    print(f"[SWARM-REPORT]  Done → {args.pdf}")


if __name__ == "__main__":
    main()
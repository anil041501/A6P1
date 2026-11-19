#!/usr/bin/env python3
"""Fig. 1 methodology flowchart: writes fig1_flowchart.dot (Graphviz, fixed layout) and renders it with
`neato -n2` to ../figures/fig1_flowchart.png (300 dpi) and a JPG copy in ../results/."""
import os, subprocess
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PT = 72.0
XL, XR, XC = 1.18, 3.62, 2.40          # column centres (in)
W, H, DY = 2.16, 0.50, 0.68             # node size and row pitch (in)
ACC, ACCF = "#c0392b", "#f6d5d2"        # accent (proposed)
GRY, GRYF = "#777777", "#ececec"

nodes, edges, panels = [], [], []
P = lambda xx, yy: f"{xx*PT:.1f},{-yy*PT:.1f}"
y = [0.0]


def row(step=DY):
    y[0] += step
    return y[0]


def node(name, x, yy, label, kind="grey", shape="box", w=W, h=H, fs=13):
    col, fill = (ACC, ACCF) if kind == "acc" else (GRY, GRYF)
    style = "rounded,filled" if shape == "box" else "filled"
    nodes.append(f'  {name} [pos="{x*PT:.1f},{-yy*PT:.1f}!", label="{label}", shape={shape}, style="{style}", '
                 f'width={w}, height={h}, fixedsize=true, color="{col}", fillcolor="{fill}", fontsize={fs}];')


def panel(name, top, bottom, title, fill, line):
    cy = (top + bottom) / 2; hh = bottom - top
    panels.append(f'  {name} [pos="{XC*PT:.1f},{-cy*PT:.1f}!", label="", shape=box, style="rounded,filled", '
                  f'width=4.86, height={hh:.2f}, fixedsize=true, color="{line}", fillcolor="{fill}", penwidth=1.0];')
    panels.append(f'  {name}_t [pos="{XC*PT:.1f},{-(top+0.15)*PT:.1f}!", label="{title}\\l", shape=plaintext, '
                  f'fontsize=13, fontname="Helvetica-Bold", width=4.62, height=0.22, fixedsize=true, margin=0];')


def edge(a, b, label="", style="solid", extra=""):
    lab = f', xlabel="{label}"' if label else ""
    edges.append(f'  {a} -> {b} [style={style}{lab}{extra}];')


def txt(name, x, yy, label, fs=12):
    nodes.append(f'  {name} [pos="{x*PT:.1f},{-yy*PT:.1f}!", label="{label}", shape=plaintext, fontsize={fs}, '
                 f'width=0.1, height=0.1, fixedsize=false, margin=0, fontcolor="#333333"];')


# ---- 1 Data layer
t1 = 0.0; r = row(0.60)
node("apps", XL, r, "Applications\\n14,000 x 21, 2025-26")
node("leak", XR, r, "Drop IDs and leaky\\nprocessing_days")
b1 = r + 0.33
# ---- 2 Pre-processing
t2 = b1 + 0.08; y[0] = t2; r = row(0.60)
node("imp", XL, r, "Median imputation,\\nwinsorise 0.5/99.5 %")
node("scr", XR, r, "Screen features\\nH, u_T, u_S, Eq. (1)")
b2 = r + 0.33
# ---- 3 PS-GAM
t3 = b2 + 0.08; y[0] = t3; r = row(0.60)
node("basis", XL, r, "B-spline bases B_j,\\ndouble penalty, Eq. (5)", "acc")
ya = r
node("lam", XR, r, "Per-term lambda_j:\\nFellner-Schall, Eq. (7)", "acc")
r = row(0.64)
node("stS", XL, r, "Stage S: P(review)\\np_S, Eq. (2)", "acc")
node("stA", XR, r, "Stage A: P(approve |\\nscreened) p_A, Eq. (3)", "acc")
r = row(0.62); yc = r
node("pirls", XC + 0.35, r, "PIRLS with step halving on the\\npenalised deviance, Eq. (6)", "acc", w=3.6, h=0.46, fs=12)
r = row(0.66); yd = r
node("conv", XL, r, "Converged?", "acc", shape="diamond", w=1.75, h=0.58, fs=12)
node("comp", XR, r, "Class probabilities\\nfrom p_S, p_A, Eq. (4)", "acc")
b3 = r + 0.36
# ---- 4 Decision automation and oversight
t4 = b3 + 0.08; y[0] = t4; r = row(0.64)
node("rc", XL, r, "Reason codes = term\\ncontributions, Eq. (8)", "acc")
yt = r
node("tau", XR, r, "P_A >= tau\\nEq. (9)", "acc", shape="diamond", w=1.75, h=0.62, fs=12)
r = row(0.66)
node("human", XL, r, "Engineer review +\\ncounterfactual hint", "acc")
node("auto", XR, r, "Auto-approve\\n(fast track)", "acc")
r = row(0.52)
node("audit", XC, r, "Audit log; retrain on drift", "acc", w=3.2, h=0.36)
b4 = r + 0.27
# ---- 5 Evaluation
t5 = b4 + 0.08; y[0] = t5; r = row(0.60)
node("base", XL, r, "Rules, logistic reg.,\\nLightGBM + TreeSHAP")
node("met", XR, r, "5-fold CV, 2026 drift:\\nF1, FAR, AOPC")
b5 = r + 0.33

panel("p1", t1, b1, "1  Data layer", "#f5f5f5", "#bbbbbb")
panel("p2", t2, b2, "2  Pre-processing (training fold only)", "#eef3f9", "#b9c8da")
panel("p3", t3, b3, "3  Proposed two-stage PS-GAM", "#fbefee", "#d9a3a3")
panel("p4", t4, b4, "4  Selective automation and human oversight", "#fbefee", "#d9a3a3")
panel("p5", t5, b5, "5  Evaluation", "#f5f5f5", "#bbbbbb")

edge("apps", "leak"); edge("leak", "imp"); edge("imp", "scr"); edge("scr", "basis")
edge("lam", "basis", style="dashed")
edges.append(f'  pirls -> lam [style=dashed, color="{ACC}", pos="e,{P(XR + 1.08, ya)} {P(XC + 2.15, yc)} {P(4.80, yc)} {P(4.80, ya)} {P(XR + 1.12, ya)}"];')
edge("basis", "stS"); edge("basis", "stA")
edge("stS", "pirls"); edge("stA", "pirls")
edge("pirls", "conv")
# "no" loop: from the left vertex of the diamond back up into the left end of the PIRLS box
x0, x1 = XL - 0.875, XC + 0.35 - 1.8
edges.append(f'  conv -> pirls [pos="e,{P(x1, yc)} {P(x0+0.02, yd)} {P(0.05, yd)} {P(0.05, yc)} {P(x1-0.08, yc)}"];')
txt("l_no", 0.34, (yc + yd) / 2 + 0.06, "no")
edge("conv", "comp"); txt("l_yes", (XL + XR) / 2 - 0.05, yd - 0.12, "yes")
edge("comp", "tau"); edge("comp", "rc", style="dashed")
edge("tau", "auto"); txt("l_yes2", XR + 0.26, yt + 0.42, "yes")
edge("tau", "human"); txt("l_no2", (XL + XR) / 2 + 0.05, yt + 0.12, "no")
edge("rc", "human", style="dashed")
edge("auto", "audit"); edge("human", "audit")
edge("audit", "met", style="dashed"); edge("base", "met")

dot = ['digraph F {',
       '  graph [splines=line, outputorder=nodesfirst, pad=0.04, bgcolor=white];',
       '  node [fontname="Helvetica", fontsize=13, margin="0.03,0.02"];',
       '  edge [fontname="Helvetica", fontsize=12, color="#555555", arrowsize=0.7, penwidth=1.1];']
dot += panels + nodes + edges + ['}']
src = os.path.join(HERE, "fig1_flowchart.dot")
open(src, "w").write("\n".join(dot))
png = os.path.join(ROOT, "figures", "fig1_flowchart.png")
subprocess.run(["neato", "-n2", "-Tpng", "-Gdpi=300", src, "-o", png], check=True)
im = Image.open(png).convert("RGB")
im.save(os.path.join(ROOT, "results", "fig1_flowchart.jpg"), quality=92)
print("fig1", im.size, "width in:", im.size[0] / 300, "aspect:", im.size[1] / im.size[0])

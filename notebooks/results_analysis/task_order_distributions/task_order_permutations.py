"""
Task Order Visualizations:
1. Bar chart of all 24 permutations (natural order, not sorted) per condition.
2. Sankey/tree flow diagram showing path densities across positions 1→2→3→4.
"""
import pandas as pd, numpy as np, matplotlib, ast, sys, itertools
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from matplotlib.path import Path as MplPath

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
OUTPUT_DIR = "."

TASKS = ['easy_mild','easy_strong','hard_mild','hard_strong']
DISP = {'easy_mild':'Low/Low','easy_strong':'Low/Strong',
        'hard_mild':'High/Low','hard_strong':'High/Strong'}
DISP_SHORT = {'easy_mild':'L/L','easy_strong':'L/S','hard_mild':'H/L','hard_strong':'H/S'}
TASK_COLORS = {'easy_mild':'#2E86AB','easy_strong':'#7FB069','hard_mild':'#E8963F','hard_strong':'#E8533F'}

CONDS = ['H','H+AI','H+AI+CF','H+AI+LLM','H+AI+GRADCAM','H+AI+SHAP']

df = pd.read_csv(DATA_PATH)
df['tasks_order'] = df['tasks_order'].apply(lambda x: ast.literal_eval(x) if isinstance(x,str) else x)
df['order_tuple'] = df['tasks_order'].apply(lambda x: tuple(x) if isinstance(x,list) else ())

# Generate all 24 permutations in natural (lexicographic) order
all_perms = list(itertools.permutations(TASKS))
perm_labels = [' → '.join(DISP_SHORT[t] for t in p) for p in all_perms]

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Bar charts — all 24 permutations, natural order
# ═══════════════════════════════════════════════════════════════════════
present_conds = [c for c in CONDS if c in df['xai_condition'].unique()]
n_conds = len(present_conds)
fig, axes = plt.subplots(1, n_conds, figsize=(5*n_conds, 8))
if n_conds == 1: axes = [axes]
fig.suptitle("Distribution of Task Order Permutations\n(all 24 in natural order)",
             fontsize=14, fontweight='bold', y=1.01)

for ci, cond in enumerate(present_conds):
    ax = axes[ci]
    sub = df[df['xai_condition']==cond]
    order_counts = sub['order_tuple'].value_counts()
    vals = [order_counts.get(p, 0) for p in all_perms]
    colors = [TASK_COLORS[p[0]] for p in all_perms]  # color by starting task

    bars = ax.barh(range(24), vals, color=colors, alpha=0.8,
                   edgecolor='white', linewidth=0.5, height=0.7)
    for i, v in enumerate(vals):
        if v > 0:
            ax.text(v + 0.2, i, str(v), va='center', fontsize=6.5, color='#555')
    ax.set_yticks(range(24))
    ax.set_yticklabels(perm_labels, fontsize=6.5, fontfamily='monospace')
    ax.invert_yaxis()
    ax.set_xlabel('Count')
    n_unique = sum(1 for v in vals if v > 0)
    ax.set_title(f"{cond}\n(n={len(sub)}, {n_unique} unique)", fontweight='bold', fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(vals)*1.25 if max(vals)>0 else 5)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_order_permutations.png", dpi=200, bbox_inches='tight')
plt.close()
print("-> fig_order_permutations.png")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Flow / alluvial tree — path densities across 4 positions
# ═══════════════════════════════════════════════════════════════════════
def draw_flow(ax, sub, cond_label):
    """Draw an alluvial/flow diagram for task order paths."""
    n_total = len(sub)
    if n_total == 0:
        ax.set_visible(False); return

    orders = sub['tasks_order'].tolist()
    orders = [o for o in orders if isinstance(o, list) and len(o)==4]
    n = len(orders)
    if n == 0:
        ax.set_visible(False); return

    # Node positions: 4 columns (positions), 4 rows (tasks) each
    pos_x = [0, 1, 2, 3]
    task_y = {t: i for i, t in enumerate(TASKS)}  # top to bottom

    # Count flows between consecutive positions
    # node_counts[pos][task] = count of participants at this position with this task
    node_counts = [{t:0 for t in TASKS} for _ in range(4)]
    # flow_counts[(pos, task_from, task_to)] = count
    flow_counts = {}
    for order in orders:
        for p in range(4):
            node_counts[p][order[p]] += 1
        for p in range(3):
            key = (p, order[p], order[p+1])
            flow_counts[key] = flow_counts.get(key, 0) + 1

    # Draw nodes as horizontal bars
    bar_width = 0.15
    bar_height_scale = 0.8 / max(1, n)  # scale so max fills ~0.8 of row height
    node_positions = {}  # (pos, task) -> (x, y_center, height)

    for p in range(4):
        x = pos_x[p]
        for t in TASKS:
            count = node_counts[p][t]
            y = task_y[t]
            h = max(count * bar_height_scale, 0.01) if count > 0 else 0
            node_positions[(p, t)] = (x, y, h)
            if count > 0:
                rect = plt.Rectangle((x - bar_width/2, y - h/2), bar_width, h,
                                     facecolor=TASK_COLORS[t], alpha=0.9,
                                     edgecolor='white', linewidth=0.5, zorder=3)
                ax.add_patch(rect)
                # Count label
                ax.text(x, y, str(count), ha='center', va='center',
                        fontsize=6, fontweight='bold', color='white', zorder=4,
                        path_effects=[pe.withStroke(linewidth=2, foreground=TASK_COLORS[t])])

    # Draw flows as curved bands
    for (p, t_from, t_to), count in flow_counts.items():
        if count == 0: continue
        x0 = pos_x[p] + bar_width/2
        x1 = pos_x[p+1] - bar_width/2
        _, y0, h0 = node_positions[(p, t_from)]
        _, y1, h1 = node_positions[(p+1, t_to)]

        # Flow thickness proportional to count
        lw = max(0.5, count / n * 15)
        alpha = min(0.6, max(0.08, count / n * 1.5))

        # Bezier curve
        mid_x = (x0 + x1) / 2
        verts = [(x0, y0), (mid_x, y0), (mid_x, y1), (x1, y1)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        path = MplPath(verts, codes)
        patch = matplotlib.patches.PathPatch(path, facecolor='none',
                                             edgecolor=TASK_COLORS[t_from],
                                             linewidth=lw, alpha=alpha, zorder=2,
                                             capstyle='round')
        ax.add_patch(patch)

    # Labels
    for p in range(4):
        ax.text(pos_x[p], -0.7, f"Position {p+1}", ha='center', fontsize=8, fontweight='bold')

    for t in TASKS:
        ax.text(-0.35, task_y[t], DISP[t], ha='right', va='center', fontsize=8,
                color=TASK_COLORS[t], fontweight='bold')

    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-1.0, 3.5)
    ax.invert_yaxis()
    ax.set_aspect('auto')
    ax.set_title(f"{cond_label} (n={n})", fontweight='bold', fontsize=11, pad=10)
    ax.axis('off')

# Create flow diagrams
n_conds = len(present_conds)
n_cols = min(3, n_conds)
n_rows = (n_conds + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
elif n_rows == 1: axes = axes.reshape(1, -1)
elif n_cols == 1: axes = axes.reshape(-1, 1)

fig.suptitle("Task Order Flow Diagram — Path Densities\n"
             "Node size and flow thickness proportional to participant count",
             fontsize=13, fontweight='bold', y=1.02)

for ci, cond in enumerate(present_conds):
    ri, col_i = ci // n_cols, ci % n_cols
    ax = axes[ri, col_i]
    sub = df[df['xai_condition']==cond]
    draw_flow(ax, sub, cond)

# Hide unused axes
for ci in range(len(present_conds), n_rows*n_cols):
    ri, col_i = ci // n_cols, ci % n_cols
    axes[ri, col_i].set_visible(False)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_order_flow.png", dpi=200, bbox_inches='tight')
plt.close()
print("-> fig_order_flow.png")

print("\nDone.")
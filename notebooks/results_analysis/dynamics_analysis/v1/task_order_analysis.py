"""
Task Order Effects Analysis for XAI Experiment
================================================
Analyzes whether the order in which participants completed tasks
(easy_mild first vs hard_strong first) affected their behavior,
for participants in "H" and "H+AI" XAI conditions only.

Dependent variables: score, reliance, overreliance, underreliance,
                     trust, cognitive load, answer times
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
from itertools import combinations
import warnings
import ast
import sys
import os

warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────────────
# Update this path to point to your CSV file
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
OUTPUT_DIR = ".."

# Style configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FFFFFF',
    'axes.edgecolor': '#CCCCCC',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#DDDDDD',
})

PALETTE = {
    'easy_mild_first': '#2E86AB',
    'hard_strong_first': '#E8533F',
    'H': '#5B8C5A',
    'H+AI': '#D4A843',
}

# ─── Data Loading & Preprocessing ────────────────────────────────────────────
print("=" * 70)
print("TASK ORDER EFFECTS ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"\nTotal participants loaded: {len(df)}")
print(f"XAI conditions present: {df['xai_condition'].unique()}")

# Filter to H and H+AI only
df = df[df['xai_condition'].isin(['H', 'H+AI'])].copy()
print(f"Participants after filtering to H and H+AI: {len(df)}")

# Parse tasks_order if stored as string
def parse_list_col(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except:
            return val
    return val

df['tasks_order'] = df['tasks_order'].apply(parse_list_col)

# Determine first task
df['first_task'] = df['tasks_order'].apply(lambda x: x[0] if isinstance(x, list) else None)

# Filter to participants who started with easy_mild or hard_strong
df = df[df['first_task'].isin(['easy_mild', 'hard_strong'])].copy()
print(f"Participants starting with easy_mild or hard_strong: {len(df)}")

# Create a readable order group label
df['order_group'] = df['first_task'].map({
    'easy_mild': 'Easy-Mild First',
    'hard_strong': 'Hard-Strong First'
})

print(f"\nBreakdown by order group and XAI condition:")
print(df.groupby(['order_group', 'xai_condition']).size().unstack(fill_value=0))

# ─── Define DVs ──────────────────────────────────────────────────────────────
# We analyze each task condition (easy_mild and hard_strong) separately,
# comparing participants who did that task first vs second.

TASKS = ['easy_mild', 'hard_strong']
DVS_CONTINUOUS = {
    'score': 'Accuracy Score',
    'reliance': 'Reliance on AI',
    'overreliance': 'Over-Reliance',
    'underreliance': 'Under-Reliance',
}
DVS_SUBJECTIVE = {
    'trust': 'Subjective Trust',
    'cogload': 'Cognitive Load',
}

# ─── Helper: Robust Statistics ───────────────────────────────────────────────
def compute_effect_size_cohens_d(g1, g2):
    """Cohen's d with pooled SD."""
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_std = np.sqrt(((n1-1)*np.var(g1, ddof=1) + (n2-1)*np.var(g2, ddof=1)) / (n1+n2-2))
    if pooled_std == 0:
        return 0
    return (np.mean(g1) - np.mean(g2)) / pooled_std

def bootstrap_ci(data, stat_func=np.mean, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval."""
    rng = np.random.default_rng(42)
    data = np.array(data)
    data = data[~np.isnan(data)]
    if len(data) < 3:
        return (np.nan, np.nan)
    boot_stats = [stat_func(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return (np.percentile(boot_stats, alpha*100), np.percentile(boot_stats, (1-alpha)*100))

def run_mann_whitney(g1, g2, label=""):
    """Mann-Whitney U test with effect size."""
    g1 = np.array(g1, dtype=float)
    g2 = np.array(g2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    if len(g1) < 3 or len(g2) < 3:
        return {'U': np.nan, 'p': np.nan, 'd': np.nan, 'n1': len(g1), 'n2': len(g2)}
    u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    d = compute_effect_size_cohens_d(g1, g2)
    # rank-biserial r
    n1, n2 = len(g1), len(g2)
    r_rb = 1 - (2*u_stat) / (n1*n2)
    return {'U': u_stat, 'p': p_val, 'd': d, 'r_rb': r_rb, 'n1': n1, 'n2': n2,
            'mean1': np.mean(g1), 'mean2': np.mean(g2),
            'median1': np.median(g1), 'median2': np.median(g2)}

def p_to_stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    if p < 0.1: return '†'
    return 'ns'

# ─── ANALYSIS 1: Task Order Effects on Each Task ─────────────────────────────
print("\n" + "=" * 70)
print("ANALYSIS 1: TASK ORDER EFFECTS ON BEHAVIORAL MEASURES")
print("=" * 70)

results_rows = []

for xai_cond in ['H', 'H+AI']:
    sub = df[df['xai_condition'] == xai_cond]
    print(f"\n{'─'*50}")
    print(f"XAI Condition: {xai_cond} (n={len(sub)})")
    print(f"{'─'*50}")

    for task in TASKS:
        print(f"\n  Task: {task}")
        # Group 1: did this task first
        # Group 2: did this task NOT first (i.e., the other was first)
        g1 = sub[sub['first_task'] == task]
        g2 = sub[sub['first_task'] != task]

        g1_label = f"{task} first"
        g2_label = f"{'hard_strong' if task == 'easy_mild' else 'easy_mild'} first"

        for dv_base, dv_label in {**DVS_CONTINUOUS, **DVS_SUBJECTIVE}.items():
            col = f"{dv_base}_{task}"
            if col not in df.columns:
                continue
            vals1 = g1[col].dropna().values
            vals2 = g2[col].dropna().values
            res = run_mann_whitney(vals1, vals2)
            sig = p_to_stars(res['p']) if not np.isnan(res['p']) else '?'
            print(f"    {dv_label:20s}: M={res.get('mean1', np.nan):.3f} vs M={res.get('mean2', np.nan):.3f}, "
                  f"U={res['U']:.1f}, p={res['p']:.4f} {sig}, d={res['d']:.3f}, r_rb={res.get('r_rb', np.nan):.3f}")
            results_rows.append({
                'xai_condition': xai_cond,
                'task': task,
                'dv': dv_label,
                'mean_task_first': res.get('mean1', np.nan),
                'mean_task_second': res.get('mean2', np.nan),
                'median_task_first': res.get('median1', np.nan),
                'median_task_second': res.get('median2', np.nan),
                'U': res['U'],
                'p': res['p'],
                'cohens_d': res['d'],
                'rank_biserial_r': res.get('r_rb', np.nan),
                'n_first': res['n1'],
                'n_second': res['n2'],
                'significance': sig,
            })

results_df = pd.DataFrame(results_rows)
results_df.to_csv(os.path.join(OUTPUT_DIR, "order_effects_stats.csv"), index=False)
print(f"\n→ Statistical results saved to order_effects_stats.csv")

# ─── ANALYSIS 2: Permutation Test (more robust for small samples) ────────────
print("\n" + "=" * 70)
print("ANALYSIS 2: PERMUTATION TESTS (10,000 permutations)")
print("=" * 70)

perm_rows = []

def permutation_test(g1, g2, n_perm=10000, stat_func=lambda a, b: np.mean(a) - np.mean(b)):
    """Two-sample permutation test."""
    g1 = np.array(g1, dtype=float)
    g2 = np.array(g2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    if len(g1) < 2 or len(g2) < 2:
        return np.nan
    observed = stat_func(g1, g2)
    combined = np.concatenate([g1, g2])
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_stat = stat_func(combined[:len(g1)], combined[len(g1):])
        if abs(perm_stat) >= abs(observed):
            count += 1
    return count / n_perm

for xai_cond in ['H', 'H+AI']:
    sub = df[df['xai_condition'] == xai_cond]
    for task in TASKS:
        g1 = sub[sub['first_task'] == task]
        g2 = sub[sub['first_task'] != task]
        for dv_base, dv_label in {**DVS_CONTINUOUS, **DVS_SUBJECTIVE}.items():
            col = f"{dv_base}_{task}"
            if col not in df.columns:
                continue
            p_perm = permutation_test(g1[col].dropna().values, g2[col].dropna().values)
            sig = p_to_stars(p_perm) if not np.isnan(p_perm) else '?'
            print(f"  [{xai_cond}] {task} | {dv_label:20s}: p_perm={p_perm:.4f} {sig}")
            perm_rows.append({
                'xai_condition': xai_cond, 'task': task, 'dv': dv_label,
                'p_permutation': p_perm, 'significance': sig
            })

perm_df = pd.DataFrame(perm_rows)
perm_df.to_csv(os.path.join(OUTPUT_DIR, "permutation_test_results.csv"), index=False)

# ─── VISUALIZATION 1: Grouped Bar Charts with CI ─────────────────────────────
print("\n→ Generating visualizations...")

all_dvs = {**DVS_CONTINUOUS, **DVS_SUBJECTIVE}
n_dvs = len(all_dvs)

fig, axes = plt.subplots(2, n_dvs, figsize=(4.2*n_dvs, 10), sharey=False)
fig.suptitle("Task Order Effects on Behavioral Measures\n(Filtered: H and H+AI conditions only)",
             fontsize=16, fontweight='bold', y=0.98)

for row_idx, xai_cond in enumerate(['H', 'H+AI']):
    sub = df[df['xai_condition'] == xai_cond]
    for col_idx, (dv_base, dv_label) in enumerate(all_dvs.items()):
        ax = axes[row_idx, col_idx]
        bar_data = []
        for task in TASKS:
            for first_task_val, group_label, color in [
                (task, f'{task}\n(done 1st)', PALETTE['easy_mild_first'] if task == 'easy_mild' else PALETTE['hard_strong_first']),
                ('hard_strong' if task == 'easy_mild' else 'easy_mild',
                 f'{task}\n(done 2nd+)',
                 PALETTE['easy_mild_first'] if task == 'easy_mild' else PALETTE['hard_strong_first']),
            ]:
                col_name = f"{dv_base}_{task}"
                if col_name not in df.columns:
                    continue
                g = sub[sub['first_task'] == first_task_val][col_name].dropna()
                ci = bootstrap_ci(g.values)
                bar_data.append({
                    'task': task.replace('_', '\n'),
                    'order': 'First' if first_task_val == task else 'Later',
                    'mean': g.mean() if len(g) > 0 else 0,
                    'ci_lo': ci[0],
                    'ci_hi': ci[1],
                    'color': color,
                    'alpha': 1.0 if first_task_val == task else 0.5,
                })

        x_positions = []
        pos = 0
        for i, bd in enumerate(bar_data):
            x_positions.append(pos)
            if i % 2 == 1:
                pos += 1.5
            else:
                pos += 0.6

        for i, bd in enumerate(bar_data):
            ax.bar(x_positions[i], bd['mean'], width=0.5,
                   color=bd['color'], alpha=bd['alpha'],
                   edgecolor='white', linewidth=0.5)
            yerr_lo = bd['mean'] - bd['ci_lo'] if not np.isnan(bd['ci_lo']) else 0
            yerr_hi = bd['ci_hi'] - bd['mean'] if not np.isnan(bd['ci_hi']) else 0
            ax.errorbar(x_positions[i], bd['mean'],
                        yerr=[[yerr_lo], [yerr_hi]],
                        fmt='none', color='#333333', capsize=3, linewidth=1.2)

        ax.set_xticks([x_positions[0]+0.25, x_positions[2]+0.25])
        ax.set_xticklabels(['easy_mild', 'hard_strong'], fontsize=9)
        ax.set_title(f"{dv_label}", fontsize=11, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel(f"{xai_cond}\n\nValue", fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add legend only once
        if row_idx == 0 and col_idx == n_dvs - 1:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='gray', alpha=1.0, label='Task done 1st'),
                Patch(facecolor='gray', alpha=0.5, label='Task done later'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUTPUT_DIR, "fig1_order_effects_bars.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  → fig1_order_effects_bars.png saved")

# ─── VISUALIZATION 2: Violin + Strip plots ───────────────────────────────────
fig, axes = plt.subplots(2, len(TASKS), figsize=(14, 10))
fig.suptitle("Distribution of Key DVs by Task Order\n(H and H+AI conditions)",
             fontsize=16, fontweight='bold', y=0.98)

for row_idx, xai_cond in enumerate(['H', 'H+AI']):
    sub = df[df['xai_condition'] == xai_cond]
    for col_idx, task in enumerate(TASKS):
        ax = axes[row_idx, col_idx]

        # Melt DVs for this task
        dvs_to_plot = ['score', 'reliance', 'overreliance', 'underreliance']
        melted = []
        for dv in dvs_to_plot:
            col_name = f"{dv}_{task}"
            if col_name not in sub.columns:
                continue
            for _, r in sub.iterrows():
                if pd.notna(r[col_name]):
                    melted.append({
                        'DV': dv.replace('_', '\n').title(),
                        'Value': r[col_name],
                        'Order': r['order_group'],
                    })
        mdf = pd.DataFrame(melted)
        if len(mdf) == 0:
            continue

        order_colors = {
            'Easy-Mild First': PALETTE['easy_mild_first'],
            'Hard-Strong First': PALETTE['hard_strong_first'],
        }
        sns.violinplot(data=mdf, x='DV', y='Value', hue='Order',
                       split=True, inner='quart', ax=ax,
                       palette=order_colors, alpha=0.7, linewidth=0.8)
        sns.stripplot(data=mdf, x='DV', y='Value', hue='Order',
                      dodge=True, ax=ax, palette=order_colors,
                      alpha=0.4, size=3, jitter=0.15, legend=False)

        ax.set_title(f"{xai_cond} — {task.replace('_', ' ').title()} Task", fontweight='bold')
        ax.set_xlabel('')
        if col_idx == 0:
            ax.set_ylabel('Value')
        else:
            ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if row_idx == 0 and col_idx == 1:
            ax.legend(title='First Task', fontsize=8, title_fontsize=9)
        else:
            ax.legend_.remove() if ax.legend_ else None

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUTPUT_DIR, "fig2_violin_distributions.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  → fig2_violin_distributions.png saved")

# ─── VISUALIZATION 3: Heatmap of Effect Sizes ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Cohen's d Effect Sizes: Task Order Effects", fontsize=15, fontweight='bold')

for idx, xai_cond in enumerate(['H', 'H+AI']):
    ax = axes[idx]
    sub_res = results_df[results_df['xai_condition'] == xai_cond]
    if len(sub_res) == 0:
        continue
    pivot = sub_res.pivot(index='dv', columns='task', values='cohens_d')
    # Add significance annotations
    annot = sub_res.pivot(index='dv', columns='task', values='significance')
    annot_combined = pivot.round(2).astype(str) + '\n' + annot.astype(str)

    sns.heatmap(pivot, annot=annot_combined, fmt='', cmap='RdBu_r', center=0,
                vmin=-1.5, vmax=1.5, ax=ax, linewidths=0.5,
                cbar_kws={'label': "Cohen's d"})
    ax.set_title(f"{xai_cond}", fontweight='bold')
    ax.set_ylabel('')
    ax.set_xlabel('')

plt.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(OUTPUT_DIR, "fig3_effect_size_heatmap.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  → fig3_effect_size_heatmap.png saved")

# ─── ANALYSIS 3: Trust vs Error Recency ──────────────────────────────────────
print("\n" + "=" * 70)
print("ANALYSIS 3: TRUST vs ERROR RECENCY")
print("=" * 70)
print("""
This analysis examines whether recent AI errors (in the last N trials)
predict trust ratings. We compute, for each participant and each task,
the number of AI errors in the last K trials before the trust rating
was collected, and correlate this with trust.
""")

trust_recency_rows = []

for xai_cond in ['H', 'H+AI']:
    sub = df[df['xai_condition'] == xai_cond]
    for task in TASKS:
        trust_col = f"trust_{task}"
        true_col = f"task_true_{task}"
        pred_col = f"ai_pred_{task}"

        if trust_col not in sub.columns or true_col not in sub.columns or pred_col not in sub.columns:
            continue

        for _, row in sub.iterrows():
            trust_val = row[trust_col]
            if pd.isna(trust_val):
                continue

            # Parse arrays
            try:
                task_true = ast.literal_eval(row[true_col]) if isinstance(row[true_col], str) else row[true_col]
                ai_pred = ast.literal_eval(row[pred_col]) if isinstance(row[pred_col], str) else row[pred_col]
            except:
                continue

            if not isinstance(task_true, (list, np.ndarray)) or not isinstance(ai_pred, (list, np.ndarray)):
                continue

            task_true = [bool(x) if not isinstance(x, bool) else x for x in task_true]
            ai_pred = [bool(x) if not isinstance(x, bool) else x for x in ai_pred]

            n_trials = len(task_true)
            # AI errors: where AI prediction != ground truth
            ai_errors = [1 if t != p else 0 for t, p in zip(task_true, ai_pred)]

            # Compute error counts in different recency windows
            for window in [2, 3, 4, 6]:
                if n_trials >= window:
                    recent_errors = sum(ai_errors[-window:])
                    all_errors = sum(ai_errors)
                    early_errors = sum(ai_errors[:window])

                    trust_recency_rows.append({
                        'xai_condition': xai_cond,
                        'task': task,
                        'order_group': row['order_group'],
                        'trust': trust_val,
                        'window': window,
                        'recent_errors': recent_errors,
                        'early_errors': early_errors,
                        'total_errors': all_errors,
                        'recent_error_rate': recent_errors / window,
                        'early_error_rate': early_errors / window,
                        'total_error_rate': all_errors / n_trials,
                    })

if trust_recency_rows:
    tr_df = pd.DataFrame(trust_recency_rows)
    tr_df.to_csv(os.path.join(OUTPUT_DIR, "trust_error_recency_data.csv"), index=False)

    # Statistical tests: correlation between recent errors and trust
    print("\nCorrelation: Trust ~ Recent AI Error Rate (Spearman)")
    print("-" * 60)
    corr_rows = []
    for xai_cond in ['H', 'H+AI']:
        for task in TASKS:
            for window in [2, 3, 4, 6]:
                subset = tr_df[(tr_df['xai_condition'] == xai_cond) &
                               (tr_df['task'] == task) &
                               (tr_df['window'] == window)]
                if len(subset) < 5:
                    continue
                # Safe Spearman: returns (nan, nan) if either array is constant
                def safe_spearmanr(a, b):
                    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
                    mask = ~(np.isnan(a) | np.isnan(b))
                    a, b = a[mask], b[mask]
                    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
                        return (np.nan, np.nan)
                    return stats.spearmanr(a, b)

                # Recent errors vs trust
                rho_recent, p_recent = safe_spearmanr(subset['recent_error_rate'], subset['trust'])
                # Early errors vs trust
                rho_early, p_early = safe_spearmanr(subset['early_error_rate'], subset['trust'])
                # Total errors vs trust
                rho_total, p_total = safe_spearmanr(subset['total_error_rate'], subset['trust'])

                print(f"  [{xai_cond}] {task} (window={window}): "
                      f"ρ_recent={rho_recent:.3f} (p={p_recent:.4f} {p_to_stars(p_recent)}), "
                      f"ρ_early={rho_early:.3f} (p={p_early:.4f}), "
                      f"ρ_total={rho_total:.3f} (p={p_total:.4f})")
                corr_rows.append({
                    'xai_condition': xai_cond, 'task': task, 'window': window,
                    'rho_recent': rho_recent, 'p_recent': p_recent,
                    'rho_early': rho_early, 'p_early': p_early,
                    'rho_total': rho_total, 'p_total': p_total,
                    'n': len(subset),
                })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(OUTPUT_DIR, "trust_error_correlations.csv"), index=False)

    # Recency bias: compare recent vs early error correlation with trust
    print("\nRecency Bias Test: Is the recent error rate a stronger predictor of trust than early error rate?")
    print("(Steiger's Z-test for comparing dependent correlations)")
    print("-" * 60)
    for xai_cond in ['H', 'H+AI']:
        for task in TASKS:
            for window in [3, 4, 6]:
                subset = tr_df[(tr_df['xai_condition'] == xai_cond) &
                               (tr_df['task'] == task) &
                               (tr_df['window'] == window)]
                if len(subset) < 10:
                    continue
                r_recent_trust, _ = safe_spearmanr(subset['recent_error_rate'], subset['trust'])
                r_early_trust, _ = safe_spearmanr(subset['early_error_rate'], subset['trust'])
                r_recent_early, _ = safe_spearmanr(subset['recent_error_rate'], subset['early_error_rate'])
                n = len(subset)

                # Skip if any correlation is NaN (e.g. constant columns)
                if np.isnan(r_recent_trust) or np.isnan(r_early_trust) or np.isnan(r_recent_early):
                    print(f"  [{xai_cond}] {task} (w={window}): skipped (constant data or insufficient variance)")
                    continue

                # Steiger's Z for comparing two dependent correlations
                # Using Williams' modification
                r_mean = (r_recent_trust + r_early_trust) / 2
                f = (1 - r_recent_early) / (2 * (1 - r_mean**2)) if (1 - r_mean**2) > 0 else 1
                h = (1 + r_recent_early * (1 - 2 * r_mean**2) - 0.5 * r_mean**2 * (1 - r_recent_early)**2)
                denom = np.sqrt(2 * (1 - r_recent_early) * h / ((n - 1) * (1 - r_mean**2)**2)) if (1 - r_mean**2) > 0 else 1
                if denom > 0:
                    z = (np.arctanh(r_recent_trust) - np.arctanh(r_early_trust)) * np.sqrt(n - 3)
                    p_steiger = 2 * (1 - stats.norm.cdf(abs(z)))
                else:
                    z = np.nan
                    p_steiger = np.nan

                print(f"  [{xai_cond}] {task} (w={window}): "
                      f"ρ_recent={r_recent_trust:.3f}, ρ_early={r_early_trust:.3f}, "
                      f"Z={z:.3f}, p={p_steiger:.4f} {p_to_stars(p_steiger) if not np.isnan(p_steiger) else '?'}")

    # ─── VISUALIZATION 4: Trust vs Error Recency ─────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Trust vs AI Error Recency\n(Recent errors in last 4 trials vs Trust rating)",
                 fontsize=15, fontweight='bold', y=0.98)

    window_plot = 4  # Focus on window of 4 for main visualization
    plot_data = tr_df[tr_df['window'] == window_plot]

    for row_idx, xai_cond in enumerate(['H', 'H+AI']):
        for col_idx, task in enumerate(TASKS):
            ax = axes[row_idx, col_idx]
            subset = plot_data[(plot_data['xai_condition'] == xai_cond) &
                               (plot_data['task'] == task)]
            if len(subset) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
                continue

            for og, color in [('Easy-Mild First', PALETTE['easy_mild_first']),
                              ('Hard-Strong First', PALETTE['hard_strong_first'])]:
                sg = subset[subset['order_group'] == og]
                if len(sg) > 0:
                    ax.scatter(sg['recent_error_rate'], sg['trust'],
                               color=color, alpha=0.6, s=50, label=og, edgecolors='white', linewidth=0.5)
                    # Regression line (guarded against NaN / constant data)
                    if len(sg) > 3:
                        x_vals = sg['recent_error_rate'].values
                        y_vals = sg['trust'].values
                        mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
                        x_vals, y_vals = x_vals[mask], y_vals[mask]
                        if len(x_vals) > 3 and np.std(x_vals) > 0 and np.std(y_vals) > 0:
                            try:
                                z = np.polyfit(x_vals, y_vals, 1)
                                p = np.poly1d(z)
                                x_range = np.linspace(x_vals.min(), x_vals.max(), 50)
                                ax.plot(x_range, p(x_range), color=color, linewidth=2, alpha=0.7, linestyle='--')
                            except (np.linalg.LinAlgError, ValueError):
                                pass  # skip regression line if fitting fails

            # Overall regression
            rho, p_val = stats.spearmanr(subset['recent_error_rate'], subset['trust'])
            ax.set_title(f"{xai_cond} — {task.replace('_', ' ').title()}\nρ={rho:.3f}, p={p_val:.4f}",
                         fontweight='bold', fontsize=11)
            ax.set_xlabel('Recent AI Error Rate (last 4 trials)')
            ax.set_ylabel('Trust Rating')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if row_idx == 0 and col_idx == 1:
                ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUTPUT_DIR, "fig4_trust_vs_error_recency.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("  → fig4_trust_vs_error_recency.png saved")

    # ─── VISUALIZATION 5: Recent vs Early Error Impact on Trust ───────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Recency Effect: Recent vs Early AI Errors' Correlation with Trust",
                 fontsize=14, fontweight='bold')

    for idx, xai_cond in enumerate(['H', 'H+AI']):
        ax = axes[idx]
        sub_corr = corr_df[corr_df['xai_condition'] == xai_cond]
        if len(sub_corr) == 0:
            continue
        x = np.arange(len(sub_corr))
        width = 0.35
        ax.bar(x - width/2, sub_corr['rho_recent'], width, label='Recent errors → Trust',
               color=PALETTE['hard_strong_first'], alpha=0.8)
        ax.bar(x + width/2, sub_corr['rho_early'], width, label='Early errors → Trust',
               color=PALETTE['easy_mild_first'], alpha=0.8)
        labels = [f"{r['task']}\nw={r['window']}" for _, r in sub_corr.iterrows()]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Spearman ρ")
        ax.set_title(f"{xai_cond}", fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUTPUT_DIR, "fig5_recency_vs_early_errors.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("  → fig5_recency_vs_early_errors.png saved")
else:
    print("  [!] Could not compute trust vs error recency (columns missing or unparseable)")

# ─── ANALYSIS 4: Answer Time Dynamics (Learning / Fatigue) ───────────────────
print("\n" + "=" * 70)
print("ANALYSIS 4: ANSWER TIME DYNAMICS BY TASK ORDER")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Answer Time Trajectories by Task Order\n(Trial-by-trial within each task)",
             fontsize=15, fontweight='bold', y=0.98)

for row_idx, xai_cond in enumerate(['H', 'H+AI']):
    sub = df[df['xai_condition'] == xai_cond]
    for col_idx, task in enumerate(TASKS):
        ax = axes[row_idx, col_idx]
        at_col = f"answer_times_{task}"
        if at_col not in sub.columns:
            continue

        for order_val, label, color in [
            (task, f'{task} first', PALETTE['easy_mild_first'] if task == 'easy_mild' else PALETTE['hard_strong_first']),
            ('hard_strong' if task == 'easy_mild' else 'easy_mild',
             f'{task} later',
             '#999999'),
        ]:
            group = sub[sub['first_task'] == order_val]
            all_times = []
            for _, row in group.iterrows():
                try:
                    times = ast.literal_eval(row[at_col]) if isinstance(row[at_col], str) else row[at_col]
                    if isinstance(times, (list, np.ndarray)):
                        all_times.append(times)
                except:
                    continue

            if not all_times:
                continue
            max_len = max(len(t) for t in all_times)
            # Pad with NaN
            padded = np.full((len(all_times), max_len), np.nan)
            for i, t in enumerate(all_times):
                padded[i, :len(t)] = t

            means = np.nanmean(padded, axis=0)
            sems = np.nanstd(padded, axis=0) / np.sqrt(np.sum(~np.isnan(padded), axis=0))
            trials = np.arange(1, max_len + 1)

            ax.plot(trials, means, color=color, linewidth=2, label=label)
            ax.fill_between(trials, means - sems, means + sems, color=color, alpha=0.15)

        ax.set_title(f"{xai_cond} — {task.replace('_', ' ').title()}", fontweight='bold')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Answer Time (s)')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUTPUT_DIR, "fig6_answer_time_trajectories.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  → fig6_answer_time_trajectories.png saved")

# ─── ANALYSIS 5: Trial-level Reliance Dynamics ──────────────────────────────
print("\n" + "=" * 70)
print("ANALYSIS 5: TRIAL-LEVEL RELIANCE DYNAMICS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Trial-by-Trial AI Agreement Rate by Task Order\n(Moving average of user agreeing with AI)",
             fontsize=15, fontweight='bold', y=0.98)

for row_idx, xai_cond in enumerate(['H', 'H+AI']):
    sub = df[df['xai_condition'] == xai_cond]
    for col_idx, task in enumerate(TASKS):
        ax = axes[row_idx, col_idx]
        ud_col = f"user_decision_{task}"
        ap_col = f"ai_pred_{task}"

        if ud_col not in sub.columns or ap_col not in sub.columns:
            continue

        for order_val, label, color in [
            (task, f'{task} first', PALETTE['easy_mild_first'] if task == 'easy_mild' else PALETTE['hard_strong_first']),
            ('hard_strong' if task == 'easy_mild' else 'easy_mild', f'{task} later', '#999999'),
        ]:
            group = sub[sub['first_task'] == order_val]
            all_agree = []
            for _, row in group.iterrows():
                try:
                    ud = ast.literal_eval(row[ud_col]) if isinstance(row[ud_col], str) else row[ud_col]
                    ap = ast.literal_eval(row[ap_col]) if isinstance(row[ap_col], str) else row[ap_col]
                    if isinstance(ud, list) and isinstance(ap, list):
                        agree = [1 if u == a else 0 for u, a in zip(ud, ap)]
                        all_agree.append(agree)
                except:
                    continue

            if not all_agree:
                continue
            max_len = max(len(a) for a in all_agree)
            padded = np.full((len(all_agree), max_len), np.nan)
            for i, a in enumerate(all_agree):
                padded[i, :len(a)] = a

            # Moving average (window=3)
            means = np.nanmean(padded, axis=0)
            window = 3
            if len(means) >= window:
                smoothed = np.convolve(means, np.ones(window)/window, mode='valid')
                trials = np.arange(window, len(means) + 1)
                ax.plot(trials, smoothed, color=color, linewidth=2, label=label)
            else:
                ax.plot(range(1, len(means)+1), means, color=color, linewidth=2, label=label)

        ax.set_title(f"{xai_cond} — {task.replace('_', ' ').title()}", fontweight='bold')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('AI Agreement Rate\n(3-trial moving avg)')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUTPUT_DIR, "fig7_reliance_dynamics.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  → fig7_reliance_dynamics.png saved")

# ─── Summary Report ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"""
Output files saved to {OUTPUT_DIR}:
  - order_effects_stats.csv        : Mann-Whitney U test results for all DVs
  - permutation_test_results.csv   : Permutation test p-values (robust)
  - trust_error_correlations.csv   : Trust ~ error recency correlations
  - trust_error_recency_data.csv   : Raw data for trust-recency analysis
  - fig1_order_effects_bars.png    : Bar charts with 95% bootstrap CIs
  - fig2_violin_distributions.png  : Violin + strip plots of distributions
  - fig3_effect_size_heatmap.png   : Cohen's d heatmap
  - fig4_trust_vs_error_recency.png: Trust vs recent error rate scatter
  - fig5_recency_vs_early_errors.png: Recency vs primacy effect on trust
  - fig6_answer_time_trajectories.png: Trial-by-trial answer times
  - fig7_reliance_dynamics.png     : Trial-by-trial AI agreement dynamics

Statistical Methods:
  1. Mann-Whitney U tests (nonparametric, robust to non-normality)
  2. Permutation tests (10,000 permutations, distribution-free)
  3. Cohen's d and rank-biserial r for effect sizes
  4. Bootstrap 95% confidence intervals (10,000 resamples)
  5. Spearman correlations for trust-recency relationships
  6. Steiger's Z-test for comparing dependent correlations
     (recent vs early error → trust)
""")
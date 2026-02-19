"""
Trust vs Error Recency — Fixed Version
========================================
Uses each participant's individual question order to reconstruct
their unique trial-by-trial experience of AI errors, then correlates
the recency of experienced errors with trust.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ast
import sys
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
OUTPUT_DIR = ".."

df = pd.read_csv(DATA_PATH)

# Filter to H+AI only (H has no AI, so no AI errors to analyze)
df = df[df['xai_condition'] == 'H+AI'].copy()

# Parse tasks_order
def parse_col(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except:
            return val
    return val

df['tasks_order'] = df['tasks_order'].apply(parse_col)
df['first_task'] = df['tasks_order'].apply(lambda x: x[0] if isinstance(x, list) else None)
df = df[df['first_task'].isin(['easy_mild', 'hard_strong'])].copy()
df['order_group'] = df['first_task'].map({
    'easy_mild': 'Easy-Mild First',
    'hard_strong': 'Hard-Strong First'
})

print(f"Participants in analysis: {len(df)}")
print(f"Order groups: {df['order_group'].value_counts().to_dict()}")

# ─── Reconstruct per-participant trial-by-trial AI error experience ──────────
TASKS = ['easy_mild', 'hard_strong']

recency_rows = []

for _, row in df.iterrows():
    for task in TASKS:
        trust_col = f"trust_{task}"
        true_col = f"task_true_{task}"
        pred_col = f"ai_pred_{task}"
        order_col = f"quest_order_{task}"
        decision_col = f"user_decision_{task}"

        trust_val = row.get(trust_col)
        if pd.isna(trust_val):
            continue

        try:
            task_true = parse_col(row[true_col])
            ai_pred = parse_col(row[pred_col])
            quest_order = parse_col(row[order_col])
            user_decision = parse_col(row[decision_col])
        except:
            continue

        if not all(isinstance(x, list) for x in [task_true, ai_pred, quest_order, user_decision]):
            continue

        n_trials = len(task_true)
        if n_trials < 4:
            continue

        # Convert to booleans
        task_true = [bool(x) for x in task_true]
        ai_pred = [bool(x) for x in ai_pred]
        user_decision = [bool(x) for x in user_decision]

        # Per-trial metrics (in the ORDER the participant experienced them)
        # ai_error[i] = 1 if AI was wrong on trial i
        ai_errors = [1 if t != p else 0 for t, p in zip(task_true, ai_pred)]
        # user_followed_ai[i] = 1 if user agreed with AI on trial i
        user_followed = [1 if u == p else 0 for u, p in zip(user_decision, ai_pred)]
        # user_was_wrong_following_ai[i] = 1 if user followed AI and AI was wrong
        user_wrong_following = [1 if (u == p and t != p) else 0
                                 for u, p, t in zip(user_decision, ai_pred, task_true)]

        for window in [2, 3, 4, 6]:
            if n_trials >= window:
                # Recent = last `window` trials the participant did
                recent_ai_errors = sum(ai_errors[-window:])
                early_ai_errors = sum(ai_errors[:window])

                # Also: did the participant experience disagreement recently?
                recent_followed = sum(user_followed[-window:])
                early_followed = sum(user_followed[:window])

                # User's recent negative experience: followed AI but was wrong
                recent_bad_follows = sum(user_wrong_following[-window:])
                early_bad_follows = sum(user_wrong_following[:window])

                recency_rows.append({
                    'task': task,
                    'order_group': row['order_group'],
                    'trust': trust_val,
                    'window': window,
                    'n_trials': n_trials,
                    # AI objective errors
                    'recent_ai_errors': recent_ai_errors,
                    'early_ai_errors': early_ai_errors,
                    'total_ai_errors': sum(ai_errors),
                    'recent_ai_error_rate': recent_ai_errors / window,
                    'early_ai_error_rate': early_ai_errors / window,
                    # User's reliance experience
                    'recent_follow_rate': recent_followed / window,
                    'early_follow_rate': early_followed / window,
                    'total_follow_rate': sum(user_followed) / n_trials,
                    # User's BAD experience (followed AI and was wrong)
                    'recent_bad_follow_rate': recent_bad_follows / window,
                    'early_bad_follow_rate': early_bad_follows / window,
                    'total_bad_follow_rate': sum(user_wrong_following) / n_trials,
                })

tr_df = pd.DataFrame(recency_rows)
print(f"\nRecency data rows: {len(tr_df)}")

# ─── Check variance in the key columns ──────────────────────────────────────
print("\n" + "=" * 70)
print("VARIANCE CHECK")
print("=" * 70)
for task in TASKS:
    for window in [3, 4, 6]:
        sub = tr_df[(tr_df['task'] == task) & (tr_df['window'] == window)]
        print(f"\n  {task}, window={window} (n={len(sub)}):")
        for col in ['recent_ai_error_rate', 'recent_follow_rate', 'recent_bad_follow_rate', 'trust']:
            vals = sub[col].dropna()
            print(f"    {col:30s}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
                  f"unique={vals.nunique()}, range=[{vals.min():.3f}, {vals.max():.3f}]")

# ─── Correlations ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CORRELATIONS: Trust ~ Error/Reliance Recency (Spearman)")
print("=" * 70)

def safe_spearmanr(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
        return (np.nan, np.nan)
    return stats.spearmanr(a, b)

def p_to_stars(p):
    if np.isnan(p): return ''
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    if p < 0.1: return '†'
    return 'ns'

corr_rows = []
predictors = [
    ('recent_ai_error_rate', 'early_ai_error_rate', 'AI Error Rate'),
    ('recent_follow_rate', 'early_follow_rate', 'AI Follow Rate'),
    ('recent_bad_follow_rate', 'early_bad_follow_rate', 'Bad Follow Rate'),
]

for task in TASKS:
    for window in [3, 4, 6]:
        sub = tr_df[(tr_df['task'] == task) & (tr_df['window'] == window)]
        if len(sub) < 5:
            continue
        print(f"\n  {task}, window={window} (n={len(sub)}):")
        for recent_col, early_col, label in predictors:
            rho_r, p_r = safe_spearmanr(sub[recent_col], sub['trust'])
            rho_e, p_e = safe_spearmanr(sub[early_col], sub['trust'])
            print(f"    {label:20s}: ρ_recent={rho_r:+.3f} (p={p_r:.4f} {p_to_stars(p_r)}) | "
                  f"ρ_early={rho_e:+.3f} (p={p_e:.4f} {p_to_stars(p_e)})")
            corr_rows.append({
                'task': task, 'window': window, 'predictor': label,
                'rho_recent': rho_r, 'p_recent': p_r,
                'rho_early': rho_e, 'p_early': p_e, 'n': len(sub),
            })

            # Steiger's Z if both correlations exist
            if not np.isnan(rho_r) and not np.isnan(rho_e):
                r12, _ = safe_spearmanr(sub[recent_col], sub[early_col])
                if not np.isnan(r12):
                    n = len(sub)
                    # Steiger Z (simplified)
                    z1, z2 = np.arctanh(rho_r), np.arctanh(rho_e)
                    denom = np.sqrt(2*(1-r12)/(n-3))
                    if denom > 0:
                        z_steiger = (z1 - z2) / denom
                        p_steiger = 2 * (1 - stats.norm.cdf(abs(z_steiger)))
                        print(f"      → Recency > Primacy? Z={z_steiger:+.3f}, p={p_steiger:.4f} {p_to_stars(p_steiger)}")

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(f"{OUTPUT_DIR}/trust_recency_correlations_fixed.csv", index=False)
tr_df.to_csv(f"{OUTPUT_DIR}/trust_recency_data_fixed.csv", index=False)

# ─── Split by order group ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CORRELATIONS SPLIT BY ORDER GROUP")
print("=" * 70)

for task in TASKS:
    for window in [4, 6]:
        for og in ['Easy-Mild First', 'Hard-Strong First']:
            sub = tr_df[(tr_df['task'] == task) & (tr_df['window'] == window)
                        & (tr_df['order_group'] == og)]
            if len(sub) < 5:
                continue
            print(f"\n  {task}, w={window}, {og} (n={len(sub)}):")
            for recent_col, early_col, label in predictors:
                rho_r, p_r = safe_spearmanr(sub[recent_col], sub['trust'])
                rho_e, p_e = safe_spearmanr(sub[early_col], sub['trust'])
                if not np.isnan(rho_r) or not np.isnan(rho_e):
                    print(f"    {label:20s}: ρ_recent={rho_r:+.3f} ({p_to_stars(p_r)}) | "
                          f"ρ_early={rho_e:+.3f} ({p_to_stars(p_e)})")

# ─── Visualizations ──────────────────────────────────────────────────────────
print("\n→ Generating visualizations...")

PALETTE = {
    'Easy-Mild First': '#2E86AB',
    'Hard-Strong First': '#E8533F',
}

# Fig A: Scatter plots of the predictor with most variance vs trust
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Trust vs Trial-Level Experience (H+AI condition)\nWindow = 4 last/first trials",
             fontsize=14, fontweight='bold', y=0.98)

window_plot = 4
plot_data = tr_df[tr_df['window'] == window_plot]

# Determine which predictor has most variance per task
for col_idx, task in enumerate(TASKS):
    task_data = plot_data[plot_data['task'] == task]

    # Row 0: recent bad follow rate (most participant-specific)
    for row_idx, (pred_col, pred_label) in enumerate([
        ('recent_bad_follow_rate', 'Recent Bad-Follow Rate\n(followed AI & was wrong, last 4 trials)'),
        ('recent_follow_rate', 'Recent AI-Follow Rate\n(agreed with AI, last 4 trials)'),
    ]):
        ax = axes[row_idx, col_idx]
        for og, color in PALETTE.items():
            sg = task_data[task_data['order_group'] == og]
            if len(sg) == 0:
                continue
            # Jitter for visibility
            jitter_x = np.random.default_rng(42).normal(0, 0.02, len(sg))
            jitter_y = np.random.default_rng(43).normal(0, 0.08, len(sg))
            ax.scatter(sg[pred_col] + jitter_x, sg['trust'] + jitter_y,
                       color=color, alpha=0.6, s=50, label=og,
                       edgecolors='white', linewidth=0.5)

            # Regression line
            x_vals = sg[pred_col].values
            y_vals = sg['trust'].values
            mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
            xv, yv = x_vals[mask], y_vals[mask]
            if len(xv) > 3 and np.std(xv) > 0:
                try:
                    z = np.polyfit(xv, yv, 1)
                    p = np.poly1d(z)
                    xr = np.linspace(xv.min(), xv.max(), 50)
                    ax.plot(xr, p(xr), color=color, linewidth=2, alpha=0.7, linestyle='--')
                except:
                    pass

        rho, p_val = safe_spearmanr(task_data[pred_col], task_data['trust'])
        sig = p_to_stars(p_val)
        ax.set_title(f"{task.replace('_', ' ').title()}\nρ={rho:+.3f}, p={p_val:.3f} {sig}",
                     fontweight='bold', fontsize=11)
        ax.set_xlabel(pred_label)
        ax.set_ylabel('Trust Rating')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if row_idx == 0 and col_idx == 1:
            ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUTPUT_DIR}/fig4_trust_vs_recency_fixed.png", dpi=200, bbox_inches='tight')
plt.close()
print("  → fig4_trust_vs_recency_fixed.png saved")

# Fig B: Comparison of recent vs early correlation strengths
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Recency vs Primacy: Which Predicts Trust More?",
             fontsize=14, fontweight='bold')

for idx, task in enumerate(TASKS):
    ax = axes[idx]
    sub_corr = corr_df[corr_df['task'] == task]
    if len(sub_corr) == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
        continue

    x = np.arange(len(sub_corr))
    width = 0.35
    bars1 = ax.bar(x - width/2, sub_corr['rho_recent'].fillna(0), width,
                   label='Recent → Trust', color='#E8533F', alpha=0.8)
    bars2 = ax.bar(x + width/2, sub_corr['rho_early'].fillna(0), width,
                   label='Early → Trust', color='#2E86AB', alpha=0.8)

    # Add significance stars
    for i, (_, r) in enumerate(sub_corr.iterrows()):
        if not np.isnan(r['rho_recent']):
            ax.text(x[i] - width/2, r['rho_recent'] + 0.02 * np.sign(r['rho_recent']),
                    p_to_stars(r['p_recent']), ha='center', fontsize=8)
        if not np.isnan(r['rho_early']):
            ax.text(x[i] + width/2, r['rho_early'] + 0.02 * np.sign(r['rho_early']),
                    p_to_stars(r['p_early']), ha='center', fontsize=8)

    labels = [f"{r['predictor']}\nw={r['window']}" for _, r in sub_corr.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_ylabel("Spearman ρ with Trust")
    ax.set_title(f"{task.replace('_', ' ').title()}", fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{OUTPUT_DIR}/fig5_recency_vs_primacy_fixed.png", dpi=200, bbox_inches='tight')
plt.close()
print("  → fig5_recency_vs_primacy_fixed.png saved")

print("\n✓ Trust-recency analysis complete.")
print(f"  Saved: trust_recency_correlations_fixed.csv, trust_recency_data_fixed.csv")
print(f"  Saved: fig4_trust_vs_recency_fixed.png, fig5_recency_vs_primacy_fixed.png")

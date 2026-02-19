"""
Task Order Effects Analysis v4
================================
Nomenclature:
  Difficulty: Low (9x9) / High (15x15)
  Time Pressure: Low (25s) / Strong (10s)
  Input mapping: easy->Low, hard->High, mild->Low pressure, strong->Strong pressure

Changes from v3:
  - Answer time analysis uses all individual trial times (not aggregated means)
  - All 4 task conditions included in answer time analysis
  - Test names shown on plots
  - Recency analysis removed
"""
import pandas as pd, numpy as np, matplotlib, ast, sys, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
warnings.filterwarnings('ignore')

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
OUTPUT_DIR = "."
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':10,
    'axes.titlesize':12,'axes.titleweight':'bold','axes.labelsize':10,
    'figure.facecolor':'#FAFAFA','axes.facecolor':'#FFFFFF','axes.edgecolor':'#CCCCCC',
    'axes.grid':True,'grid.alpha':0.25,'grid.color':'#DDDDDD'})

# ── Nomenclature ──────────────────────────────────────────────────────
ALL_TASKS = ['easy_mild', 'easy_strong', 'hard_mild', 'hard_strong']
TARGET_TASKS = ['easy_mild', 'hard_strong']

DISP = {
    'easy_mild':   'Low / Low',
    'easy_strong':  'Low / Strong',
    'hard_mild':   'High / Low',
    'hard_strong': 'High / Strong',
}
DISP_LONG = {
    'easy_mild':   'Low difficulty\nLow pressure',
    'easy_strong':  'Low difficulty\nStrong pressure',
    'hard_mild':   'High difficulty\nLow pressure',
    'hard_strong': 'High difficulty\nStrong pressure',
}
DISP_SHORT = {
    'easy_mild':   'Low/Low',
    'easy_strong':  'Low/Strong',
    'hard_mild':   'High/Low',
    'hard_strong': 'High/Strong',
}

PAL = {'easy_mild':'#2E86AB','easy_strong':'#7FB069','hard_mild':'#E8963F','hard_strong':'#E8533F'}

# ── Helpers ───────────────────────────────────────────────────────────
def parse_col(val):
    if isinstance(val, str):
        try: return ast.literal_eval(val)
        except: return val
    return val

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2: return np.nan
    p = np.sqrt(((n1-1)*np.var(g1,ddof=1)+(n2-1)*np.var(g2,ddof=1))/(n1+n2-2))
    return (np.mean(g1)-np.mean(g2))/p if p > 0 else 0

def bootstrap_ci(data, n_boot=10000, ci=0.95):
    rng = np.random.default_rng(42)
    data = np.array(data, dtype=float); data = data[~np.isnan(data)]
    if len(data) < 3: return (np.nan, np.nan)
    boot = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    a = (1-ci)/2
    return (np.percentile(boot, a*100), np.percentile(boot, (1-a)*100))

def permutation_test(g1, g2, n_perm=10000):
    g1 = np.array(g1, dtype=float); g1 = g1[~np.isnan(g1)]
    g2 = np.array(g2, dtype=float); g2 = g2[~np.isnan(g2)]
    if len(g1) < 2 or len(g2) < 2: return np.nan
    obs = np.mean(g1) - np.mean(g2)
    comb = np.concatenate([g1, g2]); rng = np.random.default_rng(42); ct = 0
    for _ in range(n_perm):
        rng.shuffle(comb)
        if abs(np.mean(comb[:len(g1)]) - np.mean(comb[len(g1):])) >= abs(obs): ct += 1
    return ct / n_perm

def p_stars(p):
    if np.isnan(p): return ''
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    if p < 0.1: return '\u2020'
    return 'ns'

def run_comp(v1, v2):
    v1 = np.array(v1, dtype=float); v1 = v1[~np.isnan(v1)]
    v2 = np.array(v2, dtype=float); v2 = v2[~np.isnan(v2)]
    if len(v1) < 3 or len(v2) < 3: return None
    u, p_mw = stats.mannwhitneyu(v1, v2, alternative='two-sided')
    d = cohens_d(v1, v2); n1, n2 = len(v1), len(v2)
    r_rb = 1 - (2*u)/(n1*n2); p_perm = permutation_test(v1, v2)
    return {'n1':n1,'n2':n2,'mean1':np.mean(v1),'mean2':np.mean(v2),
            'sd1':np.std(v1,ddof=1),'sd2':np.std(v2,ddof=1),
            'U':u,'p_mw':p_mw,'p_perm':p_perm,'d':d,'r_rb':r_rb}

# ── Load Data ─────────────────────────────────────────────────────────
print("="*70 + "\nTASK ORDER EFFECTS ANALYSIS v4\n" + "="*70)
df = pd.read_csv(DATA_PATH)
print(f"Total participants: {len(df)}")
df = df[df['xai_condition'].isin(['H','H+AI'])].copy()
print(f"After filtering to H and H+AI: {len(df)}")

df['tasks_order'] = df['tasks_order'].apply(parse_col)
df['first_task'] = df['tasks_order'].apply(lambda x: x[0] if isinstance(x, list) else None)

print(f"\nStarting task distribution:")
for t in ALL_TASKS:
    n = (df['first_task']==t).sum()
    print(f"  {DISP[t]:15s}: {n}")
print(f"\nBy condition:")
for t in ALL_TASKS:
    h = ((df['first_task']==t)&(df['xai_condition']=='H')).sum()
    hai = ((df['first_task']==t)&(df['xai_condition']=='H+AI')).sum()
    print(f"  {DISP[t]:15s}: H={h}, H+AI={hai}")

DVS = {
    'score':'Accuracy', 'reliance':'Reliance', 'overreliance':'Over-reliance',
    'underreliance':'Under-reliance', 'trust':'Trust', 'cogload':'Cognitive Load',
}

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Pairwise comparisons (all 4 starting conditions)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70 + "\nANALYSIS 1: EFFECT OF STARTING TASK ON TARGET TASK DVs\n" + "="*70)
results_rows = []
for xai_cond in ['H','H+AI']:
    sub = df[df['xai_condition']==xai_cond]
    print(f"\n{'='*60}\n  {xai_cond} (n={len(sub)})\n{'='*60}")
    for target in TARGET_TASKS:
        print(f"\n  -- Target: {DISP[target]} --")
        for dv_base, dv_label in DVS.items():
            col = f"{dv_base}_{target}"
            if col not in sub.columns: continue
            print(f"\n    {dv_label}:")
            # Kruskal-Wallis
            groups_kw = []; p_kw = np.nan
            for s in ALL_TASKS:
                g = sub[sub['first_task']==s][col].dropna()
                if len(g) >= 2: groups_kw.append(g.values)
            if len(groups_kw) >= 2:
                try:
                    H_stat, p_kw = stats.kruskal(*groups_kw)
                    N = sum(len(g) for g in groups_kw); k = len(groups_kw)
                    eta2 = (H_stat - k + 1)/(N - k) if (N-k) > 0 else np.nan
                    print(f"      Kruskal-Wallis: H={H_stat:.3f}, p={p_kw:.4f} {p_stars(p_kw)}, "
                          f"\u03b7\u00b2={eta2:.3f}")
                except: pass
            # Pairwise
            tf = sub[sub['first_task']==target][col].dropna()
            for other in ALL_TASKS:
                if other == target: continue
                of = sub[sub['first_task']==other][col].dropna()
                res = run_comp(tf.values, of.values)
                if res is None:
                    print(f"      vs {DISP_SHORT[other]:12s}: insufficient data"); continue
                print(f"      vs started {DISP_SHORT[other]:12s}: "
                      f"M={res['mean1']:.3f} vs {res['mean2']:.3f}, d={res['d']:+.3f}, "
                      f"p_MW={res['p_mw']:.4f}{p_stars(res['p_mw'])}, "
                      f"p_perm={res['p_perm']:.4f}{p_stars(res['p_perm'])}")
                results_rows.append({
                    'xai_condition':xai_cond,'target_task':target,'target_display':DISP[target],
                    'dv':dv_label,'dv_base':dv_base,
                    'target_first_mean':res['mean1'],'target_first_sd':res['sd1'],'target_first_n':res['n1'],
                    'compared_start':other,'compared_display':DISP[other],
                    'other_first_mean':res['mean2'],'other_first_sd':res['sd2'],'other_first_n':res['n2'],
                    'U':res['U'],'p_mw':res['p_mw'],'p_perm':res['p_perm'],
                    'd':res['d'],'r_rb':res['r_rb'],'kw_p':p_kw})
results_df = pd.DataFrame(results_rows)
results_df.to_csv(f"{OUTPUT_DIR}/order_effects_all_starts.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 2: First vs later (binary)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70 + "\nANALYSIS 2: TARGET FIRST vs NOT FIRST\n" + "="*70)
binary_rows = []
for xai_cond in ['H','H+AI']:
    sub = df[df['xai_condition']==xai_cond]
    print(f"\n  {xai_cond} (n={len(sub)})")
    for target in TARGET_TASKS:
        print(f"  Target: {DISP[target]}")
        gf = sub[sub['first_task']==target]; gl = sub[sub['first_task']!=target]
        for dv_base, dv_label in DVS.items():
            col = f"{dv_base}_{target}"
            if col not in sub.columns: continue
            res = run_comp(gf[col].dropna().values, gl[col].dropna().values)
            if res is None: continue
            print(f"    {dv_label:25s}: 1st M={res['mean1']:.3f}(n={res['n1']}) "
                  f"vs later M={res['mean2']:.3f}(n={res['n2']}), d={res['d']:+.3f}, "
                  f"p_MW={res['p_mw']:.4f}{p_stars(res['p_mw'])}, "
                  f"p_perm={res['p_perm']:.4f}{p_stars(res['p_perm'])}")
            binary_rows.append({
                'xai_condition':xai_cond,'target_task':target,'target_display':DISP[target],
                'dv':dv_label,'dv_base':dv_base,
                'first_mean':res['mean1'],'first_sd':res['sd1'],'first_n':res['n1'],
                'later_mean':res['mean2'],'later_sd':res['sd2'],'later_n':res['n2'],
                'U':res['U'],'p_mw':res['p_mw'],'p_perm':res['p_perm'],
                'd':res['d'],'r_rb':res['r_rb']})
pd.DataFrame(binary_rows).to_csv(f"{OUTPUT_DIR}/order_effects_first_vs_later.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Answer times — full trial-level, all 4 conditions
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70 + "\nANALYSIS 3: ANSWER TIMES (trial-level, all conditions)\n" + "="*70)

# Build long-form answer time dataframe
at_rows = []
for _, row in df.iterrows():
    for task in ALL_TASKS:
        at_col = f"answer_times_{task}"
        if at_col not in df.columns: continue
        times = parse_col(row[at_col])
        if not isinstance(times, (list, np.ndarray)): continue
        for trial_idx, t in enumerate(times):
            at_rows.append({
                'participant': row.name,
                'xai_condition': row['xai_condition'],
                'first_task': row['first_task'],
                'task': task,
                'task_display': DISP[task],
                'trial': trial_idx + 1,
                'answer_time': float(t),
            })
at_df = pd.DataFrame(at_rows)
print(f"  Total trial-level observations: {len(at_df)}")

# Per-participant mean answer time for statistical tests
at_means = at_df.groupby(['participant','xai_condition','first_task','task'])['answer_time'].mean().reset_index()
at_means.rename(columns={'answer_time':'mean_at'}, inplace=True)

# Statistical tests on answer times
at_stat_rows = []
for xai_cond in ['H','H+AI']:
    sub = at_means[at_means['xai_condition']==xai_cond]
    print(f"\n  {xai_cond}:")
    for task in ALL_TASKS:
        tsub = sub[sub['task']==task]
        # Kruskal-Wallis across 4 starting groups
        groups = []; p_kw = np.nan
        for s in ALL_TASKS:
            g = tsub[tsub['first_task']==s]['mean_at'].dropna()
            if len(g) >= 2: groups.append(g.values)
        if len(groups) >= 2:
            try:
                H_stat, p_kw = stats.kruskal(*groups)
                print(f"    {DISP[task]:15s}: KW H={H_stat:.3f}, p={p_kw:.4f} {p_stars(p_kw)}")
            except: pass
        # Task-first vs later
        tf = tsub[tsub['first_task']==task]['mean_at'].dropna()
        tl = tsub[tsub['first_task']!=task]['mean_at'].dropna()
        res = run_comp(tf.values, tl.values)
        if res:
            print(f"      First vs later: M={res['mean1']:.2f} vs {res['mean2']:.2f}, "
                  f"d={res['d']:+.3f}, p_MW={res['p_mw']:.4f}{p_stars(res['p_mw'])}, "
                  f"p_perm={res['p_perm']:.4f}{p_stars(res['p_perm'])}")
            at_stat_rows.append({
                'xai_condition':xai_cond,'task':task,'task_display':DISP[task],
                'first_mean':res['mean1'],'later_mean':res['mean2'],
                'first_n':res['n1'],'later_n':res['n2'],
                'd':res['d'],'p_mw':res['p_mw'],'p_perm':res['p_perm'],'kw_p':p_kw})

pd.DataFrame(at_stat_rows).to_csv(f"{OUTPUT_DIR}/answer_time_stats.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════
print("\n-> Generating visualizations...")

# ── Fig 1: Forest plot ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 12), sharey=True)
fig.suptitle("Effect of Starting Task on Target Task Outcomes (Cohen's d)\n"
             "Positive d = higher when target done first  |  "
             "Tests: Mann-Whitney U + Permutation (10k)",
             fontsize=12, fontweight='bold', y=0.98)
for idx, xai_cond in enumerate(['H','H+AI']):
    ax = axes[idx]
    sr = results_df[results_df['xai_condition']==xai_cond].copy()
    if len(sr)==0: continue
    sr = sr.sort_values(['target_task','dv_base','compared_start']).reset_index(drop=True)
    for i, (_, r) in enumerate(sr.iterrows()):
        sig = p_stars(r['p_perm'])
        a = 1.0 if sig in ['*','**','***'] else 0.6 if sig=='\u2020' else 0.35
        ax.barh(i, r['d'], color=PAL[r['compared_start']], alpha=a,
                height=0.7, edgecolor='white', linewidth=0.3)
        if sig and sig not in ['ns','']:
            ax.text(r['d']+(0.05 if r['d']>=0 else -0.05), i, sig,
                    va='center', ha='left' if r['d']>=0 else 'right',
                    fontsize=8, fontweight='bold')
    ax.set_yticks(range(len(sr)))
    ax.set_yticklabels([f"{r['dv']} | vs {DISP_SHORT[r['compared_start']]}"
                        for _, r in sr.iterrows()], fontsize=7)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(-0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.axvline(0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.set_xlabel("Cohen's d (permutation test, 10k)")
    ax.set_title(xai_cond, fontweight='bold', fontsize=13)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    prev = None
    for i, (_, r) in enumerate(sr.iterrows()):
        if prev and r['target_task'] != prev: ax.axhline(i-0.5, color='black', linewidth=1)
        prev = r['target_task']
    # Target task labels
    for tgt in TARGET_TASKS:
        rows_t = sr[sr['target_task']==tgt]
        if len(rows_t) > 0:
            mid = (rows_t.index[0] + rows_t.index[-1]) / 2
            ax.text(-2.2, mid, f"Target:\n{DISP[tgt]}", fontsize=9, fontweight='bold',
                    va='center', ha='center', rotation=90)
legend_elements = [Patch(facecolor=c, label=DISP[t]) for t, c in PAL.items()]
axes[1].legend(handles=legend_elements, title='Comparison:\nStarted with...',
               loc='lower right', fontsize=8, title_fontsize=9)
plt.tight_layout(rect=[0.05, 0, 1, 0.94])
fig.savefig(f"{OUTPUT_DIR}/fig1_forest_plot.png", dpi=200, bbox_inches='tight'); plt.close()
print("  -> fig1_forest_plot.png")

# ── Fig 2: Bar charts per target ──────────────────────────────────────
key_dvs = list(DVS.keys())
for target in TARGET_TASKS:
    fig, axes = plt.subplots(2, len(key_dvs), figsize=(3.5*len(key_dvs), 9))
    fig.suptitle(f"Target Task: {DISP[target]} (Difficulty / Time Pressure)\n"
                 f"Grouped by Starting Task  |  95% Bootstrap CI  |  "
                 f"Mann-Whitney U pairwise tests",
                 fontsize=13, fontweight='bold', y=0.99)
    for row_idx, xai_cond in enumerate(['H','H+AI']):
        sub = df[df['xai_condition']==xai_cond]
        for col_idx, dv_base in enumerate(key_dvs):
            ax = axes[row_idx, col_idx]; col = f"{dv_base}_{target}"
            if col not in sub.columns: ax.set_visible(False); continue
            means, clo, chi, colors = [], [], [], []
            for s in ALL_TASKS:
                g = sub[sub['first_task']==s][col].dropna()
                if len(g) >= 2:
                    m = g.mean(); ci = bootstrap_ci(g.values)
                    means.append(m); clo.append(m-ci[0]); chi.append(ci[1]-m)
                else:
                    means.append(0); clo.append(0); chi.append(0)
                colors.append(PAL[s])
            bars = ax.bar(range(4), means, color=colors, alpha=0.8, width=0.65,
                          edgecolor='white', linewidth=0.5)
            ax.errorbar(range(4), means, yerr=[clo, chi],
                        fmt='none', color='#333', capsize=3, linewidth=1)
            tidx = ALL_TASKS.index(target)
            bars[tidx].set_edgecolor('black'); bars[tidx].set_linewidth(2)
            ax.set_xticks(range(4))
            ax.set_xticklabels([DISP_LONG[t] for t in ALL_TASKS], fontsize=6)
            ax.set_title(DVS[dv_base], fontsize=10, fontweight='bold')
            if col_idx == 0: ax.set_ylabel(xai_cond, fontsize=11, fontweight='bold')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUTPUT_DIR}/fig2_bars_{target}.png", dpi=200, bbox_inches='tight'); plt.close()
    print(f"  -> fig2_bars_{target}.png")

# ── Fig 3: Answer time trajectories — ALL 4 conditions ───────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Trial-by-Trial Answer Times \u2014 H+AI condition (mean \u00b1 SEM)\n"
             "Top: grouped by starting task  |  Bottom: task done first vs later",
             fontsize=14, fontweight='bold', y=0.99)
sub_ai_at = at_df[at_df['xai_condition']=='H+AI']

for col_idx, task in enumerate(ALL_TASKS):
    td = sub_ai_at[sub_ai_at['task']==task]
    # Row 0: by starting task
    ax = axes[0, col_idx]
    for st in ALL_TASKS:
        sg = td[td['first_task']==st]
        if len(sg) == 0: continue
        trial_means = sg.groupby('trial')['answer_time'].agg(['mean','sem']).reset_index()
        ax.plot(trial_means['trial'], trial_means['mean'], color=PAL[st], linewidth=2,
                label=f"Started {DISP_SHORT[st]}", alpha=0.9)
        ax.fill_between(trial_means['trial'],
                        trial_means['mean']-trial_means['sem'],
                        trial_means['mean']+trial_means['sem'],
                        color=PAL[st], alpha=0.12)
    ax.set_title(f"{DISP[task]}", fontweight='bold')
    ax.set_xlabel('Trial'); ax.set_ylabel('Time (s)')
    if col_idx == 0: ax.legend(fontsize=6, ncol=2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Row 1: first vs later
    ax = axes[1, col_idx]
    for is_f, lab, col_c, ls in [
        (True, f'{DISP_SHORT[task]} first', PAL[task], '-'),
        (False, f'{DISP_SHORT[task]} later', '#888888', '--')]:
        if is_f:
            sg = td[td['first_task']==task]
        else:
            sg = td[td['first_task']!=task]
        if len(sg) == 0: continue
        trial_means = sg.groupby('trial')['answer_time'].agg(['mean','sem']).reset_index()
        ax.plot(trial_means['trial'], trial_means['mean'], color=col_c, linewidth=2,
                linestyle=ls, label=lab, alpha=0.9)
        ax.fill_between(trial_means['trial'],
                        trial_means['mean']-trial_means['sem'],
                        trial_means['mean']+trial_means['sem'],
                        color=col_c, alpha=0.12)
    # Add Mann-Whitney stat annotation
    at_sub = at_means[(at_means['xai_condition']=='H+AI')&(at_means['task']==task)]
    tf_vals = at_sub[at_sub['first_task']==task]['mean_at'].dropna()
    tl_vals = at_sub[at_sub['first_task']!=task]['mean_at'].dropna()
    if len(tf_vals) >= 3 and len(tl_vals) >= 3:
        u, p_mw = stats.mannwhitneyu(tf_vals, tl_vals, alternative='two-sided')
        d = cohens_d(tf_vals.values, tl_vals.values)
        ax.text(0.02, 0.98, f"Mann-Whitney U={u:.0f}\np={p_mw:.4f} {p_stars(p_mw)}\nCohen's d={d:+.2f}",
                transform=ax.transAxes, fontsize=7, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#CCC'))
    ax.set_title(f"{DISP[task]}", fontweight='bold')
    ax.set_xlabel('Trial'); ax.set_ylabel('Time (s)')
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUTPUT_DIR}/fig3_answer_time_trajectories.png", dpi=200, bbox_inches='tight'); plt.close()
print("  -> fig3_answer_time_trajectories.png")

# ── Fig 3b: Answer time distributions (violin) ───────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Answer Time Distributions by Starting Task \u2014 H+AI\n"
             "All individual trial times  |  Kruskal-Wallis omnibus test",
             fontsize=13, fontweight='bold', y=1.02)
for col_idx, task in enumerate(ALL_TASKS):
    ax = axes[col_idx]
    td = sub_ai_at[sub_ai_at['task']==task].copy()
    td['start_label'] = td['first_task'].map(DISP_SHORT)
    order = [DISP_SHORT[s] for s in ALL_TASKS]
    pal_mapped = {DISP_SHORT[s]: PAL[s] for s in ALL_TASKS}
    # violin
    parts = []
    for i, s in enumerate(ALL_TASKS):
        vals = td[td['first_task']==s]['answer_time'].dropna().values
        if len(vals) > 1:
            vp = ax.violinplot([vals], positions=[i], widths=0.7, showmeans=True, showmedians=True)
            for pc in vp['bodies']:
                pc.set_facecolor(PAL[s]); pc.set_alpha(0.5)
            for partname in ['cmeans','cmedians','cmins','cmaxes','cbars']:
                if partname in vp:
                    vp[partname].set_edgecolor('#333')
    # KW test
    groups = []
    for s in ALL_TASKS:
        g = td[td['first_task']==s]['answer_time'].dropna().values
        if len(g) >= 2: groups.append(g)
    if len(groups) >= 2:
        H_stat, p_kw = stats.kruskal(*groups)
        ax.text(0.02, 0.98, f"Kruskal-Wallis\nH={H_stat:.2f}, p={p_kw:.4f} {p_stars(p_kw)}",
                transform=ax.transAxes, fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#CCC'))
    ax.set_xticks(range(4))
    ax.set_xticklabels([DISP_SHORT[s] for s in ALL_TASKS], fontsize=8)
    ax.set_xlabel('Started with...')
    ax.set_ylabel('Answer Time (s)')
    ax.set_title(f"Task: {DISP[task]}", fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig3b_answer_time_violins.png", dpi=200, bbox_inches='tight'); plt.close()
print("  -> fig3b_answer_time_violins.png")

# ── Fig 4: Heatmaps with target reference column ─────────────────────
for xai_cond in ['H','H+AI']:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Effect Size Heatmap (Cohen's d): {xai_cond}\n"
                 f"Pairwise: target-first group vs each other starting group  |  "
                 f"Permutation test (10k iterations)",
                 fontsize=12, fontweight='bold')
    sub_cond = df[df['xai_condition']==xai_cond]

    for idx, target in enumerate(TARGET_TASKS):
        ax = axes[idx]
        sr = results_df[(results_df['xai_condition']==xai_cond) &
                        (results_df['target_task']==target)].copy()

        dv_list = list(DVS.values())
        all_starts = [target] + [t for t in ALL_TASKS if t != target]
        col_labels = [DISP_SHORT[s] + ('\n(target)' if s==target else '') for s in all_starts]

        d_matrix = np.full((len(dv_list), len(all_starts)), np.nan)
        annot_matrix = [['']*len(all_starts) for _ in range(len(dv_list))]

        for j, start in enumerate(all_starts):
            if start == target:
                for i, (dv_base, dv_label) in enumerate(DVS.items()):
                    col = f"{dv_base}_{target}"
                    if col in sub_cond.columns:
                        vals = sub_cond[sub_cond['first_task']==target][col].dropna()
                        if len(vals) >= 2:
                            d_matrix[i, j] = 0.0
                            annot_matrix[i][j] = f"M={vals.mean():.2f}\nn={len(vals)}\n(ref)"
                        else:
                            annot_matrix[i][j] = "n/a"
            else:
                for i, (dv_base, dv_label) in enumerate(DVS.items()):
                    match = sr[(sr['dv']==dv_label)&(sr['compared_start']==start)]
                    if len(match) > 0:
                        r = match.iloc[0]
                        d_matrix[i, j] = r['d']
                        annot_matrix[i][j] = f"d={r['d']:.2f}\n{p_stars(r['p_perm'])}"

        d_df = pd.DataFrame(d_matrix, index=dv_list, columns=col_labels)
        annot_arr = np.array(annot_matrix)

        sns.heatmap(d_df, annot=annot_arr, fmt='', cmap='RdBu_r', center=0,
                    vmin=-1.5, vmax=1.5, ax=ax, linewidths=0.5,
                    cbar_kws={'label': "Cohen's d"})
        ax.set_title(f"Target: {DISP[target]}", fontweight='bold')
        ax.set_ylabel(''); ax.set_xlabel('Started with...')

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fname = f"fig4_heatmap_{xai_cond.replace('+','_')}.png"
    fig.savefig(f"{OUTPUT_DIR}/{fname}", dpi=200, bbox_inches='tight'); plt.close()
    print(f"  -> {fname}")

# ── Fig 5: Position effects ──────────────────────────────────────────
print("\n-> Position analysis...")
for task in ALL_TASKS:
    df[f"position_{task}"] = df['tasks_order'].apply(
        lambda x, t=task: (x.index(t)+1) if isinstance(x, list) and t in x else np.nan)

key_dvs_pos = list(DVS.keys())
fig, axes = plt.subplots(2, len(key_dvs_pos), figsize=(3.5*len(key_dvs_pos), 9))
fig.suptitle("Effect of Task Position (1st\u20134th) on Outcomes \u2014 H+AI\n"
             "Spearman rank correlation shown below each panel",
             fontsize=13, fontweight='bold', y=0.99)
sub_ai = df[df['xai_condition']=='H+AI']
for ri, target in enumerate(TARGET_TASKS):
    for ci, dv_base in enumerate(key_dvs_pos):
        ax = axes[ri, ci]; col = f"{dv_base}_{target}"; pc = f"position_{target}"
        if col not in sub_ai.columns: ax.set_visible(False); continue
        means, clo, chi = [], [], []
        for pos in [1,2,3,4]:
            g = sub_ai[sub_ai[pc]==pos][col].dropna()
            if len(g) >= 2:
                m = g.mean(); ci_v = bootstrap_ci(g.values)
                means.append(m); clo.append(m-ci_v[0]); chi.append(ci_v[1]-m)
            else:
                means.append(np.nan); clo.append(0); chi.append(0)
        ax.bar([1,2,3,4], means, color=['#2E86AB','#5BA08E','#E8963F','#E8533F'],
               alpha=0.8, width=0.6, edgecolor='white', linewidth=0.5)
        ax.errorbar([1,2,3,4], means, yerr=[clo, chi],
                    fmt='none', color='#333', capsize=3, linewidth=1)
        valid = sub_ai[[pc, col]].dropna()
        if len(valid) > 5:
            rho, pv = stats.spearmanr(valid[pc], valid[col])
            ax.set_xlabel(f"Position\nSpearman \u03c1={rho:+.2f}, p={pv:.3f} {p_stars(pv)}", fontsize=8)
        else:
            ax.set_xlabel("Position")
        ax.set_xticks([1,2,3,4]); ax.set_xticklabels(['1st','2nd','3rd','4th'], fontsize=8)
        ax.set_title(DVS[dv_base], fontsize=10, fontweight='bold')
        if ci == 0: ax.set_ylabel(f"Target: {DISP[target]}", fontsize=9, fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUTPUT_DIR}/fig5_position_effects.png", dpi=200, bbox_inches='tight'); plt.close()
print("  -> fig5_position_effects.png")

# Position correlations
print("\n  Position-DV correlations (Spearman, H+AI):")
pcr = []
for target in TARGET_TASKS:
    pc = f"position_{target}"
    for dv_base, dv_label in DVS.items():
        col = f"{dv_base}_{target}"
        if col not in sub_ai.columns: continue
        valid = sub_ai[[pc, col]].dropna()
        if len(valid) > 5:
            rho, pv = stats.spearmanr(valid[pc], valid[col])
            print(f"    {DISP[target]:15s} | {dv_label:25s}: \u03c1={rho:+.3f}, p={pv:.4f} {p_stars(pv)}")
            pcr.append({'target':target,'target_display':DISP[target],
                        'dv':dv_label,'rho':rho,'p':pv,'n':len(valid)})
pd.DataFrame(pcr).to_csv(f"{OUTPUT_DIR}/position_correlations.csv", index=False)

# ── Fig 6: Answer time position effects (all 4 tasks) ────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Mean Answer Time by Task Position (1st\u20134th) \u2014 H+AI\n"
             "Spearman rank correlation  |  95% Bootstrap CI",
             fontsize=13, fontweight='bold', y=1.02)
for col_idx, task in enumerate(ALL_TASKS):
    ax = axes[col_idx]
    pc = f"position_{task}"
    at_col_name = f"answer_times_{task}"
    # compute per-participant mean answer time
    sub_task = sub_ai.copy()
    sub_task['_mat'] = sub_task[at_col_name].apply(
        lambda v: np.nanmean(np.array(parse_col(v), dtype=float))
        if isinstance(parse_col(v), (list, np.ndarray)) else np.nan)
    means, clo, chi = [], [], []
    for pos in [1,2,3,4]:
        g = sub_task[sub_task[pc]==pos]['_mat'].dropna()
        if len(g) >= 2:
            m = g.mean(); ci_v = bootstrap_ci(g.values)
            means.append(m); clo.append(m-ci_v[0]); chi.append(ci_v[1]-m)
        else:
            means.append(np.nan); clo.append(0); chi.append(0)
    ax.bar([1,2,3,4], means, color=['#2E86AB','#5BA08E','#E8963F','#E8533F'],
           alpha=0.8, width=0.6, edgecolor='white', linewidth=0.5)
    ax.errorbar([1,2,3,4], means, yerr=[clo, chi],
                fmt='none', color='#333', capsize=3, linewidth=1)
    valid = sub_task[[pc, '_mat']].dropna()
    if len(valid) > 5:
        rho, pv = stats.spearmanr(valid[pc], valid['_mat'])
        ax.set_xlabel(f"Position\nSpearman \u03c1={rho:+.2f}, p={pv:.3f} {p_stars(pv)}", fontsize=8)
    ax.set_xticks([1,2,3,4]); ax.set_xticklabels(['1st','2nd','3rd','4th'], fontsize=8)
    ax.set_title(f"Task: {DISP[task]}", fontweight='bold')
    ax.set_ylabel('Mean Answer Time (s)')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig6_answer_time_position.png", dpi=200, bbox_inches='tight'); plt.close()
print("  -> fig6_answer_time_position.png")

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70 + "\nANALYSIS COMPLETE\n" + "="*70)
print("""
CSV outputs:
  order_effects_all_starts.csv       Pairwise comparisons (4 starting groups)
  order_effects_first_vs_later.csv   Binary: target first vs later
  answer_time_stats.csv              Answer time first-vs-later stats (all 4 tasks)
  position_correlations.csv          Position-DV Spearman correlations

Figure outputs:
  fig1_forest_plot.png               Forest plot of Cohen's d (all DVs)
  fig2_bars_easy_mild.png            Bar charts: Low/Low target
  fig2_bars_hard_strong.png          Bar charts: High/Strong target
  fig3_answer_time_trajectories.png  Trial-by-trial answer times (all 4 tasks, H+AI)
  fig3b_answer_time_violins.png      Answer time distributions (violins, all 4 tasks)
  fig4_heatmap_H.png                 Effect size heatmap (H, with target reference column)
  fig4_heatmap_H_AI.png              Effect size heatmap (H+AI, with target reference column)
  fig5_position_effects.png          Position effects on DVs
  fig6_answer_time_position.png      Answer time by position (all 4 tasks)
""")

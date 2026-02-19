"""
Task Order Effects Analysis v5
================================
Nomenclature: Difficulty Low/High, Time Pressure Low/Strong
Input mapping: easy->Low, hard->High, mild->Low, strong->Strong
Targets: Low/Low and High/Strong only

Changes from v4:
  - Answer time trajectories reordered by question ID (quest_order_*)
    so all participants' trial N = same question
  - Matched y-axis scales across same variable
  - Distribution plots for all DVs + answer times by first task
  - Test names on all plots
  - No recency analysis
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

ALL_TASKS = ['easy_mild','easy_strong','hard_mild','hard_strong']
TARGET_TASKS = ['easy_mild','hard_strong']
DISP = {'easy_mild':'Low / Low','easy_strong':'Low / Strong',
        'hard_mild':'High / Low','hard_strong':'High / Strong'}
DISP_LONG = {'easy_mild':'Low difficulty\nLow pressure','easy_strong':'Low difficulty\nStrong pressure',
             'hard_mild':'High difficulty\nLow pressure','hard_strong':'High difficulty\nStrong pressure'}
DISP_SHORT = {'easy_mild':'Low/Low','easy_strong':'Low/Strong',
              'hard_mild':'High/Low','hard_strong':'High/Strong'}
PAL = {'easy_mild':'#2E86AB','easy_strong':'#7FB069','hard_mild':'#E8963F','hard_strong':'#E8533F'}

DVS = {'score':'Accuracy','reliance':'Reliance','overreliance':'Over-reliance',
       'underreliance':'Under-reliance','trust':'Trust','cogload':'Cognitive Load'}

def parse_col(val):
    if isinstance(val, str):
        try: return ast.literal_eval(val)
        except: return val
    return val

def cohens_d(g1, g2):
    n1,n2=len(g1),len(g2)
    if n1<2 or n2<2: return np.nan
    p=np.sqrt(((n1-1)*np.var(g1,ddof=1)+(n2-1)*np.var(g2,ddof=1))/(n1+n2-2))
    return (np.mean(g1)-np.mean(g2))/p if p>0 else 0

def bootstrap_ci(data, n_boot=10000, ci=0.95):
    rng=np.random.default_rng(42); data=np.array(data,dtype=float); data=data[~np.isnan(data)]
    if len(data)<3: return (np.nan,np.nan)
    boot=[np.mean(rng.choice(data,size=len(data),replace=True)) for _ in range(n_boot)]
    a=(1-ci)/2; return (np.percentile(boot,a*100),np.percentile(boot,(1-a)*100))

def permutation_test(g1, g2, n_perm=10000):
    g1=np.array(g1,dtype=float);g1=g1[~np.isnan(g1)]
    g2=np.array(g2,dtype=float);g2=g2[~np.isnan(g2)]
    if len(g1)<2 or len(g2)<2: return np.nan
    obs=np.mean(g1)-np.mean(g2);comb=np.concatenate([g1,g2]);rng=np.random.default_rng(42);ct=0
    for _ in range(n_perm):
        rng.shuffle(comb)
        if abs(np.mean(comb[:len(g1)])-np.mean(comb[len(g1):])) >= abs(obs): ct+=1
    return ct/n_perm

def p_stars(p):
    if np.isnan(p): return ''
    if p<0.001: return '***'
    if p<0.01: return '**'
    if p<0.05: return '*'
    if p<0.1: return '\u2020'
    return 'ns'

def run_comp(v1, v2):
    v1=np.array(v1,dtype=float);v1=v1[~np.isnan(v1)]
    v2=np.array(v2,dtype=float);v2=v2[~np.isnan(v2)]
    if len(v1)<3 or len(v2)<3: return None
    u,p_mw=stats.mannwhitneyu(v1,v2,alternative='two-sided')
    d=cohens_d(v1,v2);n1,n2=len(v1),len(v2)
    r_rb=1-(2*u)/(n1*n2);p_perm=permutation_test(v1,v2)
    return {'n1':n1,'n2':n2,'mean1':np.mean(v1),'mean2':np.mean(v2),
            'sd1':np.std(v1,ddof=1),'sd2':np.std(v2,ddof=1),
            'U':u,'p_mw':p_mw,'p_perm':p_perm,'d':d,'r_rb':r_rb}

# ── Load ──────────────────────────────────────────────────────────────
print("="*70+"\nTASK ORDER EFFECTS ANALYSIS v5\n"+"="*70)
df = pd.read_csv(DATA_PATH)
print(f"Total participants: {len(df)}")
df = df[df['xai_condition'].isin(['H','H+AI'])].copy()
print(f"After filtering to H and H+AI: {len(df)}")
df['tasks_order'] = df['tasks_order'].apply(parse_col)
df['first_task'] = df['tasks_order'].apply(lambda x: x[0] if isinstance(x,list) else None)

print(f"\nStarting task distribution:")
for t in ALL_TASKS:
    h=((df['first_task']==t)&(df['xai_condition']=='H')).sum()
    hai=((df['first_task']==t)&(df['xai_condition']=='H+AI')).sum()
    print(f"  {DISP[t]:15s}: H={h}, H+AI={hai}")

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 1 & 2: Pairwise + binary comparisons (same as v4)
# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*70+"\nANALYSIS 1: PAIRWISE COMPARISONS\n"+"="*70)
results_rows = []
for xai_cond in ['H','H+AI']:
    sub = df[df['xai_condition']==xai_cond]
    print(f"\n{'='*60}\n  {xai_cond} (n={len(sub)})\n{'='*60}")
    for target in TARGET_TASKS:
        print(f"\n  -- Target: {DISP[target]} --")
        for dv_base,dv_label in DVS.items():
            col=f"{dv_base}_{target}"
            if col not in sub.columns: continue
            print(f"\n    {dv_label}:")
            groups_kw=[];p_kw=np.nan
            for s in ALL_TASKS:
                g=sub[sub['first_task']==s][col].dropna()
                if len(g)>=2: groups_kw.append(g.values)
            if len(groups_kw)>=2:
                try:
                    H_stat,p_kw=stats.kruskal(*groups_kw)
                    N=sum(len(g) for g in groups_kw);k=len(groups_kw)
                    eta2=(H_stat-k+1)/(N-k) if (N-k)>0 else np.nan
                    print(f"      Kruskal-Wallis: H={H_stat:.3f}, p={p_kw:.4f} {p_stars(p_kw)}, eta2={eta2:.3f}")
                except: pass
            tf=sub[sub['first_task']==target][col].dropna()
            for other in ALL_TASKS:
                if other==target: continue
                of=sub[sub['first_task']==other][col].dropna()
                res=run_comp(tf.values,of.values)
                if res is None: continue
                print(f"      vs {DISP_SHORT[other]:12s}: M={res['mean1']:.3f} vs {res['mean2']:.3f}, "
                      f"d={res['d']:+.3f}, p_MW={res['p_mw']:.4f}{p_stars(res['p_mw'])}, "
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
results_df.to_csv(f"{OUTPUT_DIR}/order_effects_all_starts.csv",index=False)

print("\n"+"="*70+"\nANALYSIS 2: FIRST vs LATER\n"+"="*70)
binary_rows = []
for xai_cond in ['H','H+AI']:
    sub=df[df['xai_condition']==xai_cond]
    print(f"\n  {xai_cond} (n={len(sub)})")
    for target in TARGET_TASKS:
        print(f"  Target: {DISP[target]}")
        gf=sub[sub['first_task']==target];gl=sub[sub['first_task']!=target]
        for dv_base,dv_label in DVS.items():
            col=f"{dv_base}_{target}"
            if col not in sub.columns: continue
            res=run_comp(gf[col].dropna().values,gl[col].dropna().values)
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
pd.DataFrame(binary_rows).to_csv(f"{OUTPUT_DIR}/order_effects_first_vs_later.csv",index=False)

# ═══════════════════════════════════════════════════════════════════════
# REORDER ANSWER TIMES BY QUESTION ID
# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*70+"\nANALYSIS 3: ANSWER TIMES (reordered by question ID)\n"+"="*70)

def reorder_by_quest(times_list, quest_order_list):
    """Reorder answer times so index i = answer time for question i."""
    times = parse_col(times_list)
    order = parse_col(quest_order_list)
    if not isinstance(times,(list,np.ndarray)) or not isinstance(order,(list,np.ndarray)):
        return None
    times = list(times); order = [int(x) for x in order]
    if len(times) != len(order): return None
    n = len(times)
    reordered = [np.nan]*n
    for presentation_pos, question_id in enumerate(order):
        if 0 <= question_id < n:
            reordered[question_id] = float(times[presentation_pos])
    return reordered

# Build reordered answer time arrays
for task in TARGET_TASKS:
    at_col = f"answer_times_{task}"
    qo_col = f"quest_order_{task}"
    new_col = f"at_reordered_{task}"
    if at_col in df.columns and qo_col in df.columns:
        df[new_col] = df.apply(lambda r: reorder_by_quest(r[at_col], r[qo_col]), axis=1)
        valid = df[new_col].notna().sum()
        print(f"  {DISP[task]}: {valid} participants with reordered times")

# Also compute per-participant mean answer time
for task in TARGET_TASKS:
    at_col = f"answer_times_{task}"
    if at_col in df.columns:
        df[f"mean_at_{task}"] = df[at_col].apply(
            lambda v: np.nanmean(np.array(parse_col(v),dtype=float))
            if isinstance(parse_col(v),(list,np.ndarray)) else np.nan)



# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════
print("\n-> Generating visualizations...")

# ── Precompute matched y-axis limits ──────────────────────────────────
key_dvs = list(DVS.keys())
ylims = {}
for dv_base in key_dvs:
    all_vals = []
    for target in TARGET_TASKS:
        col = f"{dv_base}_{target}"
        if col in df.columns: all_vals.extend(df[col].dropna().values)
    if all_vals:
        mn, mx = min(all_vals), max(all_vals)
        margin = (mx - mn) * 0.15 if mx > mn else 0.1
        ylims[dv_base] = (max(0, mn - margin), mx + margin)

sub_ai = df[df['xai_condition']=='H+AI']

# ── Fig 1: Forest plot (targets only: Low/Low and High/Strong) ────────
# Filter results_df to only target tasks
fig_sr = results_df[results_df['target_task'].isin(TARGET_TASKS)].copy()
fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=True)
fig.suptitle("Effect of Starting Task on Target Task Outcomes\n"
             "Positive d = higher when target done first",
             fontsize=13, fontweight='bold', y=0.98)
for idx, xai_cond in enumerate(['H','H+AI']):
    ax = axes[idx]
    sr = fig_sr[fig_sr['xai_condition']==xai_cond].copy()
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
    ax.set_xlabel("Cohen's d")
    ax.set_title(xai_cond, fontweight='bold', fontsize=13)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    prev = None
    for i, (_, r) in enumerate(sr.iterrows()):
        if prev and r['target_task'] != prev:
            ax.axhline(i-0.5, color='black', linewidth=1)
        prev = r['target_task']
    for tgt in TARGET_TASKS:
        rt = sr[sr['target_task']==tgt]
        if len(rt) > 0:
            mid = (rt.index[0]+rt.index[-1])/2
            ax.text(-2.2, mid, f"Target:\n{DISP[tgt]}", fontsize=9, fontweight='bold',
                    va='center', ha='center', rotation=90)
axes[1].legend(handles=[Patch(facecolor=c, label=DISP[t]) for t, c in PAL.items()],
               title='Started with...', loc='lower right', fontsize=8, title_fontsize=9)
plt.tight_layout(rect=[0.05, 0, 1, 0.94])
fig.savefig(f"{OUTPUT_DIR}/fig1_forest_plot.png", dpi=200, bbox_inches='tight'); plt.close()
print("  -> fig1_forest_plot.png")

# ── Fig 2: Bar charts with matched scales ─────────────────────────────
for target in TARGET_TASKS:
    fig, axes = plt.subplots(2, len(key_dvs), figsize=(3.5*len(key_dvs), 9))
    fig.suptitle(f"Target: {DISP[target]} (Difficulty / Time Pressure)\n"
                 f"Mean values grouped by starting task  |  95% Bootstrap CI",
                 fontsize=12, fontweight='bold', y=0.99)
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
                else: means.append(0); clo.append(0); chi.append(0)
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
            if col_idx==0: ax.set_ylabel(xai_cond, fontsize=11, fontweight='bold')
            if dv_base in ylims: ax.set_ylim(ylims[dv_base])
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUTPUT_DIR}/fig2_bars_{target}.png", dpi=200, bbox_inches='tight'); plt.close()
    print(f"  -> fig2_bars_{target}.png")

# ── Fig 3: Answer time trajectories (reordered by question ID) ────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Answer Time by Question ID \u2014 H+AI\n"
             "Times reordered so each position = same question across participants (mean \u00b1 SEM)",
             fontsize=13, fontweight='bold', y=0.99)
# Shared y-limits
all_at_vals = []
for target in TARGET_TASKS:
    rc = f"at_reordered_{target}"
    if rc in sub_ai.columns:
        for v in sub_ai[rc].dropna():
            if isinstance(v, list): all_at_vals.extend([x for x in v if not np.isnan(x)])
at_ylim = (0, np.percentile(all_at_vals, 98)*1.1) if all_at_vals else (0, 25)

for col_idx, target in enumerate(TARGET_TASKS):
    rc = f"at_reordered_{target}"
    if rc not in sub_ai.columns: continue
    # Row 0: by starting task
    ax = axes[0, col_idx]
    for st in ALL_TASKS:
        grp = sub_ai[sub_ai['first_task']==st]
        valid_times = [r for r in grp[rc] if isinstance(r, list)]
        if not valid_times: continue
        n_q = max(len(t) for t in valid_times)
        pad = np.full((len(valid_times), n_q), np.nan)
        for i, t in enumerate(valid_times): pad[i,:len(t)] = t
        m = np.nanmean(pad, axis=0)
        se = np.nanstd(pad, axis=0)/np.sqrt(np.sum(~np.isnan(pad), axis=0))
        qids = np.arange(n_q)
        ax.plot(qids, m, color=PAL[st], linewidth=2,
                label=f"Started {DISP_SHORT[st]}", alpha=0.9)
        ax.fill_between(qids, m-se, m+se, color=PAL[st], alpha=0.12)
    ax.set_title(f"Target: {DISP[target]} \u2014 By Starting Task", fontweight='bold')
    ax.set_xlabel('Question ID'); ax.set_ylabel('Answer Time (s)')
    ax.set_ylim(at_ylim); ax.legend(fontsize=7, ncol=2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    # Row 1: first vs later
    ax = axes[1, col_idx]
    for is_f, lab, col_c, ls in [
        (True, f'{DISP_SHORT[target]} first', PAL[target], '-'),
        (False, f'{DISP_SHORT[target]} later', '#888', '--')]:
        grp = sub_ai[sub_ai['first_task']==target] if is_f else sub_ai[sub_ai['first_task']!=target]
        valid_times = [r for r in grp[rc] if isinstance(r, list)]
        if not valid_times: continue
        n_q = max(len(t) for t in valid_times)
        pad = np.full((len(valid_times), n_q), np.nan)
        for i, t in enumerate(valid_times): pad[i,:len(t)] = t
        m = np.nanmean(pad, axis=0)
        se = np.nanstd(pad, axis=0)/np.sqrt(np.sum(~np.isnan(pad), axis=0))
        qids = np.arange(n_q)
        ax.plot(qids, m, color=col_c, linewidth=2, linestyle=ls, label=lab, alpha=0.9)
        ax.fill_between(qids, m-se, m+se, color=col_c, alpha=0.12)
    ax.set_title(f"Target: {DISP[target]} \u2014 First vs Later", fontweight='bold')
    ax.set_xlabel('Question ID'); ax.set_ylabel('Answer Time (s)')
    ax.set_ylim(at_ylim); ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUTPUT_DIR}/fig3_answer_time_trajectories.png", dpi=200, bbox_inches='tight'); plt.close()
print("  -> fig3_answer_time_trajectories.png")

# ── Fig 4: Heatmaps with target reference column ─────────────────────
for xai_cond in ['H','H+AI']:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Effect Size Heatmap (Cohen's d): {xai_cond}\n"
                 f"d > 0 = higher when target done first vs comparison group",
                 fontsize=12, fontweight='bold')
    sub_cond = df[df['xai_condition']==xai_cond]
    for idx, target in enumerate(TARGET_TASKS):
        ax = axes[idx]
        sr = results_df[(results_df['xai_condition']==xai_cond) &
                        (results_df['target_task']==target)].copy()
        dv_list = list(DVS.values())
        all_starts = [target]+[t for t in ALL_TASKS if t!=target]
        col_labels = [DISP_SHORT[s]+('\n(target)' if s==target else '') for s in all_starts]
        d_matrix = np.full((len(dv_list), len(all_starts)), np.nan)
        annot_matrix = [['']*len(all_starts) for _ in range(len(dv_list))]
        for j, start in enumerate(all_starts):
            if start==target:
                for i, (dv_base, dv_label) in enumerate(DVS.items()):
                    col = f"{dv_base}_{target}"
                    if col in sub_cond.columns:
                        vals = sub_cond[sub_cond['first_task']==target][col].dropna()
                        if len(vals) >= 2:
                            d_matrix[i,j] = 0.0
                            annot_matrix[i][j] = f"M={vals.mean():.2f}\nn={len(vals)}\n(ref)"
                        else: annot_matrix[i][j] = "n/a"
            else:
                for i, (dv_base, dv_label) in enumerate(DVS.items()):
                    match = sr[(sr['dv']==dv_label) & (sr['compared_start']==start)]
                    if len(match) > 0:
                        r = match.iloc[0]; d_matrix[i,j] = r['d']
                        annot_matrix[i][j] = f"d={r['d']:.2f}\n{p_stars(r['p_perm'])}"
        d_df = pd.DataFrame(d_matrix, index=dv_list, columns=col_labels)
        sns.heatmap(d_df, annot=np.array(annot_matrix), fmt='', cmap='RdBu_r', center=0,
                    vmin=-1.5, vmax=1.5, ax=ax, linewidths=0.5,
                    cbar_kws={'label': "Cohen's d"})
        ax.set_title(f"Target: {DISP[target]}", fontweight='bold')
        ax.set_ylabel(''); ax.set_xlabel('Started with...')
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fname = f"fig4_heatmap_{xai_cond.replace('+','_')}.png"
    fig.savefig(f"{OUTPUT_DIR}/{fname}", dpi=200, bbox_inches='tight'); plt.close()
    print(f"  -> {fname}")

# ── Fig 5: Distribution plots — violin + histogram density ────────────
for target in TARGET_TASKS:
    all_dv_keys = list(DVS.keys()) + ['mean_at']
    all_dv_labels = list(DVS.values()) + ['Mean Answer Time (s)']
    n_dvs = len(all_dv_keys)
    fig, axes = plt.subplots(2, n_dvs, figsize=(3.5*n_dvs, 10))
    fig.suptitle(f"Distributions by Starting Task \u2014 Target: {DISP[target]} \u2014 H+AI\n"
                 f"Top: violin plots with data points  |  Bottom: density histograms",
                 fontsize=12, fontweight='bold', y=1.01)

    for ci, (dv_key, dv_label) in enumerate(zip(all_dv_keys, all_dv_labels)):
        ax_violin = axes[0, ci]
        ax_hist = axes[1, ci]
        col = f"{dv_key}_{target}"
        if col not in sub_ai.columns:
            ax_violin.set_visible(False); ax_hist.set_visible(False); continue

        groups_data = []
        groups_keys = []
        for s in ALL_TASKS:
            vals = sub_ai[sub_ai['first_task']==s][col].dropna().values
            if len(vals) >= 2:
                groups_data.append(vals)
                groups_keys.append(s)

        if len(groups_data) < 2:
            ax_violin.set_visible(False); ax_hist.set_visible(False); continue

        # ── Top row: violin + jittered points ──
        positions = list(range(len(groups_data)))
        vp = ax_violin.violinplot(groups_data, positions=positions, widths=0.7,
                                  showmeans=True, showmedians=True, showextrema=False)
        for i, pc in enumerate(vp['bodies']):
            pc.set_facecolor(PAL[groups_keys[i]]); pc.set_alpha(0.4)
        for partname in ['cmeans','cmedians']:
            if partname in vp: vp[partname].set_edgecolor('#333')
        rng = np.random.default_rng(42)
        for i, vals in enumerate(groups_data):
            jx = rng.normal(0, 0.06, len(vals))
            ax_violin.scatter(np.full(len(vals), i) + jx, vals,
                              color=PAL[groups_keys[i]], alpha=0.5, s=12, edgecolors='none')

        # KW test annotation
        if len(groups_data) >= 2:
            H_stat, p_kw = stats.kruskal(*groups_data)
            ax_violin.text(0.02, 0.98,
                           f"Kruskal-Wallis H={H_stat:.2f}\np={p_kw:.3f} {p_stars(p_kw)}",
                           transform=ax_violin.transAxes, fontsize=7, va='top', ha='left',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                     alpha=0.8, edgecolor='#CCC'))

        ax_violin.set_xticks(positions)
        ax_violin.set_xticklabels([DISP_SHORT[s] for s in groups_keys], fontsize=7,
                                  rotation=45, ha='right')
        ax_violin.set_title(dv_label, fontsize=9, fontweight='bold')
        ax_violin.spines['top'].set_visible(False); ax_violin.spines['right'].set_visible(False)

        # ── Bottom row: overlapping density histograms ──
        n_bins = 15
        # Determine shared bin range
        all_v = np.concatenate(groups_data)
        bin_range = (np.nanmin(all_v), np.nanmax(all_v))
        bins = np.linspace(bin_range[0], bin_range[1], n_bins + 1)

        for i, vals in enumerate(groups_data):
            ax_hist.hist(vals, bins=bins, density=True, alpha=0.35,
                         color=PAL[groups_keys[i]], edgecolor='white', linewidth=0.3,
                         label=DISP_SHORT[groups_keys[i]])
            # KDE line if enough data
            if len(vals) >= 5:
                try:
                    kde = stats.gaussian_kde(vals)
                    x_kde = np.linspace(bin_range[0], bin_range[1], 200)
                    ax_hist.plot(x_kde, kde(x_kde), color=PAL[groups_keys[i]],
                                 linewidth=1.5, alpha=0.8)
                except: pass

        ax_hist.set_xlabel(dv_label, fontsize=8)
        ax_hist.set_ylabel('Density', fontsize=8)
        ax_hist.spines['top'].set_visible(False); ax_hist.spines['right'].set_visible(False)
        if ci == 0:
            ax_hist.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig5_distributions_{target}.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> fig5_distributions_{target}.png")

# ── Fig 6: Answer time trial-level distributions (violin + histogram) ─
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Answer Time Distributions (all individual trials) \u2014 H+AI\n"
             "Top: violin plots  |  Bottom: density histograms by starting task",
             fontsize=13, fontweight='bold', y=1.01)
# Compute shared limits
at_shared_max = 0
for target in TARGET_TASKS:
    at_col = f"answer_times_{target}"
    if at_col not in sub_ai.columns: continue
    for _, r in sub_ai.iterrows():
        t = parse_col(r[at_col])
        if isinstance(t, (list, np.ndarray)):
            at_shared_max = max(at_shared_max, np.nanpercentile(np.array(t, dtype=float), 99))
at_shared_ylim = (0, at_shared_max * 1.1) if at_shared_max > 0 else (0, 25)

for col_idx, target in enumerate(TARGET_TASKS):
    at_col = f"answer_times_{target}"
    if at_col not in sub_ai.columns: continue
    # Collect trial times per starting group
    all_trial_times = {}
    for s in ALL_TASKS:
        grp = sub_ai[sub_ai['first_task']==s]; times = []
        for _, r in grp.iterrows():
            t = parse_col(r[at_col])
            if isinstance(t, (list, np.ndarray)):
                times.extend([float(x) for x in t if not np.isnan(float(x))])
        all_trial_times[s] = np.array(times)

    valid_keys = [s for s in ALL_TASKS if len(all_trial_times[s]) > 0]
    groups = [all_trial_times[s] for s in valid_keys]
    positions = list(range(len(groups)))

    # Top: violin
    ax = axes[0, col_idx]
    vp = ax.violinplot(groups, positions=positions, widths=0.7,
                       showmeans=True, showmedians=True, showextrema=False)
    for i, pc in enumerate(vp['bodies']):
        pc.set_facecolor(PAL[valid_keys[i]]); pc.set_alpha(0.4)
    for partname in ['cmeans','cmedians']:
        if partname in vp: vp[partname].set_edgecolor('#333')
    ax.set_xticks(positions)
    ax.set_xticklabels([DISP_SHORT[s] for s in valid_keys], fontsize=9)
    ax.set_xlabel('Started with...'); ax.set_ylabel('Answer Time (s)')
    ax.set_title(f"Target: {DISP[target]} \u2014 Violins", fontweight='bold')
    ax.set_ylim(at_shared_ylim)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Bottom: overlapping density histograms
    ax = axes[1, col_idx]
    all_v = np.concatenate(groups)
    bins = np.linspace(0, at_shared_ylim[1], 30)
    for i, s in enumerate(valid_keys):
        vals = all_trial_times[s]
        ax.hist(vals, bins=bins, density=True, alpha=0.3,
                color=PAL[s], edgecolor='white', linewidth=0.3,
                label=DISP_SHORT[s])
        if len(vals) >= 10:
            try:
                kde = stats.gaussian_kde(vals)
                x_kde = np.linspace(0, at_shared_ylim[1], 300)
                ax.plot(x_kde, kde(x_kde), color=PAL[s], linewidth=1.5, alpha=0.8)
            except: pass
    ax.set_xlabel('Answer Time (s)'); ax.set_ylabel('Density')
    ax.set_title(f"Target: {DISP[target]} \u2014 Density", fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig6_answer_time_distributions.png", dpi=200, bbox_inches='tight')
plt.close()
print("  -> fig6_answer_time_distributions.png")

# ── Fig 7: Position effects ──────────────────────────────────────────
print("\n-> Position analysis...")
for task in TARGET_TASKS:
    df[f"position_{task}"] = df['tasks_order'].apply(
        lambda x, t=task: (x.index(t)+1) if isinstance(x,list) and t in x else np.nan)

fig, axes = plt.subplots(2, len(key_dvs), figsize=(3.5*len(key_dvs), 9))
fig.suptitle("Effect of Task Position (1st\u20134th) on Outcomes \u2014 H+AI\n"
             "95% Bootstrap CI",
             fontsize=13, fontweight='bold', y=0.99)
sub_ai = df[df['xai_condition']=='H+AI']
for ri, target in enumerate(TARGET_TASKS):
    for ci, dv_base in enumerate(key_dvs):
        ax = axes[ri, ci]; col = f"{dv_base}_{target}"; pc = f"position_{target}"
        if col not in sub_ai.columns: ax.set_visible(False); continue
        means, clo, chi = [], [], []
        for pos in [1,2,3,4]:
            g = sub_ai[sub_ai[pc]==pos][col].dropna()
            if len(g) >= 2:
                m = g.mean(); ci_v = bootstrap_ci(g.values)
                means.append(m); clo.append(m-ci_v[0]); chi.append(ci_v[1]-m)
            else: means.append(np.nan); clo.append(0); chi.append(0)
        ax.bar([1,2,3,4], means, color=['#2E86AB','#5BA08E','#E8963F','#E8533F'],
               alpha=0.8, width=0.6, edgecolor='white', linewidth=0.5)
        ax.errorbar([1,2,3,4], means, yerr=[clo, chi],
                    fmt='none', color='#333', capsize=3, linewidth=1)
        # Spearman annotated on plot since it IS the test shown
        valid = sub_ai[[pc, col]].dropna()
        if len(valid) > 5:
            rho, pv = stats.spearmanr(valid[pc], valid[col])
            ax.text(0.02, 0.98,
                    f"Spearman \u03c1={rho:+.2f}\np={pv:.3f} {p_stars(pv)}",
                    transform=ax.transAxes, fontsize=7, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.8, edgecolor='#CCC'))
        ax.set_xticks([1,2,3,4])
        ax.set_xticklabels(['1st','2nd','3rd','4th'], fontsize=8)
        ax.set_xlabel('Position', fontsize=8)
        ax.set_title(DVS[dv_base], fontsize=10, fontweight='bold')
        if dv_base in ylims: ax.set_ylim(ylims[dv_base])
        if ci==0: ax.set_ylabel(f"Target: {DISP[target]}", fontsize=9, fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUTPUT_DIR}/fig7_position_effects.png", dpi=200, bbox_inches='tight'); plt.close()
print("  -> fig7_position_effects.png")

pcr = []
print("\n  Position-DV correlations (Spearman, H+AI):")
for target in TARGET_TASKS:
    pc = f"position_{target}"
    for dv_base, dv_label in DVS.items():
        col = f"{dv_base}_{target}"
        if col not in sub_ai.columns: continue
        valid = sub_ai[[pc, col]].dropna()
        if len(valid) > 5:
            rho, pv = stats.spearmanr(valid[pc], valid[col])
            print(f"    {DISP[target]:15s} | {dv_label:25s}: "
                  f"\u03c1={rho:+.3f}, p={pv:.4f} {p_stars(pv)}")
            pcr.append({'target':target,'target_display':DISP[target],
                        'dv':dv_label,'rho':rho,'p':pv,'n':len(valid)})
pd.DataFrame(pcr).to_csv(f"{OUTPUT_DIR}/position_correlations.csv", index=False)

print("\n"+"="*70+"\nCOMPLETE\n"+"="*70)
print("""
CSV: order_effects_all_starts.csv, order_effects_first_vs_later.csv, position_correlations.csv
Figures:
  fig1_forest_plot.png                Forest plot (Cohen's d, targets only)
  fig2_bars_easy_mild.png             Bar charts: Low/Low target
  fig2_bars_hard_strong.png           Bar charts: High/Strong target
  fig3_answer_time_trajectories.png   Answer times by question ID (reordered)
  fig4_heatmap_H.png                  Effect size heatmap H
  fig4_heatmap_H_AI.png               Effect size heatmap H+AI
  fig5_distributions_easy_mild.png    DV distributions: Low/Low (violin + histogram)
  fig5_distributions_hard_strong.png  DV distributions: High/Strong (violin + histogram)
  fig6_answer_time_distributions.png  Answer time distributions (violin + histogram)
  fig7_position_effects.png           Position effects
""")
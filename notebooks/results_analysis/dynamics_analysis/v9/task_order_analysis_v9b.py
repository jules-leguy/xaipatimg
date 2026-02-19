"""
Task Order Effects Analysis v9b — Pressure Exposure on Low-Pressure Tasks Only
================================================================================
Variation of Comparison A: restrict to observations on Low-pressure tasks
(easy_mild, hard_mild) and compare participants who have already been exposed
to Strong time pressure vs those who never were.

pressure_exposed = 1 if any PRIOR task had Strong time pressure
pressure_exposed = 0 if no prior task had Strong time pressure
(Current task is always Low pressure by definition of the filter.)

Between-group: Mann-Whitney + permutation.
LMM: DV ~ exposure + task + (1|participant).
Corrections: Holm-Bonferroni + Benjamini-Hochberg.
"""
import pandas as pd, numpy as np, matplotlib, ast, sys, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
OUTPUT_DIR = "."
np.random.seed(42)

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':10,
    'axes.titlesize':12,'axes.titleweight':'bold','axes.labelsize':10,
    'figure.facecolor':'#FAFAFA','axes.facecolor':'#FFFFFF','axes.edgecolor':'#CCCCCC',
    'axes.grid':True,'grid.alpha':0.25,'grid.color':'#DDDDDD'})

ALL_TASKS = ['easy_mild','easy_strong','hard_mild','hard_strong']
STRONG_TASKS = {'easy_strong','hard_strong'}
LOW_TASKS = {'easy_mild','hard_mild'}
DISP = {'easy_mild':'Low/Low','easy_strong':'Low/Strong',
        'hard_mild':'High/Low','hard_strong':'High/Strong'}
DVS = {'score':'Accuracy','reliance':'Reliance','overreliance':'Over-reliance',
       'underreliance':'Under-reliance','trust':'Trust','cogload':'Cognitive Load'}
CONDS = ['H','H+AI']

# ── Helpers ───────────────────────────────────────────────────────────
def parse_col(val):
    if isinstance(val, str):
        try: return ast.literal_eval(val)
        except: return val
    return val

def cohens_d(g1, g2):
    n1,n2=len(g1),len(g2)
    if n1<2 or n2<2: return np.nan
    p=np.sqrt(((n1-1)*np.var(g1,ddof=1)+(n2-1)*np.var(g2,ddof=1))/(n1+n2-2))
    return (np.mean(g1)-np.mean(g2))/p if p>0 else 0.0

def bootstrap_ci(data, n_boot=10000, ci=0.95):
    rng=np.random.default_rng(42); data=np.array(data,dtype=float); data=data[~np.isnan(data)]
    if len(data)<3: return (np.nan,np.nan)
    boot=[np.mean(rng.choice(data,size=len(data),replace=True)) for _ in range(n_boot)]
    a=(1-ci)/2; return (np.percentile(boot,a*100),np.percentile(boot,(1-a)*100))

def permutation_test(g1, g2, n_perm=10000):
    g1=np.array(g1,dtype=float);g1=g1[~np.isnan(g1)]
    g2=np.array(g2,dtype=float);g2=g2[~np.isnan(g2)]
    if len(g1)<2 or len(g2)<2: return np.nan
    obs=abs(np.mean(g1)-np.mean(g2));comb=np.concatenate([g1,g2]);rng=np.random.default_rng(42);ct=0
    for _ in range(n_perm):
        rng.shuffle(comb)
        if abs(np.mean(comb[:len(g1)])-np.mean(comb[len(g1):])) >= obs: ct+=1
    return ct/n_perm

def p_stars(p):
    if p is None or np.isnan(p): return ''
    if p<0.001: return '***'
    if p<0.01: return '**'
    if p<0.05: return '*'
    if p<0.1: return '\u2020'
    return 'ns'

def holm_bonferroni(pvals):
    n=len(pvals); sorted_pv=sorted(pvals,key=lambda x:x[1]); corrected={}; mx=0
    for rank,(idx,p) in enumerate(sorted_pv):
        adj=min(p*(n-rank),1.0); adj=max(adj,mx); mx=adj; corrected[idx]=adj
    return corrected

def benjamini_hochberg(pvals):
    n=len(pvals); sorted_pv=sorted(pvals,key=lambda x:x[1],reverse=True); corrected={}; mn=1.0
    for rank_desc,(idx,p) in enumerate(sorted_pv):
        rank_asc=n-rank_desc; adj=min(p*n/rank_asc,1.0); adj=min(adj,mn); mn=adj; corrected[idx]=adj
    return corrected

def correct_family(rows, p_col):
    pvals=[(i,r[p_col]) for i,r in enumerate(rows) if not np.isnan(r[p_col])]
    if len(pvals)<2:
        for r in rows: r['p_holm']=r.get(p_col,np.nan); r['p_bh']=r.get(p_col,np.nan)
        return rows
    hm=holm_bonferroni(pvals); bh=benjamini_hochberg(pvals)
    for i,r in enumerate(rows):
        r['p_holm']=hm.get(i,r.get(p_col,np.nan)); r['p_bh']=bh.get(i,r.get(p_col,np.nan))
    return rows

def reorder_by_quest(times_list, quest_order_list):
    times=parse_col(times_list); order=parse_col(quest_order_list)
    if not isinstance(times,(list,np.ndarray)) or not isinstance(order,(list,np.ndarray)): return None
    times=list(times); order=[int(x) for x in order]
    if len(times)!=len(order): return None
    n=len(times); reordered=[np.nan]*n
    for pos,qid in enumerate(order):
        if 0<=qid<n: reordered[qid]=float(times[pos])
    return reordered

# ── Load & prepare ────────────────────────────────────────────────────
print("="*70+"\nTASK ORDER EFFECTS v9b — PRESSURE EXPOSURE ON LOW-PRESSURE TASKS\n"+"="*70)
df = pd.read_csv(DATA_PATH)
print(f"Total: {len(df)}")
df = df[df['xai_condition'].isin(CONDS)].copy()
print(f"H + H+AI: {len(df)}")
df['tasks_order'] = df['tasks_order'].apply(parse_col)

# Mean answer times per task
for t in ALL_TASKS:
    at_col = f"answer_times_{t}"
    if at_col in df.columns:
        df[f"mean_at_{t}"] = df[at_col].apply(
            lambda v: np.nanmean(np.array(parse_col(v),dtype=float))
            if isinstance(parse_col(v),(list,np.ndarray)) else np.nan)
    qo_col = f"quest_order_{t}"
    if at_col in df.columns and qo_col in df.columns:
        df[f"at_reordered_{t}"] = df.apply(
            lambda r, tk=t: reorder_by_quest(r[f"answer_times_{tk}"], r[f"quest_order_{tk}"]), axis=1)

# ═══════════════════════════════════════════════════════════════════════
# BUILD: Low-pressure tasks only, with prior-exposure flag
# ═══════════════════════════════════════════════════════════════════════
print("\n-> Building task-level data: Low-pressure tasks only...")
all_dv_keys = list(DVS.keys()) + ['mean_at']
all_dv_labels = list(DVS.values()) + ['Mean Answer Time']

long_rows = []
for _, row in df.iterrows():
    pid = row.name
    cond = row['xai_condition']
    order = row['tasks_order']
    if not isinstance(order, list) or len(order)!=4: continue
    seen_strong = False
    for pos, task in enumerate(order):
        # Only keep Low-pressure tasks
        if task in LOW_TASKS:
            exposed = 1 if seen_strong else 0
            entry = {'pid':pid, 'cond':cond, 'task':task, 'task_disp':DISP[task],
                     'position':pos+1, 'prior_pressure':exposed}
            for dv_base in DVS:
                col = f"{dv_base}_{task}"
                entry[dv_base] = row[col] if col in df.columns and pd.notna(row.get(col)) else np.nan
            mat_col = f"mean_at_{task}"
            entry['mean_at'] = row[mat_col] if mat_col in df.columns and pd.notna(row.get(mat_col)) else np.nan
            long_rows.append(entry)
        # Update seen_strong after processing
        if task in STRONG_TASKS:
            seen_strong = True

df_low = pd.DataFrame(long_rows)
df_low['exposure_label'] = df_low['prior_pressure'].map({0:'Never exposed','1':'Prior Strong exposure'})
# Fix: map with int keys
df_low['exposure_label'] = df_low['prior_pressure'].map({0:'Never exposed', 1:'Prior Strong'})

print("\nObservation counts (Low-pressure tasks only):")
for cond in CONDS:
    sub = df_low[df_low['cond']==cond]
    print(f"\n  {cond} (total obs = {len(sub)}):")
    ct = sub.groupby('prior_pressure').size()
    for k,v in ct.items(): print(f"    prior_pressure={k}: {v} obs")
    print("    By task:")
    pt = sub.groupby(['task_disp','prior_pressure']).size().unstack(fill_value=0)
    print(pt.to_string())
    print("    By position:")
    pt2 = sub.groupby(['position','prior_pressure']).size().unstack(fill_value=0)
    print(pt2.to_string())

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS: Mann-Whitney + Permutation
# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*70+"\nANALYSIS: MANN-WHITNEY ON LOW-PRESSURE TASKS\n"+"="*70)

mw_rows = []
for cond in CONDS:
    sub = df_low[df_low['cond']==cond]
    print(f"\n  {cond}")
    for dv_key, dv_label in zip(all_dv_keys, all_dv_labels):
        if dv_key not in sub.columns: continue
        g0 = sub[sub['prior_pressure']==0][dv_key].dropna().values
        g1 = sub[sub['prior_pressure']==1][dv_key].dropna().values
        if len(g0)<3 or len(g1)<3: continue
        u, p_mw = stats.mannwhitneyu(g0, g1, alternative='two-sided')
        d = cohens_d(g0, g1)
        p_perm = permutation_test(g0, g1)
        mw_rows.append({
            'cond':cond,'dv':dv_label,'dv_key':dv_key,
            'mean_never':np.mean(g0),'n_never':len(g0),
            'mean_exposed':np.mean(g1),'n_exposed':len(g1),
            'U':u,'p_mw':p_mw,'p_perm':p_perm,'d':d})
        print(f"    {dv_label:22s}: Never={np.mean(g0):.3f}(n={len(g0)}) Exposed={np.mean(g1):.3f}(n={len(g1)}) d={d:+.3f} p_perm={p_perm:.4f} {p_stars(p_perm)}")

# Correct: family = per condition
for cond in CONDS:
    family = [r for r in mw_rows if r['cond']==cond]
    correct_family(family, 'p_perm')
for r in mw_rows:
    r.setdefault('p_holm', r.get('p_perm', np.nan))
    r.setdefault('p_bh', r.get('p_perm', np.nan))

# ═══════════════════════════════════════════════════════════════════════
# LMM: DV ~ prior_pressure + (1|pid) and DV ~ prior_pressure + task + (1|pid)
# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*70+"\nLMM: PRIOR PRESSURE ON LOW-PRESSURE TASKS\n"+"="*70)

lmm_rows = []
for cond in CONDS:
    sub = df_low[df_low['cond']==cond].copy()
    sub['pp'] = sub['prior_pressure'].astype(str)
    print(f"\n  {cond}")
    for dv_key, dv_label in zip(all_dv_keys, all_dv_labels):
        if dv_key not in sub.columns: continue
        sub_dv = sub[['pid','pp','task',dv_key]].dropna().copy()
        sub_dv = sub_dv.rename(columns={dv_key:'y'})
        if len(sub_dv)<10 or sub_dv['pp'].nunique()<2: continue
        for model_name, formula in [
            ('Exposure only', 'y ~ C(pp, Treatment(reference="0"))'),
            ('Exposure + task', 'y ~ C(pp, Treatment(reference="0")) + C(task)')]:
            try:
                md = smf.mixedlm(formula, sub_dv, groups=sub_dv['pid'])
                mdf = md.fit(reml=True)
                for param_name, coef in mdf.fe_params.items():
                    if 'pp' not in param_name: continue
                    pval = mdf.pvalues[param_name]
                    ci = mdf.conf_int().loc[param_name]
                    lmm_rows.append({
                        'cond':cond,'dv':dv_label,'dv_key':dv_key,
                        'model':model_name,'contrast':'Prior Strong vs Never',
                        'coef':coef,'se':mdf.bse[param_name],
                        'ci_lo':ci[0],'ci_hi':ci[1],
                        'z':mdf.tvalues[param_name],'p':pval,
                        'n_obs':int(mdf.nobs),'n_groups':sub_dv['pid'].nunique()})
                    print(f"    {dv_label:22s} | {model_name:20s}: β={coef:+.4f}, p={pval:.4f} {p_stars(pval)}")
            except Exception as e:
                print(f"    {dv_label:22s} | {model_name:20s}: FAILED — {e}")

# Correct: family = per condition × model
for cond in CONDS:
    for model_name in ['Exposure only', 'Exposure + task']:
        family = [r for r in lmm_rows if r['cond']==cond and r['model']==model_name and not np.isnan(r['p'])]
        pvals = [(lmm_rows.index(r), r['p']) for r in family]
        if len(pvals)<2: continue
        hm=holm_bonferroni(pvals); bh=benjamini_hochberg(pvals)
        for idx in hm: lmm_rows[idx]['p_holm']=hm[idx]; lmm_rows[idx]['p_bh']=bh[idx]
for r in lmm_rows:
    r.setdefault('p_holm', r.get('p', np.nan))
    r.setdefault('p_bh', r.get('p', np.nan))

pd.DataFrame(mw_rows).to_csv(f"{OUTPUT_DIR}/v9b_mw.csv", index=False)
pd.DataFrame(lmm_rows).to_csv(f"{OUTPUT_DIR}/v9b_lmm.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════
# SIGNIFICANT RESULTS
# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*70+"\nSIGNIFICANT RESULTS (p_BH < .10)\n"+"="*70)
print("\n  MW:")
for r in mw_rows:
    if r['p_bh']<0.10: print(f"    {r['cond']:5s} | {r['dv']:22s}: d={r['d']:+.3f}, p_perm={r['p_perm']:.4f}, p_holm={r['p_holm']:.4f}, p_bh={r['p_bh']:.4f}")
print("\n  LMM:")
for r in lmm_rows:
    if r['p_bh']<0.10: print(f"    {r['cond']:5s} | {r['dv']:22s} | {r['model']:20s}: β={r['coef']:+.4f}, p={r['p']:.4f}, p_holm={r['p_holm']:.4f}, p_bh={r['p_bh']:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*70+"\nGENERATING VISUALIZATIONS\n"+"="*70)

# ── Fig 1: Bar charts ────────────────────────────────────────────────
for cond in CONDS:
    sub = df_low[df_low['cond']==cond]
    n_dvs = len(all_dv_keys)
    fig, axes = plt.subplots(1, n_dvs, figsize=(2.8*n_dvs, 5))
    fig.suptitle(f"Low-Pressure Tasks: Prior Exposure Effect — {cond}\n"
                 f"Never exposed vs Previously exposed to Strong | 95% Bootstrap CI",
                 fontsize=11, fontweight='bold', y=1.03)
    for ci_idx, (dv_key, dv_label) in enumerate(zip(all_dv_keys, all_dv_labels)):
        ax = axes[ci_idx]
        if dv_key not in sub.columns: ax.set_visible(False); continue
        means, clo, chi, colors, labels = [], [], [], [], []
        for exp_val, label, color in [(0,'Never\nexposed','#2E86AB'), (1,'Prior\nStrong','#E8533F')]:
            g = sub[sub['prior_pressure']==exp_val][dv_key].dropna()
            if len(g)>=2:
                m=g.mean(); ci_v=bootstrap_ci(g.values)
                means.append(m); clo.append(m-ci_v[0]); chi.append(ci_v[1]-m)
            else: means.append(0); clo.append(0); chi.append(0)
            colors.append(color); labels.append(label)
        ax.bar(range(2), means, color=colors, alpha=0.8, width=0.55, edgecolor='white', linewidth=0.5)
        ax.errorbar(range(2), means, yerr=[clo,chi], fmt='none', color='#333', capsize=4, linewidth=1)
        ax.set_xticks(range(2)); ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(dv_label, fontsize=9, fontweight='bold')
        for xi in range(2):
            n_val = len(sub[sub['prior_pressure']==xi][dv_key].dropna())
            ylim = ax.get_ylim()
            ax.text(xi, means[xi]+chi[xi]+0.02*(ylim[1]-ylim[0]),
                    f'n={n_val}', ha='center', fontsize=6, color='#777')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fname = f"fig1_v9b_bars_{cond.replace('+','_')}.png"
    fig.savefig(f"{OUTPUT_DIR}/{fname}", dpi=200, bbox_inches='tight'); plt.close()
    print(f"  -> {fname}")

# ── Fig 2: Distributions (violin + histogram) ────────────────────────
for cond in CONDS:
    sub = df_low[df_low['cond']==cond]
    n_dvs = len(all_dv_keys)
    fig, axes = plt.subplots(2, n_dvs, figsize=(2.8*n_dvs, 9))
    fig.suptitle(f"Low-Pressure Tasks: Distributions — {cond}\n"
                 f"Top: violin + data | Bottom: density histograms",
                 fontsize=11, fontweight='bold', y=1.02)
    for ci_idx, (dv_key, dv_label) in enumerate(zip(all_dv_keys, all_dv_labels)):
        ax_v = axes[0, ci_idx]; ax_h = axes[1, ci_idx]
        if dv_key not in sub.columns: ax_v.set_visible(False); ax_h.set_visible(False); continue
        groups_data, groups_colors, groups_labels = [], [], []
        for exp_val, label, color in [(0,'Never exposed','#2E86AB'), (1,'Prior Strong','#E8533F')]:
            vals = sub[sub['prior_pressure']==exp_val][dv_key].dropna().values
            if len(vals)>=2: groups_data.append(vals); groups_colors.append(color); groups_labels.append(label)
        if len(groups_data)<2: ax_v.set_visible(False); ax_h.set_visible(False); continue
        positions = list(range(len(groups_data)))
        vp = ax_v.violinplot(groups_data, positions=positions, widths=0.7,
                             showmeans=True, showmedians=True, showextrema=False)
        for i, pc in enumerate(vp['bodies']): pc.set_facecolor(groups_colors[i]); pc.set_alpha(0.4)
        for pn in ['cmeans','cmedians']:
            if pn in vp: vp[pn].set_edgecolor('#333')
        rng = np.random.default_rng(42)
        for i, vals in enumerate(groups_data):
            jx = rng.normal(0, 0.06, len(vals))
            ax_v.scatter(np.full(len(vals),i)+jx, vals,
                         color=groups_colors[i], alpha=0.3, s=8, edgecolors='none')
        ax_v.set_xticks(positions); ax_v.set_xticklabels(groups_labels, fontsize=6)
        ax_v.set_title(dv_label, fontsize=9, fontweight='bold')
        ax_v.spines['top'].set_visible(False); ax_v.spines['right'].set_visible(False)
        all_v = np.concatenate(groups_data)
        bins = np.linspace(np.nanmin(all_v), np.nanmax(all_v), 20)
        for i, vals in enumerate(groups_data):
            ax_h.hist(vals, bins=bins, density=True, alpha=0.35,
                      color=groups_colors[i], edgecolor='white', linewidth=0.3,
                      label=groups_labels[i])
            if len(vals)>=5:
                try:
                    kde = stats.gaussian_kde(vals)
                    x_kde = np.linspace(bins[0], bins[-1], 200)
                    ax_h.plot(x_kde, kde(x_kde), color=groups_colors[i], linewidth=1.5, alpha=0.8)
                except: pass
        ax_h.set_xlabel(dv_label, fontsize=7); ax_h.set_ylabel('Density', fontsize=7)
        ax_h.spines['top'].set_visible(False); ax_h.spines['right'].set_visible(False)
        if ci_idx==0: ax_h.legend(fontsize=5)
    plt.tight_layout()
    fname = f"fig2_v9b_dist_{cond.replace('+','_')}.png"
    fig.savefig(f"{OUTPUT_DIR}/{fname}", dpi=200, bbox_inches='tight'); plt.close()
    print(f"  -> {fname}")

# ── Fig 3: Answer time trajectories ──────────────────────────────────
for cond in CONDS:
    sub_df = df[df['xai_condition']==cond]
    # Only Low-pressure tasks
    low_task_list = sorted(LOW_TASKS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Answer Time by Question ID (Low-Pressure Tasks Only) — {cond}\n"
                 f"By prior Strong pressure exposure | mean ± SEM",
                 fontsize=12, fontweight='bold', y=1.02)
    all_at = []
    for t in low_task_list:
        rc = f"at_reordered_{t}"
        if rc in sub_df.columns:
            for v in sub_df[rc].dropna():
                if isinstance(v, list): all_at.extend([x for x in v if not np.isnan(x)])
    at_ylim = (0, np.percentile(all_at, 98)*1.1) if all_at else (0, 25)

    for idx, measured_task in enumerate(low_task_list):
        ax = axes[idx]
        rc = f"at_reordered_{measured_task}"
        if rc not in sub_df.columns: continue
        for exp_val, label, color in [(0,'Never exposed','#2E86AB'), (1,'Prior Strong','#E8533F')]:
            matching_pids = df_low[(df_low['cond']==cond) &
                                   (df_low['task']==measured_task) &
                                   (df_low['prior_pressure']==exp_val)]['pid'].values
            grp = sub_df.loc[sub_df.index.isin(matching_pids)]
            valid_times = [r for r in grp[rc] if isinstance(r, list)]
            if not valid_times: continue
            n_q = max(len(t) for t in valid_times)
            pad = np.full((len(valid_times), n_q), np.nan)
            for i, t in enumerate(valid_times): pad[i,:len(t)] = t
            m = np.nanmean(pad, axis=0)
            se = np.nanstd(pad, axis=0)/np.sqrt(np.sum(~np.isnan(pad), axis=0))
            qids = np.arange(n_q)
            ax.plot(qids, m, color=color, linewidth=2, label=f"{label} (n={len(valid_times)})", alpha=0.9)
            ax.fill_between(qids, m-se, m+se, color=color, alpha=0.12)
        ax.set_title(f"On: {DISP[measured_task]}", fontweight='bold')
        ax.set_xlabel('Question ID'); ax.set_ylabel('Answer Time (s)')
        ax.set_ylim(at_ylim); ax.legend(fontsize=7)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fname = f"fig3_v9b_at_{cond.replace('+','_')}.png"
    fig.savefig(f"{OUTPUT_DIR}/{fname}", dpi=200, bbox_inches='tight'); plt.close()
    print(f"  -> {fname}")

# ── Fig 4: LMM forest plots ──────────────────────────────────────────
for model_name in ['Exposure only', 'Exposure + task']:
    for cond in CONDS:
        rows = [r for r in lmm_rows if r['cond']==cond and r['model']==model_name]
        if not rows: continue
        fig, ax = plt.subplots(figsize=(10, max(3, len(rows)*0.5+1)))
        fig.suptitle(f"LMM: Prior Strong Exposure on Low-Pressure Tasks — {cond}\n"
                     f"Model: DV ~ {model_name.lower()} + (1|pid)",
                     fontsize=11, fontweight='bold')
        for i, r in enumerate(rows):
            color = '#333'; alpha = 0.4
            if r['p_bh'] < 0.05: color = '#2E86AB'; alpha = 1.0
            elif r['p_bh'] < 0.10: color = '#E8963F'; alpha = 0.8
            ax.barh(i, r['coef'], color=color, alpha=alpha, height=0.5,
                    edgecolor='white', linewidth=0.3)
            ax.plot([r['ci_lo'], r['ci_hi']], [i, i], color='#333', linewidth=1.5,
                    solid_capstyle='round')
            sig_txt = f"p={r['p']:.3f}"
            if r['p_bh'] < 0.10: sig_txt += f" (BH:{p_stars(r['p_bh'])})"
            ax.text(max(r['ci_hi'], r['coef'])+abs(r['coef'])*0.05+0.01, i,
                    sig_txt, va='center', fontsize=7, color='#555')
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r['dv'] for r in rows], fontsize=8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('β (Prior Strong vs Never exposed)')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(rect=[0,0,1,0.88])
        m_short = model_name.replace(' ','_').replace('+','_')
        fname = f"fig4_v9b_lmm_{m_short}_{cond.replace('+','_')}.png"
        fig.savefig(f"{OUTPUT_DIR}/{fname}", dpi=200, bbox_inches='tight'); plt.close()
        print(f"  -> {fname}")

# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*70+"\nCOMPLETE\n"+"="*70)
print("""
CSVs:
  v9b_mw.csv       Mann-Whitney on Low-pressure tasks (with corrections)
  v9b_lmm.csv      LMM results (with corrections)

Figures:
  fig1_v9b_bars_*.png     Bar charts: Never vs Prior Strong
  fig2_v9b_dist_*.png     Distributions (violin + histogram)
  fig3_v9b_at_*.png       Answer time trajectories (Low-pressure tasks)
  fig4_v9b_lmm_*.png      LMM forest plots
""")

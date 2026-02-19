"""
Task Order Effects Analysis v8_first_task — Starting Group Impact (Aggregated)
====================================================================
Question: Does the starting task affect overall participant behavior?

Analysis 1: Aggregated between-group (KW + Mann-Whitney on participant means)
Analysis 2a: LMM — DV ~ starting_group + (1|participant)
Analysis 2b: LMM — DV ~ starting_group + task + (1|participant)
Corrections: Holm-Bonferroni + Benjamini-Hochberg throughout.
Figures: bars, distributions, answer time trajectories, LMM forest plots.
"""
import pandas as pd, numpy as np, matplotlib, ast, sys, warnings, itertools
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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
DISP = {'easy_mild':'Low/Low','easy_strong':'Low/Strong',
        'hard_mild':'High/Low','hard_strong':'High/Strong'}
PAL = {'easy_mild':'#2E86AB','easy_strong':'#7FB069','hard_mild':'#E8963F','hard_strong':'#E8533F'}
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

def correct_list(rows, p_col):
    pvals = [(i, r[p_col]) for i,r in enumerate(rows) if not np.isnan(r[p_col])]
    if len(pvals)<2:
        for r in rows: r['p_holm']=r.get(p_col,np.nan); r['p_bh']=r.get(p_col,np.nan)
        return rows
    hm=holm_bonferroni(pvals); bh=benjamini_hochberg(pvals)
    for i,r in enumerate(rows):
        r['p_holm']=hm.get(i, r.get(p_col,np.nan)); r['p_bh']=bh.get(i, r.get(p_col,np.nan))
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
print("="*70+"\nTASK ORDER EFFECTS v8_first_task — STARTING GROUP IMPACT\n"+"="*70)
df = pd.read_csv(DATA_PATH)
print(f"Total: {len(df)}")
df = df[df['xai_condition'].isin(CONDS)].copy()
print(f"H + H+AI: {len(df)}")
df['tasks_order'] = df['tasks_order'].apply(parse_col)
df['first_task'] = df['tasks_order'].apply(lambda x: x[0] if isinstance(x,list) else None)

print("\nStarting task distribution:")
for t in ALL_TASKS:
    counts = ' | '.join(f"{c}: {((df['first_task']==t)&(df['xai_condition']==c)).sum()}" for c in CONDS)
    print(f"  {DISP[t]:12s}: {counts}")

# Compute participant-level aggregated means across all 4 tasks
print("\n-> Computing participant-level aggregated means...")
for dv_base in DVS:
    cols = [f"{dv_base}_{t}" for t in ALL_TASKS]
    available = [c for c in cols if c in df.columns]
    if available:
        df[f"agg_{dv_base}"] = df[available].mean(axis=1)

# Aggregated mean answer time
at_cols_available = []
for t in ALL_TASKS:
    at_col = f"answer_times_{t}"
    if at_col in df.columns:
        df[f"mean_at_{t}"] = df[at_col].apply(
            lambda v: np.nanmean(np.array(parse_col(v),dtype=float))
            if isinstance(parse_col(v),(list,np.ndarray)) else np.nan)
        at_cols_available.append(f"mean_at_{t}")
if at_cols_available:
    df['agg_mean_at'] = df[at_cols_available].mean(axis=1)

# Reordered answer times
for t in ALL_TASKS:
    at_col = f"answer_times_{t}"
    qo_col = f"quest_order_{t}"
    if at_col in df.columns and qo_col in df.columns:
        df[f"at_reordered_{t}"] = df.apply(
            lambda r, tk=t: reorder_by_quest(r[f"answer_times_{tk}"], r[f"quest_order_{tk}"]), axis=1)

# Build long-format for LMM
print("-> Building long-format data for LMM...")
long_rows = []
for _, row in df.iterrows():
    pid = row.name  # use index as participant id
    cond = row['xai_condition']
    first = row['first_task']
    for task in ALL_TASKS:
        entry = {'pid': pid, 'cond': cond, 'first_task': first, 'task': task}
        for dv_base in DVS:
            col = f"{dv_base}_{task}"
            entry[dv_base] = row[col] if col in df.columns and pd.notna(row.get(col)) else np.nan
        if f"mean_at_{task}" in df.columns:
            entry['mean_at'] = row[f"mean_at_{task}"]
        long_rows.append(entry)
df_long = pd.DataFrame(long_rows)
df_long['first_task_disp'] = df_long['first_task'].map(DISP)
df_long['task_disp'] = df_long['task'].map(DISP)

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: AGGREGATED BETWEEN-GROUP
# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*70+"\nANALYSIS 1: AGGREGATED BETWEEN-GROUP (KW + Mann-Whitney)\n"+"="*70)
agg_dvs = {f"agg_{k}": v for k, v in DVS.items()}
agg_dvs['agg_mean_at'] = 'Mean Answer Time'

kw_rows = []
pw_rows = []

for cond in CONDS:
    sub = df[df['xai_condition']==cond]
    print(f"\n  {cond} (n={len(sub)})")
    for agg_col, dv_label in agg_dvs.items():
        if agg_col not in sub.columns: continue
        # KW
        groups_kw = []
        for s in ALL_TASKS:
            g = sub[sub['first_task']==s][agg_col].dropna()
            if len(g)>=2: groups_kw.append(g.values)
        H_stat, p_kw, eta2 = np.nan, np.nan, np.nan
        if len(groups_kw)>=2:
            try:
                H_stat, p_kw = stats.kruskal(*groups_kw)
                N=sum(len(g) for g in groups_kw); k=len(groups_kw)
                eta2=(H_stat-k+1)/(N-k) if (N-k)>0 else np.nan
            except: pass
        kw_rows.append({'cond':cond,'dv':dv_label,'agg_col':agg_col,
                        'H':H_stat,'p_kw':p_kw,'eta2':eta2})
        print(f"    {dv_label:22s}: KW H={H_stat:.3f}, p={p_kw:.4f} {p_stars(p_kw)}")

        # Pairwise
        for s1, s2 in itertools.combinations(ALL_TASKS, 2):
            g1 = sub[sub['first_task']==s1][agg_col].dropna().values
            g2 = sub[sub['first_task']==s2][agg_col].dropna().values
            if len(g1)<2 or len(g2)<2: continue
            u, p_mw = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            d = cohens_d(g1, g2)
            p_perm = permutation_test(g1, g2)
            pw_rows.append({
                'cond':cond,'dv':dv_label,'agg_col':agg_col,
                'start1':s1,'start1_disp':DISP[s1],'start2':s2,'start2_disp':DISP[s2],
                'mean1':np.mean(g1),'n1':len(g1),'mean2':np.mean(g2),'n2':len(g2),
                'U':u,'p_mw':p_mw,'p_perm':p_perm,'d':d})

# Correct KW: family = per condition
for cond in CONDS:
    family = [r for r in kw_rows if r['cond']==cond and not np.isnan(r['p_kw'])]
    pvals = [(kw_rows.index(r), r['p_kw']) for r in family]
    if len(pvals)<2: continue
    hm=holm_bonferroni(pvals); bh=benjamini_hochberg(pvals)
    for idx in hm: kw_rows[idx]['p_holm']=hm[idx]; kw_rows[idx]['p_bh']=bh[idx]
for r in kw_rows: r.setdefault('p_holm',r.get('p_kw',np.nan)); r.setdefault('p_bh',r.get('p_kw',np.nan))

# Correct pairwise: family = per condition × DV
for cond in CONDS:
    for dv_label in agg_dvs.values():
        family = [r for r in pw_rows if r['cond']==cond and r['dv']==dv_label and not np.isnan(r['p_perm'])]
        pvals = [(pw_rows.index(r), r['p_perm']) for r in family]
        if len(pvals)<2: continue
        hm=holm_bonferroni(pvals); bh=benjamini_hochberg(pvals)
        for idx in hm: pw_rows[idx]['p_holm']=hm[idx]; pw_rows[idx]['p_bh']=bh[idx]
for r in pw_rows: r.setdefault('p_holm',r.get('p_perm',np.nan)); r.setdefault('p_bh',r.get('p_perm',np.nan))

pd.DataFrame(kw_rows).to_csv(f"{OUTPUT_DIR}/v8_kw_omnibus.csv", index=False)
pd.DataFrame(pw_rows).to_csv(f"{OUTPUT_DIR}/v8_between_pairwise.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 2: LINEAR MIXED MODELS
# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*70+"\nANALYSIS 2: LINEAR MIXED MODELS\n"+"="*70)

lmm_results = []
all_lmm_dvs = list(DVS.keys()) + ['mean_at']
all_lmm_labels = list(DVS.values()) + ['Mean Answer Time']

for cond in CONDS:
    sub_long = df_long[df_long['cond']==cond].copy()
    print(f"\n  {cond} (n_obs={len(sub_long)}, n_subj={sub_long['pid'].nunique()})")

    for dv_base, dv_label in zip(all_lmm_dvs, all_lmm_labels):
        if dv_base not in sub_long.columns: continue
        sub_dv = sub_long[['pid','first_task','task',dv_base]].dropna().copy()
        sub_dv = sub_dv.rename(columns={dv_base:'y'})
        if len(sub_dv) < 10: continue

        # Reference = always Low/Low (easy_mild)
        ref_start = 'easy_mild'

        for model_name, formula in [
            ('Model A (start only)', 'y ~ C(first_task, Treatment(reference="{ref}"))'),
            ('Model B (start + task)', 'y ~ C(first_task, Treatment(reference="{ref}")) + C(task)')]:

            formula_filled = formula.replace('{ref}', ref_start)
            try:
                md = smf.mixedlm(formula_filled, sub_dv, groups=sub_dv['pid'])
                mdf = md.fit(reml=True)

                # Extract starting group effects
                for param_name, coef in mdf.fe_params.items():
                    if 'first_task' not in param_name: continue
                    # Parse which group this is
                    start_grp = param_name.split('[T.')[1].rstrip(']') if '[T.' in param_name else param_name
                    pval = mdf.pvalues[param_name]
                    ci = mdf.conf_int().loc[param_name]
                    lmm_results.append({
                        'cond':cond, 'dv':dv_label, 'dv_base':dv_base,
                        'model':model_name,
                        'reference':DISP.get(ref_start, ref_start),
                        'start_group':DISP.get(start_grp, start_grp),
                        'coef':coef, 'se':mdf.bse[param_name],
                        'ci_lo':ci[0], 'ci_hi':ci[1],
                        'z':mdf.tvalues[param_name], 'p':pval,
                        'aic':mdf.aic, 'bic':mdf.bic,
                        'n_obs':mdf.nobs, 'n_groups':mdf.nobs  # placeholder
                    })
                    print(f"    {dv_label:22s} | {model_name:25s} | {DISP.get(start_grp,'?'):12s} vs ref: "
                          f"β={coef:+.4f}, p={pval:.4f} {p_stars(pval)}")

                # Also store omnibus F-test approximation (Wald test for all start group params)
                start_params = [p for p in mdf.fe_params.index if 'first_task' in p]
                if len(start_params) >= 2:
                    try:
                        wald = mdf.wald_test_terms()
                        # Find the starting group term
                        for term_name in wald.table.index:
                            if 'first_task' in str(term_name):
                                chi2_val = float(wald.table.loc[term_name, 'statistic'])
                                p_wald = float(wald.table.loc[term_name, 'pvalue'])
                                df_wald = int(wald.table.loc[term_name, 'df'])
                                lmm_results.append({
                                    'cond':cond,'dv':dv_label,'dv_base':dv_base,
                                    'model':model_name,
                                    'reference':DISP.get(ref_start,ref_start),
                                    'start_group':'OMNIBUS (Wald)',
                                    'coef':np.nan,'se':np.nan,
                                    'ci_lo':np.nan,'ci_hi':np.nan,
                                    'z':chi2_val,'p':p_wald,
                                    'aic':mdf.aic,'bic':mdf.bic,
                                    'n_obs':mdf.nobs,'n_groups':sub_dv['pid'].nunique()
                                })
                                print(f"    {dv_label:22s} | {model_name:25s} | OMNIBUS Wald χ²={chi2_val:.3f}, "
                                      f"df={df_wald}, p={p_wald:.4f} {p_stars(p_wald)}")
                    except Exception as e:
                        print(f"    {dv_label:22s} | {model_name:25s} | Wald test failed: {e}")

            except Exception as e:
                print(f"    {dv_label:22s} | {model_name:25s} | FAILED: {e}")

# Correct LMM p-values: family = cond × model (pairwise contrasts only)
print("\n-> Correcting LMM p-values...")
for cond in CONDS:
    for model_name in ['Model A (start only)', 'Model B (start + task)']:
        family = [r for r in lmm_results
                  if r['cond']==cond and r['model']==model_name
                  and r['start_group']!='OMNIBUS (Wald)' and not np.isnan(r['p'])]
        pvals = [(lmm_results.index(r), r['p']) for r in family]
        if len(pvals)<2: continue
        hm=holm_bonferroni(pvals); bh=benjamini_hochberg(pvals)
        for idx in hm: lmm_results[idx]['p_holm']=hm[idx]; lmm_results[idx]['p_bh']=bh[idx]
# Correct omnibus Wald: family = cond × model
for cond in CONDS:
    for model_name in ['Model A (start only)', 'Model B (start + task)']:
        family = [r for r in lmm_results
                  if r['cond']==cond and r['model']==model_name
                  and r['start_group']=='OMNIBUS (Wald)' and not np.isnan(r['p'])]
        pvals = [(lmm_results.index(r), r['p']) for r in family]
        if len(pvals)<2: continue
        hm=holm_bonferroni(pvals); bh=benjamini_hochberg(pvals)
        for idx in hm: lmm_results[idx]['p_holm']=hm[idx]; lmm_results[idx]['p_bh']=bh[idx]

for r in lmm_results:
    r.setdefault('p_holm', r.get('p', np.nan))
    r.setdefault('p_bh', r.get('p', np.nan))

pd.DataFrame(lmm_results).to_csv(f"{OUTPUT_DIR}/v8_lmm_results.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════
# SIGNIFICANT RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*70+"\nSIGNIFICANT RESULTS (p_BH < .10)\n"+"="*70)
print("\n  KW omnibus:")
for r in kw_rows:
    if r['p_bh']<0.10: print(f"    {r['cond']:5s} | {r['dv']:22s}: H={r['H']:.3f}, p={r['p_kw']:.4f}, p_holm={r['p_holm']:.4f}, p_bh={r['p_bh']:.4f}")
print("\n  Between pairwise:")
for r in pw_rows:
    if r['p_bh']<0.10: print(f"    {r['cond']:5s} | {r['dv']:22s} | {r['start1_disp']} vs {r['start2_disp']}: d={r['d']:+.3f}, p_perm={r['p_perm']:.4f}, p_holm={r['p_holm']:.4f}, p_bh={r['p_bh']:.4f}")
print("\n  LMM omnibus (Wald):")
for r in lmm_results:
    if r['start_group']=='OMNIBUS (Wald)' and r['p_bh']<0.10:
        print(f"    {r['cond']:5s} | {r['dv']:22s} | {r['model']:25s}: χ²={r['z']:.3f}, p={r['p']:.4f}, p_holm={r['p_holm']:.4f}, p_bh={r['p_bh']:.4f}")
print("\n  LMM contrasts:")
for r in lmm_results:
    if r['start_group']!='OMNIBUS (Wald)' and r['p_bh']<0.10:
        print(f"    {r['cond']:5s} | {r['dv']:22s} | {r['model']:25s} | {r['start_group']} vs {r['reference']}: β={r['coef']:+.4f}, p={r['p']:.4f}, p_holm={r['p_holm']:.4f}, p_bh={r['p_bh']:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*70+"\nGENERATING VISUALIZATIONS\n"+"="*70)

agg_key_dvs = [f"agg_{k}" for k in DVS.keys()] + ['agg_mean_at']
agg_labels = list(DVS.values()) + ['Mean Answer Time']

# Shared y-limits per aggregated DV
ylims = {}
for agg_col, label in zip(agg_key_dvs, agg_labels):
    if agg_col in df.columns:
        vals = df[agg_col].dropna().values
        if len(vals)>0:
            mn,mx=vals.min(),vals.max(); margin=(mx-mn)*0.15 if mx>mn else 0.1
            ylims[agg_col]=(max(0,mn-margin),mx+margin)

# ── Fig 1: Bar charts (aggregated means by starting group) ───────────
for cond in CONDS:
    sub = df[df['xai_condition']==cond]
    n_dvs = len(agg_key_dvs)
    fig, axes = plt.subplots(1, n_dvs, figsize=(2.8*n_dvs, 5))
    fig.suptitle(f"Aggregated DVs by Starting Task — {cond}\n"
                 f"Participant means across all 4 tasks | 95% Bootstrap CI",
                 fontsize=12, fontweight='bold', y=1.03)
    for ci, (agg_col, label) in enumerate(zip(agg_key_dvs, agg_labels)):
        ax = axes[ci] if n_dvs > 1 else axes
        if agg_col not in sub.columns: ax.set_visible(False); continue
        means, clo, chi, colors = [], [], [], []
        for s in ALL_TASKS:
            g = sub[sub['first_task']==s][agg_col].dropna()
            if len(g)>=2:
                m=g.mean(); ci_v=bootstrap_ci(g.values)
                means.append(m); clo.append(m-ci_v[0]); chi.append(ci_v[1]-m)
            else: means.append(0); clo.append(0); chi.append(0)
            colors.append(PAL[s])
        bars = ax.bar(range(4), means, color=colors, alpha=0.8, width=0.65,
                      edgecolor='white', linewidth=0.5)
        ax.errorbar(range(4), means, yerr=[clo,chi], fmt='none', color='#333', capsize=3, linewidth=1)
        ax.set_xticks(range(4))
        ax.set_xticklabels([DISP[t] for t in ALL_TASKS], fontsize=6, rotation=45, ha='right')
        ax.set_title(label, fontsize=9, fontweight='bold')
        if agg_col in ylims: ax.set_ylim(ylims[agg_col])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fname = f"fig1_bars_{cond.replace('+','_')}.png"
    fig.savefig(f"{OUTPUT_DIR}/{fname}", dpi=200, bbox_inches='tight'); plt.close()
    print(f"  -> {fname}")

# ── Fig 2: Distributions (violin + histogram) ────────────────────────
for cond in CONDS:
    sub = df[df['xai_condition']==cond]
    n_dvs = len(agg_key_dvs)
    fig, axes = plt.subplots(2, n_dvs, figsize=(2.8*n_dvs, 9))
    fig.suptitle(f"Distributions by Starting Task — {cond}\n"
                 f"Top: violin + data points | Bottom: density histograms",
                 fontsize=12, fontweight='bold', y=1.02)
    for ci, (agg_col, label) in enumerate(zip(agg_key_dvs, agg_labels)):
        ax_v = axes[0, ci]; ax_h = axes[1, ci]
        if agg_col not in sub.columns: ax_v.set_visible(False); ax_h.set_visible(False); continue
        groups_data, groups_keys = [], []
        for s in ALL_TASKS:
            vals = sub[sub['first_task']==s][agg_col].dropna().values
            if len(vals)>=2: groups_data.append(vals); groups_keys.append(s)
        if len(groups_data)<2: ax_v.set_visible(False); ax_h.set_visible(False); continue
        # Violin
        positions = list(range(len(groups_data)))
        vp = ax_v.violinplot(groups_data, positions=positions, widths=0.7,
                             showmeans=True, showmedians=True, showextrema=False)
        for i, pc in enumerate(vp['bodies']): pc.set_facecolor(PAL[groups_keys[i]]); pc.set_alpha(0.4)
        for pn in ['cmeans','cmedians']:
            if pn in vp: vp[pn].set_edgecolor('#333')
        rng = np.random.default_rng(42)
        for i, vals in enumerate(groups_data):
            jx = rng.normal(0, 0.06, len(vals))
            ax_v.scatter(np.full(len(vals),i)+jx, vals,
                         color=PAL[groups_keys[i]], alpha=0.5, s=12, edgecolors='none')
        ax_v.set_xticks(positions)
        ax_v.set_xticklabels([DISP[s] for s in groups_keys], fontsize=6, rotation=45, ha='right')
        ax_v.set_title(label, fontsize=9, fontweight='bold')
        ax_v.spines['top'].set_visible(False); ax_v.spines['right'].set_visible(False)
        # Histogram
        all_v = np.concatenate(groups_data)
        bins = np.linspace(np.nanmin(all_v), np.nanmax(all_v), 15)
        for i, vals in enumerate(groups_data):
            ax_h.hist(vals, bins=bins, density=True, alpha=0.35,
                      color=PAL[groups_keys[i]], edgecolor='white', linewidth=0.3,
                      label=DISP[groups_keys[i]])
            if len(vals)>=5:
                try:
                    kde = stats.gaussian_kde(vals)
                    x_kde = np.linspace(bins[0], bins[-1], 200)
                    ax_h.plot(x_kde, kde(x_kde), color=PAL[groups_keys[i]], linewidth=1.5, alpha=0.8)
                except: pass
        ax_h.set_xlabel(label, fontsize=7); ax_h.set_ylabel('Density', fontsize=7)
        ax_h.spines['top'].set_visible(False); ax_h.spines['right'].set_visible(False)
        if ci==0: ax_h.legend(fontsize=5, ncol=2)
    plt.tight_layout()
    fname = f"fig2_dist_{cond.replace('+','_')}.png"
    fig.savefig(f"{OUTPUT_DIR}/{fname}", dpi=200, bbox_inches='tight'); plt.close()
    print(f"  -> {fname}")

# ── Fig 3: Answer time trajectories (pooled across all tasks, by Q ID) ─
for cond in CONDS:
    sub = df[df['xai_condition']==cond]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Answer Time by Question ID — {cond}\n"
                 f"Grouped by starting task (mean ± SEM)",
                 fontsize=13, fontweight='bold', y=0.99)
    # Shared ylim
    all_at = []
    for t in ALL_TASKS:
        rc = f"at_reordered_{t}"
        if rc in sub.columns:
            for v in sub[rc].dropna():
                if isinstance(v, list): all_at.extend([x for x in v if not np.isnan(x)])
    at_ylim = (0, np.percentile(all_at, 98)*1.1) if all_at else (0, 25)

    for idx, measured_task in enumerate(ALL_TASKS):
        ax = axes[idx//2, idx%2]
        rc = f"at_reordered_{measured_task}"
        if rc not in sub.columns: continue
        for st in ALL_TASKS:
            grp = sub[sub['first_task']==st]
            valid_times = [r for r in grp[rc] if isinstance(r, list)]
            if not valid_times: continue
            n_q = max(len(t) for t in valid_times)
            pad = np.full((len(valid_times), n_q), np.nan)
            for i, t in enumerate(valid_times): pad[i,:len(t)] = t
            m = np.nanmean(pad, axis=0)
            se = np.nanstd(pad, axis=0)/np.sqrt(np.sum(~np.isnan(pad), axis=0))
            qids = np.arange(n_q)
            ax.plot(qids, m, color=PAL[st], linewidth=2,
                    label=f"Started {DISP[st]}", alpha=0.9)
            ax.fill_between(qids, m-se, m+se, color=PAL[st], alpha=0.12)
        ax.set_title(f"On: {DISP[measured_task]}", fontweight='bold')
        ax.set_xlabel('Question ID'); ax.set_ylabel('Answer Time (s)')
        ax.set_ylim(at_ylim); ax.legend(fontsize=6, ncol=2)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0,0,1,0.94])
    fname = f"fig3_at_trajectories_{cond.replace('+','_')}.png"
    fig.savefig(f"{OUTPUT_DIR}/{fname}", dpi=200, bbox_inches='tight'); plt.close()
    print(f"  -> {fname}")

# ── Fig 4: LMM forest plots ──────────────────────────────────────────
for model_name_short, model_name in [('A','Model A (start only)'), ('B','Model B (start + task)')]:
    for cond in CONDS:
        lmm_sub = [r for r in lmm_results
                   if r['cond']==cond and r['model']==model_name and r['start_group']!='OMNIBUS (Wald)']
        if not lmm_sub: continue
        fig, ax = plt.subplots(figsize=(10, max(4, len(lmm_sub)*0.35+1)))
        fig.suptitle(f"LMM {model_name_short}: Starting Group Contrasts — {cond}\n"
                     f"{'DV ~ start + (1|pid)' if model_name_short=='A' else 'DV ~ start + task + (1|pid)'}",
                     fontsize=12, fontweight='bold')
        y_labels = []
        for i, r in enumerate(lmm_sub):
            color = '#333'
            alpha = 0.4
            if r['p_bh'] < 0.05: color = '#2E86AB'; alpha = 1.0
            elif r['p_bh'] < 0.10: color = '#E8963F'; alpha = 0.8
            ax.barh(i, r['coef'], color=color, alpha=alpha, height=0.6, edgecolor='white', linewidth=0.3)
            ax.plot([r['ci_lo'], r['ci_hi']], [i, i], color='#333', linewidth=1.5, solid_capstyle='round')
            sig_txt = f"p={r['p']:.3f}"
            if r['p_bh'] < 0.05: sig_txt += f" (BH:{p_stars(r['p_bh'])})"
            elif r['p_holm'] < 0.10: sig_txt += f" (H:{p_stars(r['p_holm'])})"
            ax.text(max(r['ci_hi'], r['coef'])+0.01, i, sig_txt, va='center', fontsize=6.5, color='#555')
            y_labels.append(f"{r['dv']} | {r['start_group']}")
        ax.set_yticks(range(len(lmm_sub)))
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel(f"β (vs reference: {lmm_sub[0]['reference']})")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(rect=[0,0,1,0.92])
        fname = f"fig4_lmm_{model_name_short}_{cond.replace('+','_')}.png"
        fig.savefig(f"{OUTPUT_DIR}/{fname}", dpi=200, bbox_inches='tight'); plt.close()
        print(f"  -> {fname}")

# ── Fig 5: LMM Wald omnibus heatmap ──────────────────────────────────
for model_name_short, model_name in [('A','Model A (start only)'), ('B','Model B (start + task)')]:
    wald_sub = [r for r in lmm_results if r['model']==model_name and r['start_group']=='OMNIBUS (Wald)']
    if not wald_sub: continue
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"LMM {model_name_short} — Wald Omnibus Test for Starting Group Effect\n"
                 f"{'DV ~ start + (1|pid)' if model_name_short=='A' else 'DV ~ start + task + (1|pid)'}",
                 fontsize=12, fontweight='bold')
    for ci, cond in enumerate(CONDS):
        ax = axes[ci]
        ws = [r for r in wald_sub if r['cond']==cond]
        if not ws: ax.set_visible(False); continue
        dv_labels = [r['dv'] for r in ws]
        p_vals = [r['p'] for r in ws]
        p_holm_vals = [r.get('p_holm',np.nan) for r in ws]
        p_bh_vals = [r.get('p_bh',np.nan) for r in ws]
        mat = np.array([-np.log10(max(p, 1e-10)) for p in p_vals]).reshape(-1, 1)
        annot_arr = np.array([f"p={p:.3f}\nH:{p_stars(ph)}/B:{p_stars(pb)}"
                              for p, ph, pb in zip(p_vals, p_holm_vals, p_bh_vals)]).reshape(-1, 1)
        hm_df = pd.DataFrame(mat, index=dv_labels, columns=['−log₁₀(p)'])
        sns.heatmap(hm_df, annot=annot_arr, fmt='', cmap='YlOrRd', ax=ax,
                    vmin=0, vmax=3, linewidths=0.5)
        ax.set_title(cond, fontweight='bold')
        ax.set_ylabel('')
    ax.text(0.01, -0.12, "H = Holm sig. / B = BH sig.", transform=axes[0].transAxes,
            fontsize=8, color='#777')
    plt.tight_layout(rect=[0,0,1,0.88])
    fname = f"fig5_wald_{model_name_short}.png"
    fig.savefig(f"{OUTPUT_DIR}/{fname}", dpi=200, bbox_inches='tight'); plt.close()
    print(f"  -> {fname}")

# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*70+"\nCOMPLETE\n"+"="*70)
print("""
CSVs:
  v8_kw_omnibus.csv              KW on aggregated means (with Holm + BH)
  v8_between_pairwise.csv        Mann-Whitney pairwise on aggregated means
  v8_lmm_results.csv             LMM contrasts + Wald omnibus (with Holm + BH)

Figures:
  fig1_bars_H.png / _H_AI.png              Bar charts (aggregated means)
  fig2_dist_H.png / _H_AI.png              Distributions (violin + histogram)
  fig3_at_trajectories_H.png / _H_AI.png   Answer time by question ID (H + H+AI)
  fig4_lmm_A_H.png / _H_AI.png             LMM Model A forest plots
  fig4_lmm_B_H.png / _H_AI.png             LMM Model B forest plots
  fig5_wald_A.png / _B.png                  Wald omnibus heatmaps
""")
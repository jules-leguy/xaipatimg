"""
Task Order Effects Analysis v10 — Exposure to Time Pressure & Difficulty
=========================================================================
Two dimensions studied independently:
  - TIME PRESSURE: Strong (easy_strong, hard_strong) vs Low (easy_mild, hard_mild)
  - DIFFICULTY:    High (hard_mild, hard_strong) vs Low (easy_mild, easy_strong)

Three comparisons per dimension:
  A:  Task-level — currently/previously exposed vs never exposed
  Ab: Task-level — restricted to UNEXPOSED-type tasks, comparing prior exposure vs none
  B:  Participant-level — first two tasks both X vs both Y (mixed excluded)

Statistics: Mann-Whitney + permutation + LMM (two variants). Corrections: Holm + BH.
Matched y-axis scales: Accuracy/Reliance/Over-reliance/Under-reliance on 0–1.
"""
import pandas as pd, numpy as np, matplotlib, ast, sys, warnings, itertools
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
DISP = {'easy_mild':'Low/Low','easy_strong':'Low/Strong',
        'hard_mild':'High/Low','hard_strong':'High/Strong'}

# Dimension definitions
DIMS = {
    'pressure': {
        'label': 'Time Pressure',
        'exposed_tasks': {'easy_strong','hard_strong'},   # Strong pressure
        'unexposed_tasks': {'easy_mild','hard_mild'},     # Low pressure
        'exposed_label': 'Strong pressure',
        'unexposed_label': 'Low pressure',
        'group_labels': {0:'Never Strong', 1:'Exposed to Strong'},
        'bar_labels': {0:'Never\nStrong', 1:'Exposed\nStrong'},
        'profile_labels': {'Both Unexposed':'Both Low press.', 'Both Exposed':'Both Strong press.'},
    },
    'difficulty': {
        'label': 'Difficulty',
        'exposed_tasks': {'hard_mild','hard_strong'},     # High difficulty
        'unexposed_tasks': {'easy_mild','easy_strong'},   # Low difficulty
        'exposed_label': 'High difficulty',
        'unexposed_label': 'Low difficulty',
        'group_labels': {0:'Never High diff.', 1:'Exposed to High diff.'},
        'bar_labels': {0:'Never\nHigh diff.', 1:'Exposed\nHigh diff.'},
        'profile_labels': {'Both Unexposed':'Both Low diff.', 'Both Exposed':'Both High diff.'},
    }
}

DVS = {'score':'Accuracy','reliance':'Reliance','overreliance':'Over-reliance',
       'underreliance':'Under-reliance','trust':'Trust','cogload':'Cognitive Load'}
CONDS = ['H','H+AI']
ALL_DV_KEYS = list(DVS.keys()) + ['mean_at']
ALL_DV_LABELS = list(DVS.values()) + ['Mean Answer Time']

# Fixed y-limits
YLIMS = {
    'score':(0,1), 'reliance':(0,1), 'overreliance':(0,1), 'underreliance':(0,1),
    'trust':None, 'cogload':None, 'mean_at':None
}

PAL = {0:'#2E86AB', 1:'#E8533F'}

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
    n=len(pvals); s=sorted(pvals,key=lambda x:x[1]); c={}; mx=0
    for rank,(idx,p) in enumerate(s):
        adj=min(p*(n-rank),1.0); adj=max(adj,mx); mx=adj; c[idx]=adj
    return c

def benjamini_hochberg(pvals):
    n=len(pvals); s=sorted(pvals,key=lambda x:x[1],reverse=True); c={}; mn=1.0
    for rd,(idx,p) in enumerate(s):
        ra=n-rd; adj=min(p*n/ra,1.0); adj=min(adj,mn); mn=adj; c[idx]=adj
    return c

def correct_family(rows, p_col):
    pvals=[(i,r[p_col]) for i,r in enumerate(rows) if not np.isnan(r[p_col])]
    if len(pvals)<2:
        for r in rows: r['p_holm']=r.get(p_col,np.nan); r['p_bh']=r.get(p_col,np.nan)
        return rows
    hm=holm_bonferroni(pvals); bh=benjamini_hochberg(pvals)
    for i,r in enumerate(rows):
        r['p_holm']=hm.get(i,r.get(p_col,np.nan)); r['p_bh']=bh.get(i,r.get(p_col,np.nan))
    return rows

def get_ylim(dv_key, data_arrays=None):
    if YLIMS.get(dv_key) is not None: return YLIMS[dv_key]
    if data_arrays:
        all_v = np.concatenate([a for a in data_arrays if len(a)>0])
        if len(all_v)>0:
            mn,mx=np.nanmin(all_v),np.nanmax(all_v)
            margin=(mx-mn)*0.15 if mx>mn else 0.5
            return (max(0,mn-margin), mx+margin)
    return None

# ── Load & prepare ────────────────────────────────────────────────────
print("="*70+"\nTASK ORDER EFFECTS v10 — PRESSURE & DIFFICULTY EXPOSURE\n"+"="*70)
df = pd.read_csv(DATA_PATH)
print(f"Total: {len(df)}")
df = df[df['xai_condition'].isin(CONDS)].copy()
print(f"H + H+AI: {len(df)}")
df['tasks_order'] = df['tasks_order'].apply(parse_col)

for t in ALL_TASKS:
    at_col = f"answer_times_{t}"
    if at_col in df.columns:
        df[f"mean_at_{t}"] = df[at_col].apply(
            lambda v: np.nanmean(np.array(parse_col(v),dtype=float))
            if isinstance(parse_col(v),(list,np.ndarray)) else np.nan)

# Compute global ylims for trust, cogload, mean_at
for dv_key in ['trust','cogload','mean_at']:
    all_v = []
    for t in ALL_TASKS:
        col = f"{dv_key}_{t}" if dv_key!='mean_at' else f"mean_at_{t}"
        if col in df.columns: all_v.extend(df[col].dropna().values)
    if all_v:
        mn,mx=min(all_v),max(all_v); margin=(mx-mn)*0.15
        YLIMS[dv_key] = (max(0,mn-margin), mx+margin)

# ═══════════════════════════════════════════════════════════════════════
# BUILD LONG-FORMAT DATA FOR EACH DIMENSION
# ═══════════════════════════════════════════════════════════════════════
all_mw_rows = []
all_lmm_rows = []

for dim_key, dim in DIMS.items():
    print(f"\n{'='*70}\n  DIMENSION: {dim['label'].upper()}\n{'='*70}")
    exposed_set = dim['exposed_tasks']
    unexposed_set = dim['unexposed_tasks']

    # ───────────────────────────────────────────────────────────────
    # COMPARISON A: task-level, exposed (current or prior) vs never
    # ───────────────────────────────────────────────────────────────
    print(f"\n  --- Comparison A: All tasks, exposed vs never ---")
    rows_a = []
    for _, row in df.iterrows():
        pid=row.name; cond=row['xai_condition']; order=row['tasks_order']
        if not isinstance(order,list) or len(order)!=4: continue
        seen_exposed = False
        for pos, task in enumerate(order):
            current_exposed = task in exposed_set
            flag = 1 if (current_exposed or seen_exposed) else 0
            entry = {'pid':pid,'cond':cond,'task':task,'task_disp':DISP[task],
                     'position':pos+1,'exposed':flag,'dim':dim_key,'comp':'A'}
            for dv_base in DVS:
                col=f"{dv_base}_{task}"
                entry[dv_base]=row[col] if col in df.columns and pd.notna(row.get(col)) else np.nan
            mc=f"mean_at_{task}"
            entry['mean_at']=row[mc] if mc in df.columns and pd.notna(row.get(mc)) else np.nan
            rows_a.append(entry)
            if current_exposed: seen_exposed=True
    df_a = pd.DataFrame(rows_a)

    for cond in CONDS:
        sub=df_a[df_a['cond']==cond]
        ct=sub.groupby('exposed').size()
        print(f"    {cond}: " + ", ".join(f"exposed={k}: {v}" for k,v in ct.items()))

    # MW + permutation
    for cond in CONDS:
        sub=df_a[df_a['cond']==cond]
        for dv_key, dv_label in zip(ALL_DV_KEYS, ALL_DV_LABELS):
            if dv_key not in sub.columns: continue
            g0=sub[sub['exposed']==0][dv_key].dropna().values
            g1=sub[sub['exposed']==1][dv_key].dropna().values
            if len(g0)<3 or len(g1)<3: continue
            u,p_mw=stats.mannwhitneyu(g0,g1,alternative='two-sided')
            d=cohens_d(g0,g1); p_perm=permutation_test(g0,g1)
            all_mw_rows.append({
                'dim':dim_key,'dim_label':dim['label'],'comp':'A','cond':cond,
                'dv':dv_label,'dv_key':dv_key,
                'mean_0':np.mean(g0),'n_0':len(g0),'mean_1':np.mean(g1),'n_1':len(g1),
                'U':u,'p_mw':p_mw,'p_perm':p_perm,'d':d,
                'label_0':dim['group_labels'][0],'label_1':dim['group_labels'][1]})
            print(f"    {cond:5s} A  {dv_label:22s}: {np.mean(g0):.3f} vs {np.mean(g1):.3f} d={d:+.3f} p={p_perm:.4f} {p_stars(p_perm)}")

    # LMM
    for cond in CONDS:
        sub=df_a[df_a['cond']==cond].copy()
        sub['exp_str']=sub['exposed'].astype(str)
        for dv_key, dv_label in zip(ALL_DV_KEYS, ALL_DV_LABELS):
            if dv_key not in sub.columns: continue
            sd=sub[['pid','exp_str','task',dv_key]].dropna().copy().rename(columns={dv_key:'y'})
            if len(sd)<10 or sd['exp_str'].nunique()<2: continue
            for mname, formula in [
                (f'A1 ({dim_key})', 'y ~ C(exp_str, Treatment(reference="0"))'),
                (f'A2 ({dim_key})', 'y ~ C(exp_str, Treatment(reference="0")) + C(task)')]:
                try:
                    mdf=smf.mixedlm(formula,sd,groups=sd['pid']).fit(reml=True)
                    for pn,coef in mdf.fe_params.items():
                        if 'exp_str' not in pn: continue
                        pval=mdf.pvalues[pn]; ci=mdf.conf_int().loc[pn]
                        all_lmm_rows.append({
                            'dim':dim_key,'dim_label':dim['label'],'comp':'A','cond':cond,
                            'dv':dv_label,'dv_key':dv_key,'model':mname,
                            'contrast':f"{dim['group_labels'][1]} vs {dim['group_labels'][0]}",
                            'coef':coef,'se':mdf.bse[pn],'ci_lo':ci[0],'ci_hi':ci[1],
                            'z':mdf.tvalues[pn],'p':pval,
                            'n_obs':int(mdf.nobs),'n_groups':sd['pid'].nunique()})
                except: pass

    # ───────────────────────────────────────────────────────────────
    # COMPARISON Ab: unexposed-type tasks only, prior exposure vs none
    # ───────────────────────────────────────────────────────────────
    print(f"\n  --- Comparison Ab: {dim['unexposed_label']} tasks only, prior exposure ---")
    rows_ab = []
    for _, row in df.iterrows():
        pid=row.name; cond=row['xai_condition']; order=row['tasks_order']
        if not isinstance(order,list) or len(order)!=4: continue
        seen_exposed = False
        for pos, task in enumerate(order):
            if task in unexposed_set:
                flag = 1 if seen_exposed else 0
                entry = {'pid':pid,'cond':cond,'task':task,'task_disp':DISP[task],
                         'position':pos+1,'prior_exposed':flag,'dim':dim_key,'comp':'Ab'}
                for dv_base in DVS:
                    col=f"{dv_base}_{task}"
                    entry[dv_base]=row[col] if col in df.columns and pd.notna(row.get(col)) else np.nan
                mc=f"mean_at_{task}"
                entry['mean_at']=row[mc] if mc in df.columns and pd.notna(row.get(mc)) else np.nan
                rows_ab.append(entry)
            if task in exposed_set: seen_exposed=True
    df_ab = pd.DataFrame(rows_ab)

    for cond in CONDS:
        sub=df_ab[df_ab['cond']==cond]
        ct=sub.groupby('prior_exposed').size()
        print(f"    {cond}: " + ", ".join(f"prior_exposed={k}: {v}" for k,v in ct.items()))

    # MW
    for cond in CONDS:
        sub=df_ab[df_ab['cond']==cond]
        for dv_key, dv_label in zip(ALL_DV_KEYS, ALL_DV_LABELS):
            if dv_key not in sub.columns: continue
            g0=sub[sub['prior_exposed']==0][dv_key].dropna().values
            g1=sub[sub['prior_exposed']==1][dv_key].dropna().values
            if len(g0)<3 or len(g1)<3: continue
            u,p_mw=stats.mannwhitneyu(g0,g1,alternative='two-sided')
            d=cohens_d(g0,g1); p_perm=permutation_test(g0,g1)
            all_mw_rows.append({
                'dim':dim_key,'dim_label':dim['label'],'comp':'Ab','cond':cond,
                'dv':dv_label,'dv_key':dv_key,
                'mean_0':np.mean(g0),'n_0':len(g0),'mean_1':np.mean(g1),'n_1':len(g1),
                'U':u,'p_mw':p_mw,'p_perm':p_perm,'d':d,
                'label_0':f"No prior {dim['exposed_label'].lower()}",
                'label_1':f"Prior {dim['exposed_label'].lower()}"})
            print(f"    {cond:5s} Ab {dv_label:22s}: {np.mean(g0):.3f} vs {np.mean(g1):.3f} d={d:+.3f} p={p_perm:.4f} {p_stars(p_perm)}")

    # LMM Ab
    for cond in CONDS:
        sub=df_ab[df_ab['cond']==cond].copy()
        sub['pe_str']=sub['prior_exposed'].astype(str)
        for dv_key, dv_label in zip(ALL_DV_KEYS, ALL_DV_LABELS):
            if dv_key not in sub.columns: continue
            sd=sub[['pid','pe_str','task',dv_key]].dropna().copy().rename(columns={dv_key:'y'})
            if len(sd)<10 or sd['pe_str'].nunique()<2: continue
            for mname, formula in [
                (f'Ab1 ({dim_key})', 'y ~ C(pe_str, Treatment(reference="0"))'),
                (f'Ab2 ({dim_key})', 'y ~ C(pe_str, Treatment(reference="0")) + C(task)')]:
                try:
                    mdf=smf.mixedlm(formula,sd,groups=sd['pid']).fit(reml=True)
                    for pn,coef in mdf.fe_params.items():
                        if 'pe_str' not in pn: continue
                        pval=mdf.pvalues[pn]; ci=mdf.conf_int().loc[pn]
                        all_lmm_rows.append({
                            'dim':dim_key,'dim_label':dim['label'],'comp':'Ab','cond':cond,
                            'dv':dv_label,'dv_key':dv_key,'model':mname,
                            'contrast':f"Prior {dim['exposed_label'].lower()} vs none",
                            'coef':coef,'se':mdf.bse[pn],'ci_lo':ci[0],'ci_hi':ci[1],
                            'z':mdf.tvalues[pn],'p':pval,
                            'n_obs':int(mdf.nobs),'n_groups':sd['pid'].nunique()})
                except: pass

    # ───────────────────────────────────────────────────────────────
    # COMPARISON B: first-two profile (Both Unexposed vs Both Exposed)
    # ───────────────────────────────────────────────────────────────
    print(f"\n  --- Comparison B: First-two profile ---")
    def get_profile(order):
        t1,t2=order[0],order[1]
        e1=t1 in exposed_set; e2=t2 in exposed_set
        if not e1 and not e2: return 'Both Unexposed'
        if e1 and e2: return 'Both Exposed'
        return 'Mixed'

    df[f'profile_{dim_key}'] = df['tasks_order'].apply(get_profile)
    for cond in CONDS:
        sub=df[df['xai_condition']==cond]
        print(f"    {cond}: {sub[f'profile_{dim_key}'].value_counts().to_dict()}")

    df_b = df[df[f'profile_{dim_key}'].isin(['Both Unexposed','Both Exposed'])].copy()

    # Aggregated DVs
    for dv_base in DVS:
        cols=[f"{dv_base}_{t}" for t in ALL_TASKS]
        available=[c for c in cols if c in df_b.columns]
        if available: df_b[f"agg_{dv_base}"] = df_b[available].mean(axis=1)
    at_cols=[f"mean_at_{t}" for t in ALL_TASKS if f"mean_at_{t}" in df_b.columns]
    if at_cols: df_b['agg_mean_at'] = df_b[at_cols].mean(axis=1)

    agg_dvs = {f"agg_{k}":v for k,v in DVS.items()}
    agg_dvs['agg_mean_at'] = 'Mean Answer Time'
    agg_dv_base_map = {f"agg_{k}":k for k in DVS}
    agg_dv_base_map['agg_mean_at'] = 'mean_at'

    # MW on aggregated
    for cond in CONDS:
        sub=df_b[df_b['xai_condition']==cond]
        for agg_col, dv_label in agg_dvs.items():
            if agg_col not in sub.columns: continue
            g0=sub[sub[f'profile_{dim_key}']=='Both Unexposed'][agg_col].dropna().values
            g1=sub[sub[f'profile_{dim_key}']=='Both Exposed'][agg_col].dropna().values
            if len(g0)<3 or len(g1)<3: continue
            u,p_mw=stats.mannwhitneyu(g0,g1,alternative='two-sided')
            d=cohens_d(g0,g1); p_perm=permutation_test(g0,g1)
            all_mw_rows.append({
                'dim':dim_key,'dim_label':dim['label'],'comp':'B','cond':cond,
                'dv':dv_label,'dv_key':agg_dv_base_map.get(agg_col,agg_col),
                'mean_0':np.mean(g0),'n_0':len(g0),'mean_1':np.mean(g1),'n_1':len(g1),
                'U':u,'p_mw':p_mw,'p_perm':p_perm,'d':d,
                'label_0':dim['profile_labels']['Both Unexposed'],
                'label_1':dim['profile_labels']['Both Exposed']})
            print(f"    {cond:5s} B  {dv_label:22s}: {np.mean(g0):.3f} vs {np.mean(g1):.3f} d={d:+.3f} p={p_perm:.4f} {p_stars(p_perm)}")

    # LMM B (task-level)
    long_b = []
    for _, row in df_b.iterrows():
        pid=row.name; cond=row['xai_condition']; profile=row[f'profile_{dim_key}']
        for task in ALL_TASKS:
            entry={'pid':pid,'cond':cond,'task':task,'profile':profile}
            for dv_base in DVS:
                col=f"{dv_base}_{task}"
                entry[dv_base]=row[col] if col in df.columns and pd.notna(row.get(col)) else np.nan
            mc=f"mean_at_{task}"
            entry['mean_at']=row[mc] if mc in df.columns and pd.notna(row.get(mc)) else np.nan
            long_b.append(entry)
    df_b_long = pd.DataFrame(long_b)

    for cond in CONDS:
        sub=df_b_long[df_b_long['cond']==cond].copy()
        if sub['profile'].nunique()<2: continue
        for dv_key, dv_label in zip(ALL_DV_KEYS, ALL_DV_LABELS):
            if dv_key not in sub.columns: continue
            sd=sub[['pid','profile','task',dv_key]].dropna().copy().rename(columns={dv_key:'y'})
            if len(sd)<10 or sd['profile'].nunique()<2: continue
            for mname, formula in [
                (f'B1 ({dim_key})', 'y ~ C(profile, Treatment(reference="Both Unexposed"))'),
                (f'B2 ({dim_key})', 'y ~ C(profile, Treatment(reference="Both Unexposed")) + C(task)')]:
                try:
                    mdf=smf.mixedlm(formula,sd,groups=sd['pid']).fit(reml=True)
                    for pn,coef in mdf.fe_params.items():
                        if 'profile' not in pn: continue
                        pval=mdf.pvalues[pn]; ci=mdf.conf_int().loc[pn]
                        all_lmm_rows.append({
                            'dim':dim_key,'dim_label':dim['label'],'comp':'B','cond':cond,
                            'dv':dv_label,'dv_key':dv_key,'model':mname,
                            'contrast':f"{dim['profile_labels']['Both Exposed']} vs {dim['profile_labels']['Both Unexposed']}",
                            'coef':coef,'se':mdf.bse[pn],'ci_lo':ci[0],'ci_hi':ci[1],
                            'z':mdf.tvalues[pn],'p':pval,
                            'n_obs':int(mdf.nobs),'n_groups':sd['pid'].nunique()})
                except: pass

# ═══════════════════════════════════════════════════════════════════════
# CORRECTIONS
# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*70+"\nAPPLYING CORRECTIONS\n"+"="*70)

# MW: family = per dim × comp × cond
for dim_key in DIMS:
    for comp in ['A','Ab','B']:
        for cond in CONDS:
            family=[r for r in all_mw_rows
                    if r['dim']==dim_key and r['comp']==comp and r['cond']==cond
                    and not np.isnan(r['p_perm'])]
            if len(family)<2: continue
            pvals=[(all_mw_rows.index(r),r['p_perm']) for r in family]
            hm=holm_bonferroni(pvals); bh=benjamini_hochberg(pvals)
            for idx in hm: all_mw_rows[idx]['p_holm']=hm[idx]; all_mw_rows[idx]['p_bh']=bh[idx]
for r in all_mw_rows:
    r.setdefault('p_holm',r.get('p_perm',np.nan)); r.setdefault('p_bh',r.get('p_perm',np.nan))

# LMM: family = per dim × comp × model × cond
models_seen = set(r['model'] for r in all_lmm_rows)
for dim_key in DIMS:
    for comp in ['A','Ab','B']:
        for mname in models_seen:
            for cond in CONDS:
                family=[r for r in all_lmm_rows
                        if r['dim']==dim_key and r['comp']==comp and r['model']==mname
                        and r['cond']==cond and not np.isnan(r['p'])]
                if len(family)<2: continue
                pvals=[(all_lmm_rows.index(r),r['p']) for r in family]
                hm=holm_bonferroni(pvals); bh=benjamini_hochberg(pvals)
                for idx in hm: all_lmm_rows[idx]['p_holm']=hm[idx]; all_lmm_rows[idx]['p_bh']=bh[idx]
for r in all_lmm_rows:
    r.setdefault('p_holm',r.get('p',np.nan)); r.setdefault('p_bh',r.get('p',np.nan))

pd.DataFrame(all_mw_rows).to_csv(f"{OUTPUT_DIR}/v10_mw_all.csv", index=False)
pd.DataFrame(all_lmm_rows).to_csv(f"{OUTPUT_DIR}/v10_lmm_all.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════
# SIGNIFICANT RESULTS
# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*70+"\nSIGNIFICANT RESULTS (p_BH < .10)\n"+"="*70)
print("\n  MW:")
for r in all_mw_rows:
    if r['p_bh']<0.10:
        print(f"    {r['dim_label']:15s} | {r['comp']:2s} | {r['cond']:5s} | {r['dv']:22s}: d={r['d']:+.3f}, p_perm={r['p_perm']:.4f}, p_holm={r['p_holm']:.4f}, p_bh={r['p_bh']:.4f}")
print("\n  LMM:")
for r in all_lmm_rows:
    if r['p_bh']<0.10:
        print(f"    {r['dim_label']:15s} | {r['comp']:2s} | {r['cond']:5s} | {r['dv']:22s} | {r['model']:18s}: β={r['coef']:+.4f}, p={r['p']:.4f}, p_holm={r['p_holm']:.4f}, p_bh={r['p_bh']:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*70+"\nGENERATING VISUALIZATIONS\n"+"="*70)

def make_bar_fig(data_dict, title, fname, dv_keys, dv_labels, ylims_dict):
    """data_dict: {group_label: {dv_key: array}}. Two groups."""
    groups = list(data_dict.keys())
    n_dvs = len(dv_keys)
    fig, axes = plt.subplots(1, n_dvs, figsize=(2.8*n_dvs, 5))
    if n_dvs==1: axes=[axes]
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.03)
    colors = ['#2E86AB','#E8533F']
    for ci, (dk, dl) in enumerate(zip(dv_keys, dv_labels)):
        ax=axes[ci]
        means,clo_l,chi_l=[],[],[]
        for gi, grp in enumerate(groups):
            vals=data_dict[grp].get(dk, np.array([]))
            if len(vals)>=2:
                m=np.nanmean(vals); ci_v=bootstrap_ci(vals)
                means.append(m); clo_l.append(m-ci_v[0]); chi_l.append(ci_v[1]-m)
            else: means.append(0); clo_l.append(0); chi_l.append(0)
        ax.bar(range(2),means,color=colors,alpha=0.8,width=0.55,edgecolor='white',linewidth=0.5)
        ax.errorbar(range(2),means,yerr=[clo_l,chi_l],fmt='none',color='#333',capsize=4,linewidth=1)
        ax.set_xticks(range(2))
        ax.set_xticklabels([g.replace(' ','\n') for g in groups], fontsize=6.5)
        ax.set_title(dl, fontsize=9, fontweight='bold')
        for xi,grp in enumerate(groups):
            n_val=len(data_dict[grp].get(dk,np.array([])))
            ylim_cur=ax.get_ylim()
            ax.text(xi,means[xi]+chi_l[xi]+0.02*(ylim_cur[1]-ylim_cur[0]),
                    f'n={n_val}',ha='center',fontsize=6,color='#777')
        yl=ylims_dict.get(dk)
        if yl: ax.set_ylim(yl)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{fname}",dpi=200,bbox_inches='tight'); plt.close()
    print(f"  -> {fname}")

def make_dist_fig(data_dict, title, fname, dv_keys, dv_labels, ylims_dict):
    groups=list(data_dict.keys()); n_dvs=len(dv_keys); colors=['#2E86AB','#E8533F']
    fig, axes = plt.subplots(2, n_dvs, figsize=(2.8*n_dvs, 9))
    if n_dvs==1: axes=axes.reshape(2,1)
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)
    for ci,(dk,dl) in enumerate(zip(dv_keys,dv_labels)):
        ax_v=axes[0,ci]; ax_h=axes[1,ci]
        gd=[data_dict[g].get(dk,np.array([])) for g in groups]
        gd=[g[~np.isnan(g)] if len(g)>0 else g for g in gd]
        valid=[i for i,g in enumerate(gd) if len(g)>=2]
        if len(valid)<2: ax_v.set_visible(False); ax_h.set_visible(False); continue
        gd_v=[gd[i] for i in valid]; gl_v=[groups[i] for i in valid]; gc_v=[colors[i] for i in valid]
        vp=ax_v.violinplot(gd_v,positions=range(len(gd_v)),widths=0.7,
                           showmeans=True,showmedians=True,showextrema=False)
        for i,pc in enumerate(vp['bodies']): pc.set_facecolor(gc_v[i]); pc.set_alpha(0.4)
        for pn in ['cmeans','cmedians']:
            if pn in vp: vp[pn].set_edgecolor('#333')
        rng=np.random.default_rng(42)
        for i,vals in enumerate(gd_v):
            jx=rng.normal(0,0.06,len(vals))
            ax_v.scatter(np.full(len(vals),i)+jx,vals,color=gc_v[i],alpha=0.3,s=8,edgecolors='none')
        ax_v.set_xticks(range(len(gd_v))); ax_v.set_xticklabels(gl_v,fontsize=6)
        ax_v.set_title(dl,fontsize=9,fontweight='bold')
        yl=ylims_dict.get(dk)
        if yl: ax_v.set_ylim(yl)
        ax_v.spines['top'].set_visible(False); ax_v.spines['right'].set_visible(False)
        all_v=np.concatenate(gd_v); bins=np.linspace(np.nanmin(all_v),np.nanmax(all_v),20)
        for i,vals in enumerate(gd_v):
            ax_h.hist(vals,bins=bins,density=True,alpha=0.35,color=gc_v[i],edgecolor='white',
                      linewidth=0.3,label=gl_v[i])
            if len(vals)>=5:
                try:
                    kde=stats.gaussian_kde(vals); xk=np.linspace(bins[0],bins[-1],200)
                    ax_h.plot(xk,kde(xk),color=gc_v[i],linewidth=1.5,alpha=0.8)
                except: pass
        ax_h.set_xlabel(dl,fontsize=7); ax_h.set_ylabel('Density',fontsize=7)
        ax_h.spines['top'].set_visible(False); ax_h.spines['right'].set_visible(False)
        if ci==0: ax_h.legend(fontsize=5)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{fname}",dpi=200,bbox_inches='tight'); plt.close()
    print(f"  -> {fname}")

def make_forest_fig(lmm_sub, title, fname, ref_label):
    if not lmm_sub: return
    fig,ax=plt.subplots(figsize=(10,max(3,len(lmm_sub)*0.5+1)))
    fig.suptitle(title,fontsize=11,fontweight='bold')
    for i,r in enumerate(lmm_sub):
        color='#333'; alpha=0.4
        if r['p_bh']<0.05: color='#2E86AB'; alpha=1.0
        elif r['p_bh']<0.10: color='#E8963F'; alpha=0.8
        ax.barh(i,r['coef'],color=color,alpha=alpha,height=0.5,edgecolor='white',linewidth=0.3)
        ax.plot([r['ci_lo'],r['ci_hi']],[i,i],color='#333',linewidth=1.5,solid_capstyle='round')
        sig_txt=f"p={r['p']:.3f}"
        if r['p_bh']<0.10: sig_txt+=f" (BH:{p_stars(r['p_bh'])})"
        ax.text(max(r['ci_hi'],r['coef'])+abs(r.get('coef',0))*0.05+0.01,i,
                sig_txt,va='center',fontsize=7,color='#555')
    ax.set_yticks(range(len(lmm_sub)))
    ax.set_yticklabels([r['dv'] for r in lmm_sub],fontsize=8)
    ax.axvline(0,color='black',linewidth=0.8)
    ax.set_xlabel(f'β (vs {ref_label})')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0,0,1,0.90])
    fig.savefig(f"{OUTPUT_DIR}/{fname}",dpi=200,bbox_inches='tight'); plt.close()
    print(f"  -> {fname}")

# ── Generate all bar + distribution + forest figures ──────────────────
# We need to rebuild the task-level dataframes for plotting
# Reuse from above: df_a was overwritten per dim. Rebuild.
for dim_key, dim in DIMS.items():
    exposed_set=dim['exposed_tasks']; unexposed_set=dim['unexposed_tasks']
    dim_short = dim_key[:5]

    # Rebuild Comp A data
    rows_a=[]
    for _,row in df.iterrows():
        pid=row.name;cond=row['xai_condition'];order=row['tasks_order']
        if not isinstance(order,list) or len(order)!=4: continue
        seen=False
        for pos,task in enumerate(order):
            cur_exp=task in exposed_set; flag=1 if (cur_exp or seen) else 0
            entry={'pid':pid,'cond':cond,'task':task,'exposed':flag}
            for dk in ALL_DV_KEYS:
                col=f"{dk}_{task}" if dk!='mean_at' else f"mean_at_{task}"
                entry[dk]=row[col] if col in df.columns and pd.notna(row.get(col)) else np.nan
            rows_a.append(entry)
            if cur_exp: seen=True
    df_a_plot=pd.DataFrame(rows_a)

    # Rebuild Comp Ab data
    rows_ab=[]
    for _,row in df.iterrows():
        pid=row.name;cond=row['xai_condition'];order=row['tasks_order']
        if not isinstance(order,list) or len(order)!=4: continue
        seen=False
        for pos,task in enumerate(order):
            if task in unexposed_set:
                flag=1 if seen else 0
                entry={'pid':pid,'cond':cond,'task':task,'prior_exposed':flag}
                for dk in ALL_DV_KEYS:
                    col=f"{dk}_{task}" if dk!='mean_at' else f"mean_at_{task}"
                    entry[dk]=row[col] if col in df.columns and pd.notna(row.get(col)) else np.nan
                rows_ab.append(entry)
            if task in exposed_set: seen=True
    df_ab_plot=pd.DataFrame(rows_ab)

    # Rebuild Comp B data
    df_b_plot=df[df[f'profile_{dim_key}'].isin(['Both Unexposed','Both Exposed'])].copy()
    for dv_base in DVS:
        cols=[f"{dv_base}_{t}" for t in ALL_TASKS]
        av=[c for c in cols if c in df_b_plot.columns]
        if av: df_b_plot[f"agg_{dv_base}"]=df_b_plot[av].mean(axis=1)
    atc=[f"mean_at_{t}" for t in ALL_TASKS if f"mean_at_{t}" in df_b_plot.columns]
    if atc: df_b_plot['agg_mean_at']=df_b_plot[atc].mean(axis=1)

    for cond in CONDS:
        cond_s=cond.replace('+','_')

        # Comp A bars + dist
        sub=df_a_plot[df_a_plot['cond']==cond]
        dd={dim['group_labels'][0]:{},dim['group_labels'][1]:{}}
        for dk in ALL_DV_KEYS:
            if dk not in sub.columns: continue
            dd[dim['group_labels'][0]][dk]=sub[sub['exposed']==0][dk].dropna().values
            dd[dim['group_labels'][1]][dk]=sub[sub['exposed']==1][dk].dropna().values
        make_bar_fig(dd,
            f"Comp. A: {dim['label']} Exposure (all tasks) — {cond}\n95% Bootstrap CI",
            f"fig_A_{dim_short}_bars_{cond_s}.png", ALL_DV_KEYS, ALL_DV_LABELS, YLIMS)
        make_dist_fig(dd,
            f"Comp. A: {dim['label']} Exposure — {cond}\nViolin + histogram",
            f"fig_A_{dim_short}_dist_{cond_s}.png", ALL_DV_KEYS, ALL_DV_LABELS, YLIMS)

        # Comp Ab bars + dist
        sub=df_ab_plot[df_ab_plot['cond']==cond]
        l0=f"No prior {dim['exposed_label'].lower()}"
        l1=f"Prior {dim['exposed_label'].lower()}"
        dd={l0:{},l1:{}}
        for dk in ALL_DV_KEYS:
            if dk not in sub.columns: continue
            dd[l0][dk]=sub[sub['prior_exposed']==0][dk].dropna().values
            dd[l1][dk]=sub[sub['prior_exposed']==1][dk].dropna().values
        task_type=dim['unexposed_label'].lower()
        make_bar_fig(dd,
            f"Comp. Ab: Prior {dim['label']} on {task_type} tasks — {cond}\n95% Bootstrap CI",
            f"fig_Ab_{dim_short}_bars_{cond_s}.png", ALL_DV_KEYS, ALL_DV_LABELS, YLIMS)
        make_dist_fig(dd,
            f"Comp. Ab: Prior {dim['label']} on {task_type} tasks — {cond}\nViolin + histogram",
            f"fig_Ab_{dim_short}_dist_{cond_s}.png", ALL_DV_KEYS, ALL_DV_LABELS, YLIMS)

        # Comp B bars + dist
        sub=df_b_plot[df_b_plot['xai_condition']==cond]
        l0b=dim['profile_labels']['Both Unexposed']
        l1b=dim['profile_labels']['Both Exposed']
        agg_keys=[f"agg_{k}" for k in DVS]+['agg_mean_at']
        agg_labs=list(DVS.values())+['Mean Answer Time']
        agg_base_keys=[k for k in DVS]+['mean_at']
        dd={l0b:{},l1b:{}}
        for ak,bk in zip(agg_keys,agg_base_keys):
            if ak not in sub.columns: continue
            dd[l0b][bk]=sub[sub[f'profile_{dim_key}']=='Both Unexposed'][ak].dropna().values
            dd[l1b][bk]=sub[sub[f'profile_{dim_key}']=='Both Exposed'][ak].dropna().values
        make_bar_fig(dd,
            f"Comp. B: First-two {dim['label']} profile — {cond}\nAggregated DVs | 95% Bootstrap CI",
            f"fig_B_{dim_short}_bars_{cond_s}.png", agg_base_keys, agg_labs, YLIMS)
        make_dist_fig(dd,
            f"Comp. B: First-two {dim['label']} profile — {cond}\nViolin + histogram",
            f"fig_B_{dim_short}_dist_{cond_s}.png", agg_base_keys, agg_labs, YLIMS)

    # LMM forest plots — pick model 2 (with task covariate)
    for comp in ['A','Ab','B']:
        model_suffix = '2'
        model_match = f'{comp}{model_suffix} ({dim_key})'
        for cond in CONDS:
            cond_s=cond.replace('+','_')
            lmm_sub=[r for r in all_lmm_rows
                     if r['dim']==dim_key and r['comp']==comp and r['model']==model_match and r['cond']==cond]
            if not lmm_sub: continue
            ref=lmm_sub[0]['contrast'].split(' vs ')[-1] if lmm_sub else '?'
            make_forest_fig(lmm_sub,
                f"LMM: Comp. {comp} {dim['label']} — {cond}\n{model_match}",
                f"fig_lmm_{comp}_{dim_short}_{cond_s}.png", ref)

print("\n"+"="*70+"\nCOMPLETE\n"+"="*70)
print("""
CSVs:
  v10_mw_all.csv       All MW/permutation results (with corrections)
  v10_lmm_all.csv      All LMM results (with corrections)

Figures (per dimension × comparison × condition):
  fig_A_*_bars_*.png / dist_*.png       Comp A: all tasks, exposed vs never
  fig_Ab_*_bars_*.png / dist_*.png      Comp Ab: unexposed tasks, prior vs none
  fig_B_*_bars_*.png / dist_*.png       Comp B: first-two profile
  fig_lmm_*_*.png                        LMM forest plots (model with task covariate)

Y-axis: Accuracy/Reliance/Over-reliance/Under-reliance fixed at 0–1.
Trust/Cognitive Load/Mean AT: shared scale across all figures.
""")

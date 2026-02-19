"""
Task Order Effects Analysis v10b — Exposure to Time Pressure & Difficulty
==========================================================================
Two dimensions studied independently:
  - TIME PRESSURE: Strong (easy_strong, hard_strong) vs Low (easy_mild, hard_mild)
  - DIFFICULTY:    High (hard_mild, hard_strong) vs Low (easy_mild, easy_strong)

Four comparisons per dimension:
  A:  Task-level — currently/previously exposed vs never exposed
  Ab: Task-level — restricted to UNEXPOSED-type tasks, comparing prior exposure vs none
  Ac: Task-level — restricted to EXPOSED-type tasks, comparing prior unexposed experience vs none
  B:  Participant-level — first two tasks both X vs both Y (mixed excluded)

Statistics: Mann-Whitney + permutation + LMM (two variants). Corrections: Holm + BH.
Matched y-axis scales: Accuracy/Reliance/Over-reliance/Under-reliance on 0–1.
No distribution plots.
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

DIMS = {
    'pressure': {
        'label': 'Time Pressure',
        'exposed_tasks': {'easy_strong','hard_strong'},
        'unexposed_tasks': {'easy_mild','hard_mild'},
        'exposed_label': 'Strong pressure',
        'unexposed_label': 'Low pressure',
        'group_labels': {0:'Never Strong', 1:'Exposed to Strong'},
        'profile_labels': {'Both Unexposed':'Both Low press.', 'Both Exposed':'Both Strong press.'},
    },
    'difficulty': {
        'label': 'Difficulty',
        'exposed_tasks': {'hard_mild','hard_strong'},
        'unexposed_tasks': {'easy_mild','easy_strong'},
        'exposed_label': 'High difficulty',
        'unexposed_label': 'Low difficulty',
        'group_labels': {0:'Never High diff.', 1:'Exposed to High diff.'},
        'profile_labels': {'Both Unexposed':'Both Low diff.', 'Both Exposed':'Both High diff.'},
    }
}

DVS = {'score':'Accuracy','reliance':'Reliance','overreliance':'Over-reliance',
       'underreliance':'Under-reliance','trust':'Trust','cogload':'Cognitive Load'}
CONDS = ['H','H+AI']
ALL_DV_KEYS = list(DVS.keys()) + ['mean_at']
ALL_DV_LABELS = list(DVS.values()) + ['Mean Answer Time']

YLIMS = {'score':(0,1),'reliance':(0,1),'overreliance':(0,1),'underreliance':(0,1),
         'trust':None,'cogload':None,'mean_at':None}

# ── Helpers ───────────────────────────────────────────────────────────
def parse_col(val):
    if isinstance(val, str):
        try: return ast.literal_eval(val)
        except: return val
    return val

def cohens_d(g1,g2):
    n1,n2=len(g1),len(g2)
    if n1<2 or n2<2: return np.nan
    p=np.sqrt(((n1-1)*np.var(g1,ddof=1)+(n2-1)*np.var(g2,ddof=1))/(n1+n2-2))
    return (np.mean(g1)-np.mean(g2))/p if p>0 else 0.0

def bootstrap_ci(data, n_boot=10000, ci=0.95):
    rng=np.random.default_rng(42); data=np.array(data,dtype=float); data=data[~np.isnan(data)]
    if len(data)<3: return (np.nan,np.nan)
    boot=[np.mean(rng.choice(data,size=len(data),replace=True)) for _ in range(n_boot)]
    a=(1-ci)/2; return (np.percentile(boot,a*100),np.percentile(boot,(1-a)*100))

def permutation_test(g1,g2,n_perm=10000):
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
    n=len(pvals);s=sorted(pvals,key=lambda x:x[1]);c={};mx=0
    for rank,(idx,p) in enumerate(s):
        adj=min(p*(n-rank),1.0);adj=max(adj,mx);mx=adj;c[idx]=adj
    return c

def benjamini_hochberg(pvals):
    n=len(pvals);s=sorted(pvals,key=lambda x:x[1],reverse=True);c={};mn=1.0
    for rd,(idx,p) in enumerate(s):
        ra=n-rd;adj=min(p*n/ra,1.0);adj=min(adj,mn);mn=adj;c[idx]=adj
    return c

def correct_family(rows, p_col):
    pvals=[(i,r[p_col]) for i,r in enumerate(rows) if not np.isnan(r[p_col])]
    if len(pvals)<2:
        for r in rows: r['p_holm']=r.get(p_col,np.nan);r['p_bh']=r.get(p_col,np.nan)
        return rows
    hm=holm_bonferroni(pvals);bh=benjamini_hochberg(pvals)
    for i,r in enumerate(rows):
        r['p_holm']=hm.get(i,r.get(p_col,np.nan));r['p_bh']=bh.get(i,r.get(p_col,np.nan))
    return rows

def run_mw(sub, flag_col, dv_keys, dv_labels, meta):
    """Run MW + permutation for binary flag_col. Returns list of row dicts."""
    rows=[]
    for dk,dl in zip(dv_keys,dv_labels):
        if dk not in sub.columns: continue
        g0=sub[sub[flag_col]==0][dk].dropna().values
        g1=sub[sub[flag_col]==1][dk].dropna().values
        if len(g0)<3 or len(g1)<3: continue
        u,p_mw=stats.mannwhitneyu(g0,g1,alternative='two-sided')
        d=cohens_d(g0,g1);p_perm=permutation_test(g0,g1)
        r={**meta,'dv':dl,'dv_key':dk,
           'mean_0':np.mean(g0),'n_0':len(g0),'mean_1':np.mean(g1),'n_1':len(g1),
           'U':u,'p_mw':p_mw,'p_perm':p_perm,'d':d}
        rows.append(r)
        print(f"    {meta['cond']:5s} {meta['comp']:2s} {dl:22s}: {np.mean(g0):.3f}(n={len(g0)}) vs {np.mean(g1):.3f}(n={len(g1)}) d={d:+.3f} p={p_perm:.4f} {p_stars(p_perm)}")
    return rows

def run_lmm(sub, flag_col, dv_keys, dv_labels, meta, has_task_var=True):
    """Run LMM. Returns list of row dicts."""
    rows=[];sub=sub.copy();sub['flag']=sub[flag_col].astype(str)
    for dk,dl in zip(dv_keys,dv_labels):
        if dk not in sub.columns: continue
        sd=sub[['pid','flag','task',dk]].dropna().copy().rename(columns={dk:'y'})
        if len(sd)<10 or sd['flag'].nunique()<2: continue
        formulas=[(f"{meta['comp']}1 ({meta['dim']})",'y ~ C(flag, Treatment(reference="0"))')]
        if has_task_var and sd['task'].nunique()>1:
            formulas.append((f"{meta['comp']}2 ({meta['dim']})",
                             'y ~ C(flag, Treatment(reference="0")) + C(task)'))
        for mname,formula in formulas:
            try:
                mdf=smf.mixedlm(formula,sd,groups=sd['pid']).fit(reml=True)
                for pn,coef in mdf.fe_params.items():
                    if 'flag' not in pn: continue
                    pval=mdf.pvalues[pn];ci=mdf.conf_int().loc[pn]
                    rows.append({**meta,'dv':dl,'dv_key':dk,'model':mname,
                                 'contrast':meta.get('contrast',''),
                                 'coef':coef,'se':mdf.bse[pn],'ci_lo':ci[0],'ci_hi':ci[1],
                                 'z':mdf.tvalues[pn],'p':pval,
                                 'n_obs':int(mdf.nobs),'n_groups':sd['pid'].nunique()})
            except: pass
    return rows

# ── Load & prepare ────────────────────────────────────────────────────
print("="*70+"\nTASK ORDER EFFECTS v10b — PRESSURE & DIFFICULTY EXPOSURE\n"+"="*70)
df = pd.read_csv(DATA_PATH)
print(f"Total: {len(df)}")
df = df[df['xai_condition'].isin(CONDS)].copy()
print(f"H + H+AI: {len(df)}")
df['tasks_order'] = df['tasks_order'].apply(parse_col)

for t in ALL_TASKS:
    at_col=f"answer_times_{t}"
    if at_col in df.columns:
        df[f"mean_at_{t}"]=df[at_col].apply(
            lambda v: np.nanmean(np.array(parse_col(v),dtype=float))
            if isinstance(parse_col(v),(list,np.ndarray)) else np.nan)

# Global ylims for trust, cogload, mean_at
for dv_key in ['trust','cogload','mean_at']:
    all_v=[]
    for t in ALL_TASKS:
        col=f"{dv_key}_{t}" if dv_key!='mean_at' else f"mean_at_{t}"
        if col in df.columns: all_v.extend(df[col].dropna().values)
    if all_v:
        mn,mx=min(all_v),max(all_v);margin=(mx-mn)*0.15
        YLIMS[dv_key]=(max(0,mn-margin),mx+margin)

# ═══════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS LOOP
# ═══════════════════════════════════════════════════════════════════════
all_mw_rows=[]
all_lmm_rows=[]

for dim_key, dim in DIMS.items():
    print(f"\n{'='*70}\n  DIMENSION: {dim['label'].upper()}\n{'='*70}")
    exposed_set=dim['exposed_tasks'];unexposed_set=dim['unexposed_tasks']

    # ── Build task-level dataframes ───────────────────────────────
    rows_a,rows_ab,rows_ac=[],[],[]
    for _,row in df.iterrows():
        pid=row.name;cond=row['xai_condition'];order=row['tasks_order']
        if not isinstance(order,list) or len(order)!=4: continue
        seen=False
        seen_unexposed=False
        for pos,task in enumerate(order):
            cur_exp=task in exposed_set
            # Collect DV values
            entry_base={'pid':pid,'cond':cond,'task':task,'task_disp':DISP[task],'position':pos+1}
            for dk in ALL_DV_KEYS:
                col=f"{dk}_{task}" if dk!='mean_at' else f"mean_at_{task}"
                entry_base[dk]=row[col] if col in df.columns and pd.notna(row.get(col)) else np.nan

            # Comp A: all tasks
            flag_a=1 if (cur_exp or seen) else 0
            rows_a.append({**entry_base,'exposed':flag_a})

            # Comp Ab: unexposed tasks only, any position
            if task in unexposed_set:
                flag_ab=1 if seen else 0
                rows_ab.append({**entry_base,'prior_exposed':flag_ab})

            # Comp Ac: exposed tasks only — has participant seen an unexposed task before?
            if task in exposed_set:
                flag_ac=1 if seen_unexposed else 0
                rows_ac.append({**entry_base,'prior_unexposed':flag_ac})

            if cur_exp: seen=True
            if task in unexposed_set: seen_unexposed=True

    df_a=pd.DataFrame(rows_a)
    df_ab=pd.DataFrame(rows_ab)
    df_ac=pd.DataFrame(rows_ac)

    # ── Print distributions ───────────────────────────────────────
    print(f"\n  Comp A (all tasks):")
    for cond in CONDS:
        ct=df_a[df_a['cond']==cond].groupby('exposed').size()
        print(f"    {cond}: "+", ".join(f"exposed={k}: {v}" for k,v in ct.items()))
    print(f"\n  Comp Ab ({dim['unexposed_label']} tasks, any position):")
    for cond in CONDS:
        ct=df_ab[df_ab['cond']==cond].groupby('prior_exposed').size()
        print(f"    {cond}: "+", ".join(f"prior={k}: {v}" for k,v in ct.items()))
    print(f"\n  Comp Ac ({dim['exposed_label']} tasks, prior {dim['unexposed_label'].lower()}):")
    for cond in CONDS:
        if len(df_ac[df_ac['cond']==cond])>0:
            ct=df_ac[df_ac['cond']==cond].groupby('prior_unexposed').size()
            print(f"    {cond}: "+", ".join(f"prior_unexposed={k}: {v}" for k,v in ct.items()))
        else:
            print(f"    {cond}: no observations")

    # ── Comp B: first-two profile ─────────────────────────────────
    def get_profile(order):
        t1,t2=order[0],order[1]
        e1=t1 in exposed_set;e2=t2 in exposed_set
        if not e1 and not e2: return 'Both Unexposed'
        if e1 and e2: return 'Both Exposed'
        return 'Mixed'
    df[f'profile_{dim_key}']=df['tasks_order'].apply(get_profile)
    df_b=df[df[f'profile_{dim_key}'].isin(['Both Unexposed','Both Exposed'])].copy()
    # Aggregated DVs
    for dv_base in DVS:
        cols=[f"{dv_base}_{t}" for t in ALL_TASKS]
        av=[c for c in cols if c in df_b.columns]
        if av: df_b[f"agg_{dv_base}"]=df_b[av].mean(axis=1)
    atc=[f"mean_at_{t}" for t in ALL_TASKS if f"mean_at_{t}" in df_b.columns]
    if atc: df_b['agg_mean_at']=df_b[atc].mean(axis=1)

    print(f"\n  Comp B (first-two profile):")
    for cond in CONDS:
        sub=df_b[df_b['xai_condition']==cond]
        print(f"    {cond}: {sub[f'profile_{dim_key}'].value_counts().to_dict()}")

    # ── Run MW + LMM for each comparison ──────────────────────────
    for comp, df_comp, flag_col, comp_desc in [
        ('A',  df_a,  'exposed',          f"All tasks: exposed vs never"),
        ('Ab', df_ab, 'prior_exposed',    f"{dim['unexposed_label']} tasks: prior exposure vs none"),
        ('Ac', df_ac, 'prior_unexposed',f"{dim['exposed_label']} tasks: prior {dim['unexposed_label'].lower()} vs none"),
    ]:
        print(f"\n  --- {comp}: {comp_desc} ---")
        l0_label = {
            'A': dim['group_labels'][0],
            'Ab': f"No prior {dim['exposed_label'].lower()}",
            'Ac': f"Only {dim['exposed_label'].lower()} so far",
        }[comp]
        l1_label = {
            'A': dim['group_labels'][1],
            'Ab': f"Prior {dim['exposed_label'].lower()}",
            'Ac': f"Prior {dim['unexposed_label'].lower()}",
        }[comp]
        contrast_str = f"{l1_label} vs {l0_label}"

        for cond in CONDS:
            sub=df_comp[df_comp['cond']==cond]
            if len(sub)<6 or sub[flag_col].nunique()<2: continue
            meta={'dim':dim_key,'dim_label':dim['label'],'comp':comp,'cond':cond,
                  'label_0':l0_label,'label_1':l1_label,'contrast':contrast_str}
            all_mw_rows.extend(run_mw(sub,flag_col,ALL_DV_KEYS,ALL_DV_LABELS,meta))
            all_lmm_rows.extend(run_lmm(sub,flag_col,ALL_DV_KEYS,ALL_DV_LABELS,meta,has_task_var=True))

    # Comp B: MW on aggregated + LMM on task-level
    print(f"\n  --- B: First-two profile ---")
    agg_keys=[f"agg_{k}" for k in DVS]+['agg_mean_at']
    agg_labs=list(DVS.values())+['Mean Answer Time']
    agg_base=[k for k in DVS]+['mean_at']

    for cond in CONDS:
        sub=df_b[df_b['xai_condition']==cond].copy()
        if sub[f'profile_{dim_key}'].nunique()<2: continue
        # Create binary flag: 0=Both Unexposed, 1=Both Exposed
        sub['b_flag']=(sub[f'profile_{dim_key}']=='Both Exposed').astype(int)
        l0=dim['profile_labels']['Both Unexposed']
        l1=dim['profile_labels']['Both Exposed']
        meta={'dim':dim_key,'dim_label':dim['label'],'comp':'B','cond':cond,
              'label_0':l0,'label_1':l1,'contrast':f"{l1} vs {l0}"}
        # MW on aggregated
        for ak,al,bk in zip(agg_keys,agg_labs,agg_base):
            if ak not in sub.columns: continue
            g0=sub[sub['b_flag']==0][ak].dropna().values
            g1=sub[sub['b_flag']==1][ak].dropna().values
            if len(g0)<3 or len(g1)<3: continue
            u,p_mw=stats.mannwhitneyu(g0,g1,alternative='two-sided')
            d=cohens_d(g0,g1);p_perm=permutation_test(g0,g1)
            all_mw_rows.append({**meta,'dv':al,'dv_key':bk,
                                'mean_0':np.mean(g0),'n_0':len(g0),'mean_1':np.mean(g1),'n_1':len(g1),
                                'U':u,'p_mw':p_mw,'p_perm':p_perm,'d':d})
            print(f"    {cond:5s} B  {al:22s}: {np.mean(g0):.3f}(n={len(g0)}) vs {np.mean(g1):.3f}(n={len(g1)}) d={d:+.3f} p={p_perm:.4f} {p_stars(p_perm)}")

        # LMM on task-level
        long_b=[]
        for _,row in sub.iterrows():
            pid=row.name
            for task in ALL_TASKS:
                entry={'pid':pid,'cond':cond,'task':task,'b_flag':row['b_flag']}
                for dk in ALL_DV_KEYS:
                    col=f"{dk}_{task}" if dk!='mean_at' else f"mean_at_{task}"
                    entry[dk]=row[col] if col in df.columns and pd.notna(row.get(col)) else np.nan
                long_b.append(entry)
        df_bl=pd.DataFrame(long_b)
        all_lmm_rows.extend(run_lmm(df_bl,'b_flag',ALL_DV_KEYS,ALL_DV_LABELS,meta,has_task_var=True))

# ═══════════════════════════════════════════════════════════════════════
# CORRECTIONS
# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*70+"\nAPPLYING CORRECTIONS\n"+"="*70)

for dim_key in DIMS:
    for comp in ['A','Ab','Ac','B']:
        for cond in CONDS:
            family=[r for r in all_mw_rows
                    if r['dim']==dim_key and r['comp']==comp and r['cond']==cond
                    and not np.isnan(r['p_perm'])]
            if len(family)<2: continue
            pvals=[(all_mw_rows.index(r),r['p_perm']) for r in family]
            hm=holm_bonferroni(pvals);bh=benjamini_hochberg(pvals)
            for idx in hm: all_mw_rows[idx]['p_holm']=hm[idx];all_mw_rows[idx]['p_bh']=bh[idx]
for r in all_mw_rows:
    r.setdefault('p_holm',r.get('p_perm',np.nan));r.setdefault('p_bh',r.get('p_perm',np.nan))

models_seen=set(r['model'] for r in all_lmm_rows)
for dim_key in DIMS:
    for comp in ['A','Ab','Ac','B']:
        for mname in models_seen:
            for cond in CONDS:
                family=[r for r in all_lmm_rows
                        if r['dim']==dim_key and r['comp']==comp and r['model']==mname
                        and r['cond']==cond and not np.isnan(r['p'])]
                if len(family)<2: continue
                pvals=[(all_lmm_rows.index(r),r['p']) for r in family]
                hm=holm_bonferroni(pvals);bh=benjamini_hochberg(pvals)
                for idx in hm: all_lmm_rows[idx]['p_holm']=hm[idx];all_lmm_rows[idx]['p_bh']=bh[idx]
for r in all_lmm_rows:
    r.setdefault('p_holm',r.get('p',np.nan));r.setdefault('p_bh',r.get('p',np.nan))

pd.DataFrame(all_mw_rows).to_csv(f"{OUTPUT_DIR}/v10b_mw_all.csv",index=False)
pd.DataFrame(all_lmm_rows).to_csv(f"{OUTPUT_DIR}/v10b_lmm_all.csv",index=False)

# ═══════════════════════════════════════════════════════════════════════
# SIGNIFICANT RESULTS
# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*70+"\nSIGNIFICANT RESULTS (p_BH < .10)\n"+"="*70)
print("\n  MW:")
for r in sorted(all_mw_rows, key=lambda x:(x['dim'],x['comp'],x['cond'])):
    if r['p_bh']<0.10:
        print(f"    {r['dim_label']:15s} | {r['comp']:2s} | {r['cond']:5s} | {r['dv']:22s}: d={r['d']:+.3f}, p={r['p_perm']:.4f}, p_H={r['p_holm']:.4f}, p_BH={r['p_bh']:.4f}")
print("\n  LMM:")
for r in sorted(all_lmm_rows, key=lambda x:(x['dim'],x['comp'],x['cond'])):
    if r['p_bh']<0.10:
        print(f"    {r['dim_label']:15s} | {r['comp']:2s} | {r['cond']:5s} | {r['dv']:22s} | {r['model']:18s}: β={r['coef']:+.4f}, p={r['p']:.4f}, p_H={r['p_holm']:.4f}, p_BH={r['p_bh']:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATIONS — Bar charts + LMM forest plots only
# ═══════════════════════════════════════════════════════════════════════
print("\n"+"="*70+"\nGENERATING VISUALIZATIONS\n"+"="*70)

def make_bar_fig(data_dict, title, fname, dv_keys, dv_labels, ylims_dict):
    groups=list(data_dict.keys());n_dvs=len(dv_keys);colors=['#2E86AB','#E8533F']
    fig,axes=plt.subplots(1,n_dvs,figsize=(2.8*n_dvs,5))
    if n_dvs==1: axes=[axes]
    fig.suptitle(title,fontsize=11,fontweight='bold',y=1.03)
    for ci,(dk,dl) in enumerate(zip(dv_keys,dv_labels)):
        ax=axes[ci]
        means,clo_l,chi_l=[],[],[]
        for gi,grp in enumerate(groups):
            vals=data_dict[grp].get(dk,np.array([]))
            if len(vals)>=2:
                m=np.nanmean(vals);ci_v=bootstrap_ci(vals)
                means.append(m);clo_l.append(m-ci_v[0]);chi_l.append(ci_v[1]-m)
            else: means.append(0);clo_l.append(0);chi_l.append(0)
        ax.bar(range(2),means,color=colors,alpha=0.8,width=0.55,edgecolor='white',linewidth=0.5)
        ax.errorbar(range(2),means,yerr=[clo_l,chi_l],fmt='none',color='#333',capsize=4,linewidth=1)
        ax.set_xticks(range(2))
        ax.set_xticklabels([g.replace(' ','\n') for g in groups],fontsize=6.5)
        ax.set_title(dl,fontsize=9,fontweight='bold')
        for xi,grp in enumerate(groups):
            n_val=len(data_dict[grp].get(dk,np.array([])))
            yl=ax.get_ylim()
            ax.text(xi,means[xi]+chi_l[xi]+0.02*(yl[1]-yl[0]),f'n={n_val}',ha='center',fontsize=6,color='#777')
        yl_fixed=ylims_dict.get(dk)
        if yl_fixed: ax.set_ylim(yl_fixed)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{fname}",dpi=200,bbox_inches='tight');plt.close()
    print(f"  -> {fname}")

def make_forest_fig(lmm_sub, title, fname, ref_label):
    if not lmm_sub: return
    fig,ax=plt.subplots(figsize=(10,max(3,len(lmm_sub)*0.5+1)))
    fig.suptitle(title,fontsize=11,fontweight='bold')
    for i,r in enumerate(lmm_sub):
        color='#333';alpha=0.4
        if r['p_bh']<0.05: color='#2E86AB';alpha=1.0
        elif r['p_bh']<0.10: color='#E8963F';alpha=0.8
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
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0,0,1,0.90])
    fig.savefig(f"{OUTPUT_DIR}/{fname}",dpi=200,bbox_inches='tight');plt.close()
    print(f"  -> {fname}")

# ── Generate bar charts and forest plots ──────────────────────────────
for dim_key, dim in DIMS.items():
    exposed_set=dim['exposed_tasks'];unexposed_set=dim['unexposed_tasks']
    ds=dim_key[:5]

    # Rebuild data for plotting (same logic as analysis)
    rows_a,rows_ab,rows_ac=[],[],[]
    for _,row in df.iterrows():
        pid=row.name;cond=row['xai_condition'];order=row['tasks_order']
        if not isinstance(order,list) or len(order)!=4: continue
        seen=False;seen_unexp=False
        for pos,task in enumerate(order):
            cur_exp=task in exposed_set
            eb={'pid':pid,'cond':cond,'task':task,'position':pos+1}
            for dk in ALL_DV_KEYS:
                col=f"{dk}_{task}" if dk!='mean_at' else f"mean_at_{task}"
                eb[dk]=row[col] if col in df.columns and pd.notna(row.get(col)) else np.nan
            rows_a.append({**eb,'exposed':1 if (cur_exp or seen) else 0})
            if task in unexposed_set:
                rows_ab.append({**eb,'prior_exposed':1 if seen else 0})
            if task in exposed_set:
                rows_ac.append({**eb,'prior_unexposed':1 if seen_unexp else 0})
            if cur_exp: seen=True
            if task in unexposed_set: seen_unexp=True
    df_a_p=pd.DataFrame(rows_a);df_ab_p=pd.DataFrame(rows_ab);df_ac_p=pd.DataFrame(rows_ac)

    # Comp B data
    df_b_p=df[df[f'profile_{dim_key}'].isin(['Both Unexposed','Both Exposed'])].copy()
    for dv_base in DVS:
        cols=[f"{dv_base}_{t}" for t in ALL_TASKS]
        av=[c for c in cols if c in df_b_p.columns]
        if av: df_b_p[f"agg_{dv_base}"]=df_b_p[av].mean(axis=1)
    atc=[f"mean_at_{t}" for t in ALL_TASKS if f"mean_at_{t}" in df_b_p.columns]
    if atc: df_b_p['agg_mean_at']=df_b_p[atc].mean(axis=1)

    plot_specs = [
        ('A', df_a_p, 'exposed', dim['group_labels'][0], dim['group_labels'][1],
         f"Comp. A: {dim['label']} Exposure (all tasks)", ALL_DV_KEYS, ALL_DV_LABELS),
        ('Ab', df_ab_p, 'prior_exposed',
         f"No prior {dim['exposed_label'].lower()}", f"Prior {dim['exposed_label'].lower()}",
         f"Comp. Ab: Prior {dim['label']} on {dim['unexposed_label'].lower()} tasks",
         ALL_DV_KEYS, ALL_DV_LABELS),
        ('Ac', df_ac_p, 'prior_unexposed',
         f"Only {dim['exposed_label'].lower()} so far", f"Prior {dim['unexposed_label'].lower()}",
         f"Comp. Ac: Prior {dim['unexposed_label']} on {dim['exposed_label'].lower()} tasks",
         ALL_DV_KEYS, ALL_DV_LABELS),
    ]

    for comp, df_p, flag_col, l0, l1, title_base, dk_list, dl_list in plot_specs:
        for cond in CONDS:
            sub=df_p[df_p['cond']==cond]
            if len(sub)<6 or flag_col not in sub.columns or sub[flag_col].nunique()<2: continue
            cs=cond.replace('+','_')
            dd={l0:{},l1:{}}
            for dk in dk_list:
                if dk not in sub.columns: continue
                dd[l0][dk]=sub[sub[flag_col]==0][dk].dropna().values
                dd[l1][dk]=sub[sub[flag_col]==1][dk].dropna().values
            make_bar_fig(dd,f"{title_base} — {cond}\n95% Bootstrap CI",
                         f"fig_{comp}_{ds}_bars_{cs}.png",dk_list,dl_list,YLIMS)

    # Comp B bars
    agg_base_keys=[k for k in DVS]+['mean_at']
    agg_labs=list(DVS.values())+['Mean Answer Time']
    agg_keys=[f"agg_{k}" for k in DVS]+['agg_mean_at']
    for cond in CONDS:
        sub=df_b_p[df_b_p['xai_condition']==cond]
        if sub[f'profile_{dim_key}'].nunique()<2: continue
        cs=cond.replace('+','_')
        l0=dim['profile_labels']['Both Unexposed'];l1=dim['profile_labels']['Both Exposed']
        dd={l0:{},l1:{}}
        for ak,bk in zip(agg_keys,agg_base_keys):
            if ak not in sub.columns: continue
            dd[l0][bk]=sub[sub[f'profile_{dim_key}']=='Both Unexposed'][ak].dropna().values
            dd[l1][bk]=sub[sub[f'profile_{dim_key}']=='Both Exposed'][ak].dropna().values
        make_bar_fig(dd,f"Comp. B: First-two {dim['label']} profile — {cond}\nAggregated DVs | 95% Bootstrap CI",
                     f"fig_B_{ds}_bars_{cs}.png",agg_base_keys,agg_labs,YLIMS)

    # LMM forest plots — use model with task covariate where available
    for comp in ['A','Ab','Ac','B']:
        # Find the most complete model for this comp
        model2 = f'{comp}2 ({dim_key})'
        model1 = f'{comp}1 ({dim_key})'
        for cond in CONDS:
            cs=cond.replace('+','_')
            lmm_sub=[r for r in all_lmm_rows
                     if r['dim']==dim_key and r['comp']==comp and r['cond']==cond
                     and r['model']==model2]
            if not lmm_sub:
                lmm_sub=[r for r in all_lmm_rows
                         if r['dim']==dim_key and r['comp']==comp and r['cond']==cond
                         and r['model']==model1]
            if not lmm_sub: continue
            ref=lmm_sub[0]['contrast'].split(' vs ')[-1] if lmm_sub else '?'
            make_forest_fig(lmm_sub,
                            f"LMM: Comp. {comp} {dim['label']} — {cond}\n{lmm_sub[0]['model']}",
                            f"fig_lmm_{comp}_{ds}_{cs}.png",ref)

print("\n"+"="*70+"\nCOMPLETE\n"+"="*70)
print("""
CSVs:
  v10b_mw_all.csv       All MW/permutation results (A, Ab, Ac, B × 2 dims)
  v10b_lmm_all.csv      All LMM results (A, Ab, Ac, B × 2 dims)

Figures (per dim × comp × cond):
  fig_A_*_bars_*.png        Comp A: all tasks, exposed vs never
  fig_Ab_*_bars_*.png       Comp Ab: unexposed tasks, prior exposure vs none
  fig_Ac_*_bars_*.png       Comp Ac: exposed tasks, prior unexposed vs none
  fig_B_*_bars_*.png        Comp B: first-two profile (aggregated)
  fig_lmm_*_*.png           LMM forest plots

Y-axis: Accuracy/Reliance/Over-/Under-reliance fixed 0–1.
Trust/Cognitive Load/Mean AT: shared global scale.
""")
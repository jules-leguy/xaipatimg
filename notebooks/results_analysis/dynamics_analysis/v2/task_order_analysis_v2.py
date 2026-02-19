"""
Task Order Effects Analysis v2 - Full preregistration-informed analysis
"""
import pandas as pd, numpy as np, matplotlib, ast, sys, os, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
warnings.filterwarnings('ignore')

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
OUTPUT_DIR = "."
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':10,'axes.titlesize':12,'axes.titleweight':'bold','axes.labelsize':10,'figure.facecolor':'#FAFAFA','axes.facecolor':'#FFFFFF','axes.edgecolor':'#CCCCCC','axes.grid':True,'grid.alpha':0.25,'grid.color':'#DDDDDD'})
PALETTE_ORDER = {'easy_mild':'#2E86AB','easy_strong':'#7FB069','hard_mild':'#E8963F','hard_strong':'#E8533F'}
TASK_LABELS = {'easy_mild':'Easy / Low pressure','easy_strong':'Easy / High pressure','hard_mild':'Hard / Low pressure','hard_strong':'Hard / High pressure'}
ALL_TASKS = ['easy_mild','easy_strong','hard_mild','hard_strong']
TARGET_TASKS = ['easy_mild','hard_strong']

print("="*70+"\nTASK ORDER EFFECTS ANALYSIS v2\n"+"="*70)
df = pd.read_csv(DATA_PATH)
print(f"Total participants: {len(df)}")
df = df[df['xai_condition'].isin(['H','H+AI'])].copy()
print(f"After filtering to H and H+AI: {len(df)}")

def parse_col(val):
    if isinstance(val, str):
        try: return ast.literal_eval(val)
        except: return val
    return val

df['tasks_order'] = df['tasks_order'].apply(parse_col)
df['first_task'] = df['tasks_order'].apply(lambda x: x[0] if isinstance(x, list) else None)

for task in ALL_TASKS:
    at_col = f"answer_times_{task}"
    if at_col in df.columns:
        df[f"mean_answer_time_{task}"] = df[at_col].apply(lambda v: np.nanmean(np.array(parse_col(v),dtype=float)) if isinstance(parse_col(v),(list,np.ndarray)) else np.nan)

print(f"\nStarting task distribution:\n{df['first_task'].value_counts().to_string()}")
print(f"\nBreakdown:\n{pd.crosstab(df['first_task'], df['xai_condition']).to_string()}")

DVS = {'score':'Accuracy','reliance':'Reliance','overreliance':'Over-reliance','underreliance':'Under-reliance','trust':'Trust','cogload':'Cognitive Load','mean_answer_time':'Mean Answer Time (s)'}

def safe_spearmanr(a, b):
    a,b = np.array(a,dtype=float),np.array(b,dtype=float)
    mask = ~(np.isnan(a)|np.isnan(b)); a,b = a[mask],b[mask]
    if len(a)<5 or np.std(a)==0 or np.std(b)==0: return (np.nan,np.nan)
    return stats.spearmanr(a, b)

def cohens_d(g1, g2):
    n1,n2 = len(g1),len(g2)
    if n1<2 or n2<2: return np.nan
    pooled = np.sqrt(((n1-1)*np.var(g1,ddof=1)+(n2-1)*np.var(g2,ddof=1))/(n1+n2-2))
    return (np.mean(g1)-np.mean(g2))/pooled if pooled>0 else 0

def bootstrap_ci(data, n_boot=10000, ci=0.95):
    rng = np.random.default_rng(42); data = np.array(data,dtype=float); data = data[~np.isnan(data)]
    if len(data)<3: return (np.nan,np.nan)
    boot = [np.mean(rng.choice(data,size=len(data),replace=True)) for _ in range(n_boot)]
    alpha = (1-ci)/2
    return (np.percentile(boot,alpha*100), np.percentile(boot,(1-alpha)*100))

def permutation_test(g1, g2, n_perm=10000):
    g1 = np.array(g1,dtype=float); g1=g1[~np.isnan(g1)]
    g2 = np.array(g2,dtype=float); g2=g2[~np.isnan(g2)]
    if len(g1)<2 or len(g2)<2: return np.nan
    observed = np.mean(g1)-np.mean(g2); combined = np.concatenate([g1,g2])
    rng = np.random.default_rng(42); count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        if abs(np.mean(combined[:len(g1)])-np.mean(combined[len(g1):])) >= abs(observed): count+=1
    return count/n_perm

def p_to_stars(p):
    if np.isnan(p): return ''
    if p<0.001: return '***'
    if p<0.01: return '**'
    if p<0.05: return '*'
    if p<0.1: return '\u2020'
    return 'ns'

def run_comparison(vals1, vals2):
    v1 = np.array(vals1,dtype=float); v1=v1[~np.isnan(v1)]
    v2 = np.array(vals2,dtype=float); v2=v2[~np.isnan(v2)]
    if len(v1)<3 or len(v2)<3: return None
    u,p_mw = stats.mannwhitneyu(v1,v2,alternative='two-sided')
    d = cohens_d(v1,v2); n1,n2 = len(v1),len(v2)
    r_rb = 1-(2*u)/(n1*n2); p_perm = permutation_test(v1,v2)
    return {'n1':n1,'n2':n2,'mean1':np.mean(v1),'mean2':np.mean(v2),'median1':np.median(v1),'median2':np.median(v2),'sd1':np.std(v1,ddof=1),'sd2':np.std(v2,ddof=1),'U':u,'p_mw':p_mw,'p_perm':p_perm,'d':d,'r_rb':r_rb}

# ===== ANALYSIS 1: All 4 starting conditions =====
print("\n"+"="*70+"\nANALYSIS 1: EFFECT OF STARTING TASK ON TARGET TASK OUTCOMES\n"+"="*70)
results_rows = []
for xai_cond in ['H','H+AI']:
    sub = df[df['xai_condition']==xai_cond]
    print(f"\n{'='*60}\n  {xai_cond} (n={len(sub)})\n{'='*60}")
    for target in TARGET_TASKS:
        print(f"\n  -- Target: {TASK_LABELS[target]} --")
        for dv_base, dv_label in DVS.items():
            col = f"{dv_base}_{target}"
            if col not in sub.columns: continue
            print(f"\n    {dv_label}:")
            groups_kw = []; p_kw = np.nan
            for start in ALL_TASKS:
                g = sub[sub['first_task']==start][col].dropna()
                if len(g)>=2: groups_kw.append(g.values)
            if len(groups_kw)>=2:
                try:
                    H_stat,p_kw = stats.kruskal(*groups_kw)
                    N = sum(len(g) for g in groups_kw); k = len(groups_kw)
                    eta2 = (H_stat-k+1)/(N-k) if (N-k)>0 else np.nan
                    print(f"      Kruskal-Wallis: H={H_stat:.3f}, p={p_kw:.4f} {p_to_stars(p_kw)}, eta2={eta2:.3f}")
                except: print("      Kruskal-Wallis: error")
            target_first = sub[sub['first_task']==target][col].dropna()
            for other in ALL_TASKS:
                if other==target: continue
                other_first = sub[sub['first_task']==other][col].dropna()
                res = run_comparison(target_first.values, other_first.values)
                if res is None: print(f"      vs {other:15s}: insufficient data"); continue
                print(f"      vs started-{other:15s}: M={res['mean1']:.3f} vs {res['mean2']:.3f}, d={res['d']:+.3f}, p_MW={res['p_mw']:.4f}{p_to_stars(res['p_mw'])}, p_perm={res['p_perm']:.4f}{p_to_stars(res['p_perm'])}")
                results_rows.append({'xai_condition':xai_cond,'target_task':target,'dv':dv_label,'dv_base':dv_base,'group_target_first_mean':res['mean1'],'group_target_first_sd':res['sd1'],'group_target_first_n':res['n1'],'compared_start':other,'group_other_first_mean':res['mean2'],'group_other_first_sd':res['sd2'],'group_other_first_n':res['n2'],'U':res['U'],'p_mw':res['p_mw'],'p_perm':res['p_perm'],'d':res['d'],'r_rb':res['r_rb'],'kw_p':p_kw})

results_df = pd.DataFrame(results_rows)
results_df.to_csv(f"{OUTPUT_DIR}/order_effects_all_starts.csv", index=False)

# ===== ANALYSIS 2: First vs later (binary) =====
print("\n"+"="*70+"\nANALYSIS 2: TARGET FIRST vs NOT FIRST\n"+"="*70)
binary_rows = []
for xai_cond in ['H','H+AI']:
    sub = df[df['xai_condition']==xai_cond]
    print(f"\n  {xai_cond} (n={len(sub)})")
    for target in TARGET_TASKS:
        print(f"  Target: {TASK_LABELS[target]}")
        g_first = sub[sub['first_task']==target]; g_later = sub[sub['first_task']!=target]
        for dv_base, dv_label in DVS.items():
            col = f"{dv_base}_{target}"
            if col not in sub.columns: continue
            res = run_comparison(g_first[col].dropna().values, g_later[col].dropna().values)
            if res is None: continue
            print(f"    {dv_label:25s}: 1st M={res['mean1']:.3f}(n={res['n1']}) vs later M={res['mean2']:.3f}(n={res['n2']}), d={res['d']:+.3f}, p_MW={res['p_mw']:.4f}{p_to_stars(res['p_mw'])}, p_perm={res['p_perm']:.4f}{p_to_stars(res['p_perm'])}")
            binary_rows.append({'xai_condition':xai_cond,'target_task':target,'dv':dv_label,'dv_base':dv_base,'first_mean':res['mean1'],'first_sd':res['sd1'],'first_n':res['n1'],'later_mean':res['mean2'],'later_sd':res['sd2'],'later_n':res['n2'],'U':res['U'],'p_mw':res['p_mw'],'p_perm':res['p_perm'],'d':res['d'],'r_rb':res['r_rb']})
pd.DataFrame(binary_rows).to_csv(f"{OUTPUT_DIR}/order_effects_first_vs_later.csv", index=False)

# ===== VISUALIZATIONS =====
print("\n-> Generating visualizations...")

# Fig 1: Forest plot
fig, axes = plt.subplots(1,2,figsize=(16,12),sharey=True)
fig.suptitle("Effect of Starting Task on Target Task Outcomes (Cohen's d)\nPositive d = higher when target done first",fontsize=13,fontweight='bold',y=0.98)
for idx, xai_cond in enumerate(['H','H+AI']):
    ax = axes[idx]; sub_res = results_df[results_df['xai_condition']==xai_cond].copy()
    if len(sub_res)==0: continue
    sub_res = sub_res.sort_values(['target_task','dv_base','compared_start']).reset_index(drop=True)
    for i,(_,r) in enumerate(sub_res.iterrows()):
        sig = p_to_stars(r['p_perm']); alpha = 1.0 if sig in ['*','**','***'] else 0.6 if sig=='\u2020' else 0.35
        ax.barh(i,r['d'],color=PALETTE_ORDER[r['compared_start']],alpha=alpha,height=0.7,edgecolor='white',linewidth=0.3)
        if sig and sig not in ['ns','']: ax.text(r['d']+(0.05 if r['d']>=0 else -0.05),i,sig,va='center',ha='left' if r['d']>=0 else 'right',fontsize=8,fontweight='bold')
    ax.set_yticks(range(len(sub_res))); ax.set_yticklabels([f"{r['dv']} | vs {r['compared_start']}" for _,r in sub_res.iterrows()],fontsize=7)
    ax.axvline(0,color='black',linewidth=0.8); ax.axvline(-0.5,color='gray',linewidth=0.5,linestyle='--',alpha=0.4); ax.axvline(0.5,color='gray',linewidth=0.5,linestyle='--',alpha=0.4)
    ax.set_xlabel("Cohen's d"); ax.set_title(xai_cond,fontweight='bold',fontsize=13); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    prev = None
    for i,(_,r) in enumerate(sub_res.iterrows()):
        if prev and r['target_task']!=prev: ax.axhline(i-0.5,color='black',linewidth=1)
        prev = r['target_task']
axes[1].legend(handles=[Patch(facecolor=c,label=TASK_LABELS[t]) for t,c in PALETTE_ORDER.items()],title='Started with...',loc='lower right',fontsize=8,title_fontsize=9)
plt.tight_layout(rect=[0.02,0,1,0.94]); fig.savefig(f"{OUTPUT_DIR}/fig1_forest_plot.png",dpi=200,bbox_inches='tight'); plt.close()
print("  -> fig1_forest_plot.png")

# Fig 2: Bar charts
key_dvs = ['score','reliance','overreliance','underreliance','trust','cogload','mean_answer_time']
for target in TARGET_TASKS:
    fig, axes = plt.subplots(2,len(key_dvs),figsize=(3.5*len(key_dvs),9))
    fig.suptitle(f"Target: {TASK_LABELS[target]} - By Starting Task (95% Bootstrap CI)",fontsize=14,fontweight='bold',y=0.99)
    for row_idx,xai_cond in enumerate(['H','H+AI']):
        sub = df[df['xai_condition']==xai_cond]
        for col_idx,dv_base in enumerate(key_dvs):
            ax = axes[row_idx,col_idx]; col = f"{dv_base}_{target}"
            if col not in sub.columns: ax.set_visible(False); continue
            means,cis_lo,cis_hi,colors = [],[],[],[]
            for start in ALL_TASKS:
                g = sub[sub['first_task']==start][col].dropna()
                if len(g)>=2:
                    m=g.mean(); ci=bootstrap_ci(g.values); means.append(m); cis_lo.append(m-ci[0]); cis_hi.append(ci[1]-m)
                else: means.append(0); cis_lo.append(0); cis_hi.append(0)
                colors.append(PALETTE_ORDER[start])
            bars = ax.bar(range(4),means,color=colors,alpha=0.8,width=0.65,edgecolor='white',linewidth=0.5)
            ax.errorbar(range(4),means,yerr=[cis_lo,cis_hi],fmt='none',color='#333333',capsize=3,linewidth=1)
            bars[ALL_TASKS.index(target)].set_edgecolor('black'); bars[ALL_TASKS.index(target)].set_linewidth(2)
            ax.set_xticks(range(4)); ax.set_xticklabels([t.replace('_','\n') for t in ALL_TASKS],fontsize=7)
            ax.set_title(DVS[dv_base],fontsize=10,fontweight='bold')
            if col_idx==0: ax.set_ylabel(xai_cond,fontsize=11,fontweight='bold')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0,0,1,0.94]); fig.savefig(f"{OUTPUT_DIR}/fig2_bars_{target}.png",dpi=200,bbox_inches='tight'); plt.close()
    print(f"  -> fig2_bars_{target}.png")

# Fig 3: Answer time trajectories
fig, axes = plt.subplots(2,2,figsize=(14,10))
fig.suptitle("Answer Time Trajectories (H+AI, mean +/- SEM)",fontsize=14,fontweight='bold',y=0.98)
sub_ai = df[df['xai_condition']=='H+AI']
for col_idx, target in enumerate(TARGET_TASKS):
    at_col = f"answer_times_{target}"
    if at_col not in sub_ai.columns: continue
    for row_idx, mode in enumerate(['by_start','binary']):
        ax = axes[row_idx,col_idx]
        if mode=='by_start':
            for st in ALL_TASKS:
                grp = sub_ai[sub_ai['first_task']==st]; times_list = []
                for _,r in grp.iterrows():
                    t = parse_col(r[at_col])
                    if isinstance(t,(list,np.ndarray)): times_list.append(list(t))
                if not times_list: continue
                ml = max(len(t) for t in times_list); pad = np.full((len(times_list),ml),np.nan)
                for i,t in enumerate(times_list): pad[i,:len(t)] = t
                m = np.nanmean(pad,axis=0); se = np.nanstd(pad,axis=0)/np.sqrt(np.sum(~np.isnan(pad),axis=0))
                tr = np.arange(1,ml+1); ax.plot(tr,m,color=PALETTE_ORDER[st],linewidth=2,label=f"Start={st}",alpha=0.9); ax.fill_between(tr,m-se,m+se,color=PALETTE_ORDER[st],alpha=0.12)
            ax.set_title(f"{TASK_LABELS[target]} - By Start",fontweight='bold'); ax.legend(fontsize=7,ncol=2)
        else:
            for is_f,lab,col_c,ls in [(True,f'{target} 1st',PALETTE_ORDER[target],'-'),(False,f'{target} later','#888','--')]:
                grp = sub_ai[sub_ai['first_task']==target] if is_f else sub_ai[sub_ai['first_task']!=target]; times_list = []
                for _,r in grp.iterrows():
                    t = parse_col(r[at_col])
                    if isinstance(t,(list,np.ndarray)): times_list.append(list(t))
                if not times_list: continue
                ml = max(len(t) for t in times_list); pad = np.full((len(times_list),ml),np.nan)
                for i,t in enumerate(times_list): pad[i,:len(t)] = t
                m = np.nanmean(pad,axis=0); se = np.nanstd(pad,axis=0)/np.sqrt(np.sum(~np.isnan(pad),axis=0))
                tr = np.arange(1,ml+1); ax.plot(tr,m,color=col_c,linewidth=2,linestyle=ls,label=lab,alpha=0.9); ax.fill_between(tr,m-se,m+se,color=col_c,alpha=0.12)
            ax.set_title(f"{TASK_LABELS[target]} - 1st vs Later",fontweight='bold'); ax.legend(fontsize=8)
        ax.set_xlabel('Trial'); ax.set_ylabel('Time (s)'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(rect=[0,0,1,0.93]); fig.savefig(f"{OUTPUT_DIR}/fig3_answer_time_trajectories.png",dpi=200,bbox_inches='tight'); plt.close()
print("  -> fig3_answer_time_trajectories.png")

# Fig 4: Heatmaps
for xai_cond in ['H','H+AI']:
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    fig.suptitle(f"Effect Size Heatmap: {xai_cond}",fontsize=13,fontweight='bold')
    for idx,target in enumerate(TARGET_TASKS):
        ax = axes[idx]; sr = results_df[(results_df['xai_condition']==xai_cond)&(results_df['target_task']==target)].copy()
        if len(sr)==0: ax.text(0.5,0.5,'No data',transform=ax.transAxes,ha='center'); continue
        pivot = sr.pivot(index='dv',columns='compared_start',values='d')
        sr['annot'] = sr.apply(lambda r: f"{r['d']:.2f}\n{p_to_stars(r['p_perm'])}",axis=1)
        annot_p = sr.pivot(index='dv',columns='compared_start',values='annot')
        sns.heatmap(pivot,annot=annot_p,fmt='',cmap='RdBu_r',center=0,vmin=-1.5,vmax=1.5,ax=ax,linewidths=0.5,cbar_kws={'label':"Cohen's d"})
        ax.set_title(f"Target: {TASK_LABELS[target]}",fontweight='bold'); ax.set_ylabel(''); ax.set_xlabel('Started with...')
    plt.tight_layout(rect=[0,0,1,0.90]); fig.savefig(f"{OUTPUT_DIR}/fig4_heatmap_{xai_cond.replace('+','_')}.png",dpi=200,bbox_inches='tight'); plt.close()
    print(f"  -> fig4_heatmap_{xai_cond.replace('+','_')}.png")

# ===== ANALYSIS 3: Trust vs Error Recency =====
print("\n"+"="*70+"\nANALYSIS 3: TRUST vs ERROR RECENCY (H+AI)\n"+"="*70)
sub_ai = df[df['xai_condition']=='H+AI']; recency_rows = []
for _,row in sub_ai.iterrows():
    for task in TARGET_TASKS:
        tv = row.get(f"trust_{task}")
        if pd.isna(tv): continue
        try:
            tt=parse_col(row[f"task_true_{task}"]); ap=parse_col(row[f"ai_pred_{task}"]); ud=parse_col(row[f"user_decision_{task}"])
        except: continue
        if not all(isinstance(x,list) for x in [tt,ap,ud]): continue
        n=len(tt)
        if n<4: continue
        tt=[bool(x) for x in tt]; ap=[bool(x) for x in ap]; ud=[bool(x) for x in ud]
        uf=[1 if u==p else 0 for u,p in zip(ud,ap)]; bf=[1 if (u==p and t!=p) else 0 for u,p,t in zip(ud,ap,tt)]
        for w in [3,4,6]:
            if n>=w:
                recency_rows.append({'task':task,'first_task':row['first_task'],'trust':tv,'window':w,'recent_follow_rate':sum(uf[-w:])/w,'early_follow_rate':sum(uf[:w])/w,'recent_bad_follow_rate':sum(bf[-w:])/w,'early_bad_follow_rate':sum(bf[:w])/w,'total_follow_rate':sum(uf)/n})

if recency_rows:
    tr_df = pd.DataFrame(recency_rows); tr_df.to_csv(f"{OUTPUT_DIR}/trust_recency_data_v2.csv",index=False)
    predictors = [('recent_follow_rate','early_follow_rate','AI Follow Rate'),('recent_bad_follow_rate','early_bad_follow_rate','Bad Follow Rate')]
    rcr = []
    for task in TARGET_TASKS:
        for w in [3,4,6]:
            sub = tr_df[(tr_df['task']==task)&(tr_df['window']==w)]
            if len(sub)<5: continue
            print(f"\n  {TASK_LABELS[task]}, w={w} (n={len(sub)}):")
            for rc,ec,lab in predictors:
                rr,pr = safe_spearmanr(sub[rc],sub['trust']); re,pe = safe_spearmanr(sub[ec],sub['trust'])
                print(f"    {lab:20s}: rho_recent={rr:+.3f} (p={pr:.4f} {p_to_stars(pr)}) | rho_early={re:+.3f} (p={pe:.4f} {p_to_stars(pe)})")
                rcr.append({'task':task,'window':w,'predictor':lab,'rho_recent':rr,'p_recent':pr,'rho_early':re,'p_early':pe,'n':len(sub)})
                if not np.isnan(rr) and not np.isnan(re):
                    r12,_ = safe_spearmanr(sub[rc],sub[ec])
                    if not np.isnan(r12):
                        denom = np.sqrt(2*(1-r12)/(len(sub)-3))
                        if denom>0:
                            zs = (np.arctanh(rr)-np.arctanh(re))/denom; ps = 2*(1-stats.norm.cdf(abs(zs)))
                            print(f"      -> Steiger Z={zs:+.3f}, p={ps:.4f} {p_to_stars(ps)}")
    pd.DataFrame(rcr).to_csv(f"{OUTPUT_DIR}/trust_recency_correlations_v2.csv",index=False)

    # Fig 5
    fig, axes = plt.subplots(2,2,figsize=(14,10)); fig.suptitle("Trust vs Recent/Early AI-Follow Rate (H+AI, w=4)",fontsize=14,fontweight='bold',y=0.98)
    pdata = tr_df[tr_df['window']==4]
    for ci,task in enumerate(TARGET_TASKS):
        for ri,(pred,lbl) in enumerate([('recent_follow_rate','Recent Follow Rate (last 4)'),('early_follow_rate','Early Follow Rate (first 4)')]):
            ax = axes[ri,ci]; td = pdata[pdata['task']==task]
            if len(td)<3: continue
            for st in ALL_TASKS:
                sg = td[td['first_task']==st]
                if len(sg)==0: continue
                jx=np.random.default_rng(42).normal(0,0.015,len(sg)); jy=np.random.default_rng(43).normal(0,0.06,len(sg))
                ax.scatter(sg[pred]+jx,sg['trust']+jy,color=PALETTE_ORDER[st],alpha=0.6,s=40,label=f"Start={st}",edgecolors='white',linewidth=0.3)
            rho,pv = safe_spearmanr(td[pred],td['trust'])
            ax.set_title(f"{TASK_LABELS[task]}\nrho={rho:+.3f}, p={pv:.3f} {p_to_stars(pv)}",fontweight='bold',fontsize=10)
            ax.set_xlabel(lbl); ax.set_ylabel('Trust'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            if ri==0 and ci==1: ax.legend(fontsize=7,ncol=2)
    plt.tight_layout(rect=[0,0,1,0.93]); fig.savefig(f"{OUTPUT_DIR}/fig5_trust_recency_v2.png",dpi=200,bbox_inches='tight'); plt.close()
    print("  -> fig5_trust_recency_v2.png")

# ===== ANALYSIS 4: Position effects =====
print("\n-> Position analysis...")
for task in TARGET_TASKS:
    df[f"position_{task}"] = df['tasks_order'].apply(lambda x, t=task: (x.index(t)+1) if isinstance(x,list) and t in x else np.nan)

fig, axes = plt.subplots(2,len(key_dvs),figsize=(3.5*len(key_dvs),9))
fig.suptitle("Position Effects (1st-4th) on Outcomes - H+AI",fontsize=14,fontweight='bold',y=0.99)
sub_ai = df[df['xai_condition']=='H+AI']
for ri,target in enumerate(TARGET_TASKS):
    for ci,dv_base in enumerate(key_dvs):
        ax = axes[ri,ci]; col = f"{dv_base}_{target}"; pc = f"position_{target}"
        if col not in sub_ai.columns: ax.set_visible(False); continue
        means,clo,chi = [],[],[]
        for pos in [1,2,3,4]:
            g = sub_ai[sub_ai[pc]==pos][col].dropna()
            if len(g)>=2:
                m=g.mean(); ci_v=bootstrap_ci(g.values); means.append(m); clo.append(m-ci_v[0]); chi.append(ci_v[1]-m)
            else: means.append(np.nan); clo.append(0); chi.append(0)
        ax.bar([1,2,3,4],means,color=['#2E86AB','#5BA08E','#E8963F','#E8533F'],alpha=0.8,width=0.6,edgecolor='white',linewidth=0.5)
        ax.errorbar([1,2,3,4],means,yerr=[clo,chi],fmt='none',color='#333',capsize=3,linewidth=1)
        valid = sub_ai[[pc,col]].dropna()
        if len(valid)>5:
            rho,pv = stats.spearmanr(valid[pc],valid[col]); ax.set_xlabel(f"Pos (rho={rho:+.2f}, p={pv:.3f})",fontsize=8)
        else: ax.set_xlabel("Position")
        ax.set_xticks([1,2,3,4]); ax.set_xticklabels(['1st','2nd','3rd','4th'],fontsize=8)
        ax.set_title(DVS[dv_base],fontsize=10,fontweight='bold')
        if ci==0: ax.set_ylabel(TASK_LABELS[target],fontsize=10,fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(rect=[0,0,1,0.94]); fig.savefig(f"{OUTPUT_DIR}/fig6_position_effects.png",dpi=200,bbox_inches='tight'); plt.close()
print("  -> fig6_position_effects.png")

print("\n  Position-DV correlations (Spearman, H+AI):")
pcr = []
for target in TARGET_TASKS:
    pc = f"position_{target}"
    for dv_base,dv_label in DVS.items():
        col = f"{dv_base}_{target}"
        if col not in sub_ai.columns: continue
        valid = sub_ai[[pc,col]].dropna()
        if len(valid)>5:
            rho,pv = stats.spearmanr(valid[pc],valid[col])
            print(f"    {TASK_LABELS[target]:25s} | {dv_label:25s}: rho={rho:+.3f}, p={pv:.4f} {p_to_stars(pv)}")
            pcr.append({'target':target,'dv':dv_label,'rho':rho,'p':pv,'n':len(valid)})
pd.DataFrame(pcr).to_csv(f"{OUTPUT_DIR}/position_correlations.csv",index=False)

print("\n"+"="*70+"\nANALYSIS COMPLETE\n"+"="*70)
print("Files: order_effects_all_starts.csv, order_effects_first_vs_later.csv,")
print("trust_recency_*.csv, position_correlations.csv")
print("Figures: fig1-fig6 (.png)")
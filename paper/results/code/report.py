"""Generate figures/tables only from completed, saved experiment outputs."""
from __future__ import annotations
import argparse, json, math, shutil, subprocess, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from bare import (HERE, DEVICE, BOXES, MEAN, STD, Encoder, Fusion, load_utility,
                  views_from_image, policy_features, metrics, save_json, gpu_sample)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':140,'savefig.dpi':240,'axes.grid':True,'grid.alpha':.18})

def tex(s):
    return ''.join({'_':r'\_','%':r'\%','&':r'\&','#':r'\#'}.get(c,c) for c in str(s))

def table(paper,name,caption,headers,rows,wide=False):
    env='table*' if wide else 'table'
    # X columns allow wrapped prose; numeric columns use bounded centered text.
    first='p{27mm}' if not wide else ('p{31mm}' if len(headers)<=6 else 'p{28mm}')
    spec='@{}'+first+'X'*(len(headers)-1)+'@{}'
    width=r'\textwidth' if wide else r'\columnwidth'
    body=[r'\begin{'+env+'}[!t]',r'\centering\footnotesize',r'\caption{'+caption+'}',r'\label{tab:'+name+'}',
          r'\begin{tabularx}{'+width+'}{'+spec+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    for row in rows: body.append(' & '.join(str(v) for v in row)+r' \\')
    body += [r'\bottomrule',r'\end{tabularx}',r'\end{'+env+'}']
    (paper/'tables'/f'{name}.tex').write_text('\n'.join(body),encoding='utf-8')

def figure(paper,name,caption,wide=False,height=None):
    env='figure*' if wide else 'figure';width=r'\textwidth' if wide else r'\columnwidth'
    options='width='+width
    if height: options+=',height='+height+',keepaspectratio'
    return '\n'.join([r'\begin{'+env+'}[!t]',r'\centering',r'\includegraphics['+options+']{figures/'+name+'.png}',
                      r'\caption{'+caption+'}',r'\label{fig:'+name+'}',r'\end{'+env+'}'])

def savefig(paper,name,fig):
    fig.savefig(paper/'figures'/f'{name}.png',bbox_inches='tight',facecolor='white');plt.close(fig)

def original_views(path,args):
    with Image.open(path) as im:
        a=np.asarray(im)
        if a.ndim==3:a=np.asarray(im.convert('L'))
        lo,hi=np.percentile(a,[.5,99.5]);a=np.clip((a.astype(np.float32)-lo)/max(float(hi-lo),1)*255,0,255).astype(np.uint8)
        im=Image.fromarray(a);im.thumbnail((1280,1280),Image.Resampling.LANCZOS)
        return views_from_image(im,False,args.global_size,args.local_size)

@torch.no_grad()
def infer_real(enc,fusion,policy,row,args,tau=-float('inf'),budget=3):
    views,desc=original_views(row.path,args);desc=desc.unsqueeze(0).to(DEVICE)
    sex=torch.tensor([int(row.sex)],device=DEVICE);z=torch.zeros(1,10,128,device=DEVICE)
    selected=torch.zeros((1,1),device=DEVICE,dtype=torch.long)
    with torch.autocast(DEVICE.type,dtype=torch.float16):z[:,0]=enc(views[0].unsqueeze(0).to(DEVICE)).float()
    mean,p,_=fusion(z[:,:1],sex,selected)
    for _ in range(budget):
        before,p,h,fs=policy_features(fusion,z,sex,desc,selected)
        model,mu,sd=policy;gains=model((fs[0]-mu)/sd)
        for j in selected[0].tolist():
            if j>0:gains[j-1]=-float('inf')
        gain,idx=gains.max(0);j=int(idx)+1
        if float(gain)<=tau:break
        with torch.autocast(DEVICE.type,dtype=torch.float16):z[:,j]=enc(views[j].unsqueeze(0).to(DEVICE)).float()
        selected=torch.cat([selected,torch.tensor([[j]],device=DEVICE)],1)
        mean,p,_=fusion(z.gather(1,selected[:,:,None].expand(-1,-1,128)),sex,selected)
    return float(mean),selected[0].tolist()

def latency_benchmark(enc,fusion,policy,manifest,args,run,tau):
    # IDs are chosen before looking at accuracy, with five warmups and 50 measurements.
    rows=manifest[manifest.split=='holdout'].sort_values('id').iloc[:55]
    times=[];consistency=[];peak=[]
    cached_pred=pd.read_csv(run/'predictions_policy_three.csv').set_index('id')
    for label,threshold,budget in [('global',-float('inf'),0),('policy_three',-float('inf'),3),('adaptive',tau,3)]:
        values=[];counts=[];torch.cuda.reset_peak_memory_stats()
        for i,row in enumerate(rows.itertuples()):
            torch.cuda.synchronize();start=time.perf_counter()
            mean,selected=infer_real(enc,fusion,policy,row,args,threshold,budget)
            torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
            if i>=5:values.append(elapsed);counts.append(len(selected)-1)
            if label=='policy_three':consistency.append(abs(mean-float(cached_pred.loc[row.id,'pred'])))
        times.append(dict(mode=label,n=len(values),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),
                          mean_ms=float(np.mean(values)),mean_crops=float(np.mean(counts)),
                          allocated_mib=torch.cuda.max_memory_allocated()/2**20,reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                          protocol='original PNG decode + normalization + CPU view construction + CUDA encoding + policy/fusion; batch 1; 5 warmups then 50 images'))
    save_json(run/'latency.json',times)
    save_json(run/'inference_consistency.json',dict(n=len(consistency),max_abs_difference_months=max(consistency),
              mean_abs_difference_months=float(np.mean(consistency)),note='FP16 batch-one raw inference vs FP16 batched cached-token predictions; no unseen tokens are passed to acquisition features'))
    return times

def profile_flops(enc,fusion,policy,manifest,args,run):
    row=next(manifest[manifest.split=='holdout'].sort_values('id').itertuples())
    # PyTorch counts supported convolution/matmul operations, not every nonlinear/attention primitive.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
        infer_real(enc,fusion,policy,row,args,budget=3)
        torch.cuda.synchronize()
    total=sum(e.flops for e in prof.key_averages())
    record=dict(counted_flops=int(total),counted_gflops=total/1e9,image_id=int(row.id),batch=1,local_crops=3,
                convention='One multiply and one add count as two FLOPs; PyTorch supported convolution/matmul events only. Unsupported attention/nonlinear operations omitted; not a complete total.')
    save_json(run/'flops.json',record);return record

def gradcam_examples(enc,fusion,manifest,predictions,args,paper,run):
    enc.eval();fusion.eval();cases=[]
    # First ID in each prespecified age band. No selection by saliency or error.
    hold=manifest[manifest.split=='holdout'].sort_values('id').set_index('id')
    ordered=predictions.sort_values('id')
    for lower,upper in [(0,48),(48,96),(96,144),(144,192),(192,241)]:
        subset=ordered[(ordered.true>=lower)&(ordered.true<upper)]
        if subset.empty:continue
        record=subset.iloc[0];row=hold.loc[int(record.id)]
        views,desc=original_views(row.path,args)
        selected=[0]+[int(j) for j in str(record.selected).split(',') if j and j!='nan']
        activations={};handles=[]
        def hook(module,inputs,output):activations['current']=output
        handles.append(enc.features[-1].register_forward_hook(hook))
        zs=[];maps=[]
        # Retain at most four batch-one branches for explanation; training uses replay.
        for j in selected:
            z=enc(views[j].unsqueeze(0).to(DEVICE));zs.append(z);maps.append(activations['current'])
        z=torch.stack(zs,1);sex=torch.tensor([int(row.sex)],device=DEVICE)
        mean,p,_=fusion(z,sex,torch.tensor(selected,device=DEVICE))
        gradients=torch.autograd.grad(mean.sum(),maps,retain_graph=False)
        cams=[]
        for a,g in zip(maps,gradients):
            alpha=g.mean((2,3),keepdim=True);cam=(a*alpha).sum(1).relu()[0].detach().cpu().numpy()
            if cam.max()>0:cam=cam/cam.max()
            cams.append(cam)
        for handle in handles:handle.remove()
        global_display=(views[0]*STD+MEAN)[0].numpy()
        local_index=1 if len(selected)>1 else 0
        local_display=(views[selected[local_index]]*STD+MEAN)[0].numpy()
        fig,axes=plt.subplots(1,4,figsize=(12,3.4))
        axes[0].imshow(global_display,cmap='gray',vmin=0,vmax=1)
        for pos,j in enumerate(selected[1:]):
            x,y,w,h=BOXES[j].tolist();sz=global_display.shape[0]
            axes[0].add_patch(Rectangle(((x-w/2)*sz,(y-h/2)*sz),w*sz,h*sz,fill=False,edgecolor=['#00d4c7','#ffce54','#ff6688'][pos],linewidth=1.2))
            axes[0].text((x-w/2)*sz,(y-h/2)*sz,str(j),color='white',fontsize=9)
        axes[0].set_title('Acquired regions')
        axes[1].imshow(global_display,cmap='gray');axes[1].imshow(cams[0],cmap='jet',alpha=.45,extent=(0,global_display.shape[1],global_display.shape[0],0),vmin=0,vmax=1);axes[1].set_title('Whole-hand Grad-CAM')
        axes[2].imshow(local_display,cmap='gray');axes[2].set_title(f'First selected crop {selected[local_index]}')
        axes[3].imshow(local_display,cmap='gray');axes[3].imshow(cams[local_index],cmap='jet',alpha=.45,extent=(0,local_display.shape[1],local_display.shape[0],0),vmin=0,vmax=1);axes[3].set_title('Regional Grad-CAM')
        for ax in axes:ax.axis('off')
        fig.suptitle(f'Image {int(record.id)} | Reference {record.true:.0f} months | FP16 evaluation {record.pred:.2f} months | FP32 attribution {float(mean):.2f} months',fontsize=10)
        fig.tight_layout();name=f'gradcam_{len(cases)+1}';savefig(paper,name,fig)
        cases.append(dict(id=int(record.id),true=float(record.true),pred=float(record.pred),attribution_pred=float(mean),
                  age_band=f'{lower}--{upper}',selected=selected,figure=name,positive_map_nonzero=[bool(c.max()>0) for c in cams]))
        del zs,z,maps,gradients;torch.cuda.empty_cache()
    if len(cases)<5:raise RuntimeError('Fewer than five age bands available for Grad-CAM')
    save_json(run/'gradcam_cases.json',cases);return cases

def bootstrap_table(run,predictions):
    primary=predictions['policy_three'].set_index('id');rng=np.random.default_rng(42);rows=[]
    for label in ['global_baseline','fixed_three','random_three','policy_no_disagreement','policy_adaptive']:
        other=predictions[label].set_index('id').loc[primary.index]
        d=(primary.pred-primary.true).abs().to_numpy()-(other.pred-other.true).abs().to_numpy()
        means=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)])
        lo,hi=np.percentile(means,[2.5,97.5]);rows.append(dict(comparator=label,difference=float(d.mean()),ci_low=float(lo),ci_high=float(hi),replicates=2000))
    pd.DataFrame(rows).to_csv(run/'paired_bootstrap.csv',index=False);return rows

def copy_ieee_template(paper):
    template=HERE/'ieee_template'/'ACCESS_latex_template_20260513'
    for f in template.iterdir():
        if f.suffix in ('.cls','.sty','.fd','.tfm','.pfb','.map','.bst') or f.name in ('logo.png','notaglinelogo.png','bullet.png'):
            shutil.copy2(f,paper/f.name)

def make_report(run,enc,fusion,policy,args):
    paper=HERE/'paper';(paper/'figures').mkdir(parents=True,exist_ok=True);(paper/'tables').mkdir(exist_ok=True)
    copy_ieee_template(paper)
    metrics_all=json.loads((run/'metrics.json').read_text());audit=json.loads((run/'audit.json').read_text());history=pd.read_csv(run/'history.csv')
    hardware=json.loads((run/'hardware.json').read_text());profile=json.loads((run/'profile.json').read_text());training=json.loads((run/'training_summary.json').read_text())
    calibration=json.loads((run/'calibration.json').read_text());replay=json.loads((run/'replay_check.json').read_text());manifest=pd.read_csv(run/'manifest.csv')
    predictions={p.stem.replace('predictions_',''):pd.read_csv(p) for p in run.glob('predictions_*.csv')}
    main=predictions['policy_three'];tau=-float('inf') if calibration['threshold']=='always' else float(calibration['threshold'])
    timings=latency_benchmark(enc,fusion,policy,manifest,args,run,tau)
    flops=profile_flops(enc,fusion,policy,manifest,args,run)
    cams=gradcam_examples(enc,fusion,manifest,main,args,paper,run)
    bootstrap=bootstrap_table(run,predictions)
    # At least six measured results plots, plus five individual Grad-CAM figures.
    fig,ax=plt.subplots(figsize=(6,3.5))
    for split in ['fit','dev','calibration','holdout']:
        ax.hist(manifest[manifest.split==split].age,bins=np.arange(0,253,12),histtype='step',linewidth=1.4,label=split)
    ax.set(xlabel='Reference bone age (months)',ylabel='Images',title='Age distribution by frozen partition');ax.legend();savefig(paper,'age_distribution',fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    x=np.arange(1,len(history)+1)
    axes[0].plot(x,history.loss,'o-',color='#007c83');axes[0].set(xlabel='Completed epoch',ylabel='Training objective')
    axes[1].plot(x,history.dev_mae,'o-',color='#17324d');axes[1].set(xlabel='Completed epoch',ylabel='Development MAE (months)')
    for ax in axes:ax.axvline(training['global_epochs']+.5,color='#cc6677',linestyle='--',label='Multiview starts');ax.legend()
    fig.tight_layout();savefig(paper,'training_curves',fig)
    fig,ax=plt.subplots(figsize=(4.8,4))
    ax.scatter(main.true,main.pred,s=9,alpha=.35,c=main.sex,cmap='coolwarm');ax.plot([0,240],[0,240],'k--',linewidth=1)
    ax.set(xlabel='Reference (months)',ylabel='Prediction (months)',title='Sealed holdout: always-three policy');savefig(paper,'prediction_scatter',fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4));err=main.pred-main.true
    axes[0].hist(err,bins=35,color='#007c83',alpha=.8);axes[0].set(xlabel='Signed error (months)',ylabel='Images')
    axes[1].scatter((main.pred+main.true)/2,err,s=7,alpha=.3);axes[1].axhline(err.mean(),color='black')
    for v in [err.mean()-1.96*err.std(ddof=1),err.mean()+1.96*err.std(ddof=1)]:axes[1].axhline(v,color='#cc6677',linestyle='--')
    axes[1].set(xlabel='Mean of prediction and reference',ylabel='Prediction minus reference');fig.tight_layout();savefig(paper,'errors_agreement',fig)
    fig,ax=plt.subplots(figsize=(6,3.5));labels=['global_multiview','policy_one','policy_two','policy_three','policy_adaptive']
    ax.plot([metrics_all[l]['mean_crops'] for l in labels[:4]],[metrics_all[l]['mae'] for l in labels[:4]],'o-',color='#007c83')
    for l in labels:ax.scatter(metrics_all[l]['mean_crops'],metrics_all[l]['mae']);ax.annotate(l.replace('policy_',''),(metrics_all[l]['mean_crops'],metrics_all[l]['mae']),xytext=(4,5),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Mean acquired local crops',ylabel='Holdout MAE (months)',title='Accuracy versus evidence budget');savefig(paper,'budget_frontier',fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    for label,color in [('global_baseline','#cc6677'),('fixed_three','#88ccee'),('policy_three','#007c83')]:
        p=predictions[label];ages=np.arange(0,241,48);vals=[np.abs(p[(p.true>=a)&(p.true<a+48)].pred-p[(p.true>=a)&(p.true<a+48)].true).mean() for a in ages[:-1]]
        axes[0].plot(np.arange(5),vals,'o-',label=label.replace('_',' '),color=color)
    axes[0].set_xticks(np.arange(5),['0--47','48--95','96--143','144--191','192--240'],rotation=25);axes[0].set(ylabel='MAE (months)',xlabel='Reference age band');axes[0].legend(fontsize=8)
    count=np.zeros(9,dtype=int)
    for s in main.selected.fillna(''):
        for j in str(s).split(','):
            if j:count[int(j)-1]+=1
    axes[1].imshow(count.reshape(3,3),cmap='Blues')
    for j,c in enumerate(count):axes[1].text(j%3,j//3,f'{j+1}\n{c}',ha='center',va='center',color='black')
    axes[1].set_title('Acquisition counts on 3 Ã— 3 grid');axes[1].set_xticks([]);axes[1].set_yticks([])
    fig.tight_layout();savefig(paper,'subgroups_selection',fig)
    diag=torch.load(run/'utility_diagnostics.pt',map_location='cpu',weights_only=False);ck=torch.load(run/'utility.pt',map_location='cpu',weights_only=False)
    ids=diag['validation_indices'];model,mu,sd=policy
    with torch.no_grad():
        upred=torch.cat([model(((diag['features'][ix].to(DEVICE)-mu)/sd)).cpu() for ix in ids.split(1024)]).numpy()
    utrue=diag['targets'][ids].numpy()
    fig,ax=plt.subplots(figsize=(5,4));take=np.random.default_rng(42).choice(len(utrue),min(4000,len(utrue)),replace=False)
    ax.scatter(utrue[take],upred[take],s=5,alpha=.15,color='#007c83');lim=max(1,float(np.percentile(np.abs(utrue),99)))
    ax.plot([-lim,lim],[-lim,lim],'k--');ax.set(xlabel='Actual model-defined gain (months)',ylabel='Predicted gain (months)',title='Fitting-only utility diagnostic split')
    savefig(paper,'utility_diagnostics',fig)
    utility_stats=dict(n=len(utrue),mae=float(np.abs(upred-utrue).mean()),rmse=float(np.sqrt(np.mean((upred-utrue)**2))),
                       positive_fraction=float(np.mean(utrue>0)*100),sign_accuracy=float(np.mean((upred>0)==(utrue>0))*100))
    save_json(run/'utility_statistics.json',utility_stats)
    # Machine-readable tables accompany every LaTeX table.
    def emit(name,caption,headers,rows,wide=False):
        table(paper,name,caption,headers,rows,wide)
        pd.DataFrame(rows,columns=headers).to_csv(run/f'table_{name}.csv',index=False)
    splitrows=[]
    for split in ['fit','dev','calibration','holdout','excluded']:
        d=manifest[manifest.split==split]
        splitrows.append([split,len(d),f'{d.age.min():.0f}--{d.age.max():.0f}' if len(d) else '--',int((d.sex==0).sum()),int((d.sex==1).sum())])
    emit('partitions','Frozen dataset partitions after hash-based exclusions. F = recorded female; M = recorded male.',
         ['Partition','$N$','Age (mo)','F','M'],splitrows)
    order=['global_baseline','global_multiview','fixed_three','random_three','policy_one','policy_two','policy_three','policy_no_disagreement','policy_adaptive']
    rows=[[tex(l.replace('_',' ')),f"{metrics_all[l]['mae']:.3f}",f"{metrics_all[l]['rmse']:.3f}",f"{metrics_all[l]['bias']:.3f}",f"{metrics_all[l]['within6']:.2f}",f"{metrics_all[l]['within12']:.2f}",f"{metrics_all[l]['mean_crops']:.2f}"] for l in order]
    emit('overall','Measured holdout results. All rows except the separately trained global baseline share the same selected image checkpoint; these are inference variants, not independently retrained architecture ablations.',
         ['Configuration','MAE','RMSE','Bias','$\leq6$ mo (\%)','$\leq12$ mo (\%)','Crops'],rows,True)
    rows=[]
    for sex,label in [(0,'Female'),(1,'Male')]:
        m=metrics(main[main.sex==sex]);rows.append([label,m['n'],f"{m['mae']:.3f}",f"{m['rmse']:.3f}",f"{m['bias']:.3f}"])
    emit('sex','Recorded-sex subgroup results for always-three BARE. Differences are descriptive; demographic fairness and clinical independence are not established.',
         ['Group','$N$','MAE','RMSE','Bias'],rows)
    rows=[]
    for a in range(0,240,48):
        d=main[(main.true>=a)&(main.true<(a+48 if a<192 else 241))];m=metrics(d)
        rows.append([f'{a}--{a+47 if a<192 else 240}',m['n'],f"{m['mae']:.3f}",f"{m['rmse']:.3f}",f"{m['within6']:.1f}"])
    emit('age_groups','Age-band holdout results. Ages and errors are in months.', ['Age','$N$','MAE','RMSE','$\leq6$ (\%)'],rows)
    rows=[[str(c['tau']),f"{c['mae']:.3f}",f"{c['rmse']:.3f}",f"{c['mean_crops']:.2f}",'Yes' if c['tau']==calibration['threshold'] else 'No'] for c in calibration['rows']]
    emit('calibration','Stopping-threshold calibration results, not holdout results. Select minimum crop count within 0.15 months MAE of the always-three policy.',
         ['$\tau$ (mo)','MAE','RMSE','Crops','Chosen'],rows)
    rows=[]
    for r in profile:
        if r['oom']:rows.append([r['mode'],'OOM','OOM','OOM','OOM'])
        else:rows.append([r['mode'],f"{r['allocated_mib']:.1f}",f"{r['reserved_mib']:.1f}",f"{r['sampled_device_peak_mib']:.0f}",f"{r['seconds_update']:.3f}"])
    emit('memory','Same-hardware admission profiling: 22 complete synthetic-input optimizer updates per mode; two timing warmups excluded. Memory is MiB; whole-device use is sampled, not a guaranteed continuous peak.',
         ['Mode','Allocated','Reserved','Device','s/update'],rows)
    rows=[[tex(r['mode']),f"{r['median_ms']:.2f}",f"{r['p95_ms']:.2f}",f"{r['mean_crops']:.2f}",f"{r['reserved_mib']:.1f}"] for r in timings]
    emit('latency','Original-image batch-one pipeline latency: decoding, normalization, CPU views, GPU encoders, policy and fusion. Five warmups then 50 IDs; ms latency and MiB reserved memory.',
         ['Mode','Median','P95','Crops','Memory'],rows)
    emit('replay','Deterministic three-view FP32 replay equivalence test at 64-pixel resolution. This numerical check does not establish statistical accuracy.',
         ['Quantity','Measured value'],[[tex(k.replace('_',' ')),f'{v:.3e}' if isinstance(v,float) else tex(v)] for k,v in replay.items()])
    rows=[[tex(r['comparator'].replace('_',' ')),f"{r['difference']:.3f}",f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"] for r in bootstrap]
    emit('bootstrap','Paired case-bootstrap MAE differences: always-three BARE minus comparator, in months; 2,000 replicates, 95\% percentile intervals. Negative values favor BARE. One training seed; no patient-level resampling is possible.',
         ['Comparator','$\Delta$ MAE','95\% CI'],rows)
    emit('utility','Fitting-only internal utility diagnostic set. Rows within an image may cross this diagnostic split; it is not an independent generalization estimate.',
         ['Quantity','Value'],[[tex(k.replace('_',' ')),f'{v:.3f}' if isinstance(v,float) else v] for k,v in utility_stats.items()])
    rows=[[str(c['id']),f"{c['true']:.0f}",f"{c['pred']:.2f}",f"{c['attribution_pred']:.2f}",','.join(map(str,c['selected'][1:]))] for c in cams]
    emit('gradcam_cases','Five predetermined Grad-CAM cases: smallest image ID in each 48-month band. Attribution runs in FP32; evaluation uses FP16.',
         ['ID','Target','Eval.','Attrib.','Crop IDs'],rows)
    # Conservative literature comparison; do not infer architecture parameters from backbone alone.
    lit=[
      ['Chen et al.','chen','2022','Attention crops + LDL','Annotation-free attention','Region discovery; distribution learning','No explicit marginal gain or laptop memory objective'],
      ['RFGA-Net','rfga','2024','InceptionV3 + RAA/FFA','Sex + attention','Fine-grained feature discrimination','No measured 4 GB training in source'],
      ['CFJLNet','cfjl','2022','Coarse/fine feature fusion','Attention','Cross-scale joint learning','Evidence is fused rather than budget-acquired'],
      ['Self-supervised attention','selfsupervised','2021','Region consistency','Weak annotation','Region consistency objectives','Acquisition utility is not supervised by error reduction'],
      ['Divide/conquer','lightweight','2023','Lightweight regional CNNs','Regional decomposition','Compact regional inference','Protocol and resource accounting differ'],
      ['BoGFF-Net','bogff','2025','Coarse-to-fine attention','Sex + triplet loss','Multiscale regression','Different encoder and fusion costs'],
      ['BoNet+','bonetplus','2026','Two-stream global/local','Keypoints / bounding boxes','Global transformer plus RFAConv','Additional annotations and RTX 4090 training'],
      ['DSGTF','stage','2026','Graph + stage-aware fusion','17 keypoints','Developmental-stage evidence fusion','Different 70/15/15 split and RTX 4090'],
      ['BARE','', 'This run','Shared '+args.backbone+' + utility','Sex; fixed nine-window grid','Marginal error-gain acquisition + replay','Single seed; prospective novelty; no SOTA claim']]
    rows=[[tex(x[0])+(r'~\cite{'+x[1]+'}' if x[1] else ''),x[2],tex(x[3]),tex(x[4]),tex(x[5]),tex(x[6])] for x in lit]
    emit('literature','Literature survey: model families, evidence sources and relation to BARE. This is a focused survey, not a systematic review.',
         ['Work','Year','Architecture','Supervision','Contribution','Relation / limitation'],rows,True)
    sota=[
      ['Chen et al.~\\cite{chen}','4.40','4.78','Official test 200','Attention-guided multiview','No keypoints','NR','NR','NR','NR','NR','NR','NR','NR'],
      ['RFGA-Net~\\cite{rfga}','3.34','4.02','Official test 200','InceptionV3 + RAA/FFA','Sex; attention','$576^2$','2$\\times$3090','36','100','NR','NR','NR','NR'],
      ['CFJLNet~\\cite{cfjl}','4.07','NR','Official test (as cited)','Coarse/fine attention','Attention','NR','NR','NR','NR','NR','NR','NR','NR'],
      ['BoGFF-Net~\\cite{bogff}','3.91','NR','Official test 200','Multiscale self-attention','Sex; triplet objective','NR','NR','NR','NR','NR','NR','NR','NR'],
      ['Divide/conquer~\\cite{lightweight}','3.90','NR','RSNA test; 0--20 yr','Lightweight regional CNNs','Regional decomposition','NR','NR','NR','NR','NR','NR','NR','NR'],
      ['BoNet+~\\cite{bonetplus}','3.81','NR','Official test 200','InceptionV3 + transformer','Keypoints / boxes','$500^2$','RTX 4090','17','150','NR','NR','NR','NR'],
      ['DSGTF~\\cite{stage}','4.82','NR','Custom test 1,892','InceptionV3 + EfficientNet-B1','17 keypoints','$299^2/240^2$','RTX 4090','8','100','NR','NR','NR','115 ms'],
      ['BARE (measured)',f"{metrics_all['policy_three']['mae']:.3f}",f"{metrics_all['policy_three']['rmse']:.3f}",'Official validation 1,425',args.backbone,'No keypoints',f'${args.global_size}^2/{args.local_size}^2$','RTX 3050 4 GB',str(args.batch)+' (eff. 8)',str(training['completed_epochs']),f"{(hardware['parameters_image_model']+sum(p.numel() for p in policy[0].parameters()))/1e6:.3f} M",f"{flops['counted_gflops']:.3f} G*",f"{training['max_reserved_mib']:.0f} MiB",f"{next(r['median_ms'] for r in timings if r['mode']=='policy_three'):.2f} ms"]]
    emit('sota_accuracy','Published strong-method comparison. Published values are not reproduced; test labels/splits and auxiliary supervision differ. BARE uses the untouched official validation set, not the absent 200-image official test. RFGA RMSE uses Table 1 (4.02); its narrative also prints 4.03. NR = not verified/reported in consulted material.',
         ['Method','MAE (mo)','RMSE (mo)','Evaluation','Backbone / fusion','Auxiliary input'],[r[:6] for r in sota],True)
    emit('sota_resources','Resource and hyperparameter comparison for the same methods. NR values are not inferred. Batch is image microbatch; BARE effective optimizer batch is eight. Parameters count the shared image model and one policy. VRAM is allocator-reserved for BARE. G* is profiled convolution/matmul GFLOPs only, excluding unsupported operations, not a complete total. Latencies on different GPUs are not directly comparable.',
         ['Method','Resolution','GPU','Batch','Epochs','Parameters','FLOPs','VRAM','Latency'],[[r[0]]+r[6:] for r in sota],True)
    optimization=[
      ['Chen et al.~\\cite{chen}','NR','NR','Distribution + regression','NR','NR','NR','NR'],
      ['RFGA-Net~\\cite{rfga}','SGD','NR','L1','12,611','1,425','200','NR'],
      ['CFJLNet~\\cite{cfjl}','NR','NR','NR','NR','NR','NR','NR'],
      ['BoGFF-Net~\\cite{bogff}','NR','NR','Triplet / regression','NR','NR','200','NR'],
      ['Divide/conquer~\\cite{lightweight}','NR','NR','NR','NR','NR','NR','NR'],
      ['BoNet+~\\cite{bonetplus}','Adam','$10^{-3}$','Smooth L1','12,611','1,425','200','NR'],
      ['DSGTF~\\cite{stage}','AdamW','NR','Multitask','8,827','1,892','1,892','NR'],
      ['BARE (measured)','AdamW','$10^{-4}$ / $3\\times10^{-4}$','MAE + CE + CDF',str(audit['counts']['fit']),str(audit['counts']['dev'])+' / '+str(audit['counts']['calibration']),'1,425 validation',f"{training['image_training_seconds']/3600:.2f} h"]]
    emit('sota_optimization','Optimization and dataset-size comparison. BARE learning rates shown are initial global encoder/head rates; multiview rates are halved. BARE development/calibration are internal training partitions, and its evaluation set is supplied validation. Training time excludes final caches, policy fitting and reporting. NR values are not guessed.',
         ['Method','Optimizer','Initial LR','Objective','Fitting $N$','Dev / cal $N$','Evaluation $N$','Train time'],optimization,True)
    pd.DataFrame(sota,columns=['method','mae','rmse','evaluation','architecture','supervision','resolution','gpu','batch','epochs','parameters','flops','vram','latency']).to_csv(run/'sota_comparison.csv',index=False)
    write_manuscript(paper,run,metrics_all,training,hardware,timings,calibration,cams,audit,bootstrap)
    compile_and_package(paper,run)

def write_manuscript(paper,run,m,train,hw,timings,cal,cams,audit,bootstrap):
    base=m['global_baseline'];main=m['policy_three'];fixed=m['fixed_three'];adaptive=m['policy_adaptive']
    delta=main['mae']-base['mae'];effect='lower' if delta<0 else 'higher'
    result=f"The always-three acquisition policy attained {main['mae']:.3f} months MAE and {main['rmse']:.3f} months RMSE on {main['n']} sealed validation images. Its MAE was {abs(delta):.3f} months {effect} than the separately trained global baseline ({base['mae']:.3f} months)."
    body=r'''% Preserve TeX's year register: the IEEE Access class defines a conflicting year setter.
\let\BAREprimitiveyear\year
\documentclass[nolineno]{ieeeaccess}
\let\year\BAREprimitiveyear
\renewcommand{\theyear}{2026}
\vol{Draft}
\usepackage{cite,amsmath,amssymb,graphicx,booktabs,tabularx,array}
\usepackage{algorithm,algpseudocode}
\usepackage{tikz}\usetikzlibrary{arrows.meta,positioning}
% RGB fallback avoids incompatible spot-color definitions when TikZ loads xcolor.
\definecolor{accessblue}{RGB}{0,103,143}
\usepackage[hidelinks]{hyperref}
\newcommand{\sg}{\operatorname{stopgrad}}
\newcommand{\MAE}{\operatorname{MAE}}
\newcommand{\softmax}{\operatorname{softmax}}
\setlength{\emergencystretch}{1em}
\begin{document}
\history{Experimental research draft. Run date: 30 September 2026. Not accepted or published.}
\doi{}
\title{BARE: Budgeted Regional Evidence Acquisition with Gradient-Replayed Training for Bone Age Estimation on a 4 GB GPU}
\author{}
\address{}
\corresp{}
\markboth{BARE: Research Draft}{BARE: Research Draft}
\begin{abstract}
Memory constraints make high-resolution multiview bone age estimation difficult to train on consumer laptops. This study implements Budgeted Acquisition of Regional Evidence, a shared-encoder framework that estimates the marginal value of an additional image crop from the evidence already acquired. A @BACKBONE@ encoder supplies global and local tokens to a compact attention module and an age-distribution head. A separate utility network learns signed reductions in absolute prediction error using fitting images only. Deterministic token-gradient replay retains one encoder view's activation graph at a time. The implementation was executed on an NVIDIA RTX 3050 laptop with 4 GB video memory using a frozen fitting, development, calibration and final-evaluation protocol. @RESULT@ Adaptive stopping was calibrated separately and reduced the mean crop count to @CROPS@, with holdout MAE @ADAPTIVE@ months. Results include memory admission, numerical replay checks, batch-one latency, subgroup analyses and five predetermined gradient-attribution cases. The experiment uses one training seed and an overnight budget; it does not demonstrate official challenge-test superiority or clinical validity. The contribution is a reproducible evaluation of state-conditioned regional acquisition under a concrete hardware constraint, with clearly qualified comparisons to published methods.
\end{abstract}
\begin{keywords}Bone age estimation, counterfactual utility, gradient replay, medical image analysis, memory-constrained learning, regional evidence acquisition.\end{keywords}
\titlepgskip=-15pt
\maketitle

\section{Introduction}
\PARstart{P}{ediatric} bone age estimation uses skeletal appearance in hand radiographs as a developmental indicator. Public benchmarks accelerated convolutional approaches, but their evaluation protocols and use of auxiliary annotations vary~\cite{halabi,larson,spampinato,lee,iglovikov,bonexpert}. The RSNA challenge separates 12,611 training, 1,425 validation and 200 test images~\cite{halabi}. A method evaluated on the public validation labels cannot be ranked directly against methods evaluated on the smaller official test set.

Attention-guided localization, local/global fusion and distribution learning are established bone-age techniques~\cite{chen,cfjl,selfsupervised,rfga,bonet,bogff,bonetplus,integral}. Regional CNNs can also be designed for lightweight operation~\cite{lightweight}. BARE investigates a narrower question: can the expected \emph{incremental} value of a crop be learned without encoding that unseen crop at inference, while preserving detailed regional supervision under a laptop memory budget?

Our implementation contributes three testable elements. First, it supervises crop utility with the change in model absolute error caused by adding a crop to a sampled evidence subset. Second, a small evidence-set model combines ordinal age learning with local descriptors, recorded sex, and regional disagreement. Third, replay separates fusion differentiation from encoder backpropagation, applying the existing gradient-caching principle~\cite{gradcache} to shared multiview bone-age training. The combination is a candidate research contribution; this paper does not claim that gradient caching, attention, or distribution learning were invented here.

\section{Related work}
\subsection{Global, regional, and developmental evidence}
Earlier systems established global convolutional prediction and hand segmentation~\cite{spampinato,lee,larson,iglovikov}. Attention-guided crops reduce reliance on explicit localization annotations~\cite{chen,selfsupervised}. Coarse/fine fusion, fine-grained discrimination and multiscale regression exploit local skeletal detail~\cite{cfjl,rfga,bogff}. Two-stream and pose-guided methods use additional regional structure~\cite{bonet,bonetplus}; developmental stage-aware graph fusion is also prior art~\cite{stage}. Ensemble gains illustrate that complementary predictions can matter~\cite{ensemble}. Table~\ref{tab:literature} positions BARE relative to these families.
\input{tables/literature.tex}

\subsection{Efficient representation and memory control}
MobileNet inverted residuals and architecture search motivate the shared encoder~\cite{mobilenetv2,mobilenetv3}; residual networks, EfficientNet, squeeze/excitation and channel/spatial attention supply alternative designs~\cite{resnet,efficientnet,squeeze,cbam}. Dense prediction architectures such as U-Net address segmentation at an additional training cost~\cite{unet}. Transformer, vision transformer and shifted-window work motivate token fusion~\cite{transformer,vit,swin}, although this implementation attends only to a small set of pooled view descriptors.
Mixed precision and activation recomputation are complementary memory techniques~\cite{mixed,checkpoint}. Batch and layer normalization have different behavior under small batches~\cite{batchnorm,layernorm}. We freeze encoder batch-normalization statistics and use layer-normalized evidence fusion. PyTorch provides the executed training substrate~\cite{pytorch}; AdamW supplies the optimizer~\cite{adamw}.

\subsection{Ordinal outputs, robustness and interpretation}
Label distributions and ordinal relationships are useful for age prediction~\cite{chen,coral}; integral prediction is another established strategy~\cite{integral}. Entropy and regional disagreement are not automatically calibrated confidence~\cite{calibration,uncertainty}. Distribution shift and radiographic perturbations can degrade bone-age algorithms~\cite{stress,bias}. We therefore provide descriptive error analyses and regression-target Grad-CAM~\cite{gradcam}, while avoiding claims that saliency demonstrates clinical reasoning or causal anatomy.

\section{Materials and methods}
\subsection{Dataset, audit and frozen partitions}
We resolve every CSV image ID to one PNG, validate label range and the explicit true/false sex flag, and decode every radiograph. Original file hashes, decoded-pixel hashes and low-resolution difference hashes are recorded. Exact cross-holdout pixel duplicates and identical difference-hash matches are excluded from fitting/development/calibration. Within the supplied training set, approximate sex/12-month stratification uses ten grouped folds: eight fitting folds, one development fold and one calibration fold. The supplied validation set is sealed for final reporting. Patient independence and non-identical near duplicates remain unverified. Hash grouping is conservative and can also group different images with similar low-resolution structure.

Image preprocessing normalizes the 0.5th--99.5th percentile range to an 8-bit grayscale PNG cache, caps the master image's longest side at 1,280 pixels, and letterboxes the hand. This normalization is lossless relative to the stored 8-bit cache, but conversion from an original higher-bit-depth image is an explicit quantization. The global view is $@GLOBALSIZE@\times@GLOBALSIZE@$; nine overlapping crops are extracted before downsampling to $@LOCALSIZE@\times@LOCALSIZE@$. Centers are $(0.25,0.50,0.75)^2$ with normalized side 0.45. The windows are a spatial cover, not verified anatomical landmarks.

\subsection{Shared encoder and evidence-set fusion}
A generic ImageNet-pretrained @BACKBONE@~\cite{resnet,mobilenetv3} has its classifier removed. The @WIDTH@-dimensional pooled feature is projected to 128 dimensions; grayscale is repeated into three input channels. For view $j$ with coordinate vector $b_j$, its token is
\begin{equation}
z_j=P_\theta\operatorname{GAP}(f_\theta(x_j)),\quad
t_j=z_j+W_b b_j+e_{v_j}.
\end{equation}
Two pre-normalized transformer blocks with four heads and 256-unit feedforward layers operate on the view tokens plus a sex-conditioned query. For evidence subset $S$, the query representation $h_S$ predicts
\begin{equation}
p_{S,m}=\softmax(\ell(h_S))_m,\quad
\hat y_S=\sum_{m=0}^{240}m\,p_{S,m}.
\end{equation}
Recorded sex is a 16-dimensional learned embedding. No chronological age enters the model. Global-only and random subset paths share parameters with the full ten-view path.

\begin{figure}[!t]\centering
\begin{tikzpicture}[node distance=6mm,box/.style={draw,rounded corners,align=center,text width=65mm,minimum height=9mm,font=\small},arr/.style={-{Latex},thick}]
\node[box](a){Hand radiograph / normalized CPU master};
\node[box,below=of a](b){Global view + nine candidate windows};
\node[box,below=of b](c){One shared @BACKBONE@ encoder\\128-D tokens, one view graph at a time};
\node[box,below=of c](d){Evidence-set attention + recorded sex};
\node[box,below=of d](e){Age distribution + signed crop-utility head};
\node[box,below=of e](f){Acquire next crop / stop at budget 3\\Expected age in months};
\draw[arr](a)--(b);\draw[arr](b)--(c);\draw[arr](c)--(d);\draw[arr](d)--(e);\draw[arr](e)--(f);
\draw[arr,dashed](f.east)--++(.45,0)|-(b.east);
\end{tikzpicture}
\caption{Implemented BARE architecture. Training enumerates all ten views with replay; deployment encodes only the global image and acquired regions.}\label{fig:architecture}\end{figure}

\subsection{Age and acquisition losses}
A normalized Gaussian target $q_m(y)\propto\exp[-(m-y)^2/8]$ supplies soft labels with a two-month width. The objective combines absolute age error, cross entropy and ordinal CDF distance:
\begin{equation}
\begin{split}
\mathcal L_{\rm age}(S,y)&=|\hat y_S-y|/12
-0.1\sum_m q_m(y)\log p_{S,m}\\
&\quad+\frac{0.5}{240}\sum_{m=0}^{239}|F_{p_S}(m)-F_q(m)|.
\end{split}
\end{equation}
Average this loss over the global-only subset, all ten views and two random subsets containing one to three local views. A 0.1-weighted mean singleton-view loss provides weak regional supervision. Background crops may receive noisy age supervision; a retrained weight ablation remains future work.

For fitting image $i$, subset $S$ and unacquired crop $j$, define the signed gain
\begin{equation}
u^*_{i,j\mid S}=|\hat y_{i,S}-y_i|-|\hat y_{i,S\cup\{j\}}-y_i|.
\end{equation}
Negative gains are retained. These are model-defined counterfactual comparisons rather than causal effects. A two-hidden-layer utility MLP predicts this gain from detached fused representation, age distribution, sex embedding, subset size, candidate geometry, normalized regional disagreement and entropy, and three cheap low-resolution statistics (mean, standard deviation, gradient energy). No unseen high-resolution token enters the utility input. The fitting-only utility objective is Huber loss with a two-month transition. Input means and scales are estimated from fitting-only utility data.

\subsection{Replay and numerical correctness}
Graph-free encoder passes first cache token values. Detached token leaves then allow differentiation of the complete fusion objective. With $g_j=\partial\mathcal L/\partial z_j$ cached, one view is replayed at a time:
\begin{equation}
\nabla_\theta\mathcal L
=\sum_j J_\theta(z_j)^\top g_j
=\nabla_\theta\sum_j\langle z_j,\sg(g_j)\rangle.
\end{equation}
Weights, augmented inputs and normalization states remain fixed until the complete optimizer update. Encoder stochastic layers are absent and running BN statistics are frozen. Fusion dropout is sampled once in the token-leaf pass; the encoder replay uses the resulting cached gradients. Under mixed precision, token gradients already include the gradient-scaler factor and are not scaled twice.

\begin{algorithm}[!t]\caption{One multiview replay update}\label{alg:replay}
\begin{algorithmic}[1]
\State Sample and retain a single master augmentation
\For{each global/local view $j$}
\State Cache $z_j=f_\theta(x_j)$ without an encoder graph
\EndFor
\State Create gradient-enabled token leaves; compute subset losses
\State Differentiate fusion once; cache scaled token gradients $g_j$
\For{each view $j$}
\State Recompute the same $z_j$ with encoder graph
\State Backpropagate $\langle z_j,\sg(g_j)\rangle$ and release graph
\EndFor
\State After the accumulation window: unscale, clip, step once
\end{algorithmic}\end{algorithm}

\subsection{Training, calibration and final inference}
The executed run uses seed 42, AdamW weight decay $10^{-4}$, @GLOBALBATCH@-image global batches and @LOCALBATCH@-image multiview batches, each accumulated to eight images per optimizer update, FP16 encoder autocast, FP32 fusion/losses, and a cosine learning-rate schedule. Global encoder/head learning rates start at $10^{-4}/3\times10^{-4}$; multiview rates start at $5\times10^{-5}/1.5\times10^{-4}$. Mild brightness/contrast jitter and $\pm10^\circ$ rotations are applied coherently to the master image during multiview training. Global-only loading instead letterboxes and augments at global resolution, avoiding unnecessary master-resolution transforms. DataLoader prefetching is limited to one batch per worker. There is no scale augmentation in this implementation. Global warmup is followed by multiview training, with an overnight wall-time allowance and development-based checkpoint selection. The selected checkpoint completed @EPOCHS@ multiview epochs after @GLOBAL@ global epochs.

Unlike the initial concept's more ambitious schedule, this executed version uses one fitting-only deterministic token cache per selected checkpoint and random subsets to fit the policy; augmented utility caches and trajectory-refinement cycles are not implemented. This difference is documented rather than implied to have been tested. A no-disagreement policy is trained on the same utility targets, with that feature masked.

Starting with the global token, select the unobserved crop with greatest predicted signed gain, encode it, update the evidence state and repeat up to three times. For adaptive inference, stop when the greatest gain is no larger than $\tau$. The calibration partition chooses among always-acquire, 0, 0.25, 0.5 and 1.0 months, minimizing crop count within 0.15 months MAE of the always-three policy. Model selection, fitting normalization and threshold selection precede all final holdout predictions.

\section{Results}
\subsection{Dataset accounting and optimization}
Table~\ref{tab:partitions} gives exact measured counts, including exclusions. Figure~\ref{fig:age_distribution} shows age imbalance. The mean losses and development errors in Fig.~\ref{fig:training_curves} describe optimization of this one-seed run; they are not estimates of uncertainty across repeated training.
\input{tables/partitions.tex}
@FIG_DATA@
@FIG_TRAIN@

\subsection{Predictive performance and evidence budget}
@RESULT@ Table~\ref{tab:overall} includes the separately trained global baseline and same-checkpoint inference variants. The three fixed crops have MAE @FIXED@ months. The greedy utility policy is evaluated even if it is worse than a baseline: there is no final-holdout-based switching. Figures~\ref{fig:prediction_scatter} and \ref{fig:errors_agreement} show prediction agreement and signed errors. Gaussian limits in the agreement plot are descriptive and should not be interpreted as clinically acceptable bounds.
\input{tables/overall.tex}
@FIG_SCATTER@
@FIG_ERRORS@
Figure~\ref{fig:budget_frontier} and Table~\ref{tab:calibration} quantify the budget/stopping tradeoff. The adaptive configuration uses @CROPS@ mean local crops and obtains @ADAPTIVE@ months holdout MAE. The calibration-set tolerance is not a guarantee on unseen patients.
\input{tables/calibration.tex}
@FIG_BUDGET@

\subsection{Subgroups and paired comparisons}
Tables~\ref{tab:sex} and \ref{tab:age_groups} describe errors by recorded sex and age band. Figure~\ref{fig:subgroups_selection} combines age-band comparisons with actual acquisition frequencies. Rare age ranges warrant particular caution. Table~\ref{tab:bootstrap} reports case-level paired bootstrap intervals. These intervals quantify holdout sampling conditional on one trained checkpoint, not training-seed variability. Image-level resampling cannot substitute for patient-group resampling when patient identifiers are unavailable.
\input{tables/sex.tex}
\input{tables/age_groups.tex}
\input{tables/bootstrap.tex}
@FIG_SUBGROUP@

\subsection{Memory, latency and replay fidelity}
The shared image model has @PARAMS@ parameters. The run's largest recorded allocator-reserved memory was @VRAM@ MiB; whole-device samples are tracked separately. Table~\ref{tab:memory} compares joint and replay training on identical hardware and synthetic inputs, including optimizer steps. The joint and replay admission outcomes are reported individually; an out-of-memory configuration is marked OOM. A configuration whose allocated or reserved CUDA memory exceeds the dedicated device capacity is not considered to fit the 4 GB budget, even if the Windows driver permits its completion. Replay retains one branch graph at a time. It is a memory admission experiment rather than a predictive-accuracy experiment. Table~\ref{tab:replay} verifies deterministic FP32 gradients on a smaller three-view case. Table~\ref{tab:latency} measures the original-image batch-one pipeline, including CPU preprocessing. Always-three median latency is @LATENCY@ ms. Full encoding and training require more work than small cached-token fusion alone, so cache-based prediction timings are not presented as deployment latency.
\input{tables/memory.tex}
\input{tables/replay.tex}
\input{tables/latency.tex}

\subsection{Utility diagnostics and five Grad-CAM cases}
Figure~\ref{fig:utility_diagnostics} and Table~\ref{tab:utility} show fitting-only diagnostic utility prediction. Because rows from one image can appear in both diagnostic subsets, this internal split is not independent image-level validation; final policy performance is assessed on the sealed holdout. The no-disagreement inference variant in Table~\ref{tab:overall} uses a separately fitted policy with a common image encoder. It is not a retrained image-model ablation.
\input{tables/utility.tex}
@FIG_UTILITY@
Five predetermined cases appear in Figs.~\ref{fig:gradcam_1}--\ref{fig:gradcam_5} and Table~\ref{tab:gradcam_cases}. We take the smallest image ID in each prespecified 48-month band, without choosing attractive maps or low errors. Grad-CAM differentiates the final expected age with respect to the shared encoder's last convolutional maps along a \emph{fixed acquired subset}. It does not differentiate the discrete acquisition decision. For activation channel $k$,
\begin{equation}
\alpha_k=\frac1{HW}\sum_{u,v}\frac{\partial\hat y}{\partial A^k_{uv}},\quad
G=\operatorname{ReLU}\!\left(\sum_k\alpha_kA^k\right).
\end{equation}
Positive attribution highlights features locally increasing the scalar predicted age; omitted negative attribution may also carry evidence. Maps are min/max normalized per view, and their colors are not comparable magnitudes across patients. FP32 attribution predictions and FP16 evaluation predictions are reported separately. Saliency is a qualitative diagnostic and does not establish causal or anatomically correct reasoning.
\input{tables/gradcam_cases.tex}
@GRADCAM_FIGURES@

\subsection{Comparison with strong published methods}
Tables~\ref{tab:sota_accuracy}, \ref{tab:sota_resources} and \ref{tab:sota_optimization} give reported accuracy, model/resource characteristics and optimization/dataset context. They deliberately retain protocol labels and NR entries. Published MAEs are not reproduced here, and many methods use the official 200-image test set or additional landmarks. BARE's final set is the official 1,425-image validation set. DSGTF uses its own 70/15/15 partition. GPU-dependent latency and unavailable FLOPs/parameter counts prevent an all-numeric resource ranking. No numerical comparison in these tables supports a claim of official-test SOTA.
\input{tables/sota_accuracy.tex}
\input{tables/sota_resources.tex}
\input{tables/sota_optimization.tex}
\clearpage

\section{Discussion}
The appropriate interpretation is determined by the measured comparison, not by the intended novelty. In this run, BARE's always-three MAE is @DIRECTION@ the trained global baseline. @INTERPRETATION@ The distinction between saliency, single-crop informativeness and \emph{marginal} value is still important: the policy learns error changes conditional on an existing subset, and additional evidence can be redundant or harmful. However, these utility targets depend on the current image model and can become obsolete after image-model refinement.

Replay controls simultaneous activation retention but increases encoder work. A graph-free cache pass followed by replay forwards/backwards is slower than ordinary joint backpropagation when sufficient memory exists. This makes the accuracy--memory--time frontier the relevant resource question. The memory profile here must not be generalized to larger backbones, resolutions, batches or other laptop processes without profiling them.

The spatial grid avoids extra keypoint labels but cannot guarantee that a window contains the biologically relevant region. Preprocessing quantization and full-frame letterboxing may affect local detail. Frozen normalization facilitates deterministic replay but may limit adaptation of a generic pretrained encoder to radiographs. Weak singleton age targets can penalize background crops. Utility proxies may not identify unseen regions that differ primarily in fine detail. These choices require independently retrained ablations before attributing gains to individual components.

\section{Limitations and reproducibility}
The experiment completes one training seed under an overnight allowance and does not prove convergence. It has no independent external hospital cohort, no prospective assessment, no verified patient identifiers and no official 200-image challenge test. Broad robustness sweeps, multiple-seed retraining, trajectory-aware policy refitting, utility-cache augmentation and formal anatomical localization assessment are not completed. Entropy/disagreement are acquisition features, not calibrated clinical uncertainty. Case-level bootstrap intervals cannot address unmeasured site or population shifts.

The release includes Python source, a frozen manifest, configuration, checkpoints, per-epoch logs, numeric tables, per-image holdout predictions, utility provenance, five saliency panels, IEEE Access class/fonts and the complete LaTeX project. The paper archive excludes source radiographs, pretrained weights and training caches. Missing comparison parameters are explicitly NR; authors should verify source details and update the contemporaneous literature comparison before any submission. Author, affiliation and corresponding-author fields are blank at the user's request.

\section{Conclusion}
This executed study evaluates memory-bounded multiview bone-age learning and model-defined crop utility on a consumer CUDA laptop. @RESULT@ The results support only the measured scope of this one-seed validation experiment. More complete evidence is needed to establish official-test competitiveness, generalization or clinical usefulness.

\section*{Acknowledgment}
The RSNA Pediatric Bone Age Challenge data are credited to the original challenge organizers and contributing institutions; use is governed by the dataset terms. OpenAI Codex assisted with the research concept, implementation, literature retrieval, manuscript drafting, reporting code and compilation checks. Plots and attribution images are computed from the executed experiment rather than generated as illustrative predictions. Human authors must review the scientific claims, references, authorship, data-use terms and reproducibility before submission.
\input{references.tex}
\EOD
\end{document}
'''
    figures={'@FIG_DATA@':figure(paper,'age_distribution','Age distribution in the frozen partitions, computed from the audited manifest.'),
             '@FIG_TRAIN@':figure(paper,'training_curves','Executed training objectives and development MAE. Dashed line marks transition from global to multiview training.',True),
             '@FIG_SCATTER@':figure(paper,'prediction_scatter','Always-three BARE predictions versus reference ages on the sealed holdout; colors denote recorded sex.'),
             '@FIG_ERRORS@':figure(paper,'errors_agreement','Signed error distribution and prediction--reference agreement. Horizontal dashed lines denote descriptive mean $\pm1.96$ standard deviations.',True),
             '@FIG_BUDGET@':figure(paper,'budget_frontier','Measured holdout MAE versus acquired evidence budget. The adaptive threshold was selected using calibration images only.'),
             '@FIG_SUBGROUP@':figure(paper,'subgroups_selection','Age-band MAE and acquired-window counts. Grid positions correspond to candidate locations on the letterboxed image.',True),
             '@FIG_UTILITY@':figure(paper,'utility_diagnostics','Predicted versus actual model-defined marginal gain on the fitting-only internal diagnostic subset; not an independent image-level validation set.'),
             '@GRADCAM_FIGURES@':'\n'.join(figure(paper,c['figure'],f"Grad-CAM case {i+1}, image {c['id']}, reference {c['true']:.0f} months. Whole-hand and first acquired crop show positive scalar-age attribution. Selection is fixed during differentiation; color intensity is normalized per map.",True) for i,c in enumerate(cams))}
    config=json.loads((run/'config.json').read_text())
    backbone=config.get('backbone','mobilenet_v3_large')
    replacements={'@GLOBALSIZE@':str(config.get('global_size',384)),'@LOCALSIZE@':str(config.get('local_size',256)),'@BACKBONE@':backbone.replace('_',' ').title(),'@WIDTH@':str({'resnet50':2048,'efficientnet_b2':1408,'mobilenet_v3_large':960}[backbone]),'@GLOBALBATCH@':str(config.get('global_batch',8)),'@LOCALBATCH@':str(config.get('batch',2)),'@RESULT@':result,'@CROPS@':f"{adaptive['mean_crops']:.2f}",'@ADAPTIVE@':f"{adaptive['mae']:.3f}",'@FIXED@':f"{fixed['mae']:.3f}",
                  '@EPOCHS@':str(train['multiview_epochs']),'@GLOBAL@':str(train['global_epochs']),'@PARAMS@':f"{hw['parameters_image_model']:,}",
                  '@VRAM@':f"{train['max_reserved_mib']:.1f}",'@LATENCY@':f"{next(t['median_ms'] for t in timings if t['mode']=='policy_three'):.2f}",
                  '@DIRECTION@':'lower than' if delta<0 else 'higher than',
                  '@INTERPRETATION@':'This is consistent with the benefit of added regional evidence in this setting, while the paired fixed/random comparisons determine whether learned selection adds value.' if delta<0 else 'The intended accuracy improvement is not demonstrated against that baseline. The result should be reported as a negative or inconclusive acquisition experiment rather than a SOTA advance.'}
    for k,v in {**figures,**replacements}.items():body=body.replace(k,v)
    if '@RESULT@' in body:raise ValueError('Unfilled report value')
    (paper/'bare_ieee_access.tex').write_text(body,encoding='utf-8')
    (paper/'README.md').write_text('Compile bare_ieee_access.tex with pdfLaTeX twice. All required IEEE template files, figures and tables are included. Results are generated from one executed run; see results/metrics.json and run configuration. Published comparisons use different protocols and NR means not verified/reported. No official-test SOTA claim is made. Author details must be supplied before submission.\n',encoding='utf-8')

def compile_and_package(paper,run):
    # Multi-file project requires the existing local MiKTeX distribution; standalone editor compiler may not resolve assets.
    exe=Path(r'C:\Users\AVISHEK\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe')
    diagnostic=dict(success=False,attempted=False)
    if exe.exists():
        diagnostic['attempted']=True
        try:
            for iteration in range(2):
                proc=subprocess.run([str(exe),'--disable-installer','-interaction=nonstopmode','-halt-on-error','bare_ieee_access.tex'],cwd=paper,capture_output=True,text=True,timeout=180)
                (run/f'latex_compile_{iteration+1}.txt').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
                if proc.returncode:break
            diagnostic.update(success=proc.returncode==0,returncode=proc.returncode)
        except Exception as ex:diagnostic['error']=str(ex)
    save_json(run/'latex_compile.json',diagnostic)
    results=paper/'results';results.mkdir(exist_ok=True)
    for f in run.iterdir():
        if f.suffix in ('.csv','.json') and f.name!='manifest.csv':shutil.copy2(f,results/f.name)
    # Manifest carries local image paths; reproducibility subset keeps IDs, partition, hashes and metadata only.
    pd.read_csv(run/'manifest.csv').drop(columns=['path','cached']).to_csv(results/'manifest.csv',index=False)
    provenance=HERE/'references_verified.json'
    if provenance.exists():shutil.copy2(provenance,results/provenance.name)
    for f in HERE.glob('*.py'):shutil.copy2(f,results/f.name)
    (results/'environment.txt').write_text(subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    archive=HERE/'BARE_IEEE_Access_LaTeX.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in paper.rglob('*'):
            if f.is_file() and f.suffix not in ('.aux','.log','.out','.synctex','.tmp'):z.write(f,f.relative_to(paper))
    save_json(run/'artifact_inventory.json',dict(zip=str(archive),main=str(paper/'bare_ieee_access.tex'),
              references=len(json.loads(provenance.read_text(encoding='utf-8'))),results_figures=len(list((paper/'figures').glob('*.png'))),
              gradcam_figures=len(list((paper/'figures').glob('gradcam_*.png'))),tables=len(list((paper/'tables').glob('*.tex'))),
              compile=diagnostic,pdf_visual_qa='pending human/agent inspection; not confirmed by compilation alone'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(HERE/'runs'/'overnight'));args0=p.parse_args();run=Path(args0.run)
    config=json.loads((run/'config.json').read_text());args=argparse.Namespace(**config);torch.set_num_threads(4)
    ck=torch.load(run/'best.pt',map_location=DEVICE,weights_only=False);enc=Encoder(False,backbone=config.get('backbone','mobilenet_v3_large')).to(DEVICE);enc.load_state_dict(ck['encoder']);enc.eval()
    fusion=Fusion().to(DEVICE);fusion.load_state_dict(ck['fusion']);fusion.eval();policy=load_utility(run/'utility.pt')
    make_report(run,enc,fusion,policy,args)

if __name__=='__main__':main()

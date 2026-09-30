"""Populate the open paper with actual early-checkpoint DEVELOPMENT results.

Never reads holdout images for prediction, never claims the full acquisition model is fitted.
Final overnight report replaces this interim manuscript in the same .tex path.
"""
import argparse, ast, hashlib, json, shutil, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from bare import HERE, DEVICE, Encoder, Fusion, loader, metrics, save_json, MEAN, STD
from report import table, figure, savefig, tex, original_views

def constant_rows(name):
    tree=ast.parse((HERE/'report.py').read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            result=[]
            for element in node.value.elts:
                try:result.append(ast.literal_eval(element))
                except (ValueError,TypeError):break
            return result
    raise ValueError(name)

def main():
    torch.set_num_threads(2)
    run=HERE/'runs'/'overnight';out=run/'preliminary';out.mkdir(exist_ok=True)
    paper=HERE/'paper';(paper/'figures').mkdir(exist_ok=True);(paper/'tables').mkdir(exist_ok=True)
    args=argparse.Namespace(**json.loads((run/'config.json').read_text()))
    manifest=pd.read_csv(run/'manifest.csv');dev=manifest[manifest.split=='dev'].sort_values('id')
    # Snapshot a saved checkpoint; independently record hash, epoch and evaluation protocol.
    snapshot=out/'global_checkpoint_snapshot.pt'
    for retry in range(3):
        try:
            shutil.copy2(run/'global_best.pt',snapshot)
            ck=torch.load(snapshot,map_location=DEVICE,weights_only=False);break
        except Exception:
            if retry==2:raise
            time.sleep(1)
    enc=Encoder(False).to(DEVICE).eval();enc.load_state_dict(ck['encoder'])
    fusion=Fusion().to(DEVICE).eval();fusion.load_state_dict(ck['fusion'])
    rows=[]
    with torch.no_grad():
        for views,sex,y,desc,ids in loader(dev,batch=8,workers=0,global_only=True,sizes=(args.global_size,args.local_size)):
            with torch.autocast(DEVICE.type,dtype=torch.float16):z=enc(views[0].to(DEVICE)).float().unsqueeze(1)
            mean,p,h=fusion(z,sex.to(DEVICE),torch.tensor([0],device=DEVICE))
            rows.extend(dict(id=int(i),true=float(t),sex=int(s),pred=float(pred),crops=0) for i,t,s,pred in zip(ids,y,sex,mean.cpu()))
    predictions=pd.DataFrame(rows);predictions.to_csv(out/'development_predictions.csv',index=False)
    m=metrics(predictions);fit=manifest[manifest.split=='fit'];median=float(fit.age.median())
    global_prior=predictions.copy();global_prior['pred']=median
    sex_prior=predictions.copy();medians=fit.groupby('sex').age.median();sex_prior['pred']=sex_prior.sex.map(medians)
    hardware=json.loads((run/'hardware.json').read_text());profile=json.loads((run/'profile.json').read_text())
    replay=json.loads((run/'replay_check.json').read_text());checks=json.loads((run/'implementation_checks.json').read_text())
    history=pd.read_csv(run/'history.csv')
    provenance=dict(protocol='development-only preliminary global warmup; NOT final BARE or official test',
          n=len(predictions),checkpoint_epoch=ck['epoch'],checkpoint_phase=ck['phase'],
          checkpoint_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),metrics=m,
          stage_at_capture=json.loads((run/'status.json').read_text()),author_fields='blank',
          source_rows=dev.id.tolist(),figure_case_selection='smallest development image ID in each prespecified 48-month age band',
          holdout_evaluated=False)
    save_json(out/'provenance.json',provenance)
    print('MEASURED DEVELOPMENT METRICS',m,flush=True)
    # Six non-saliency results figures, each derived from real logs/data/predictions.
    fig,ax=plt.subplots(figsize=(6,3.5))
    for split in ['fit','dev','calibration','holdout']:
        ax.hist(manifest[manifest.split==split].age,bins=np.arange(0,253,12),histtype='step',linewidth=1.5,label=split)
    ax.set(xlabel='Reference age (months)',ylabel='Images',title='Audited partition age distributions');ax.legend();savefig(paper,'prelim_data',fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4));x=np.arange(1,len(history)+1)
    axes[0].plot(x,history.loss,'o-',color='#007c83');axes[0].set(xlabel='Completed global epoch',ylabel='Mean training objective')
    axes[1].plot(x,history.dev_mae,'o-',color='#17324d');axes[1].set(xlabel='Completed global epoch',ylabel='Development MAE (months)')
    fig.tight_layout();savefig(paper,'prelim_learning',fig)
    fig,ax=plt.subplots(figsize=(5,4));ax.scatter(predictions.true,predictions.pred,c=predictions.sex,s=9,alpha=.35,cmap='coolwarm')
    ax.plot([0,240],[0,240],'k--');ax.set(xlabel='Reference age (months)',ylabel='Predicted age (months)',title='Early global checkpoint: development only');savefig(paper,'prelim_scatter',fig)
    error=predictions.pred-predictions.true
    fig,axes=plt.subplots(1,2,figsize=(9,3.4));axes[0].hist(error,bins=35,color='#007c83');axes[0].set(xlabel='Signed error (months)',ylabel='Images')
    axes[1].scatter((predictions.pred+predictions.true)/2,error,s=8,alpha=.3)
    axes[1].axhline(error.mean(),color='black')
    for v in [error.mean()-1.96*error.std(ddof=1),error.mean()+1.96*error.std(ddof=1)]:axes[1].axhline(v,color='#cc6677',linestyle='--')
    axes[1].set(xlabel='Mean reference/predicted age',ylabel='Prediction minus reference');fig.tight_layout();savefig(paper,'prelim_agreement',fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    for label,data,color in [('Global checkpoint',predictions,'#007c83'),('Fitting median',global_prior,'#cc6677'),('Fitting sex median',sex_prior,'#88ccee')]:
        vals=[metrics(data[(data.true>=a)&(data.true<(a+48 if a<192 else 241))])['mae'] for a in range(0,240,48)]
        axes[0].plot(range(5),vals,'o-',label=label,color=color)
        ae=np.sort(np.abs(data.pred-data.true));axes[1].plot(ae,np.arange(1,len(ae)+1)/len(ae),label=label,color=color)
    axes[0].set_xticks(range(5),['0--47','48--95','96--143','144--191','192--240'],rotation=20)
    axes[0].set(xlabel='Age band (months)',ylabel='Development MAE (months)');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Absolute error (months)',ylabel='Empirical cumulative fraction');axes[1].set_xlim(0,60);axes[1].legend(fontsize=8)
    fig.tight_layout();savefig(paper,'prelim_subgroups',fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4));labels=[r['mode'] for r in profile]
    axes[0].bar(labels,[r['reserved_mib'] for r in profile],color=['#cc6677','#007c83']);axes[0].set(ylabel='Peak reserved CUDA memory (MiB)')
    axes[1].bar(labels,[r['seconds_update'] for r in profile],color=['#cc6677','#007c83']);axes[1].set(ylabel='Seconds per optimizer update')
    fig.suptitle('22-update admission profile, batch 2 and ten views');fig.tight_layout();savefig(paper,'prelim_resources',fig)
    # Five genuine regression Grad-CAM images on DEVELOPMENT data from the frozen snapshot.
    cases=[];lookup=dev.set_index('id')
    for lower in range(0,240,48):
        part=predictions[(predictions.true>=lower)&(predictions.true<(lower+48 if lower<192 else 241))].sort_values('id')
        if part.empty:raise RuntimeError('Missing prespecified age band for saliency')
        case=part.iloc[0];row=lookup.loc[int(case.id)];views,_=original_views(row.path,args)
        activations={}
        def hook(module,inputs,output):activations['a']=output
        handle=enc.features[-1].register_forward_hook(hook)
        z=enc(views[0].unsqueeze(0).to(DEVICE)).unsqueeze(1)
        mean,p,h=fusion(z,torch.tensor([int(case.sex)],device=DEVICE),torch.tensor([0],device=DEVICE))
        a=activations['a'];g=torch.autograd.grad(mean.sum(),a)[0]
        cam=(a*g.mean((2,3),keepdim=True)).sum(1).relu()[0].detach().cpu().numpy()
        if cam.max()>0:cam=cam/cam.max()
        im=(views[0]*STD+MEAN)[0].numpy();fig,axes=plt.subplots(1,3,figsize=(10,3.4))
        axes[0].imshow(im,cmap='gray',vmin=0,vmax=1);axes[0].set_title('Normalized whole hand')
        axes[1].imshow(cam,cmap='jet',vmin=0,vmax=1);axes[1].set_title('Positive age Grad-CAM')
        axes[2].imshow(im,cmap='gray',vmin=0,vmax=1);axes[2].imshow(cam,cmap='jet',alpha=.45,extent=(0,im.shape[1],im.shape[0],0),vmin=0,vmax=1);axes[2].set_title('Attribution overlay')
        for ax in axes:ax.axis('off')
        fig.suptitle(f'Development image {int(case.id)} | Reference {case.true:.0f} mo | FP16 evaluation {case.pred:.2f} mo | FP32 attribution {float(mean):.2f} mo',fontsize=10)
        fig.tight_layout();name=f'prelim_gradcam_{len(cases)+1}';savefig(paper,name,fig)
        cases.append(dict(id=int(case.id),target=float(case.true),pred=float(case.pred),attribution=float(mean),band=f'{lower}--{lower+47 if lower<192 else 240}',figure=name,positive_map_nonzero=bool(cam.max()>0)))
        handle.remove();del z,a,g;torch.cuda.empty_cache()
    save_json(out/'gradcam_cases.json',cases)
    # Tables: measured preliminary results plus protocol-qualified published comparisons.
    def emit(name,caption,headers,rows,wide=False):
        table(paper,'prelim_'+name,caption,headers,rows,wide)
        pd.DataFrame(rows,columns=headers).to_csv(out/f'{name}.csv',index=False)
    partition_rows=[]
    for split in ['fit','dev','calibration','holdout','excluded']:
        d=manifest[manifest.split==split]
        partition_rows.append([split,len(d),int((d.sex==0).sum()),int((d.sex==1).sum()),f'{d.age.min():.0f}--{d.age.max():.0f}'])
    emit('partitions','Audited partitions. Only development predictions are used in this interim results section; the final holdout remains unevaluated.',
         ['Partition','$N$','Female','Male','Age (mo)'],partition_rows)
    scores=[('Early global checkpoint',m),('Fitting median',metrics(global_prior)),('Fitting sex median',metrics(sex_prior))]
    emit('metrics',f'Development-only results for a frozen global warmup checkpoint from epoch {ck["epoch"]}. This checkpoint was selected on development MAE, so these are optimistic model-selection results, not final held-out performance.',
         ['Method','MAE','RMSE','Bias','$\leq6$ mo (\%)','$\leq12$ mo (\%)'],[[name,f"{s['mae']:.3f}",f"{s['rmse']:.3f}",f"{s['bias']:.3f}",f"{s['within6']:.2f}",f"{s['within12']:.2f}"] for name,s in scores],True)
    emit('sex','Development recorded-sex subgroups for the same early global checkpoint.',
         ['Group','$N$','MAE','RMSE','Bias'],[[label,(s:=metrics(predictions[predictions.sex==sex]))['n'],f"{s['mae']:.3f}",f"{s['rmse']:.3f}",f"{s['bias']:.3f}"] for sex,label in [(0,'Female'),(1,'Male')]])
    age_rows=[]
    for a in range(0,240,48):
        s=metrics(predictions[(predictions.true>=a)&(predictions.true<(a+48 if a<192 else 241))])
        age_rows.append([f'{a}--{a+47 if a<192 else 240}',s['n'],f"{s['mae']:.3f}",f"{s['rmse']:.3f}",f"{s['within6']:.1f}"])
    emit('age','Development age-band results in months; small subgroups should be interpreted cautiously.',
         ['Age (mo)','$N$','MAE','RMSE','$\leq6$ (\%)'],age_rows)
    emit('epochs','Completed global-training epochs at capture time. Training objectives, checkpoint development MAE and training duration are actual logged values; multiview training is not yet reported here.',
         ['Epoch','Train loss','Dev MAE','Seconds','Reserved MiB'],[[int(r.epoch),f'{r.loss:.3f}',f'{r.dev_mae:.3f}',f'{r.seconds:.1f}',f'{r.reserved_mib:.1f}'] for r in history.itertuples()])
    emit('memory','Ten-view synthetic-input admission profile on the RTX 3050, batch two; 22 complete optimizer updates per mode, first two excluded from timing. Both modes fit 4 GB. Device use is sampled, not a continuously measured peak.',
         ['Mode','Allocated MiB','Reserved MiB','Device MiB','s/update'],[[r['mode'],f"{r['allocated_mib']:.1f}",f"{r['reserved_mib']:.1f}",f"{r['sampled_device_peak_mib']:.0f}",f"{r['seconds_update']:.3f}"] for r in profile])
    emit('replay','Actual deterministic FP32 gradient-replay equivalence test, three views at 64-pixel resolution.',
         ['Quantity','Value'],[[tex(k.replace('_',' ')),f'{v:.3e}' if isinstance(v,float) else tex(v)] for k,v in replay.items()])
    emit('checks','Executed software checks. Unseen-isolation zero means altering unavailable encoded crops leaves policy input features exactly unchanged.',
         ['Check','Measured outcome'],[[tex(k.replace('_',' ')),tex(v)] for k,v in checks.items()])
    emit('cam_cases','Five genuine Grad-CAM cases from development data. Lowest ID in each prespecified age band; no selection by low error or attractive saliency. FP32 attribution and FP16 evaluation are listed separately.',
         ['ID','Age band','Target','Eval.','Attrib.'],[[str(c['id']),c['band'],f"{c['target']:.0f}",f"{c['pred']:.2f}",f"{c['attribution']:.2f}"] for c in cases])
    rng=np.random.default_rng(42);bootstrap_rows=[]
    for label,data in [('Fitting median',global_prior),('Fitting sex median',sex_prior)]:
        difference=np.abs(predictions.pred-predictions.true).to_numpy()-np.abs(data.pred-data.true).to_numpy()
        values=np.array([difference[rng.integers(0,len(difference),len(difference))].mean() for _ in range(2000)])
        lo,hi=np.percentile(values,[2.5,97.5]);bootstrap_rows.append([label,f'{difference.mean():.3f}',f'[{lo:.3f}, {hi:.3f}]'])
    emit('bootstrap','Development case-bootstrap difference in MAE (early global model minus fitting-only age prior), 2,000 replicates. Negative favors the image model. Development selection and one training seed limit inference; these are not independent test-set confidence intervals.',
         ['Comparator','$\Delta$ MAE','95\% interval'],bootstrap_rows)
    lit=constant_rows('lit')
    lit[-1]=['BARE','', 'Preliminary','Shared MobileNetV3 + utility','Sex; nine-window grid','Marginal evidence acquisition + replay','Global checkpoint evaluated; policy results pending']
    emit('literature','Focused literature survey of bone-age methods, supervision and relationship to BARE. This is not an exhaustive systematic review.',
         ['Work','Year','Architecture','Supervision','Contribution','Relation / limitation'],[[tex(r[0])+(r'~\cite{'+r[1]+'}' if r[1] else ''),r[2],tex(r[3]),tex(r[4]),tex(r[5]),tex(r[6])] for r in lit],True)
    sota=constant_rows('sota')
    sota.append(['Early global checkpoint',f"{m['mae']:.3f}",f"{m['rmse']:.3f}",f'Development {len(predictions):,}','MobileNetV3 + attention','Sex; no keypoints','$384^2$','RTX 3050 4 GB','8',str(ck['epoch']),f"{hardware['parameters_image_model']/1e6:.3f} M",'NR',f"{float(history.reserved_mib.max()):.0f} MiB",'NR'])
    emit('sota_accuracy','Published strong-method accuracy and supervision comparison. Published official-test results are not directly comparable to this development-only early checkpoint. RFGA RMSE uses Table 1 (4.02). NR = not verified/reported in consulted material; no SOTA claim.',
         ['Method','MAE (mo)','RMSE (mo)','Evaluation','Backbone / fusion','Auxiliary inputs'],[r[:6] for r in sota],True)
    emit('sota_resources','Published model/resource comparison. Values marked NR are not inferred. Warmup uses only the global image; profile-table replay memory refers to a separate ten-view experiment. VRAM here is peak allocator-reserved warmup memory. Latencies on different hardware are not comparable.',
         ['Method','Resolution','GPU','Batch','Epochs','Parameters','FLOPs','VRAM','Latency'],[[r[0]]+r[6:] for r in sota],True)
    optimization=constant_rows('optimization')
    optimization.append(['Early global checkpoint','AdamW','$10^{-4}$ / $3\times10^{-4}$','MAE + CE + CDF',str(len(fit)),str(len(dev)), 'Dev only',f'{history[history.epoch<=ck["epoch"]].seconds.sum()/3600:.2f} h'])
    emit('sota_optimization','Optimization and data-size comparison. Early-checkpoint learning rates are encoder/head initial values. Reported training duration includes its completed warmup epochs only, excluding audit and ongoing subsequent training. NR values are not guessed.',
         ['Method','Optimizer','Initial LR','Objective','Fitting $N$','Dev / cal $N$','Evaluation','Train time'],optimization,True)
    write_paper(paper,out,provenance,m,cases,history)
    print('PRELIMINARY PAPER SOURCE WRITTEN',flush=True)

def write_paper(paper,out,provenance,m,cases,history):
    path=paper/'bare_ieee_access.tex';source=path.read_text(encoding='utf-8')
    if r'\section{Results: pending execution}' not in source:
        raise RuntimeError('Existing paper is no longer the pending draft; avoid overwriting final results')
    (out/'previous_methods_draft.tex').write_text(source,encoding='utf-8')
    source=source[:source.index(r'\section{Results: pending execution}')]
    start=source.index(r'\begin{abstract}');end=source.index(r'\end{abstract}')+len(r'\end{abstract}')
    abstract=r'''\begin{abstract}
High-resolution multiview bone age models can be difficult to train under a laptop video-memory constraint. This paper presents the implementation of Budgeted Acquisition of Regional Evidence, which combines a shared MobileNetV3-Large encoder, small evidence-set attention, ordinal age-distribution learning, model-defined crop-utility supervision and deterministic token-gradient replay. This version reports preliminary executed results from global warmup and completed resource/correctness experiments; full multiview acquisition training is still running. A frozen early global checkpoint achieved @MAE@ months mean absolute error and @RMSE@ months root mean squared error on @N@ development images. These model-selection results are not independent test performance. A ten-view training admission profile measured 260 MiB reserved memory with replay versus 704 MiB with joint backpropagation, while both configurations fitted the four-gigabyte RTX 3050. A deterministic gradient test obtained relative error of approximately $4.54\times10^{-8}$. Results include development subgroups, fitting-only age-prior comparisons, optimization logs, software checks, resource measurements and five predetermined regression Grad-CAM cases. Forty verified references, a focused literature survey and protocol-qualified published accuracy/resource comparisons accompany the analysis. The official validation holdout remains unevaluated. The reported scope does not establish final acquisition accuracy, official-test state of the art, external generalization or clinical validity.
\end{abstract}'''
    source=source[:start]+abstract+source[end:]
    source=source.replace('A focused literature survey will accompany the completed results.',r'Table~\ref{tab:prelim_literature} surveys the related methods.')
    source=source.replace(r'\subsection{Efficient representation and memory control}',r'\input{tables/prelim_literature.tex}'+'\n\n'+r'\subsection{Efficient representation and memory control}')
    source=source.replace('The executed run uses seed 42','The current global warmup uses seed 42; the implemented subsequent multiview stage is configured as follows. The overall run uses seed 42')
    source=source.replace('this executed version uses one fitting-only deterministic token cache','the implemented downstream procedure will use one fitting-only deterministic token cache')
    source=source.replace('A no-disagreement policy is trained on the same utility targets, with that feature masked.','A no-disagreement policy is implemented for subsequent fitting on the same utility targets, with that feature masked; it is not yet evaluated in these interim results.')
    source+=r'''
\section{Results: preliminary executed experiments}
\subsection{Scope, checkpoint and dataset accounting}
This results section reports an actual frozen global-warmup checkpoint, not the completed multiview acquisition model. The snapshot is from global epoch @EPOCH@ and predicts only from the whole-hand image plus recorded sex. It was chosen using development MAE during continuing training. The @N@ development images therefore serve both selection and descriptive evaluation; the resulting metrics are optimistic and must not be interpreted as independent test accuracy. No calibration or official-validation holdout predictions are used here. Table~\ref{tab:prelim_partitions} and Fig.~\ref{fig:prelim_data} document the audited dataset. Nineteen training rows with matching low-resolution hashes across the sealed holdout were conservatively excluded; no exact cross-holdout decoded-pixel duplicates were found.
\input{tables/prelim_partitions.tex}
@F_DATA@

\subsection{Prediction metrics and optimization}
The early global checkpoint obtained MAE @MAE@ months and RMSE @RMSE@ months, with signed bias @BIAS@ months. Table~\ref{tab:prelim_metrics} compares its predictions against age priors whose medians are estimated from fitting labels only. The sex-specific prior selects the fitting median for the recorded sex and uses no image evidence. These priors are regression baselines rather than published deep-learning competitors. Figures~\ref{fig:prelim_scatter} and \ref{fig:prelim_agreement} show actual development predictions and errors. The agreement limits are descriptive, not clinical acceptance bounds.
\input{tables/prelim_metrics.tex}
@F_SCATTER@
@F_AGREEMENT@
Figure~\ref{fig:prelim_learning} and Table~\ref{tab:prelim_epochs} show completed global optimization epochs at capture time. A temporary status-logging error occurred after the first epoch checkpoint was saved; it was repaired, and the process resumed from that checkpoint. No final holdout was evaluated during that repair. Current training is ongoing; later checkpoints and the utility policy are outside this snapshot's reported scope.
\input{tables/prelim_epochs.tex}
@F_LEARNING@

\subsection{Subgroups and sampling diagnostics}
Tables~\ref{tab:prelim_sex} and \ref{tab:prelim_age} report recorded-sex and age-band metrics for the same development predictions. Figure~\ref{fig:prelim_subgroups} shows age-dependent errors and empirical absolute-error cumulative distributions, with the latter displayed up to 60 months. Small age groups and development-set selection limit interpretation. Table~\ref{tab:prelim_bootstrap} gives paired case-bootstrap differences against fitting-only median priors. The intervals condition on this single checkpoint and selected development set, so they do not estimate independent generalization or training-seed variability.
\input{tables/prelim_sex.tex}
\input{tables/prelim_age.tex}
\input{tables/prelim_bootstrap.tex}
@F_SUBGROUP@

\subsection{GPU resources and implementation checks}
Table~\ref{tab:prelim_memory} and Fig.~\ref{fig:prelim_resources} report an executed ten-view synthetic-input training profile on the RTX 3050. Twenty-two complete optimizer updates were profiled, with the first two excluded from timing. Joint backpropagation reserved 704 MiB and replay reserved 260 MiB, but replay took more time per update. Both fitted the 4 GB device: this is a headroom/compute tradeoff rather than proof that joint training was impossible. These memory measurements are separate from the global checkpoint's accuracy experiment. CUDA allocator metrics do not include every driver allocation; sampled device use is listed separately.
\input{tables/prelim_memory.tex}
@F_RESOURCES@
The three-view deterministic FP32 replay test in Table~\ref{tab:prelim_replay} obtained relative encoder-gradient error $4.54\times10^{-8}$ and zero fusion-gradient discrepancy. It verifies the implementation on a small case, not convergence or clinical prediction quality. Table~\ref{tab:prelim_checks} lists executed acquisition-input, crop-budget, stopping and sex-parsing checks. The policy-isolation check uses test tensors rather than trained policy-accuracy results.
\input{tables/prelim_replay.tex}
\input{tables/prelim_checks.tex}

\subsection{Five genuine development Grad-CAM images}
Figures~\ref{fig:prelim_gradcam_1}--\ref{fig:prelim_gradcam_5} contain five actual hand radiographs with regression Grad-CAM computed from the frozen trained global checkpoint. Cases are selected by the smallest image ID in each prespecified 48-month development age band, not by prediction error or attractiveness. Table~\ref{tab:prelim_cam_cases} identifies every case and distinguishes FP16 evaluation from FP32 attribution. For the predicted expected age and last convolutional maps,
\begin{equation}
\alpha_k=\frac1{HW}\sum_{u,v}\frac{\partial\hat y}{\partial A^k_{uv}},\qquad
G=\operatorname{ReLU}\left(\sum_k\alpha_kA^k\right).
\end{equation}
Each map is normalized by its positive maximum. Positive scalar-age saliency indicates features locally increasing age; negative contributions are omitted. Color values are not comparable magnitudes across cases. These are whole-hand maps from the global warmup model, not trained acquisition-policy or local-crop explanations. Saliency cannot establish causal anatomy, clinical reasoning or localization correctness.
\input{tables/prelim_cam_cases.tex}
@F_CAMS@

\subsection{Comprehensive published-method comparison}
Tables~\ref{tab:prelim_sota_accuracy}, \ref{tab:prelim_sota_resources} and \ref{tab:prelim_sota_optimization} cover reported accuracy, evaluation protocol, architecture, supervision, resolution, GPU, batch size, epochs, parameters, FLOPs, VRAM, latency, optimizer, learning rate, objective, data sizes and training duration. NR means not verified/reported in the consulted source material; these missing values are not estimated from backbone names. Published results use different official/custom test partitions and annotations. Our early checkpoint uses development data, so no direct SOTA ranking is valid. Resource profiles and prediction metrics likewise describe separate executed experiments, and latencies measured on different GPUs are not comparable.
\input{tables/prelim_sota_accuracy.tex}
\input{tables/prelim_sota_resources.tex}
\input{tables/prelim_sota_optimization.tex}
\clearpage

\section{Discussion and limitations}
The executed evidence supports the correctness of the replay implementation and a measured reduction in simultaneous training-memory reservation, with increased compute. It also confirms that the early global model learns image-dependent age information beyond fitting-only median priors. This observation does not establish the value of counterfactual crop acquisition: multiview training, utility fitting, calibration and sealed-holdout evaluation are still running and must be assessed separately.

The development set selects the reported checkpoint; consequently, all prediction and subgroup values in this version are model-selection diagnostics. There is one seed, no external institution cohort, no official 200-image challenge test and no verified patient-level grouping. Difference-hash grouping is conservative but is not a complete non-identical near-duplicate audit. The 8-bit normalized cache may quantize higher-bit-depth originals. Five saliency cases do not validate attention over the entire population. Numerical replay checks and lightweight memory profiles should not be conflated with a clinically validated accuracy result.

The final reporting pipeline will replace this interim section in the same source file after the complete BARE run finishes. It will use the frozen calibration threshold and sealed 1,425-image validation holdout, with explicitly labeled global, fixed/random crop and acquisition-policy variants. This version preserves the prospective architecture and equations for reproducibility, while reporting only experiments that have already completed. All author, affiliation and correspondence fields remain blank at the user's request.

\section{Conclusion}
BARE has been implemented and is undergoing overnight evaluation on a 4 GB CUDA laptop. The frozen early global checkpoint has development MAE @MAE@ months; this is a preliminary model-selection result. Actual resource and numerical checks demonstrate a memory/compute tradeoff for replay and close gradient agreement. Final acquisition accuracy, official-test superiority, clinical validity and external generalization are not established by this interim study.

\section*{Acknowledgment}
The RSNA Pediatric Bone Age Challenge organizers and original contributing institutions are acknowledged for the dataset. OpenAI Codex assisted with the research concept, code, reference retrieval, manuscript drafting and reporting. The figures and five Grad-CAM cases are computed from actual data and a trained checkpoint, not synthetic predictions. Human researchers must review scientific claims and data-use terms before submission.
\input{references.tex}
\EOD
\end{document}
'''
    captions={
      '@F_DATA@':figure(paper,'prelim_data','Age distributions from the audited manifest. Holdout labels are shown for dataset accounting only; holdout images are not predicted in this interim study.'),
      '@F_SCATTER@':figure(paper,'prelim_scatter','Actual predictions of the frozen global warmup checkpoint on development images, colored by recorded sex.'),
      '@F_AGREEMENT@':figure(paper,'prelim_agreement','Development signed-error distribution and prediction--reference agreement; dashed limits are mean $\pm1.96$ standard deviations, not clinical acceptance bounds.',True),
      '@F_LEARNING@':figure(paper,'prelim_learning','Completed global warmup losses and development MAE at capture time. Full multiview acquisition training is not represented.',True),
      '@F_SUBGROUP@':figure(paper,'prelim_subgroups','Development age-band error and empirical absolute-error distributions versus fitting-only median priors.',True),
      '@F_RESOURCES@':figure(paper,'prelim_resources','Actual same-hardware ten-view admission measurements: replay reserves less CUDA memory while taking more time per optimizer update.',True),
      '@F_CAMS@':'\n'.join(figure(paper,c['figure'],f"Development Grad-CAM case {i+1}, image {c['id']}, reference age {c['target']:.0f} months. Original normalized hand, positive expected-age attribution and overlay from the frozen global checkpoint. This is not a learned crop-policy explanation.",True) for i,c in enumerate(cases))}
    values={'@MAE@':f"{m['mae']:.3f}",'@RMSE@':f"{m['rmse']:.3f}",'@BIAS@':f"{m['bias']:.3f}",'@N@':str(m['n']),'@EPOCH@':str(provenance['checkpoint_epoch'])}
    for key,value in {**captions,**values}.items():source=source.replace(key,value)
    if '@F_' in source:raise RuntimeError('Unfilled figure marker')
    path.write_text(source,encoding='utf-8')
    save_json(out/'inventory.json',dict(references=40,results_figures=11,gradcam_images=5,results_tables=13,literature_tables=1,
              source=str(path),scope='preliminary development-only global checkpoint; final acquisition model still training'))

if __name__=='__main__':main()

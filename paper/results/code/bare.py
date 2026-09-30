"""BARE: memory-bounded regional evidence acquisition. No evaluation labels train models."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, subprocess, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights, resnet50, ResNet50_Weights, efficientnet_b2, EfficientNet_B2_Weights
from torchvision.ops import StochasticDepth
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CENTERS = [(x, y) for y in (.25, .5, .75) for x in (.25, .5, .75)]
BOXES = torch.tensor([[.5, .5, 1., 1.]] + [[x, y, .45, .45] for x, y in CENTERS])
MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]

def save_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, default=str, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def status(run, phase, **kwargs):
    data = dict(phase=phase, time=time.strftime('%Y-%m-%dT%H:%M:%S'), pid=os.getpid(), **kwargs)
    save_json(run / 'status.json', data)
    print(json.dumps(data, default=str), flush=True)

def parse_sex(value):
    text = str(value).strip().lower()
    if text not in ('true', 'false'): raise ValueError(f'Invalid sex flag: {value}')
    return int(text == 'true')

def audit(root, run, workers=8):
    manifest_file = run / 'manifest.csv'
    if manifest_file.exists():
        return pd.read_csv(manifest_file)
    cache = run / 'image_cache'; cache.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(root/'Bone+Age+Training+Set+Annotations'/'train.csv').rename(columns={'boneage':'age'})
    va = pd.read_csv(root/'Bone+Age+Validation+Set'/'Validation Dataset.csv').rename(columns={'Image ID':'id','Bone Age (months)':'age'})
    tr['origin'] = 'training'; va['origin'] = 'holdout'
    assert set(tr.id).isdisjoint(set(va.id)), 'ID overlap'
    tr_index = {}
    for p in (root/'Bone+Age+Training+Set').rglob('*.png'): tr_index.setdefault(int(p.stem), []).append(p)
    va_index = {}
    for p in (root/'Bone+Age+Validation+Set').rglob('*.png'): va_index.setdefault(int(p.stem), []).append(p)
    data = pd.concat([tr, va], ignore_index=True)
    def process(row):
        idx = tr_index if row.origin == 'training' else va_index
        paths = idx.get(int(row.id), [])
        if len(paths) != 1: raise ValueError(f'{row.id}: expected one image, found {len(paths)}')
        age = float(row.age)
        if not np.isfinite(age) or not 0 <= age <= 240: raise ValueError(f'Invalid age {age}')
        with Image.open(paths[0]) as source:
            arr = np.asarray(source)
            if arr.ndim == 3: arr = np.asarray(source.convert('L'))
            if arr.ndim != 2: raise ValueError('Not a 2D radiograph')
            pixel_hash = hashlib.sha256(str(arr.shape).encode()+arr.tobytes()).hexdigest()
            lo, hi = np.percentile(arr, [.5, 99.5])
            norm = np.clip((arr.astype(np.float32)-lo)/max(float(hi-lo),1.)*255,0,255).astype(np.uint8)
            im = Image.fromarray(norm)
            width, height = im.size
            tiny = np.asarray(im.resize((9,8), Image.Resampling.BILINEAR))
            dhash = np.packbits(tiny[:,1:] > tiny[:,:-1]).tobytes().hex()
            im.thumbnail((1280,1280), Image.Resampling.LANCZOS)
            # Explicit preprocessing choice: lossless normalized PNG cache, full original SHA below.
            dest = cache/f'{row.origin}_{int(row.id)}.png'
            im.save(dest, compress_level=1)
        return dict(id=int(row.id), age=age, sex=parse_sex(row.male), origin=row.origin,
                    path=str(paths[0]), cached=str(dest), width=width, height=height,
                    pixel_hash=pixel_hash, dhash=dhash, file_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest())
    records=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(process, data.itertuples(index=False))):
            records.append(result)
            if (i+1)%500 == 0: status(run,'audit',processed=i+1,total=len(data))
    df = pd.DataFrame(records)
    hold = df[df.origin=='holdout']
    crossed = set(hold.pixel_hash) & set(df[df.origin=='training'].pixel_hash)
    df['split'] = np.where(df.origin=='holdout','holdout','excluded')
    eligible = df[(df.origin=='training') & ~df.pixel_hash.isin(crossed)].copy()
    # Conservative perceptual-hash grouping within training prevents identical dHash groups crossing splits.
    bins = (eligible.age//12).clip(upper=19).astype(int)
    strata = eligible.sex.astype(str)+'_'+bins.astype(str)
    # Fold grouping is approximate for rare age/sex strata; counts are reported, not assumed balanced.
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (_, val) in enumerate(sgkf.split(eligible, strata, groups=eligible.dhash)):
        df.loc[eligible.index[val], 'split'] = 'fit' if fold<8 else ('dev' if fold==8 else 'calibration')
    df.to_csv(manifest_file,index=False)
    cross_dhash = set(hold.dhash) & set(eligible.dhash)
    # Exclude exact matching low-resolution perceptual hashes across the sealed holdout too.
    excluded_perceptual = df[(df.origin=='training') & df.dhash.isin(cross_dhash)].index
    df.loc[excluded_perceptual,'split']='excluded'
    df.to_csv(manifest_file,index=False)
    save_json(run/'audit.json',dict(rows=len(df),counts=df.split.value_counts().to_dict(),
              exact_cross_split_hashes=len(crossed),perceptual_cross_split_hashes=len(cross_dhash),
              perceptual_excluded_training_rows=len(excluded_perceptual),
              note='Exact matching dHash groups excluded across holdout; non-identical near duplicates and patient independence remain unverified.',
              train_age_min=float(tr.age.min()),train_age_max=float(tr.age.max())))
    return df

def views_from_image(image, augment=False, global_size=384, local_size=256, global_only=False):
    canvas_size = global_size if global_only else 1280
    image = ImageOps.pad(image.convert('L'), (canvas_size,canvas_size), color=0, method=Image.Resampling.BILINEAR)
    if augment:
        image = image.rotate(random.uniform(-10,10), resample=Image.Resampling.BILINEAR)
        image = ImageEnhance.Contrast(image).enhance(random.uniform(.9,1.1))
        image = ImageEnhance.Brightness(image).enhance(random.uniform(.9,1.1))
    small = image.resize((global_size,global_size),Image.Resampling.BILINEAR)
    images=[small]
    desc=[np.zeros(3,dtype=np.float32)]
    for x,y in CENTERS:
        if global_only:
            desc.append(np.zeros(3,dtype=np.float32))
            continue
        box=tuple(int(v*1280) for v in (x-.225,y-.225,x+.225,y+.225))
        proxy = np.asarray(small.crop(tuple(int(v*global_size) for v in (x-.225,y-.225,x+.225,y+.225))),dtype=np.float32)/255
        desc.append(np.array([proxy.mean(),proxy.std(),np.abs(np.diff(proxy,axis=0)).mean()+np.abs(np.diff(proxy,axis=1)).mean()],dtype=np.float32))
        if not global_only: images.append(image.crop(box).resize((local_size,local_size),Image.Resampling.BILINEAR))
    def tensor(im):
        a=torch.from_numpy(np.asarray(im,dtype=np.float32).copy()/255).unsqueeze(0).repeat(3,1,1)
        return (a-MEAN)/STD
    return [tensor(im) for im in images], torch.tensor(np.stack(desc))

class HandDataset(Dataset):
    def __init__(self, df, augment=False, global_only=False, sizes=(384,256)):
        self.df=df.reset_index(drop=True); self.augment=augment; self.global_only=global_only; self.sizes=sizes
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        row=self.df.iloc[i]
        with Image.open(row.cached) as im:
            views,desc=views_from_image(im,self.augment,*self.sizes,self.global_only)
        return views, torch.tensor(row.sex,dtype=torch.long), torch.tensor(row.age,dtype=torch.float32), desc, int(row.id)

def loader(df, batch=2, workers=2, **kwargs):
    return DataLoader(HandDataset(df,**kwargs),batch_size=batch,shuffle=kwargs.get('augment',False),
                      num_workers=workers,pin_memory=DEVICE.type=='cuda',persistent_workers=workers>0,
                      **({'prefetch_factor':1} if workers else {}))

class Encoder(nn.Module):
    def __init__(self, pretrained=True, backbone='mobilenet_v3_large'):
        super().__init__()
        self.backbone=backbone
        if backbone=='resnet50':
            base=resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            self.features=nn.Sequential(*list(base.children())[:-2]); width=2048
        elif backbone=='efficientnet_b2':
            base=efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT if pretrained else None)
            self.features=base.features; width=1408
            for module in self.features.modules():
                if isinstance(module,StochasticDepth): module.p=0.0
        else:
            base=mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
            self.features=base.features; width=960
        self.project=nn.Linear(width,128)
    def train(self,mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m,nn.BatchNorm2d): m.eval()
        return self
    def forward(self,x): return self.project(self.features(x).mean((2,3)))

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('boxes',BOXES.clone())
        self.coordinates=nn.Linear(4,128); self.view_type=nn.Embedding(2,128)
        self.sex=nn.Embedding(2,16); self.query=nn.Linear(16,128)
        layer=nn.TransformerEncoderLayer(128,4,256,.1,activation='gelu',batch_first=True,norm_first=True)
        self.attention=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.output=nn.Linear(128,241)
        self.register_buffer('ages',torch.arange(241,dtype=torch.float32))
    def forward(self,z,sex,ids=None):
        if ids is None: ids=torch.arange(z.shape[1],device=z.device).expand(z.shape[0],-1)
        if ids.ndim==1: ids=ids.expand(z.shape[0],-1)
        token=z.float()+self.coordinates(self.boxes[ids])+self.view_type((ids>0).long())
        query=self.query(self.sex(sex)).unsqueeze(1)
        h=self.attention(torch.cat([query,token],1))[:,0]
        logits=self.output(h); p=logits.softmax(-1)
        return (p*self.ages).sum(-1),p,h

def age_loss(mean,p,y):
    ages=torch.arange(241,device=p.device,dtype=torch.float32)
    q=torch.exp(-.5*((ages[None,:]-y[:,None])/2.)**2); q=q/q.sum(-1,keepdim=True)
    ce=-(q*p.clamp_min(1e-8).log()).sum(-1).mean()
    ordinal=(p.cumsum(-1)[:,:-1]-q.cumsum(-1)[:,:-1]).abs().mean()
    return (mean-y).abs().mean()/12+.1*ce+.5*ordinal

def main_loss(fusion,z,sex,y,global_only=False):
    if global_only:
        mean,p,_=fusion(z,sex,torch.tensor([0],device=z.device)); return age_loss(mean,p,y)
    subsets=[[0],list(range(10))]
    for _ in range(2): subsets.append([0]+random.sample(range(1,10),random.randint(1,3)))
    loss=0.
    for ids in subsets:
        mean,p,_=fusion(z[:,ids],sex,torch.tensor(ids,device=z.device)); loss=loss+age_loss(mean,p,y)/len(subsets)
    b=z.shape[0]
    ids=torch.arange(10,device=z.device).repeat(b).reshape(-1,1)
    mean,p,_=fusion(z.reshape(b*10,1,128),sex.repeat_interleave(10),ids)
    loss=loss+.1*age_loss(mean,p,y.repeat_interleave(10))
    return loss

def replay_equivalence(run,backbone='mobilenet_v3_large'):
    seed_all(99)
    enc=Encoder(pretrained=False,backbone=backbone).to(DEVICE).eval(); fusion=Fusion().to(DEVICE).eval()
    inputs=[torch.randn(1,3,64,64,device=DEVICE) for _ in range(3)]
    sex=torch.tensor([1],device=DEVICE); y=torch.tensor([120.],device=DEVICE)
    zs=torch.stack([enc(x) for x in inputs],1)
    mean,p,_=fusion(zs,sex)
    age_loss(mean,p,y).backward()
    joint={n:p.grad.detach().clone() for n,p in enc.named_parameters() if p.grad is not None}
    head_joint={n:p.grad.detach().clone() for n,p in fusion.named_parameters() if p.grad is not None}
    enc.zero_grad(set_to_none=True);fusion.zero_grad(set_to_none=True)
    with torch.no_grad(): zs=torch.stack([enc(x) for x in inputs],1)
    zs.requires_grad_(True); mean,p,_=fusion(zs,sex);age_loss(mean,p,y).backward();g=zs.grad.detach()
    for j,x in enumerate(inputs): (enc(x)*g[:,j]).sum().backward()
    absolute=max(float((p.grad-joint[n]).abs().max()) for n,p in enc.named_parameters() if n in joint)
    rel=math.sqrt(sum(float((p.grad-joint[n]).square().sum()) for n,p in enc.named_parameters() if n in joint))/max(math.sqrt(sum(float(v.square().sum()) for v in joint.values())),1e-12)
    head_error=max(float((p.grad-head_joint[n]).abs().max()) for n,p in fusion.named_parameters() if n in head_joint)
    result=dict(encoder_max_absolute_error=absolute,encoder_relative_l2_error=rel,fusion_max_absolute_error=head_error,passed=rel<1e-4,precision='FP32',views=3,resolution=64)
    save_json(run/'replay_check.json',result)
    if not result['passed']: raise RuntimeError(f'Replay check failed: {result}')
    del enc,fusion,zs;torch.cuda.empty_cache()
    return result

def gpu_sample():
    try:
        out=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip().split(',')
        return dict(device_used_mib=float(out[0]),temperature_c=float(out[1]),power_w=float(out[2]))
    except Exception: return {}

def profile(run,sizes=(384,256),batch=2,backbone='mobilenet_v3_large'):
    seed_all(42); enc=Encoder(backbone=backbone).to(DEVICE).train(); fusion=Fusion().to(DEVICE).train()
    opt=torch.optim.AdamW(list(enc.parameters())+list(fusion.parameters()),lr=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=DEVICE.type=='cuda')
    sex=torch.ones(batch,device=DEVICE,dtype=torch.long);y=torch.ones(batch,device=DEVICE)*100
    modes=[]
    for mode in ('replay','joint'):
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();times=[];device_peak=0
        try:
            for iteration in range(22):
                torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
                # CPU inputs avoid simultaneous GPU view-image retention.
                xs=[torch.randn(batch,3,sizes[0] if j==0 else sizes[1],sizes[0] if j==0 else sizes[1]) for j in range(10)]
                if mode=='joint':
                    with torch.autocast(DEVICE.type,dtype=torch.float16,enabled=DEVICE.type=='cuda'):
                        z=torch.stack([enc(x.to(DEVICE)) for x in xs],1)
                    loss=main_loss(fusion,z,sex,y);scaler.scale(loss).backward()
                else:
                    with torch.no_grad(),torch.autocast(DEVICE.type,dtype=torch.float16,enabled=DEVICE.type=='cuda'):
                        z=torch.stack([enc(x.to(DEVICE)) for x in xs],1).float()
                    z.requires_grad_(True);loss=main_loss(fusion,z,sex,y);scaler.scale(loss).backward();g=z.grad.detach()
                    for j,x in enumerate(xs):
                        with torch.autocast(DEVICE.type,dtype=torch.float16,enabled=DEVICE.type=='cuda'): zz=enc(x.to(DEVICE))
                        (zz.float()*g[:,j]).sum().backward()
                scaler.unscale_(opt);nn.utils.clip_grad_norm_(list(enc.parameters())+list(fusion.parameters()),1)
                scaler.step(opt);scaler.update();torch.cuda.synchronize()
                if iteration>=2: times.append(time.perf_counter()-start)
                device_peak=max(device_peak,gpu_sample().get('device_used_mib',0))
            modes.append(dict(mode=mode,batch=batch,global_size=sizes[0],local_size=sizes[1],
                        allocated_mib=torch.cuda.max_memory_allocated()/2**20,reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                        sampled_device_peak_mib=device_peak,seconds_update=float(np.mean(times)),oom=False))
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True);torch.cuda.empty_cache()
            modes.append(dict(mode=mode,batch=batch,global_size=sizes[0],local_size=sizes[1],oom=True))
    save_json(run/'profile.json',modes)
    params=sum(p.numel() for p in enc.parameters())+sum(p.numel() for p in fusion.parameters())
    save_json(run/'hardware.json',dict(gpu=torch.cuda.get_device_name(),total_mib=torch.cuda.get_device_properties(0).total_memory/2**20,
        torch=torch.__version__,cuda=torch.version.cuda,parameters_image_model=params,device=gpu_sample(),profile='22 complete updates per mode, first two timing warmups excluded'))
    del enc,fusion,opt;torch.cuda.empty_cache()
    replay=next(x for x in modes if x['mode']=='replay')
    if replay['oom'] or replay.get('reserved_mib',9999)>3600: raise RuntimeError('Admission failed; lower batch/resolution and rerun profile')
    return modes

@torch.no_grad()
def cache_tokens(enc, df, run, name, workers=2, sizes=(384,256),batch=12):
    enc.eval();zout=[];sexes=[];ys=[];descs=[];ids=[];start=time.perf_counter()
    data=loader(df,batch=batch,workers=workers,sizes=sizes)
    for i,(views,sex,y,desc,imgid) in enumerate(data):
        with torch.autocast(DEVICE.type,dtype=torch.float16,enabled=DEVICE.type=='cuda'):
            z=torch.stack([enc(v.to(DEVICE,non_blocking=True)) for v in views],1)
        zout.append(z.float().cpu());sexes.append(sex);ys.append(y);descs.append(desc);ids.append(imgid)
        if i%100==0: status(run,'cache_'+name,batch=i,total_batches=len(data))
    cache=dict(z=torch.cat(zout),sex=torch.cat(sexes),y=torch.cat(ys),desc=torch.cat(descs),id=torch.cat(ids),seconds=time.perf_counter()-start)
    torch.save(cache,run/f'cache_{name}.pt');return cache

@torch.no_grad()
def evaluate_global(enc,fusion,df,workers=2,sizes=(384,256)):
    enc.eval();fusion.eval();pred=[];true=[]
    for views,sex,y,desc,ids in loader(df,batch=4,workers=workers,global_only=True,sizes=sizes):
        with torch.autocast(DEVICE.type,dtype=torch.float16,enabled=DEVICE.type=='cuda'): z=enc(views[0].to(DEVICE)).float().unsqueeze(1)
        mean,_,_=fusion(z,sex.to(DEVICE),torch.tensor([0],device=DEVICE));pred.extend(mean.cpu().tolist());true.extend(y.tolist())
    return float(np.mean(np.abs(np.array(pred)-true)))

def train_image_model(df,run,args):
    seed_all(args.seed);enc=Encoder(backbone=args.backbone).to(DEVICE);fusion=Fusion().to(DEVICE)
    opt=torch.optim.AdamW([{'params':enc.parameters(),'lr':1e-4},{'params':fusion.parameters(),'lr':3e-4}],weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=DEVICE.type=='cuda')
    fit=df[df.split=='fit'];dev=df[df.split=='dev'];history=[];started=time.time();best=float('inf');best_global=float('inf')
    resume_ck=None
    if args.resume and (run/'training_summary.json').exists() and (run/'best.pt').exists():
        ck=torch.load(run/'best.pt',map_location=DEVICE,weights_only=False)
        enc.load_state_dict(ck['encoder']);fusion.load_state_dict(ck['fusion']);enc.eval();fusion.eval()
        return enc,fusion
    if args.resume and (run/'last.pt').exists():
        resume_ck=torch.load(run/'last.pt',map_location=DEVICE,weights_only=False)
        history=pd.read_csv(run/'history.csv').to_dict('records')
        best_global=min([r['dev_mae'] for r in history if r['phase']=='global'] or [float('inf')])
        best=min([r['dev_mae'] for r in history if r['phase']=='multiview'] or [float('inf')])
    # Reserve cache/policy/evaluation time; fixed maximum epochs plus wall time bound.
    deadline=started+args.hours*3600*.75
    for phase,epochs,batch,accum in [('global',args.global_epochs,args.global_batch,max(1,8//args.global_batch)),('multiview',args.multiview_epochs,args.batch,max(1,8//args.batch))]:
        if resume_ck and resume_ck['phase']=='multiview' and phase=='global': continue
        data=loader(fit,batch=batch,workers=args.workers,augment=True,global_only=phase=='global',sizes=(args.global_size,args.local_size))
        if phase=='multiview':
            ck=torch.load(run/'global_best.pt',map_location=DEVICE,weights_only=False)
            enc.load_state_dict(ck['encoder']);fusion.load_state_dict(ck['fusion'])
            opt=torch.optim.AdamW([{'params':enc.parameters(),'lr':5e-5},{'params':fusion.parameters(),'lr':1.5e-4}],weight_decay=1e-4)
        start_epoch=0
        if resume_ck and resume_ck['phase']==phase:
            enc.load_state_dict(resume_ck['encoder']);fusion.load_state_dict(resume_ck['fusion'])
            opt.load_state_dict(resume_ck['optimizer']);scaler.load_state_dict(resume_ck['scaler']);start_epoch=resume_ck['epoch']
            for group,base in zip(opt.param_groups,([1e-4,3e-4] if phase=='global' else [5e-5,1.5e-4])):
                group['lr']=base*(.1+.9*(1+math.cos(math.pi*start_epoch/epochs))/2)
            if 'rng_torch' in resume_ck:
                torch.set_rng_state(resume_ck['rng_torch'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in resume_ck['rng_cuda']])
                random.setstate(resume_ck['rng_python']);np.random.set_state(resume_ck['rng_numpy'])
        stale=0
        for epoch in range(start_epoch,epochs):
            if phase=='multiview' and epoch>=2 and time.time()>deadline: break
            enc.train();fusion.train();opt.zero_grad(set_to_none=True);losses=[];tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            running_device_peak=0;skipped=0
            for i,(views,sex,y,desc,ids) in enumerate(data):
                sex=sex.to(DEVICE);y=y.to(DEVICE)
                actual_window=min(accum,len(data)-(i//accum)*accum)
                if phase=='global':
                    with torch.autocast(DEVICE.type,dtype=torch.float16,enabled=DEVICE.type=='cuda'): z=enc(views[0].to(DEVICE)).float().unsqueeze(1)
                    loss=main_loss(fusion,z,sex,y,True);scaler.scale(loss/actual_window).backward()
                else:
                    with torch.no_grad(),torch.autocast(DEVICE.type,dtype=torch.float16,enabled=DEVICE.type=='cuda'):
                        z=torch.stack([enc(v.to(DEVICE,non_blocking=True)) for v in views],1).float()
                    z.requires_grad_(True);loss=main_loss(fusion,z,sex,y);scaler.scale(loss/actual_window).backward();g=z.grad.detach()
                    for j,v in enumerate(views):
                        with torch.autocast(DEVICE.type,dtype=torch.float16,enabled=DEVICE.type=='cuda'): zz=enc(v.to(DEVICE,non_blocking=True))
                        (zz.float()*g[:,j]).sum().backward()
                losses.append(float(loss.detach()))
                if (i+1)%accum==0 or i+1==len(data):
                    scaler.unscale_(opt);nn.utils.clip_grad_norm_(list(enc.parameters())+list(fusion.parameters()),1)
                    oldscale=scaler.get_scale();scaler.step(opt);scaler.update();skipped+=int(scaler.get_scale()<oldscale);opt.zero_grad(set_to_none=True)
                if i%100==0:
                    running_device_peak=max(running_device_peak,gpu_sample().get('device_used_mib',0))
                    status(run,'train_'+phase,epoch=epoch+1,batch=i,total_batches=len(data),loss=float(np.mean(losses[-100:])),**gpu_sample())
            train_seconds=time.perf_counter()-tick
            if phase=='global': val_mae=evaluate_global(enc,fusion,dev,args.workers,(args.global_size,args.local_size))
            else:
                c=cache_tokens(enc,dev,run,'dev_epoch',args.workers,(args.global_size,args.local_size))
                val_mae=evaluate_cached(fusion,c,mode='fixed')['mae']
            row=dict(phase=phase,epoch=epoch+1,loss=float(np.mean(losses)),dev_mae=val_mae,seconds=train_seconds,
                     allocated_mib=torch.cuda.max_memory_allocated()/2**20,reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                     sampled_device_peak_mib=running_device_peak,skipped_updates=skipped)
            history.append(row);pd.DataFrame(history).to_csv(run/'history.csv',index=False)
            ck=dict(encoder=enc.state_dict(),fusion=fusion.state_dict(),args=vars(args),phase=phase,epoch=epoch+1,dev_mae=val_mae)
            torch.save(dict(**ck,optimizer=opt.state_dict(),scaler=scaler.state_dict(),rng_torch=torch.get_rng_state(),
                       rng_cuda=torch.cuda.get_rng_state_all(),rng_python=random.getstate(),rng_numpy=np.random.get_state()),run/'last.pt')
            target=run/('global_best.pt' if phase=='global' else 'best.pt')
            improve=val_mae<(best_global if phase=='global' else best)
            if improve:
                torch.save(ck,target);stale=0
                if phase=='global': best_global=val_mae
                else: best=val_mae
            else: stale+=1
            status(run,'epoch_complete',training_phase=row['phase'],**{k:v for k,v in row.items() if k!='phase'})
            fraction=(epoch+1)/epochs
            for group,base in zip(opt.param_groups,([1e-4,3e-4] if phase=='global' else [5e-5,1.5e-4])):
                group['lr']=base*(.1+.9*(1+math.cos(math.pi*fraction))/2)
            if stale>=8: break
    selected=torch.load(run/'best.pt',map_location=DEVICE,weights_only=False)
    enc.load_state_dict(selected['encoder']);fusion.load_state_dict(selected['fusion']);enc.eval();fusion.eval()
    save_json(run/'training_summary.json',dict(seed=args.seed,completed_epochs=len(history),global_epochs=sum(x['phase']=='global' for x in history),
              multiview_epochs=sum(x['phase']=='multiview' for x in history),image_training_seconds=time.time()-started,
              selection='minimum development fixed-crop MAE',selected_epoch=selected['epoch'],selected_dev_mae=selected['dev_mae'],
              max_reserved_mib=max(x['reserved_mib'] for x in history),sampled_device_peak_mib=max(x['sampled_device_peak_mib'] for x in history),
              independent_seeds_completed=1))
    return enc,fusion

class Utility(nn.Module):
    def __init__(self,disagreement=True):
        super().__init__();self.disagreement=disagreement
        self.net=nn.Sequential(nn.Linear(128+241+1+1+16+4+3+1,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,1))
    def forward(self,features):
        if not self.disagreement:
            features=features.clone();features[:,128+241]=0
        return self.net(features).squeeze(-1)

def policy_features(fusion,z,sex,desc,selected):
    # selected: [batch, evidence count], contains global 0. No unseen encoded token enters policy.
    zz=z.gather(1,selected[:,:,None].expand(-1,-1,128))
    mean,p,h=fusion(zz,sex,selected)
    b,k=selected.shape
    mu,_,_=fusion(zz.reshape(b*k,1,128),sex.repeat_interleave(k),selected.reshape(-1,1))
    disagreement=mu.reshape(b,k).std(-1,unbiased=False)/12
    entropy=-(p*p.clamp_min(1e-8).log()).sum(-1)/math.log(241)
    state=torch.cat([h,p,disagreement[:,None],entropy[:,None],fusion.sex(sex),torch.full((b,1),k/10,device=z.device)],-1)
    # reorder scalar count to the end after geometry/proxies, matching Utility input contract
    state_no_count=state[:,:-1]
    features=torch.cat([state_no_count[:,None,:].expand(-1,9,-1),fusion.boxes[1:][None].expand(b,-1,-1),desc[:,1:],state[:,-1:,None].expand(-1,1,9).transpose(1,2)],-1)
    return mean,p,h,features

@torch.no_grad()
def build_utility_data(fusion,cache,run):
    fusion.eval();features=[];targets=[];rng=random.Random(42)
    for start in range(0,len(cache['y']),128):
        z=cache['z'][start:start+128].to(DEVICE);sex=cache['sex'][start:start+128].to(DEVICE);y=cache['y'][start:start+128].to(DEVICE);desc=cache['desc'][start:start+128].to(DEVICE)
        for count in (0,1,2):
            # Random subsets provide initial state coverage; target provenance records selected checkpoint.
            ids=[0]+rng.sample(range(1,10),count);selected=torch.tensor(ids,device=DEVICE).expand(len(z),-1)
            before,_,_,fs=policy_features(fusion,z,sex,desc,selected)
            for j in range(1,10):
                if j in ids: continue
                ids2=torch.tensor(ids+[j],device=DEVICE).expand(len(z),-1)
                after,_,_=fusion(z[:,ids+[j]],sex,ids2)
                targets.append(((before-y).abs()-(after-y).abs()).cpu());features.append(fs[:,j-1].cpu())
        if start%1024==0: status(run,'utility_targets',processed=start,total=len(cache['y']))
    return torch.cat(features),torch.cat(targets)

def train_utility(fusion,cache,dev_cache,run,args):
    x,y=build_utility_data(fusion,cache,run)
    # Fitting-only internal policy split; official dev used for trajectory comparison, not utility fitting.
    generator=torch.Generator().manual_seed(args.seed);perm=torch.randperm(len(y),generator=generator);cut=int(.9*len(y))
    train_ids,val_ids=perm[:cut],perm[cut:]
    mu=x[train_ids].mean(0);sd=x[train_ids].std(0).clamp_min(.01)
    normal=(x-mu)/sd
    pack=dict(mu=mu,sd=sd,source_checkpoint_sha256=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),records=len(y),
              target_type='signed absolute-error reduction in months',trajectory_refinement=False,cache_augmentation=False)
    utility_history=[]
    for use_disagreement,name in [(True,'utility'),(False,'utility_no_disagreement')]:
        model=Utility(use_disagreement).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        best=float('inf')
        for epoch in range(args.utility_epochs):
            model.train();losses=[]
            order=train_ids[torch.randperm(len(train_ids),generator=generator)]
            for ids in order.split(512):
                xx=normal[ids].to(DEVICE)
                # Mask raw disagreement after normalization by setting that normalized feature to zero.
                pred=model(xx);loss=nn.functional.huber_loss(pred,y[ids].to(DEVICE),delta=2)
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss))
            model.eval()
            with torch.no_grad():
                vv=torch.cat([model(normal[ids].to(DEVICE)).cpu() for ids in val_ids.split(1024)])
                val=float(nn.functional.l1_loss(vv,y[val_ids]))
            utility_history.append(dict(model=name,epoch=epoch+1,loss=float(np.mean(losses)),utility_mae=val))
            if val<best:
                best=val;torch.save(dict(**pack,state=model.state_dict(),disagreement=use_disagreement),run/f'{name}.pt')
            status(run,'train_'+name,epoch=epoch+1,utility_mae=val)
    pd.DataFrame(utility_history).to_csv(run/'utility_history.csv',index=False)
    torch.save(dict(features=x,targets=y,validation_indices=val_ids),run/'utility_diagnostics.pt')
    return load_utility(run/'utility.pt')

def load_utility(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);model=Utility(ck['disagreement']).to(DEVICE);model.load_state_dict(ck['state']);model.eval()
    return model,ck['mu'].to(DEVICE),ck['sd'].to(DEVICE)

@torch.no_grad()
def predict_cached(fusion,cache,mode='policy',policy=None,tau=-float('inf'),budget=3,batch=128):
    fusion.eval();outputs=[];rng=np.random.default_rng(2026)
    for start in range(0,len(cache['y']),batch):
        z=cache['z'][start:start+batch].to(DEVICE);sex=cache['sex'][start:start+batch].to(DEVICE);desc=cache['desc'][start:start+batch].to(DEVICE);b=len(z)
        selected=torch.zeros((b,1),dtype=torch.long,device=DEVICE);chosen=[];gainlogs=[];stopped=torch.zeros(b,dtype=torch.bool,device=DEVICE)
        first,_,_=fusion(z[:,:1],sex,selected);final=first.clone();counts=torch.zeros(b,device=DEVICE,dtype=torch.long)
        for step in range(budget if mode!='global' else 0):
            before,p,h,features=policy_features(fusion,z,sex,desc,selected)
            if mode=='fixed': next_ids=torch.full((b,),[2,5,8][step],device=DEVICE,dtype=torch.long);gains=torch.zeros(b,device=DEVICE)
            elif mode=='random':
                if step==0: random_ids=torch.tensor(np.stack([rng.choice(np.arange(1,10),3,replace=False) for _ in range(b)]),device=DEVICE)
                next_ids=random_ids[:,step];gains=torch.zeros(b,device=DEVICE)
            else:
                model,mu,sd=policy;raw=model(((features-mu)/sd).reshape(b*9,-1)).reshape(b,9)
                for col in range(selected.shape[1]):
                    ids=selected[:,col]-1;valid=ids>=0;raw[torch.arange(b,device=DEVICE)[valid],ids[valid]]=-float('inf')
                gains,idx=raw.max(-1);next_ids=idx+1
                stopped=stopped|(gains<=tau)
            selected=torch.cat([selected,next_ids[:,None]],1)
            zz=z.gather(1,selected[:,:,None].expand(-1,-1,128));mean,_,_=fusion(zz,sex,selected)
            active=~stopped;final=torch.where(active,mean,final);counts+=active.long()
            chosen.append(next_ids.cpu());gainlogs.append(gains.cpu())
        for k in range(b):
            outputs.append(dict(id=int(cache['id'][start+k]),true=float(cache['y'][start+k]),sex=int(cache['sex'][start+k]),
                pred=float(final[k]),global_pred=float(first[k]),crops=int(counts[k]),
                selected=','.join(str(int(c[k])) for c in chosen[:int(counts[k])]),
                predicted_gains=','.join(f'{float(g[k]):.5f}' for g in gainlogs[:int(counts[k])]),mode=mode))
    return pd.DataFrame(outputs)

def metrics(pred):
    e=pred.pred.to_numpy()-pred.true.to_numpy();ae=np.abs(e)
    return dict(n=len(pred),mae=float(ae.mean()),rmse=float(np.sqrt(np.mean(e**2))),bias=float(e.mean()),
                within6=float(np.mean(ae<=6)*100),within12=float(np.mean(ae<=12)*100),median_ae=float(np.median(ae)),
                p95_ae=float(np.percentile(ae,95)),mean_crops=float(pred.crops.mean()))

def evaluate_cached(fusion,cache,**kwargs): return metrics(predict_cached(fusion,cache,**kwargs))

def run_experiment(args):
    run=Path(args.run).resolve();run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);seed_all(args.seed);save_json(run/'config.json',vars(args));status(run,'starting',device=str(DEVICE))
    if DEVICE.type!='cuda': raise RuntimeError('This experiment requires a CUDA GPU')
    df=audit(Path(args.root),run,args.audit_workers)
    if not args.resume or not (run/'profile.json').exists():
        replay_equivalence(run,args.backbone);profile(run,(args.global_size,args.local_size),args.batch,args.backbone)
    if args.command=='verify': status(run,'verified');return
    enc,fusion=train_image_model(df,run,args)
    fit_cache=cache_tokens(enc,df[df.split=='fit'],run,'fit',args.workers,(args.global_size,args.local_size))
    dev_cache=cache_tokens(enc,df[df.split=='dev'],run,'dev',args.workers,(args.global_size,args.local_size))
    policy=train_utility(fusion,fit_cache,dev_cache,run,args)
    # Model selection between policy and fixed always-three using development only.
    dev_fixed=evaluate_cached(fusion,dev_cache,mode='fixed');dev_policy=evaluate_cached(fusion,dev_cache,policy=policy)
    save_json(run/'development_selection.json',dict(fixed=dev_fixed,policy=dev_policy,primary='policy',note='BARE policy remains primary even if fixed baseline is better; no holdout-based switching'))
    cal_cache=cache_tokens(enc,df[df.split=='calibration'],run,'calibration',args.workers,(args.global_size,args.local_size))
    fixed=evaluate_cached(fusion,cal_cache,policy=policy)
    calibration=[]
    for tau in (-float('inf'),0.,.25,.5,1.):
        m=evaluate_cached(fusion,cal_cache,policy=policy,tau=tau);calibration.append(dict(tau='always' if tau==-float('inf') else tau,**m))
    feasible=[c for c in calibration if c['mae']<=fixed['mae']+.15]
    selected=min(feasible,key=lambda c:(c['mean_crops'],c['mae']));tau=-float('inf') if selected['tau']=='always' else float(selected['tau'])
    save_json(run/'calibration.json',dict(threshold=selected['tau'],rows=calibration,tolerance_months=.15))
    # Official validation is read for final predictions only after every training/calibration choice is frozen.
    hold_cache=cache_tokens(enc,df[df.split=='holdout'],run,'holdout',args.workers,(args.global_size,args.local_size))
    summary={}
    no_dis=load_utility(run/'utility_no_disagreement.pt')
    for label,kwargs in [('global_multiview',dict(mode='global')),('fixed_three',dict(mode='fixed')),('random_three',dict(mode='random')),
             ('policy_one',dict(policy=policy,budget=1)),('policy_two',dict(policy=policy,budget=2)),('policy_three',dict(policy=policy)),
             ('policy_no_disagreement',dict(policy=no_dis)),('policy_adaptive',dict(policy=policy,tau=tau))]:
        pred=predict_cached(fusion,hold_cache,**kwargs);pred.to_csv(run/f'predictions_{label}.csv',index=False);summary[label]=metrics(pred)
    # Separately trained global-only baseline checkpoint, evaluated once on same holdout inputs.
    globalck=torch.load(run/'global_best.pt',map_location=DEVICE,weights_only=False)
    globalenc=Encoder(pretrained=False,backbone=args.backbone).to(DEVICE);globalenc.load_state_dict(globalck['encoder']);globalenc.eval()
    globalfusion=Fusion().to(DEVICE);globalfusion.load_state_dict(globalck['fusion']);globalfusion.eval();baseline=[]
    with torch.no_grad():
        for views,sex,y,desc,ids in loader(df[df.split=='holdout'],batch=4,workers=args.workers,global_only=True,sizes=(args.global_size,args.local_size)):
            with torch.autocast(DEVICE.type,dtype=torch.float16): z=globalenc(views[0].to(DEVICE)).float().unsqueeze(1)
            mean,_,_=globalfusion(z,sex.to(DEVICE),torch.tensor([0],device=DEVICE))
            baseline.extend(dict(id=int(i),true=float(t),sex=int(s),pred=float(p),crops=0) for i,t,s,p in zip(ids,y,sex,mean.cpu()))
    base=pd.DataFrame(baseline);base.to_csv(run/'predictions_global_baseline.csv',index=False);summary['global_baseline']=metrics(base)
    del globalenc,globalfusion;torch.cuda.empty_cache()
    save_json(run/'metrics.json',summary);status(run,'analysis',metrics=summary)
    # report.py performs real end-to-end latency, Grad-CAM and table/figure generation.
    from report import make_report
    make_report(run,enc,fusion,policy,args)
    status(run,'complete',paper=str(HERE/'paper'/'bare_ieee_access.tex'),zip=str(HERE/'BARE_IEEE_Access_LaTeX.zip'))

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['verify','run']);p.add_argument('--root',default=str(HERE.parent))
    p.add_argument('--run',default=str(HERE/'runs'/'overnight'));p.add_argument('--hours',type=float,default=8)
    p.add_argument('--global-epochs',type=int,default=8);p.add_argument('--multiview-epochs',type=int,default=24)
    p.add_argument('--utility-epochs',type=int,default=10);p.add_argument('--seed',type=int,default=42);p.add_argument('--batch',type=int,default=2)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--audit-workers',type=int,default=8)
    p.add_argument('--global-size',type=int,default=384);p.add_argument('--local-size',type=int,default=256)
    p.add_argument('--resume',action='store_true',help='Resume last saved epoch, or reuse completed image-model selection')
    p.add_argument('--backbone',choices=['mobilenet_v3_large','efficientnet_b2','resnet50'],default='mobilenet_v3_large')
    p.add_argument('--global-batch',type=int,default=8)
    args=p.parse_args();run=Path(args.run)
    try: run_experiment(args)
    except Exception as ex:
        run.mkdir(parents=True,exist_ok=True);(run/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        status(run,'failed',error=str(ex));raise

if __name__=='__main__': main()

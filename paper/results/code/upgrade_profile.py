"""Measured admission of larger encoders; optimizer states and replay included."""
import gc,time,json
from pathlib import Path
import torch
import bare

def main():
    torch.set_num_threads(4)
    out=bare.HERE/'runs'/'upgraded';out.mkdir(exist_ok=True)
    rows=[]
    for name,batch in [('resnet50',4)]:
        enc=fusion=opt=z=g=loss=zz=None
        try:
            enc=bare.Encoder(backbone=name).cuda().train();fusion=bare.Fusion().cuda().train()
            opt=torch.optim.AdamW(list(enc.parameters())+list(fusion.parameters()),lr=1e-4)
            scaler=torch.amp.GradScaler('cuda');torch.cuda.reset_peak_memory_stats();times=[]
            xs=[torch.randn(batch,3,640 if j==0 else 384,640 if j==0 else 384) for j in range(10)]
            sex=torch.ones(batch,device='cuda',dtype=torch.long);y=torch.full((batch,),100.,device='cuda')
            for step in range(5):
                torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                    z=torch.stack([enc(x.cuda()) for x in xs],1).float()
                z.requires_grad_(True);loss=bare.main_loss(fusion,z,sex,y);scaler.scale(loss).backward();g=z.grad.detach()
                for j,x in enumerate(xs):
                    with torch.autocast('cuda',dtype=torch.float16):zz=enc(x.cuda())
                    (zz.float()*g[:,j]).sum().backward()
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(fusion.parameters()),1)
                scaler.step(opt);scaler.update();torch.cuda.synchronize()
                if step>0:times.append(time.perf_counter()-t)
            row=dict(backbone=name,batch=batch,global_size=640,local_size=384,parameters=sum(p.numel() for p in enc.parameters())+sum(p.numel() for p in fusion.parameters()),
                     reserved_mib=torch.cuda.max_memory_reserved()/2**20,allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                     seconds_microbatch=sum(times)/len(times),images_per_second=batch/(sum(times)/len(times)),oom=False)
        except torch.OutOfMemoryError:
            row=dict(backbone=name,batch=batch,oom=True)
        rows.append(row);print(json.dumps(row),flush=True);bare.save_json(out/'selected_profile.json',rows)
        del enc,fusion,opt,z,g,loss,zz;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()

"""Train the GUI-domain encoder initialization used before LGWM main training."""
import argparse,json,math,os,time
from pathlib import Path
import numpy as np
_CACHE_ROOT = Path(__file__).resolve().parents[1] / "outputs/cache"
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(_CACHE_ROOT / "torch"))
import torch
import torch.distributed as dist
from lgwm.data.transitions import find_index,sha256_file
from lgwm.train.stage_a import FrameDataset,StageAModel,collate_frames,sample_block_mask
from lgwm.train.trainer import rng_state,set_rng_state
ROOT=Path(__file__).resolve().parents[1]
SOURCES=('miniwob_plus_plus','mobileworld_aitw','android_control','gui_odyssey','amex')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--index-dir',type=Path,action='append',required=True)
    p.add_argument('--run-name',required=True)
    p.add_argument('--encoder',default='vit_base_patch14_dinov2.lvd142m')
    p.add_argument('--backbone-weights',type=Path,help='Optional local DINOv2 safetensors; otherwise use the pretrained timm model')
    p.add_argument('--epochs',type=float,default=2.)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--stop-at',type=int)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--log-every',type=int,default=200)
    p.add_argument('--checkpoint-every',type=int,default=2000)
    args=p.parse_args()
    if Path(args.run_name).name!=args.run_name or args.run_name in ('.','..'):p.error('run-name must be a directory name')
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    if args.batch_size<=0 or args.batch_size%world:p.error('batch size must be positive and divisible by world size')
    torch.cuda.set_device(local)
    if world>1:dist.init_process_group('nccl')
    device=torch.device('cuda',local);torch.manual_seed(2027+rank)
    run=ROOT/'outputs/runs'/args.run_name
    if rank==0 and run.exists() and not args.resume:p.error('run directory already exists; pass --resume or choose a new --run-name')
    indexes=[find_index(args.index_dir,s,'train') for s in SOURCES]
    ds=FrameDataset(indexes,args.data_root)
    steps=int(args.epochs*len(ds)/args.batch_size)
    if steps<=0:raise ValueError('Insufficient frames for the requested schedule')
    if rank==0:run.mkdir(parents=True,exist_ok=True)
    if world>1:dist.barrier()
    sampler=torch.utils.data.distributed.DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=True,seed=2027,drop_last=True)
    dl=torch.utils.data.DataLoader(ds,batch_size=args.batch_size//world,sampler=sampler,num_workers=args.workers,collate_fn=collate_frames,pin_memory=True,drop_last=True,persistent_workers=args.workers>0,prefetch_factor=4 if args.workers else None)
    model=StageAModel(args.encoder,pretrained=not args.resume and not args.backbone_weights).to(device)
    if args.backbone_weights and not args.resume:
        from safetensors.torch import load_file
        from timm.models.vision_transformer import checkpoint_filter_fn
        state=checkpoint_filter_fn(load_file(str(args.backbone_weights)),model.vit)
        model.vit.load_state_dict(state,strict=True)
        model.target.load_state_dict(model.vit.state_dict(),strict=True)
    net=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local]) if world>1 else model
    opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=args.lr,betas=(.9,.95),weight_decay=.05)
    step=epoch=offset=0
    signature={'encoder':args.encoder,'epochs':args.epochs,'batch_size':args.batch_size,'lr':args.lr,'world_size':world,'backbone_sha256':sha256_file(args.backbone_weights) if args.backbone_weights else None,'index_sha256':[sha256_file(p) for p in indexes]}
    if args.resume:
        state=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=True)
        if state['signature']!=signature:raise ValueError('Resume recipe or input indexes differ')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);step=state['step'];epoch=state['epoch'];offset=state['offset']
        set_rng_state(state['rng'],device)
    elif rank==0:
        (run/'config.json').write_text(json.dumps({'signature':signature,'data_root':str(args.data_root.resolve()),'frames':len(ds),'schedule_steps':steps,'seed':2027,'deviations':{k:getattr(args,k) for k,v in {'batch_size':256,'epochs':2.,'lr':1e-4}.items() if getattr(args,k)!=v}},indent=2)+'\n')
    stop=min(args.stop_at or steps,steps)
    if rank==0:print(json.dumps({'unique_frames':len(ds),'steps':steps,'stop_at':stop}),flush=True)
    sampler.set_epoch(epoch);iterator=iter(dl)
    for _ in range(offset):next(iterator)
    def save():
        if rank:return
        state={'signature':signature,'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'epoch':epoch,'offset':offset,'rng':rng_state(device)}
        tmp=run/'checkpoint.pt.partial';torch.save(state,tmp);tmp.replace(run/'checkpoint.pt')
        tmp=run/'stage_a_encoder.pt.partial';torch.save({'vit':model.vit.state_dict(),'step':step,'encoder_name':args.encoder,'epochs':args.epochs,'frames':len(ds),'complete':step==steps},tmp);tmp.replace(run/'stage_a_encoder.pt')
    while step<stop:
        try:x=next(iterator)
        except StopIteration:
            epoch+=1;offset=0;sampler.set_epoch(epoch);iterator=iter(dl);x=next(iterator)
        offset+=1;x=x.to(device,non_blocking=True)
        rng=np.random.default_rng(2027*1_000_003+step)
        mask=torch.from_numpy(sample_block_mask(rng)).to(device).unsqueeze(0).expand(x.shape[0],-1).contiguous()
        lr=args.lr*min(1.,(step+1)/max(steps*.05,1))*(.5*(1+math.cos(math.pi*step/steps))*.9+.1)
        for group in opt.param_groups:group['lr']=lr
        with torch.autocast('cuda',torch.bfloat16):loss=net(x,mask)
        if not torch.isfinite(loss):raise FloatingPointError('Stage A loss is not finite')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True)
        model.ema(.996+.004*step/steps);step+=1
        if rank==0 and step%args.log_every==0:
            rec={'step':step,'loss':loss.item(),'grad_norm':norm.item(),'lr':lr,'mask_ratio':mask[0].float().mean().item()}
            with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            print(json.dumps(rec),flush=True)
        if step%args.checkpoint_every==0:save()
    save()
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()

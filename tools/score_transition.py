"""Score a before/action/after GUI transition with the released online weights."""
import argparse,json,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from safetensors.numpy import load_file
from lgwm.inference import Verifier
from lgwm.data.semantic_action import ACTION_TO_ID,DIRECTION_LITERAL
ROOT=Path(__file__).resolve().parents[1]
TEXT_MODEL='sentence-transformers/all-MiniLM-L6-v2'
TEXT_REVISION='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def action_inputs(action,device,text_vector=None):
    kind=action.get('action_type')
    if kind not in ACTION_TO_ID:raise ValueError('Unknown action_type: '+str(kind))
    coords=[];present=[]
    for x,y in [('x','y'),('x2','y2')]:
        a,b=action.get(x),action.get(y)
        if (a is None)!=(b is None):raise ValueError('Both coordinates in a pair must be present')
        if a is not None and not (math.isfinite(a) and math.isfinite(b) and 0<=a<=1 and 0<=b<=1):raise ValueError('Coordinates must be finite and normalized to [0,1]')
        coords.append([a or 0.,b or 0.]);present.append(a is not None)
    text=(action.get('text') or '')[:64]
    vector=np.zeros(384,np.float32) if text_vector is None else np.asarray(text_vector,dtype=np.float32)
    if vector.shape!=(384,) or not np.isfinite(vector).all():raise ValueError('Expected one finite 384-dimensional text embedding')
    if text and text_vector is None:raise ValueError('Nonempty action text requires its MiniLM embedding')
    direction=(action.get('source_action') or {}).get('direction',action.get('direction')) if kind=='scroll' else None
    if direction is not None and direction not in DIRECTION_LITERAL:raise ValueError('Unknown scroll direction')
    def t(v,dtype):return torch.tensor(v,dtype=dtype,device=device)
    return SimpleNamespace(a_type=t([ACTION_TO_ID[kind]],torch.long),a_coord=t([coords[0]],torch.float32),a_has_coord=t([present[0]],torch.bool),a_coord2=t([coords[1]],torch.float32),a_has_coord2=t([present[1]],torch.bool),a_dir=t([DIRECTION_LITERAL.get(direction,-1)],torch.long),a_text=t(vector[None],torch.float32),a_has_text=t([bool(text)],torch.bool))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--action',type=Path,required=True,help='Canonical action JSON')
    p.add_argument('--weights',type=Path,default=ROOT/'weights/lgwm-online')
    p.add_argument('--harm-head',type=Path,help='Optional compatible linear head directory')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--precision',choices=['bf16','fp32'],default='fp32')
    args=p.parse_args();action=json.loads(args.action.read_text());text=(action.get('text') or '')[:64]
    vector=None
    if text:
        from sentence_transformers import SentenceTransformer
        encoder=SentenceTransformer(TEXT_MODEL,revision=TEXT_REVISION,device=args.device,cache_folder=str(ROOT/'outputs/cache/text-model'))
        vector=encoder.encode(text,normalize_embeddings=True,convert_to_numpy=True)
    a=action_inputs(action,args.device,vector)
    def frame(path):
        with Image.open(path) as im:array=np.array(im.convert('RGB'),copy=True)
        return torch.from_numpy(array).permute(2,0,1).unsqueeze(0).to(args.device)
    model=Verifier.from_pretrained(args.weights,device=args.device)
    torch.backends.cuda.matmul.allow_tf32=False
    with torch.autocast('cuda' if args.device.startswith('cuda') else 'cpu',dtype=torch.bfloat16,enabled=args.precision=='bf16'):
        result=model(frame(args.before),frame(args.after),a)
    output={'integrity':float(result['integrity'][0]),'observed_encoder':'online','precision':args.precision}
    if args.harm_head:
        import hashlib
        config=json.loads((args.harm_head/'config.json').read_text())
        with (args.weights/'model.safetensors').open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if config['world_model_weights_sha256']!=digest or config['observed_encoder']!='online':raise ValueError('Harm head and online weights do not match')
        head=load_file(str(args.harm_head/'head.safetensors'))
        x=result['residual_features'].float().cpu().numpy().copy();x-=head['mean'];x/=head['scale']
        logits=x@head['coef']+head['intercept']
        output['harm_score']=float(torch.from_numpy(np.asarray(logits)).sigmoid().reshape(-1)[0])
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()

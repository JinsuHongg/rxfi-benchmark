#!/usr/bin/env python3
"""Manually benchmark candidate QR batch sizes; never chooses one automatically."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch, yaml
from train_vit_quantile_regression import attach_images, autocast, loader, pinball, read_rows, validate_config, SuryaDataset

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--batch-sizes",nargs="+",type=int,default=[16,32,64,128]); parser.add_argument("--steps",type=int,default=5); parser.add_argument("--precision",choices=["32","16-mixed","bf16-mixed"]); args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    config=yaml.safe_load(Path(args.config).read_text()); config["precision"]=args.precision or config["precision"]; spec=validate_config(config); rows,_=read_rows(config,spec); records,_=attach_images(rows,config,spec); device=torch.device("cuda"); quantiles=torch.tensor(config["quantiles"],device=device)
    for batch_size in args.batch_sizes:
        config["batch_size"]=batch_size; torch.cuda.empty_cache()
        try:
            import timm
            model=timm.create_model(config["model_name"],pretrained=False,in_chans=config["in_chans"],num_classes=3).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"]); iterator=iter(loader(SuryaDataset(records["train"],config["zarr_path"]),config,True)); torch.cuda.reset_peak_memory_stats(device); began=time.perf_counter()
            for _ in range(args.steps):
                x,y=next(iterator); x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
                with autocast(config["precision"]): loss=pinball(model(x),y,quantiles)
                loss.backward(); optimizer.step()
            torch.cuda.synchronize(device); elapsed=time.perf_counter()-began; print(json.dumps({"batch_size":batch_size,"precision":config["precision"],"peak_vram_mib":round(torch.cuda.max_memory_allocated()/2**20,2),"iteration_seconds":elapsed/args.steps,"samples_per_second":batch_size*args.steps/elapsed,"finite_loss":bool(torch.isfinite(loss)),"oom":False}))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(json.dumps({"batch_size":batch_size,"precision":config["precision"],"oom":True}))
if __name__=="__main__": main()

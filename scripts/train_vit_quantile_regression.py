#!/usr/bin/env python3
"""Config-driven ViT-Small quantile regression for NOAA/NCEI targets."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, subprocess, time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm, torch, yaml, zarr
from torch import nn
from torch.utils.data import DataLoader, Dataset

SPLITS = ("train", "validation", "test", "leaky_validation")
SOURCE_HASHES = {"train":"2ec7b8f39367f8340a39889bc66525aff303410d7b7ce6c12a55ea346b55e865","validation":"803d2e5584fe9bbe23bc02cbed1b06fb47520e4863c2b22b5f09f9d5c654c658","test":"40ddef01aebe23e5ee460717a08b7392827eacca2852af074d5f1533f59ebd4b","leaky_validation":"03134a82a53891d25761774c5aad52f77e01673195f7cfd28c0dc061bfe5849e"}
PRECISIONS = {"32", "16-mixed", "bf16-mixed"}

def channel_selection(config, actual_order):
    selected = config.get("selected_channels", actual_order)
    if not selected or len(set(selected)) != len(selected) or any(name not in actual_order for name in selected): raise ValueError("selected_channels must be a non-empty, unique subset of canonical channel_order")
    indices = [actual_order.index(name) for name in selected]
    if int(config["in_chans"]) != len(indices): raise ValueError("in_chans must equal selected channel count")
    return list(selected), indices

def channel_selection(config, actual_order):
    selected = config.get("selected_channels", actual_order)
    if not selected or len(set(selected)) != len(selected) or any(name not in actual_order for name in selected): raise ValueError("selected_channels must be a non-empty, unique subset of canonical channel_order")
    indices = [actual_order.index(name) for name in selected]
    if int(config["in_chans"]) != len(indices): raise ValueError("in_chans must equal selected channel count")
    return list(selected), indices

@dataclass(frozen=True)
class Record:
    split: str; original_row_index: int; timestamp: str; year: int; zarr_index: int; raw: float; transformed: float; flare_class: str

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()

def timestamp_ns(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")); parsed = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000_000_000)

def validate_transform(spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict) or spec.get("name") not in {"log10_scaled", "log10_1p_scaled", "log10_1p"}: raise ValueError("Unsupported target_transform.name")
    result = {"name": spec["name"]}
    if spec["name"] != "log10_1p":
        scale = float(spec.get("scale", 0))
        if not math.isfinite(scale) or scale <= 0: raise ValueError("Scaled transform requires positive scale")
        result["scale"] = scale
    return result

def forward(raw: float, spec: dict[str, Any]) -> float | None:
    if not math.isfinite(raw): return None
    if spec["name"] == "log10_scaled": value = math.log10(raw / spec["scale"]) if raw > 0 else math.nan
    elif spec["name"] == "log10_1p_scaled": value = math.log10(1 + raw / spec["scale"]) if raw >= 0 else math.nan
    else: value = math.log10(1 + raw) if raw >= 0 else math.nan
    return value if math.isfinite(value) else None

def inverse(value: float, spec: dict[str, Any]) -> float:
    if spec["name"] == "log10_scaled": return spec["scale"] * 10 ** value
    if spec["name"] == "log10_1p_scaled": return spec["scale"] * (10 ** value - 1)
    return 10 ** value - 1

def transform_tests() -> None:
    cases = [(1e-8,{"name":"log10_scaled","scale":1e-8},0.),(1e-7,{"name":"log10_scaled","scale":1e-8},1.),(1e-6,{"name":"log10_scaled","scale":1e-8},2.),(0.,{"name":"log10_1p_scaled","scale":1e-8},0.),(0.,{"name":"log10_1p"},0.)]
    for raw, spec, expected in cases:
        spec = validate_transform(spec); value = forward(raw, spec); assert value is not None and math.isclose(value, expected, abs_tol=1e-12) and math.isclose(inverse(value, spec), raw, abs_tol=1e-12)

def target(row: dict[str,str], config: dict[str,Any], spec: dict[str,Any]) -> tuple[float,float] | None:
    try: raw = float(row[config["target_column"]])
    except (KeyError, ValueError): return None
    value = forward(raw, spec); return (raw, value) if value is not None else None

def read_rows(config: dict[str,Any], spec: dict[str,Any]) -> tuple[dict[str,list[dict[str,str]]],dict[str,Any]]:
    result = {}; meta = {"source_split_sha256":{},"derived_target_sha256":{},"target_validity":{}}
    for split in SPLITS:
        source = Path(config["source_split_dir"]) / f"{split}.csv"; derived = Path(config["derived_target_dir"]) / f"{split}_targets.csv"
        sh, dh = sha256(source), sha256(derived)
        if sh != SOURCE_HASHES[split]: raise RuntimeError(f"Frozen {split} hash changed")
        with source.open(newline="", encoding="utf-8") as h: source_rows = list(csv.DictReader(h))
        with derived.open(newline="", encoding="utf-8") as h: rows = list(csv.DictReader(h))
        if len(source_rows) != len(rows): raise RuntimeError(f"{split} row count mismatch")
        for i,(a,b) in enumerate(zip(source_rows,rows)):
            if a["timestamp"] != b["timestamp"] or int(b.get("original_row_index",i)) != i: raise RuntimeError(f"{split} row alignment mismatch at {i}")
        valid = sum(target(row,config,spec) is not None for row in rows); result[split] = rows; meta["source_split_sha256"][split] = sh; meta["derived_target_sha256"][split] = dh; meta["target_validity"][split] = {"total":len(rows),"valid":valid,"excluded_missing_or_invalid":len(rows)-valid}
    return result,meta

def attach_images(rows: dict[str,list[dict[str,str]]], config: dict[str,Any], spec: dict[str,Any]) -> tuple[dict[str,list[Record]],dict[str,Any]]:
    needed: dict[int,set[int]] = defaultdict(set)
    for values in rows.values():
        for row in values:
            if target(row,config,spec) is not None: needed[datetime.fromisoformat(row["timestamp"]).year].add(timestamp_ns(row["timestamp"]))
    positions = {}; availability = {"per_year":{},"missing_or_unreadable":[]}; root = Path(config["zarr_path"])
    for year, stamps in needed.items():
        group_path = root / str(year) / "dataset"
        if not group_path.exists(): positions[year] = {}; availability["missing_or_unreadable"] += [{"year":year,"timestamp_ns":stamp} for stamp in stamps]; continue
        group = zarr.open_group(str(group_path),mode="r"); images,times = group["images"],group["time"]
        if list(images.attrs["channel_names"]) != config["channel_order"] or tuple(images.shape[1:]) != (len(config["channel_order"]),int(config["image_size"]),int(config["image_size"])): raise RuntimeError(f"{year} image schema mismatch")
        found = {int(stamp):i for i,stamp in enumerate(np.asarray(times[:],dtype=np.int64))}; missing = stamps-found.keys(); positions[year]=found; availability["per_year"][str(year)]={"requested":len(stamps),"available":len(stamps)-len(missing)}; availability["missing_or_unreadable"] += [{"year":year,"timestamp_ns":stamp} for stamp in missing]
    unavailable = {(x["year"],x["timestamp_ns"]) for x in availability["missing_or_unreadable"]}; availability["missing_or_unreadable"]=[]; availability["canonical_channel_order"] = config["channel_order"]; availability["selected_channel_names"], availability["selected_channel_indices"] = channel_selection(config, config["channel_order"]); result={}
    for split, values in rows.items():
        records=[]
        for i,row in enumerate(values):
            value=target(row,config,spec)
            if value is None: continue
            raw,z=value; stamp=timestamp_ns(row["timestamp"]); year=datetime.fromisoformat(row["timestamp"]).year
            if (year,stamp) in unavailable: availability["missing_or_unreadable"].append({"split":split,"timestamp":row["timestamp"],"year":year,"reason":"timestamp_not_in_zarr"}); continue
            records.append(Record(split,int(row.get("original_row_index",i)),row["timestamp"],year,positions[year][stamp],raw,z,row["max_flare_class"]))
        result[split]=records
    return result,availability

class SuryaDataset(Dataset):
    def __init__(self,records:list[Record],zarr_path:str,channel_indices:list[int]): self.records,self.zarr_path,self.channel_indices,self.arrays=records,Path(zarr_path),channel_indices,{}
    def __len__(self): return len(self.records)
    def __getitem__(self,index):
        record=self.records[index]
        if record.year not in self.arrays: self.arrays[record.year]=zarr.open_group(str(self.zarr_path/str(record.year)/"dataset"),mode="r")["images"]
        image=np.asarray(self.arrays[record.year][record.zarr_index],dtype=np.float32)
        if image.shape != (13,224,224) or not np.isfinite(image).all(): raise RuntimeError(f"Unreadable image {record.timestamp}")
        image = image[self.channel_indices]
        return torch.from_numpy((image-image.mean((1,2),keepdims=True))/np.maximum(image.std((1,2),keepdims=True),1e-6)),torch.tensor(record.transformed,dtype=torch.float32)

def loader(dataset,config,shuffle): return DataLoader(dataset,batch_size=int(config["batch_size"]),shuffle=shuffle,num_workers=int(config["num_workers"]),pin_memory=True)
def pinball(pred,y,q,reduction="mean"):
    values=torch.maximum(q*(y[:,None]-pred),(q-1)*(y[:,None]-pred)); return values.mean() if reduction=="mean" else values
def autocast(precision): return torch.amp.autocast("cuda",dtype=torch.bfloat16 if precision=="bf16-mixed" else torch.float16,enabled=precision!="32")

def metrics(pred,y,q):
    loss=pinball(pred,y,q,"none").mean(0); result={"pinball_loss":float(loss.mean()),"mae_q50":float((y-pred[:,1]).abs().mean()),"mse_q50":float(((y-pred[:,1])**2).mean())}
    for i,tau in enumerate(q): result[f"pinball_q{int(tau*100):02d}"]=float(loss[i]); result[f"fraction_target_le_q{int(tau*100):02d}"]=float((y<=pred[:,i]).float().mean())
    result.update({"cross_q05_gt_q50":float((pred[:,0]>pred[:,1]).float().mean()),"cross_q50_gt_q95":float((pred[:,1]>pred[:,2]).float().mean()),"cross_q05_gt_q95":float((pred[:,0]>pred[:,2]).float().mean())}); return result

def run_epoch(model,dl,opt,scaler,device,q,config):
    train=opt is not None; model.train(train); total=defaultdict(float)
    if train: opt.zero_grad(set_to_none=True)
    for step,(x,y) in enumerate(dl):
        x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
        with torch.set_grad_enabled(train),autocast(config["precision"]): pred=model(x); loss=pinball(pred,y,q)
        if not torch.isfinite(loss): raise RuntimeError("Non-finite pinball loss")
        if train:
            scaler.scale(loss/int(config["gradient_accumulation_steps"])).backward()
            if (step+1)%int(config["gradient_accumulation_steps"])==0 or step+1==len(dl): scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        for name,value in metrics(pred.detach(),y.detach(),q).items(): total[name]+=value*len(y)
    out={name:value/len(dl.dataset) for name,value in total.items()};out["rmse_q50"]=math.sqrt(out.pop("mse_q50"));return out

def write_csv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True); fields=fields or (list(rows[0]) if rows else [])
    with path.open("w",newline="",encoding="utf-8") as h: writer=csv.DictWriter(h,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)

def plot_curves(path,history):
    epochs=[row["epoch"] for row in history];fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(epochs,[row["train_pinball_loss"] for row in history],label="train");axes[0].plot(epochs,[row["validation_pinball_loss"] for row in history],label="validation");axes[0].set(xlabel="epoch",ylabel="pinball loss");axes[0].legend()
    axes[1].plot(epochs,[row["validation_mae_q50"] for row in history],label="MAE q50");axes[1].plot(epochs,[row["validation_rmse_q50"] for row in history],label="RMSE q50");axes[1].set(xlabel="epoch",ylabel="transformed-target error");axes[1].legend();fig.tight_layout();fig.savefig(path,format="svg");plt.close(fig)

def exports(model,dl,records,config,spec,device):
    model.eval();rows=[];i=0
    with torch.no_grad():
        for x,_ in dl:
            with autocast(config["precision"]): values=model(x.to(device,non_blocking=True)).float().cpu().numpy()
            for value in values:
                record=records[i];i+=1;raw=[inverse(float(v),spec) for v in value]
                rows.append({"split":record.split,"original_row_index":record.original_row_index,"timestamp":record.timestamp,"target_name":config["target_column"],"target_raw":format(record.raw,".12g"),"target_transformed":format(record.transformed,".12g"),"max_flare_class":record.flare_class,"q05":format(value[0],".12g"),"q50":format(value[1],".12g"),"q95":format(value[2],".12g"),"q05_raw":format(raw[0],".12g"),"q50_raw":format(raw[1],".12g"),"q95_raw":format(raw[2],".12g")})
    return rows

def git_commit():
    try:return subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):return None

def validate_config(config):
    spec=validate_transform(config.get("target_transform"));required={"experiment_name","output_dir","target_column","quantiles","model_name","checkpoint_metric","precision","batch_size","num_workers","gradient_accumulation_steps"}
    if required-set(config):raise ValueError(f"Missing config fields: {sorted(required-set(config))}")
    frozen = {"model_name":"vit_small_patch16_224", "image_size":224, "in_chans":int(config["in_chans"]), "channels":int(config["in_chans"]), "pretrained":False, "quantiles":[0.05,0.5,0.95], "optimizer":"adamw", "learning_rate":1e-4, "lr_policy":"constant", "weight_decay":0.01, "epochs":10, "seed":0, "checkpoint_metric":"validation_pinball_loss"}
    missing = set(frozen) - set(config)
    if missing: raise ValueError(f"Missing frozen config fields: {sorted(missing)}")
    if any(config[key] != value for key, value in frozen.items()): raise ValueError("Model/training/checkpoint settings differ from the frozen QR experiment family")
    if config["in_chans"] != config["channels"]: raise ValueError("channels must equal in_chans")
    if int(config["batch_size"]) <= 0 or int(config["num_workers"]) < 0 or int(config["gradient_accumulation_steps"]) <= 0: raise ValueError("Batch size and gradient accumulation must be positive; num_workers cannot be negative")
    if config["precision"] not in PRECISIONS:raise ValueError("Unsupported precision")
    return spec

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/vit_small_224_max_peak_flux_qr.yaml");parser.add_argument("--smoke-only",action="store_true");parser.add_argument("--validate-transforms",action="store_true");parser.add_argument("--batch-size",type=int);parser.add_argument("--precision",choices=sorted(PRECISIONS));parser.add_argument("--num-workers",type=int);parser.add_argument("--gradient-accumulation-steps",type=int);args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());
    for name in ("batch_size","precision","num_workers","gradient_accumulation_steps"):
        if getattr(args,name) is not None:config[name]=getattr(args,name)
    spec=validate_config(config)
    if args.validate_transforms:transform_tests();print(json.dumps({"transform_tests":"passed","transform":spec}));return
    if not torch.cuda.is_available():raise RuntimeError("CUDA GPU is required; refusing CPU fallback")
    output=Path(config["output_dir"]);output.mkdir(parents=True,exist_ok=True);(output/"checkpoints").mkdir(exist_ok=True);(output/"predictions").mkdir(exist_ok=True)
    random.seed(config["seed"]);np.random.seed(config["seed"]);torch.manual_seed(config["seed"]);torch.cuda.manual_seed_all(config["seed"])
    rows,integrity=read_rows(config,spec);hashes=dict(integrity["derived_target_sha256"]);records,availability=attach_images(rows,config,spec);datasets={s:SuryaDataset(r,config["zarr_path"],availability["selected_channel_indices"]) for s,r in records.items()};loaders={s:loader(d,config,s=="train") for s,d in datasets.items()};device=torch.device("cuda");q=torch.tensor(config["quantiles"],device=device)
    cohort={s:{"total_split_rows":integrity["target_validity"][s]["total"],"valid_target_rows":integrity["target_validity"][s]["valid"],"image_target_available_rows":len(records[s]),"missing_target_rows":integrity["target_validity"][s]["excluded_missing_or_invalid"],"missing_image_rows":integrity["target_validity"][s]["valid"]-len(records[s])} for s in SPLITS};print(json.dumps({"cohort":cohort}),flush=True);meta={"git_commit":git_commit(),"config_path":args.config,"config":config,"target_transform":spec,"integrity":integrity,"cohort":cohort,"legacy_suryabench_label_used_as_target":False,"torch_version":torch.__version__,"cuda_version":torch.version.cuda,"gpu_name":torch.cuda.get_device_name(device),"selected_channel_names":availability["selected_channel_names"],"selected_channel_indices":availability["selected_channel_indices"]}
    (output/"resolved_config.yaml").write_text(yaml.safe_dump(config,sort_keys=False));(output/"transform_metadata.json").write_text(json.dumps(spec,indent=2)+"\n");(output/"experiment_metadata.json").write_text(json.dumps(meta,indent=2)+"\n");write_csv(output/"cohort_summary.csv",[{"split":s,**cohort[s]} for s in SPLITS]);write_csv(output/"missing_or_unreadable_images.csv",availability["missing_or_unreadable"],["split","timestamp","year","reason"])
    fields=["split","original_row_index","timestamp","target_name","target_raw","target_transformed","max_flare_class","q05","q50","q95","q05_raw","q50_raw","q95_raw"];(output/"prediction_schema.json").write_text(json.dumps({"columns":fields,"quantiles":config["quantiles"],"target_transform":spec},indent=2)+"\n")
    model=timm.create_model(config["model_name"],pretrained=False,in_chans=config["in_chans"],num_classes=3).to(device)
    if args.smoke_only:return
    scaler=torch.amp.GradScaler("cuda",enabled=config["precision"]=="16-mixed");opt=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"]);history=[];best,best_epoch=math.inf,0
    for epoch in range(1,config["epochs"]+1):
        start=time.perf_counter();train=run_epoch(model,loaders["train"],opt,scaler,device,q,config);valid=run_epoch(model,loaders["validation"],None,scaler,device,q,config);row={"epoch":epoch,**{"train_"+k:v for k,v in train.items()},**{"validation_"+k:v for k,v in valid.items()},"elapsed_seconds":round(time.perf_counter()-start,2)};history.append(row);write_csv(output/"training_log.csv",history)
        if valid["pinball_loss"]<best:best,best_epoch=valid["pinball_loss"],epoch;torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"target_transform":spec,"quantiles":config["quantiles"]},output/"checkpoints"/"best_validation_pinball_loss.pt")
    plot_curves(output/"learning_curves.svg",history);state=torch.load(output/"checkpoints"/"best_validation_pinball_loss.pt",map_location=device,weights_only=False);model.load_state_dict(state["model_state_dict"]);summary=[]
    for split in ("validation","test","leaky_validation"):
        result=run_epoch(model,loaders[split],None,scaler,device,q,config);summary.append({"split":split,"selected_epoch":best_epoch,**result});write_csv(output/"predictions"/f"{split}_predictions.csv",exports(model,loaders[split],records[split],config,spec,device),fields)
    write_csv(output/"summary_metrics.csv",summary);(output/"checkpoint_metadata.json").write_text(json.dumps({"best_validation_epoch":best_epoch,"best_validation_pinball_loss":best,"selection_split":"validation"},indent=2)+"\n")
    if hashes!={s:sha256(Path(config["derived_target_dir"])/f"{s}_targets.csv") for s in SPLITS}:raise RuntimeError("Derived targets changed during training")
if __name__=="__main__":main()

#!/usr/bin/env python3
"""Audit Project 2 config channel names/indices against Zarr metadata; no training."""
import json
from pathlib import Path
import yaml, zarr
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import train_vit_classifier as classifier
configs=sorted((ROOT/'configs/project2').glob('*.yaml'))
seen={}
for p in configs:
 c=yaml.safe_load(p.read_text()); names=list(c['selected_channels']); canonical=list(c['channel_order']); indices=[canonical.index(x) for x in names]
 if len(names)!=c['in_chans'] or len(set(names))!=len(names): raise SystemExit(f'bad selection: {p}')
 seen["+".join(names)]={'selected_channels':names,'selected_channel_indices':indices,'input_channel_count':c['in_chans']}
group=zarr.open_group(str(yaml.safe_load(configs[0].read_text())['zarr_path']+'/2010/dataset'),mode='r')
actual=list(group['images'].attrs['channel_names'])
if actual != yaml.safe_load(configs[0].read_text())['channel_order']: raise SystemExit('Zarr canonical order mismatch')
cohorts = {}; usable_keys = {}
for p in configs:
 c=yaml.safe_load(p.read_text())
 if c["target_column"] == "max_flare_class":
  rows, _ = classifier.load_rows(c)
  labels = sorted({r[c["target_column"]] for values in rows.values() for r in values}, key=classifier.class_sort_key)
  records, _ = classifier.attach_zarr_indices(rows, c, {label:i for i,label in enumerate(labels)})
  condition = "all13" if len(c["selected_channels"]) == 13 else c["selected_channels"][0]; usable_keys[condition] = {split:{record.timestamp for record in records[split]} for split in rows}; cohorts[condition] = {split:{"total_split_rows":len(rows[split]),"target_valid_rows":len(rows[split]),"selected_channel_image_available_rows":len(records[split]),"image_target_usable_rows":len(records[split])} for split in rows}
common_usable_cohort = {split: len(set.intersection(*(usable_keys[name][split] for name in usable_keys))) for split in ("train", "validation", "test", "leaky_validation")}
out={"canonical_channel_order":actual,"canonical_indices":{x:i for i,x in enumerate(actual)},"configurations":seen,"normalization":"per_sample_per_channel_zscore after selected-channel slicing","cohort_availability":cohorts,"common_usable_cohort":common_usable_cohort}
(ROOT/"outputs/project2_channel_audit.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))

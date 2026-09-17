#!/usr/bin/env python3
"""Post-training Project 2 comparison scaffold; refuses missing completed artifacts."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
for task, suffix in [('classification','max_flare_class'),('regression','cumulative_peak_flux_qr')]:
 rows=[]
 for channel in ('hmi_m','aia131','aia193','all13'):
  directory=ROOT/'outputs/project2'/f'vit_small_224_{channel}_{suffix}'
  summary=directory/'summary_metrics.csv'
  if not summary.exists(): raise SystemExit(f'Missing completed output: {summary}; do not infer placeholder results.')
  row=pd.read_csv(summary).query("split == 'test'").iloc[0].to_dict(); row['channel_configuration']=channel; rows.append(row)
 pd.DataFrame(rows).to_csv(ROOT/'outputs/project2'/f'table_{task}_comparison.csv',index=False)

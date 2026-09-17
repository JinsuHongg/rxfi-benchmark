# Project 2 classification label audit

The completed preliminary classifier used 310 exact NOAA magnitude strings (for example `B1.0`, `C1.2`, and `M2.1`) as separate softmax classes. This is directly verified by each saved `class_mapping.json` and explains the near-zero macro-F1/balanced accuracy. It is not the intended ordinal severity task.

The corrected target representation is the validated mapping `FQ -> FQ`, `A* -> A`, `B* -> B`, `C* -> C`, `M* -> M`, and `X* -> X`, with fixed order `FQ=0, A=1, B=2, C=3, M=4, X=5`. Unknown or malformed values are rejected. The frozen usable-cohort counts are: train `10410/0/10438/17268/6202/716`, validation `395/0/644/1020/332/31`, test `5307/0/4477/9600/7628/968`, and leaky validation `678/0/827/1387/803/60` in that order. There are no A-band examples, but A remains in the fixed six-class vocabulary.

Four corrected `_ordinal6` channel-condition configs preserve the channels, splits, ViT-Small protocol, cross-entropy loss, and validation macro-F1 selection. The original exact-string results are retained as provenance and must not be used as the primary Project 2 classification comparison. QR outputs are unaffected.

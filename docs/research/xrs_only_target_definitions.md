# XRS-only 24-hour regression targets

## Status and scope

This document records the approved research definition for a new set of four
continuous targets. The target values will be calculated from science-quality,
one-minute GOES XRS Level 2 XRS-B observations only. The NOAA flare catalog is
not an input to this target-generation procedure; it remains available for
separate operational-skill evaluation on the test set.

This is a target specification, not an implementation or a claim about the
currently saved split CSVs. Existing split targets and their generator use the
legacy catalog-based definitions described in
[24h_target_generation.md](24h_target_generation.md). The new targets have not
yet been generated.

## Windows and quality filtering

For a forecast-window start time `s`, the prediction window is the half-open
interval `[s, s + 24 h)`. Let `F_i` denote an eligible one-minute XRS-B flux
sample in that interval. At minimum, eligible measurements must be finite,
positive, and have `xrsb_flag == 0`.

The background is fixed for the whole prediction window. It is computed from
the preceding, non-overlapping 24-hour reference interval
`[s - 48 h, s - 24 h)`, using the mean of the lower 50% of eligible XRS-B
measurements. If the sorted eligible reference samples are
`R_(1) <= ... <= R_(N)`, then

\[
B_s = \frac{1}{\lceil N/2\rceil}\sum_{j=1}^{\lceil N/2\rceil} R_{(j)}.
\]

The reference interval is based on forecast-window start `s`, not on a flare
catalog start time. No flare intervals are masked because the catalog is not
used to construct these targets.

## Four targets

For the eligible prediction-window samples `F_i`, define:

| Target | Definition | Units |
| --- | --- | --- |
| `max_xflux` | `max_i F_i` | W m^-2 |
| `fluence` | `60 s * sum_i F_i` | J m^-2 |
| `max_background_deviation` | `max_i (F_i - B_s)` | W m^-2 |
| `total_background_deviation` | `60 s * sum_i (F_i - B_s)` | J m^-2 |

The sums are over the one-minute measurements in `[s, s + 24 h)`. The 60-second
factor treats each one-minute average as representing one minute of exposure,
so the resulting fluence is a cadence-weighted discrete approximation to a
time integral. A bare sample sum may also be reported as an intermediate
quantity, but it is cadence-dependent and does not have fluence units.

Background deviation is signed. Negative sample-level deviations and negative
window targets are retained; values below background are part of the measured
signal and are not clipped to zero.

The name `max_background_deviation` describes the largest signed difference
from the fixed background. It is not a fractional increase. The distinct
fractional-excess definition would be `(F - B_s) / B_s`; that RXFI-style ratio
is not one of the four new targets.

## Relationship between targets

With one fixed background `B_s` and a complete 24-hour series of 1,440
one-minute bins, the two background-deviation targets are algebraically
related to flux targets:

\[
\texttt{max\_background\_deviation} = \texttt{max\_xflux} - B_s,
\]

\[
\texttt{total\_background\_deviation} = \texttt{fluence} - 86{,}400\,B_s.
\]

Therefore, the four targets provide different views of flux and baseline
context, but are not four statistically independent quantities. These
identities require complete, uniformly sampled coverage; the final missing
data policy may change how they are computed for incomplete windows.

## Decisions still open before implementation

- **Coverage and missing minutes:** define an acceptable coverage threshold
  and whether short gaps are interpolated, left missing, or otherwise handled.
  Summing only available samples without adjustment would make fluence depend
  on data gaps.
- **Satellite selection and transitions:** define which GOES satellite is used
  when multiple satellites overlap, and how handovers or unavailable channels
  are handled without relying on catalog-selected satellite IDs.
- **Output columns and regeneration:** implement the XRS-only generator, choose
  versioned output locations/names, regenerate split targets, and validate
  provenance before treating them as current data products.

## Provenance and evaluation role of the catalog

Input observations are NOAA/NCEI science-quality GOES XRS Level 2 one-minute
XRS-B measurements. The catalog remains useful for identifying observed flare
events when evaluating operational performance (for example, event detection
or M/X-class skill) on held-out test data. It does not define the four
regression values above.

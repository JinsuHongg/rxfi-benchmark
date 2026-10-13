# Lower-50% background results

These outputs are calculated from the checksum-pinned NOAA/NCEI flare report and GOES XRS Level 2 `avg1m` inputs using [`configs/background_lower50.json`](../../configs/background_lower50.json). The method averages the lowest half of valid, non-flare measurements in the 24 hours before each event start and calculates project RXFI as `(peak_flux - background) / background`.

## Current run

Run fingerprint: `a7e506fa45a732128ad3192cc023661000c6dab37f9633c320b21dab4ff1cb55`

- Events: 41,723 (peak dates 2010-01-01 through 2025-12-31)
- Lower-50% background and RXFI available: 41,723 (100%)
- Median lower-50% background: `6.767769e-7 W m^-2`
- Median lower-50% RXFI: `1.42337`; 90th percentile: `7.51969`
- Uncomputable RXFI: 0; negative RXFI: 261 (0.63%), all A/B/C-class
- M-class: 3,022 events; median RXFI `10.1785`
- X-class: 167 events; median RXFI `105.233`
- Median lower-50% background / NOAA catalog `background_irrad`: `0.8348` across 41,721 comparable events

The event-level values, inputs, and exact config are in [`runs/a7e506fa45a732128ad3192cc023661000c6dab37f9633c320b21dab4ff1cb55/`](runs/a7e506fa45a732128ad3192cc023661000c6dab37f9633c320b21dab4ff1cb55/). See [`docs/research/lower50_background_rxfi.md`](../../docs/research/lower50_background_rxfi.md) for the method definition and caveats.

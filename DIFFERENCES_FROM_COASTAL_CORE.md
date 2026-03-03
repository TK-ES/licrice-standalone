# Differences from coastal-core-main

This document records where `licrice-standalone` intentionally diverges from
`coastal-core-main` (`pyTC`).

## 1. Radius estimation (`licrice/tracks/radius.py`)

Both use RF models (RandomForestRegressor) trained on IBTrACS to estimate
`rmstore_estimated` (RMW) and `storm_radius_estimated` (ROCI). The logic in
`estimate_rmw_climada`, `estimate_rmw_licrice`, `create_radius_reg_dataset`, and
`estimate_radii` is verbatim from coastal-core.

**Standalone-only additions:**
- `get_radius_ratio_models(ds, model_dir)` — takes a cleaned `ds` directly instead of
  calling `ibtracs.load_processed_ibtracs("ALL", ps)`. Saves `.pkl` files to `model_dir`.
- `load_radius_models(model_dir)` — loads saved `.pkl` files; returns
  `(rmw_to_rad, rad_to_rmw, rmw, cols)` where `cols` ensures feature alignment between
  training (global IBTrACS) and inference (domain-filtered) datasets.

**Coastal-core `get_radius_ratio_models(ps, save)`** takes a `Settings` object and an
optional `save` flag; does not return `cols`. The standalone version returns `cols` and
caches models at `params/radius/` automatically.

## 2. `download()` signature (`licrice/io/ibtracs.py`)

coastal-core: `download(ps=ps)` — uses `ps.IBTRACS_URL` and `ps.DIR_TRACKS_RAW_HIST`

licrice-standalone: `download(url, outdir)` — takes explicit parameters.

## 3. No `Settings` object

coastal-core has a central `pyTC.settings.Settings` object that stores paths, model
versions, and geography config. licrice-standalone passes all config as explicit function
arguments.

## 4. No GCS / cloud I/O

`rhg_compute_tools` and Google Cloud Storage integrations are not present.
`check_finished_zarr_workflow` is simplified (no `tmpstore` file-system checks that use
`gcsfs`).

## 5. Import paths

All `from pyTC import ...` imports are replaced with `from licrice import ...`.

## 6. `_latlon_to_geosph_vector` (`licrice/spatial.py`)

coastal-core uses `climada.util.coordinates.latlon_to_geosph_vector`.
licrice-standalone has an inline replacement that avoids the CLIMADA dependency.

## 7. `convert_units` (`licrice/utilities.py`)

coastal-core uses `pint`-based unit conversion and also accepts `**unit_kwargs` and
DataArrays with pint quantification (lines 155–160 of pyTC/utilities.py).
licrice-standalone uses a lookup table (`_UNIT_CONVERSIONS`) with the same supported
conversions and identical basic call signature; `**unit_kwargs` and pint DataArrays are
not supported.

## 8. `smooth_fill` default `time_dim`

Both coastal-core and licrice-standalone default `time_dim="time"` in `smooth_fill`.
The inner helper `_smooth_interp_w_other_data_inner` defaults to `time_dim="date_time"`
in both (it is always called with an explicit value from `smooth_fill`).

## 9. `geoclaw_convert()` (`licrice/utilities.py`)

New in licrice-standalone; not present in coastal-core. Replaces
`clawpack.geoclaw.units.convert` to avoid the clawpack dependency. Uses the same
`_UNIT_CONVERSIONS` lookup table.

## 10. `preprocess_ibtracs()` (`licrice/io/ibtracs.py`)

New in licrice-standalone. coastal-core performs this step in a notebook; there is no
equivalent standalone function. Wraps `format_clean` + RF radius training/loading +
`estimate_radii` + zarr save. Signature: `preprocess_ibtracs(nc_path, zarr_outpath,
overwrite=False)` — no `params` dict (radius fill fallback removed in favour of RF
model which fills all gaps).

## 11. I/O helpers absent

`load_processed_ibtracs`, `get_processed_ibtracs_path`, `load_raw_ibtracs`,
`get_raw_ibtracs_path` from coastal-core `pyTC/io/tracks/ibtracs.py` are not included;
they depend on `Settings` paths and GCS and are not needed for the standalone pipeline.

## 12. Many coastal-core `tracks/` functions omitted

licrice-standalone includes only the subset of `pyTC/tracks/` needed for
IBTrACS→LICRICE: `velocity`, `utils` (core functions), and `radius`. Omitted functions
include `set_missing_storm_rad_to_1000km`, `randomly_sample_storm_size`,
`update_with_random_storm_radius_sampling` (geoclaw/synthetic-track only), and many
utility functions in `pyTC/tracks/utils.py` that require geopandas, shapely, or pyTC
internal modules.

## 13. Reference output

The zarrs in `licrice_result_data/` were generated from historical IBTrACS tracks using
coastal-core. The `"notes"` attribute in those files reads "Synthetic storm track wind
field summary statistics" — this is a misleading label copied verbatim from the
coastal-core notebook, which hardcoded that string regardless of track source.
`run_licrice.py` corrects this to "Historical IBTrACS storm track wind field summary
statistics."

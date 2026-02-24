"""IBTrACS preprocessing for LICRICE.

Converts a raw IBTrACS netCDF file (e.g. IBTrACS.ALL.v04r01.nc) into the
intermediate zarr format expected by licrice's preprocess.load_tracks / _clean_ibtracs.

The output zarr has dimensions (storm, time) with integer indices and variables:
  latstore, longstore, v_circular, v_total, datetime, pstore (hPa),
  rmstore_estimated (km), storm_radius_estimated (km), sid, season

This matches the format produced by coastal-core-main's
`pyTC.io.tracks.ibtracs.format_clean` + `pyTC.tracks.radius.estimate_radii`.
The radius estimation here uses physics-based formulas rather than a random forest.
"""

import numpy as np
import xarray as xr

from licrice.tracks import velocity as vel

# ---------------------------------------------------------------------------
# Agency wind-averaging-period table
# Source: IBTrACS v4 Technical Details (WMO_TD_1555_en.pdf)
# ---------------------------------------------------------------------------
AGENCY_AVERAGING_PERIOD = {
    "usa": 1,
    "cma": 2,
    "bom": 10,
    "wellington": 10,
    "nadi": 10,
    "tokyo": 10,
    "reunion": 10,
    "newdelhi": 3,
    "hko": 10,
    "ds824": 1,
    "td9636": 1,
    "td9635": 1,
    "neumann": 10,
    "mlc": 1,
}

# Divisor to convert X-minute average to 1-minute average
# (from WMO recommendations; see pyTC.tracks.velocity.AVERAGING_PERIOD_CONVERSION_DIVISOR)
_AVG_PERIOD_DIVISOR = {1: 1.0, 2: 0.96, 3: 0.92, 10: 0.88}

_NM_TO_KM = 1.852       # nautical miles → km
_KTS_TO_MS = 0.514444   # knots → m/s


# ---------------------------------------------------------------------------
# Physics-based radius estimation
# ---------------------------------------------------------------------------

def estimate_rmw_climada(pres_hpa):
    """Estimate RMW (km) from central pressure using CLIMADA piecewise model.

    Borrowed from climada_python/climada/hazard/tc_tracks.py.

    Parameters
    ----------
    pres_hpa : array-like
        Central pressure in hPa. NaN-safe.
    """
    pres_l = [872, 940, 980, 1021]
    rmw_l = [14.907318, 15.726927, 25.742142, 56.856522]
    ermw = pres_hpa * 0  # preserve NaN structure
    for i, p_i in enumerate(pres_l):
        s0 = 1.0 / (p_i - pres_l[i - 1]) if i > 0 else 0.0
        s1 = 1.0 / (pres_l[i + 1] - p_i) if i + 1 < len(pres_l) else 0.0
        ermw += rmw_l[i] * np.fmax(
            0,
            1 - s0 * np.fmax(0, p_i - pres_hpa) - s1 * np.fmax(0, pres_hpa - p_i),
        )
    return ermw


def estimate_rmw_licrice(v_circular_ms, lat_deg):
    """Estimate RMW (km) from circular wind speed and latitude (LICRICE formula).

    Parameters
    ----------
    v_circular_ms : array-like
        Maximum circular wind speed in m/s.
    lat_deg : array-like
        Latitude in degrees.
    """
    return 63.273 - 0.8683 * v_circular_ms + 1.07 * np.abs(lat_deg)


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def format_ibtracs(raw_ds, params):
    """Convert a raw IBTrACS xarray.Dataset to the licrice intermediate format.

    Parameters
    ----------
    raw_ds : xarray.Dataset
        Loaded from ``xr.open_dataset('IBTrACS.ALL.v04r01.nc')``.
        Expected dimensions: storm, date_time.  Expected coordinates: time (datetime),
        lat, lon.
    params : dict
        LICRICE params dict (loaded from params/licrice/v1.1.json).
        Uses ``params["ibtracs"]["missing_roci_fill_km"]`` (default 400).

    Returns
    -------
    xarray.Dataset
        Dimensions: storm (int index), time (int index 0..date_time-1).
        Variables: latstore, longstore, v_circular, v_total, datetime,
                   pstore (hPa), rmstore_estimated (km), storm_radius_estimated (km),
                   sid (str), season (uint16).
        Units attributes are set on pstore, rmstore_estimated,
        storm_radius_estimated so that _clean_ibtracs() assertions pass.
    """
    missing_roci_fill = params.get("ibtracs", {}).get("missing_roci_fill_km", 400.0)

    # ------------------------------------------------------------------ #
    # 1. Determine USA agency aliases for wind averaging-period lookup
    # ------------------------------------------------------------------ #
    usa_agency_raw = raw_ds.usa_agency.values  # (storm, date_time), bytes |S32
    usa_agencies = set(usa_agency_raw.astype("U32").ravel()) - {""}
    usa_agencies.add("atcf")  # explicit extra alias mentioned in coastal-core

    # ------------------------------------------------------------------ #
    # 2. Build per-observation wind conversion factor (X-min → 1-min)
    # ------------------------------------------------------------------ #
    wmo_agency_raw = raw_ds.wmo_agency.values  # (storm, date_time), bytes |S19
    wmo_agency_str = wmo_agency_raw.astype("U19")  # decode to unicode in-place

    # Default: assume 10-minute averaging (conservative; applies to most non-US basins)
    factors = np.full(wmo_agency_str.shape, 1.0 / _AVG_PERIOD_DIVISOR[10], dtype=np.float32)

    # Apply known agency averaging periods
    for agency, period in AGENCY_AVERAGING_PERIOD.items():
        mask = wmo_agency_str == agency
        factors[mask] = 1.0 / _AVG_PERIOD_DIVISOR[period]

    # Override with 1-minute for any USA alias (e.g., 'hurdat_atl', 'jtwc', 'atcf')
    for ua in usa_agencies:
        mask = wmo_agency_str == ua
        factors[mask] = 1.0  # already 1-minute

    # Unknown / empty agency → keep 10-min default (already set above)

    # ------------------------------------------------------------------ #
    # 3. Pool wind speed (kts, 1-min sustained)
    # ------------------------------------------------------------------ #
    wmo_wind = raw_ds.wmo_wind.values * factors  # kts, 1-min
    usa_wind = raw_ds.usa_wind.values            # kts, 1-min (always)
    wind_1min_kts = np.where(np.isnan(wmo_wind), usa_wind, wmo_wind)
    v_total_ms = (wind_1min_kts * _KTS_TO_MS).astype(np.float32)
    v_total_ms[v_total_ms < 0] = np.nan

    # ------------------------------------------------------------------ #
    # 4. Pool pressure (hPa)
    # ------------------------------------------------------------------ #
    pres_hpa = raw_ds.wmo_pres.values.copy()
    usa_pres = raw_ds.usa_pres.values
    pres_hpa = np.where(np.isnan(pres_hpa), usa_pres, pres_hpa).astype(np.float32)

    # ------------------------------------------------------------------ #
    # 5. Pool RMW (nautical miles → km)
    # ------------------------------------------------------------------ #
    rmw_nm = raw_ds.usa_rmw.values.copy()
    for vname in ["reunion_rmw", "bom_rmw"]:
        if vname in raw_ds.data_vars:
            rmw_nm = np.where(np.isnan(rmw_nm), raw_ds[vname].values, rmw_nm)
    rmw_km = (rmw_nm * _NM_TO_KM).astype(np.float32)
    rmw_km[rmw_km < 0] = np.nan

    # ------------------------------------------------------------------ #
    # 6. Pool ROCI (nautical miles → km)
    # ------------------------------------------------------------------ #
    roci_nm = raw_ds.usa_roci.values.copy()
    for vname in ["bom_roci", "td9635_roci"]:
        if vname in raw_ds.data_vars:
            roci_nm = np.where(np.isnan(roci_nm), raw_ds[vname].values, roci_nm)
    roci_km = (roci_nm * _NM_TO_KM).astype(np.float32)
    roci_km[roci_km < 0] = np.nan

    # ------------------------------------------------------------------ #
    # 7. Standardize coordinates and create dataset
    # ------------------------------------------------------------------ #
    n_storms, n_times = raw_ds.sizes["storm"], raw_ds.sizes["date_time"]

    lat = raw_ds.lat.values.astype(np.float32)   # (storm, date_time)
    lon = raw_ds.lon.values.astype(np.float32)   # (storm, date_time)
    lon = ((lon + 180) % 360) - 180              # normalize to [-180, 180]

    # The IBTrACS "time" coordinate holds datetime values; rename date_time → time
    datetime_vals = raw_ds.time.values  # (storm, date_time), datetime64[ns]
    # Round to second precision (coastal-core-main: newds["time"] = newds.time.dt.round("s"))
    # This fixes sub-second offsets in IBTrACS timestamps that cause interp boundary issues.
    datetime_ns = datetime_vals.astype("datetime64[ns]")
    nat_mask = np.isnat(datetime_ns)
    sec = np.where(
        nat_mask,
        np.iinfo(np.int64).min,
        (datetime_ns.astype(np.int64) + 500_000_000) // 1_000_000_000,
    )
    datetime_ns = sec.astype("datetime64[s]").astype("datetime64[ns]")

    sids_str = np.array(
        [s.decode("utf-8").strip() if isinstance(s, bytes) else str(s).strip()
         for s in raw_ds.sid.values],
        dtype=object,
    )
    seasons = raw_ds.season.values.astype(np.uint16)

    ds = xr.Dataset(
        {
            "latstore": (["storm", "time"], lat),
            "longstore": (["storm", "time"], lon),
            "v_total": (["storm", "time"], v_total_ms, {"units": "m/s"}),
            "pstore": (["storm", "time"], pres_hpa, {"units": "hPa"}),
            "rmstore": (["storm", "time"], rmw_km, {"units": "km"}),
            "storm_radius": (["storm", "time"], roci_km, {"units": "km"}),
            "datetime": (["storm", "time"], datetime_ns),
            "sid": (["storm"], sids_str),
            "season": (["storm"], seasons),
        },
        coords={
            "storm": np.arange(n_storms),
            "time": np.arange(n_times),
        },
    )

    # ------------------------------------------------------------------ #
    # 8. Drop storms with < 2 valid wind observations
    # ------------------------------------------------------------------ #
    n_valid = (ds.v_total.notnull() & ds.latstore.notnull()).sum("time")
    ds = ds.isel(storm=(n_valid >= 2).values)

    # ------------------------------------------------------------------ #
    # 9. Drop time slots with NaT datetime, then interpolate interior NaN
    #    values using datetime as the coordinate (matches coastal-core-main
    #    format_clean: dropna on datetime + interpolate_nans(use_coordinate="datetime"))
    # ------------------------------------------------------------------ #
    _vars_to_interp = ["latstore", "longstore", "v_total", "pstore", "rmstore", "storm_radius"]
    ds = ds.set_coords("datetime")
    storm_list = []
    for i in range(ds.sizes["storm"]):
        storm_i = ds.isel(storm=i).dropna(dim="time", subset=["datetime"])
        for var in _vars_to_interp:
            attrs = storm_i[var].attrs.copy()
            storm_i[var] = (
                storm_i[var]
                .interpolate_na(dim="time", use_coordinate="datetime")
                .ffill("time")   # fill leading NaN (not handled by interpolate_na)
                .bfill("time")   # fill trailing NaN
            )
            storm_i[var].attrs.update(attrs)
        storm_list.append(storm_i.reset_coords("datetime"))
    ds = xr.concat(storm_list, dim="storm", join="outer")

    # ------------------------------------------------------------------ #
    # 10. Calculate translational and circular velocity
    # ------------------------------------------------------------------ #
    ds = vel.calculate_v_trans_x_y(ds, lat_var="latstore", lon_var="longstore")
    ds = vel.calculate_v_circular(ds, lat_var="latstore", lon_var="longstore")

    # Drop storms where circular wind never exceeds 0
    ds = ds.isel(storm=(ds.v_circular.max("time") > 0).values)

    # ------------------------------------------------------------------ #
    # 11. Estimate missing radii (physics-based; no random forest)
    # ------------------------------------------------------------------ #
    # RMW: use observed, fill with pressure-based then lat/wind-based estimate
    rmw_est_climada = estimate_rmw_climada(ds.pstore)  # km, xr.DataArray
    rmw_est_licrice = estimate_rmw_licrice(ds.v_circular, ds.latstore)  # km
    rmw_est = rmw_est_climada.fillna(rmw_est_licrice).clip(min=1.0)
    # Fill rmw: observed → climada estimate → licrice estimate.
    # Then ffill/bfill within the valid obs window so that leading/trailing
    # NaN (e.g. first obs missing pressure AND wind) get filled from neighbours.
    # Finally mask trailing NaT slots so they are not confused with valid obs.
    ds["rmstore_estimated"] = (
        ds.rmstore.fillna(rmw_est)
        .where(ds.datetime.notnull())
        .ffill("time")
        .bfill("time")
        .clip(min=1.0)
        .astype(np.float32)
    )

    # ROCI: use observed, fill remainder with the fixed fallback value,
    # then mask trailing NaT slots (same rationale as rmstore_estimated).
    ds["storm_radius_estimated"] = (
        ds.storm_radius.fillna(missing_roci_fill)
        .where(ds.datetime.notnull())
        .astype(np.float32)
    )

    # Ensure ROCI >= RMW everywhere
    ds["storm_radius_estimated"] = xr.concat(
        [ds.storm_radius_estimated, ds.rmstore_estimated], dim="_var"
    ).max("_var")

    # Clip negative estimated values (shouldn't occur, but safety check)
    ds["rmstore_estimated"] = ds.rmstore_estimated.clip(min=1.0)
    ds["storm_radius_estimated"] = ds.storm_radius_estimated.clip(min=1.0)

    # Set units attributes required by _clean_ibtracs assertions
    ds["rmstore_estimated"].attrs["units"] = "km"
    ds["storm_radius_estimated"].attrs["units"] = "km"
    ds["pstore"].attrs["units"] = "hPa"

    return ds


def preprocess_ibtracs(nc_path, zarr_outpath, params, overwrite=False):
    """Preprocess a raw IBTrACS netCDF file and save as zarr.

    The output zarr is in the intermediate format that licrice's load_tracks /
    _clean_ibtracs expects.  Run this once before running run_licrice_on_trackset.

    Parameters
    ----------
    nc_path : str or Path
        Path to the raw IBTrACS netCDF file.
    zarr_outpath : str or Path
        Destination zarr directory path.
    params : dict
        LICRICE params (from params/licrice/v1.1.json).
    overwrite : bool, optional
        If True, overwrite an existing zarr. Default False.
    """
    import pathlib

    zarr_outpath = pathlib.Path(zarr_outpath)
    if zarr_outpath.exists() and not overwrite:
        print(f"Preprocessed zarr already exists at {zarr_outpath}. Skipping.")
        return

    print(f"Loading {nc_path} ...")
    raw_ds = xr.open_dataset(str(nc_path))

    print("Preprocessing IBTrACS tracks ...")
    ds = format_ibtracs(raw_ds, params)

    # Chunk for efficient zarr storage (storm dimension chunked, time kept whole)
    n_storms = ds.sizes["storm"]
    chunk_size = min(50, n_storms)
    ds = ds.chunk({"storm": chunk_size, "time": ds.sizes["time"]})

    print(f"Saving preprocessed tracks to {zarr_outpath} ...")
    ds.to_zarr(str(zarr_outpath), mode="w", consolidated=True)
    print("Done.")

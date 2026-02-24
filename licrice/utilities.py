"""Utility functions for LICRICE.

Extracted from pyTC.utilities with only the functions needed by the LICRICE wind model.
The pint/clawpack dependencies have been replaced with a simple lookup table.
"""

import numpy as np
import xarray as xr


# Simple unit conversion lookup (replaces pint/clawpack dependency)
_UNIT_CONVERSIONS = {
    ("kts", "m/s"): 0.514444,
    ("m/s", "kts"): 1 / 0.514444,
    ("km", "m"): 1000.0,
    ("m", "km"): 0.001,
    ("hpa", "pa"): 100.0,
    ("pa", "hpa"): 0.01,
    ("mb", "pa"): 100.0,
    ("mbar", "pa"): 100.0,
}


def convert_units(data, from_to):
    """Converts data from one unit to another.

    Parameters
    ----------
    data : scalar or array
        Original value to convert from
    from_to : tuple
        2-tuple of strings, as (Units to convert from, Units to convert to)

    Returns
    -------
    converted : scalar or array

    """
    key = (from_to[0].lower(), from_to[1].lower())
    if key not in _UNIT_CONVERSIONS:
        raise ValueError(f"Unsupported unit conversion: {from_to[0]} -> {from_to[1]}")
    return data * _UNIT_CONVERSIONS[key]


def geoclaw_convert(data, old_units, new_units):
    """Simple unit conversion replacing clawpack.geoclaw.units.convert.

    Supports km<->m and hPa/mb<->Pa conversions needed by LICRICE.
    """
    key = (old_units.lower(), new_units.lower())
    if key not in _UNIT_CONVERSIONS:
        raise ValueError(f"Unsupported unit conversion: {old_units} -> {new_units}")
    return data * _UNIT_CONVERSIONS[key]


def bin_data(da, res_spatial):
    """Bin data.

    Parameters
    ----------
    da : :py:class:`xarray.DataArray`
        Data to get binned
    res_spatial : float
        bin size

    Returns
    -------
    `da` binned in bins of size `res_spatial`

    """
    return (np.floor(da / res_spatial) * res_spatial) + res_spatial / 2


def check_finished_zarr_workflow(
    finalstore=None,
    tmpstore=None,
    varname=None,
    final_selector={},
    mask=None,
    check_final=True,
    check_temp=True,
    how="all",
):
    def _check_notnull(da, how):
        out = da.notnull()
        if how == "all":
            return out.all().item()
        elif how == "any":
            return out.any().item()
        raise ValueError(how)

    finished = False
    temp = False
    if check_final:
        finished = xr.open_zarr(finalstore, chunks=None)[varname].sel(
            final_selector,
            drop=True,
        )
        if mask is not None:
            finished = finished.where(mask, 1)
        finished = _check_notnull(finished, how)

    if finished:
        return True
    if check_temp:
        if tmpstore.fs.isdir(tmpstore.root):
            try:
                temp = xr.open_zarr(tmpstore, chunks=None)
                if mask is not None:
                    temp = temp.where(mask, 1)
                if (
                    varname in temp.data_vars
                    and "year" in temp.dims
                    and len(temp.year) > 0
                ):
                    finished = _check_notnull(temp[varname], how)
            except Exception:
                ...
    return finished

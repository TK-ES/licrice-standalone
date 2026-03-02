import pickle  # NEW
from pathlib import Path  # NEW

import numpy as np  # NEW
import pandas as pd  # NEW
import xarray as xr  # NEW
from joblib import dump, load  # NEW
from sklearn.ensemble import RandomForestRegressor  # NEW
from sklearn.preprocessing import OneHotEncoder  # NEW

from licrice.utilities import smooth_fill  # NEW


def estimate_rmw(pres, v_circular, lat, time_dim="time"):  # NEW
    """Estimate RMW using a central pressure-based regression first and, if P is  # NEW
    unavailable, then using the lat+Vmax regression from LICRICE.  # NEW

    Parameters  # NEW
    ----------  # NEW
    pres : float or :class:`xarray.DataArray`  # NEW
        Maximum azimuthal wind, indexed by ``storm`` and ``time``  # NEW
    v_circular : float or :class:`xarray.DataArray`  # NEW
        Central Pressure estimates, indexed by ``storm`` and ``time``  # NEW
    lat : float or :class:`xarray.DataArray`  # NEW
        Latitude, indexed by ``storm`` and ``time``  # NEW

    Returns  # NEW
    -------  # NEW
    float or :class:`numpy.ndarray` or :class:`xarray.DataArray`  # NEW
        Estimated Radius of Maximum Wind, in meters, indexed by ``storm`` and ``time``  # NEW

    """  # NEW
    pres_ermw = estimate_rmw_climada(pres).interpolate_na(dim="time")  # NEW
    lat_ermw = estimate_rmw_licrice(v_circular, lat).interpolate_na(dim="time")  # NEW
    return smooth_fill(pres_ermw, lat_ermw)  # NEW


def estimate_rmw_climada(pres):  # NEW
    """Uses central pressure to estimate RMW (in km). Borrowed from  # NEW
    https://github.com/CLIMADA-project/climada_python/blob/main/climada/hazard/tc_tracks.py#L1067.  # NEW

    Parameters  # NEW
    ----------  # NEW
    pres : float or :class:`numpy.ndarray` or :class:`xarray.DataArray`  # NEW
        Central Pressure estimates, in hPa.  # NEW

    Returns  # NEW
    -------  # NEW
    float or :class:`numpy.ndarray` or :class:`xarray.DataArray`  # NEW
        Estimated Radius of Maximum Wind, in km  # NEW

    """  # NEW
    pres_l = [872, 940, 980, 1021]  # NEW
    rmw_l = [14.907318, 15.726927, 25.742142, 56.856522]  # NEW

    ermw = pres * 0  # maintain nans  # NEW
    for i, pres_l_i in enumerate(pres_l):  # NEW
        slope_0 = 1.0 / (pres_l_i - pres_l[i - 1]) if i > 0 else 0  # NEW
        slope_1 = 1.0 / (pres_l[i + 1] - pres_l_i) if i + 1 < len(pres_l) else 0  # NEW
        ermw += rmw_l[i] * np.fmax(  # NEW
            0,  # NEW
            (  # NEW
                1  # NEW
                - slope_0 * np.fmax(0, pres_l_i - pres)  # NEW
                - slope_1 * np.fmax(0, pres - pres_l_i)  # NEW
            ),  # NEW
        )  # NEW
    return ermw  # NEW


def estimate_rmw_licrice(v_circular, lat):  # NEW
    """Combines latitude, translational velocity, and maximum wind speed to estimate the  # NEW
    radius of maximum wind.  # NEW

    Eqn from p.3 of LICRICE docs  # NEW

    Parameters  # NEW
    ----------  # NEW
    v_circular : float or :class:`numpy.ndarray` or :class:`xarray.DataArray`  # NEW
        Maximum azimuthal wind, in m/s  # NEW
    lat : float or :class:`numpy.ndarray` or :class:`xarray.DataArray`  # NEW
        Latitude, in degrees  # NEW

    Returns  # NEW
    -------  # NEW
    float or :class:`numpy.ndarray` or :class:`xarray.DataArray`  # NEW
        Estimated Radius of Maximum Wind, in km  # NEW

    """  # NEW
    ermw = 63.273 - 0.8683 * v_circular + 1.07 * np.abs(lat)  # NEW

    return ermw  # NEW


def create_radius_reg_dataset(  # NEW
    ds,  # NEW
    rmw_var="rmstore",  # NEW
    radius_var="storm_radius",  # NEW
    reg_cols=None,  # NEW
):  # NEW
    rat_estimator = (  # NEW
        ds[  # NEW
            [  # NEW
                rmw_var,  # NEW
                radius_var,  # NEW
                "latstore",  # NEW
                "basin",  # NEW
                "subbasin",  # NEW
                "v_circular",  # NEW
                "v_total",  # NEW
                "dist2land",  # NEW
                "nature",  # NEW
                "v_trans_x",  # NEW
                "v_trans_y",  # NEW
                "longstore",  # NEW
            ]  # NEW
        ]  # NEW
        .to_dataframe()  # NEW
        .rename(columns={rmw_var: "rmstore", radius_var: "storm_radius"})  # NEW
    )  # NEW

    rat_estimator = rat_estimator.dropna(  # NEW
        how="any",  # NEW
        subset=[  # NEW
            v for v in rat_estimator.columns if v not in ["rmstore", "storm_radius"]  # NEW
        ],  # NEW
    )  # NEW
    rat_estimator["hemisphere"] = (rat_estimator.latstore > 0).astype(np.uint8)  # NEW
    rat_estimator["abslat"] = np.abs(rat_estimator.latstore)  # NEW
    rat_estimator["subbasin"] = rat_estimator.basin + rat_estimator.subbasin  # NEW
    rat_estimator = rat_estimator.drop(columns=["latstore", "basin"])  # NEW

    categoricals = ["subbasin", "nature"]  # NEW
    enc = OneHotEncoder(sparse_output=False, drop="first")  # NEW
    enc.fit(rat_estimator[categoricals])  # NEW
    basins = pd.DataFrame(  # NEW
        enc.transform(rat_estimator[categoricals]),  # NEW
        columns=enc.get_feature_names_out(categoricals),  # NEW
        index=rat_estimator.index,  # NEW
    )  # NEW
    out = pd.concat((rat_estimator.drop(columns=categoricals), basins), axis=1)  # NEW

    if reg_cols is not None:  # NEW
        addl_cols = pd.DataFrame(  # NEW
            0,  # NEW
            index=out.index,  # NEW
            columns=[c for c in reg_cols if c not in out.columns],  # NEW
        )  # NEW
        out = pd.concat((out, addl_cols), axis=1).reindex(columns=reg_cols)  # NEW
    return out  # NEW


def estimate_radii(ds, rmw_to_rad, rad_to_rmw, rmw, reg_cols=None):  # NEW
    # fix contexts in which outer radius is smaller than RMW in ibtracs data  # NEW
    ds["storm_radius_estimated"] = (  # NEW
        xr.concat((ds.storm_radius, ds.rmstore), dim="var")  # NEW
        .max(dim="var")  # NEW
        .where(ds.storm_radius.notnull())  # NEW
    )  # NEW

    # use ratios of storm radius and RMW observed in storm to extrapolate when only one  # NEW
    # of the two variables is missing at start or end of storm  # NEW
    ds["rmstore_estimated"] = smooth_fill(  # NEW
        ds.rmstore,  # NEW
        ds.storm_radius,  # NEW
        fill_all_null=False,  # NEW
    )  # NEW
    ds["storm_radius_estimated"] = smooth_fill(  # NEW
        ds.storm_radius_estimated,  # NEW
        ds.rmstore,  # NEW
        fill_all_null=False,  # NEW
    )  # NEW

    # Fill in RMW using relationship to other vars INCLUDING ROCI  # NEW
    X = create_radius_reg_dataset(  # NEW
        ds,  # NEW
        rmw_var="rmstore_estimated",  # NEW
        radius_var="storm_radius_estimated",  # NEW
        reg_cols=reg_cols,  # NEW
    )  # NEW
    X = X[X.storm_radius.notnull()]  # NEW
    if len(X):  # NEW
        rmw_estimated = (  # NEW
            pd.Series(rad_to_rmw.predict(X.drop(columns="rmstore")), index=X.index)  # NEW
            .to_xarray()  # NEW
            .reindex(time=ds.time)  # NEW
        )  # NEW
        rmw_estimated = xr.concat(  # NEW
            (rmw_estimated, ds.storm_radius_estimated),  # NEW
            dim="var",  # NEW
        ).min(dim="var")  # NEW
        ds["rmstore_estimated"] = smooth_fill(ds.rmstore_estimated, rmw_estimated)  # NEW

    # Fill in RMW using relationship to other vars WHEN ROCI NOT AVAILABLE  # NEW
    X = create_radius_reg_dataset(  # NEW
        ds,  # NEW
        rmw_var="rmstore_estimated",  # NEW
        radius_var="storm_radius_estimated",  # NEW
        reg_cols=reg_cols,  # NEW
    )  # NEW
    rmw_estimated = (  # NEW
        pd.Series(  # NEW
            rmw.predict(X.drop(columns=["rmstore", "storm_radius"])),  # NEW
            index=X.index,  # NEW
        )  # NEW
        .to_xarray()  # NEW
        .reindex(time=ds.time)  # NEW
    )  # NEW
    rmw_estimated = xr.concat(  # NEW
        (rmw_estimated, ds.storm_radius_estimated),  # NEW
        dim="var",  # NEW
    ).min(dim="var")  # NEW
    ds["rmstore_estimated"] = smooth_fill(ds.rmstore_estimated, rmw_estimated)  # NEW

    # Fill in ROCI using relationship to other vars INCLUDING RMW  # NEW
    X = create_radius_reg_dataset(  # NEW
        ds,  # NEW
        rmw_var="rmstore_estimated",  # NEW
        radius_var="storm_radius_estimated",  # NEW
        reg_cols=reg_cols,  # NEW
    )  # NEW
    assert X.rmstore.notnull().all()  # NEW
    storm_radius_estimated = (  # NEW
        pd.Series(rmw_to_rad.predict(X.drop(columns="storm_radius")), index=X.index)  # NEW
        .to_xarray()  # NEW
        .reindex(time=ds.time)  # NEW
    )  # NEW
    storm_radius_estimated = xr.concat(  # NEW
        (storm_radius_estimated, ds.rmstore_estimated),  # NEW
        dim="var",  # NEW
    ).max(dim="var")  # NEW
    ds["storm_radius_estimated"] = smooth_fill(  # NEW
        ds.storm_radius_estimated,  # NEW
        storm_radius_estimated,  # NEW
    )  # NEW

    ds["rmstore_estimated"].attrs.update(  # NEW
        {  # NEW
            "long_name": "radius of maximum wind estimated",  # NEW
            "units": "km",  # NEW
            "method": (  # NEW
                "When obs are available, use obs. When unavailable, check central "  # NEW
                "pressure. If available, use pcen-->rmw relationship to estimate rmw. "  # NEW
                "If not, use lat+v_circ-->rmw relationship (LICRICE) to estimate rmw. "  # NEW
                "After estimation, bias correct estimates to first and last observed "  # NEW
                "values. Cap at observed ROCI."  # NEW
            ),  # NEW
        }  # NEW
    )  # NEW

    ds["storm_radius_estimated"].attrs.update(  # NEW
        {  # NEW
            "long_name": "radius of last closed isobar estimated",  # NEW
            "units": "km",  # NEW
            "method": (  # NEW
                "When obs are available, use obs. When unavailable for some time "  # NEW
                "points, extrapolate RMW->ROCI relationship from last observed ROCI. "  # NEW
                "When unavailable for any time points, leave as NaN. Clip so that it "  # NEW
                "is at least as big as rmstore_estimated."  # NEW
            ),  # NEW
        }  # NEW
    )  # NEW

    # ensure no negative estimated values and no missing values when lat/long is  # NEW
    # non-missing  # NEW
    test = ds[["storm_radius_estimated", "rmstore_estimated"]]  # NEW
    assert ((test > 0) | test.isnull()).to_array().all().item()  # NEW
    assert (  # NEW
        (test.to_array(dim="var").notnull().all(dim="var") | ds.latstore.isnull())  # NEW
        .all()  # NEW
        .item()  # NEW
    )  # NEW

    return ds  # NEW


def get_radius_ratio_models(ds, model_dir=None):  # NEW
    """Train RF models for RMW and ROCI estimation and optionally save to disk.  # NEW

    Adapted from coastal-core's get_radius_ratio_models(ps, save). Takes ``ds``  # NEW
    directly instead of loading via ibtracs.load_processed_ibtracs.  # NEW

    Parameters  # NEW
    ----------  # NEW
    ds : xarray.Dataset  # NEW
        Cleaned track dataset (output of format_clean).  # NEW
    model_dir : Path or None  # NEW
        Directory to save .pkl model files. If None, models are not saved.  # NEW

    Returns  # NEW
    -------  # NEW
    tuple  # NEW
        (rmw_to_rad, rad_to_rmw, rad, rmw, cols)  # NEW

    """  # NEW
    df = create_radius_reg_dataset(ds)  # NEW
    cols = list(df.columns)  # NEW

    this_df = df.dropna(subset=["storm_radius", "rmstore"], how="any")  # NEW
    rmw_to_rad = RandomForestRegressor(random_state=0, oob_score=True)  # NEW
    rmw_to_rad.fit(this_df.drop(columns="storm_radius"), this_df.storm_radius)  # NEW
    rad_to_rmw = RandomForestRegressor(random_state=0, oob_score=True)  # NEW
    rad_to_rmw.fit(this_df.drop(columns="rmstore"), this_df.rmstore)  # NEW

    this_df = df.dropna(subset=["storm_radius"])  # NEW
    rad = RandomForestRegressor(random_state=0, oob_score=True)  # NEW
    rad.fit(this_df.drop(columns=["storm_radius", "rmstore"]), this_df.storm_radius)  # NEW

    this_df = df.dropna(subset=["rmstore"])  # NEW
    rmw = RandomForestRegressor(random_state=0, oob_score=True)  # NEW
    rmw.fit(this_df.drop(columns=["storm_radius", "rmstore"]), this_df.rmstore)  # NEW

    if model_dir is not None:  # NEW
        model_dir = Path(model_dir)  # NEW
        model_dir.mkdir(parents=True, exist_ok=True)  # NEW
        for obj, name in (  # NEW
            (rmw_to_rad, "rmw_to_rad"),  # NEW
            (rad_to_rmw, "rad_to_rmw"),  # NEW
            (rad, "rad"),  # NEW
            (rmw, "rmw"),  # NEW
        ):  # NEW
            with (model_dir / f"{name}.pkl").open("wb") as f:  # NEW
                dump(obj, f)  # NEW
        with (model_dir / "cols.pkl").open("wb") as f:  # NEW
            pickle.dump(cols, f)  # NEW

    return rmw_to_rad, rad_to_rmw, rad, rmw, cols  # NEW


def load_radius_models(model_dir):  # NEW
    """Load RF radius models from disk.  # NEW

    Parameters  # NEW
    ----------  # NEW
    model_dir : Path or str  # NEW
        Directory containing rmw_to_rad.pkl, rad_to_rmw.pkl, rmw.pkl, cols.pkl.  # NEW

    Returns  # NEW
    -------  # NEW
    tuple  # NEW
        (rmw_to_rad, rad_to_rmw, rmw, cols)  # NEW

    """  # NEW
    model_dir = Path(model_dir)  # NEW
    rmw_to_rad = load(model_dir / "rmw_to_rad.pkl")  # NEW
    rad_to_rmw = load(model_dir / "rad_to_rmw.pkl")  # NEW
    rmw = load(model_dir / "rmw.pkl")  # NEW
    with (model_dir / "cols.pkl").open("rb") as f:  # NEW
        cols = pickle.load(f)  # NEW
    return rmw_to_rad, rad_to_rmw, rmw, cols  # NEW

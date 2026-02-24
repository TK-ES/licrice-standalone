"""Example script demonstrating how to use the standalone LICRICE wind model.

LICRICE calculates two wind hazard metrics over a spatial grid:
  - MAXS: Maximum sustained wind speed (m/s)
  - PDDI: Power Dissipation Density Index (m^3/s^2)

Usage:
  1. Provide a zarr trackset (IBTrACS or Emanuel synthetic)
  2. Define a bounding box (xlim, ylim)
  3. Load LICRICE parameters
  4. Run the model

This example creates synthetic test data and runs LICRICE on it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from licrice.licrice.run import run_licrice_on_track
from licrice.licrice.preprocess import timesteps_to_pixelsteps
from licrice.licrice.utils import get_output_grid
from licrice.tracks.velocity import calculate_v_trans_x_y, calculate_v_total


def create_synthetic_storm():
    """Create a simple synthetic storm dataset for testing."""
    n_times = 20

    # Storm moves from (280, 25) to (285, 30) - roughly Caribbean to US East Coast
    lons = np.linspace(-80, -75, n_times)
    lats = np.linspace(25, 30, n_times)

    # Simple pressure and wind profiles
    v_circular = np.concatenate([
        np.linspace(15, 50, n_times // 2),
        np.linspace(50, 20, n_times // 2),
    ])

    # RMW and storm radius in meters
    rmw = np.full(n_times, 40e3)  # 40 km
    radius = np.full(n_times, 300e3)  # 300 km

    # Pressure in Pa
    pres = np.full(n_times, 98000.0)

    # Timestamps every 3 hours
    start = np.datetime64("2020-08-15T00:00:00")
    datetimes = [start + np.timedelta64(3 * i, "h") for i in range(n_times)]

    ds = xr.Dataset(
        {
            "storm_lat": (("storm", "time"), [lats]),
            "storm_lon": (("storm", "time"), [lons]),
            "v_circular": (("storm", "time"), [v_circular]),
            "rmw": (("storm", "time"), [rmw]),
            "radius": (("storm", "time"), [radius]),
            "pres": (("storm", "time"), [pres]),
            "datetime": (("storm", "time"), [datetimes]),
        },
        coords={
            "storm": ["test_storm_001"],
            "time": np.arange(n_times),
        },
    )

    # Add unit attributes
    ds["rmw"].attrs["units"] = "m"
    ds["radius"].attrs["units"] = "m"
    ds["pres"].attrs["units"] = "Pa"
    ds["v_circular"].attrs["units"] = "m/s"

    return ds


def main():
    # Load LICRICE parameters
    params_path = Path(__file__).parent / "params" / "licrice" / "v1.1.json"
    with open(params_path) as f:
        params = json.load(f)

    print("LICRICE Standalone - Wind Field Model")
    print("=" * 50)
    print(f"Vortex function: {params['wind']['vortex_func']}")
    print(f"Grid resolution: {params['grid']['res_spatial_deg']} degrees")
    print()

    # Create synthetic test storm
    print("Creating synthetic storm data...")
    ds = create_synthetic_storm()
    print(f"  Storm: {ds.storm.values[0]}")
    print(f"  Time steps: {len(ds.time)}")
    print(f"  Max wind: {float(ds.v_circular.max()):.1f} m/s")
    print()

    # Define bounding box
    xlim = [-82, -73]
    ylim = [23, 32]

    # Convert timesteps to pixel steps
    print("Converting timesteps to pixel steps...")
    ds_pixel = timesteps_to_pixelsteps(ds, params)
    print(f"  Pixel steps: {len(ds_pixel.time)}")
    print()

    # Add translational velocities
    print("Calculating translational velocities...")
    ds_pixel = calculate_v_trans_x_y(
        ds_pixel,
        lat_var="storm_lat",
        lon_var="storm_lon",
    )
    ds_pixel = calculate_v_total(
        ds_pixel,
        "storm_lat",
        "storm_lon",
        baroclinic_effect=False,
    )
    print(f"  Max v_total: {float(ds_pixel.v_total.max()):.1f} m/s")
    print()

    # Run LICRICE on single storm
    print("Running LICRICE...")
    result = run_licrice_on_track(
        ds_pixel.isel(storm=0),
        xlim,
        ylim,
        params,
    )

    print()
    print("Results:")
    print(f"  MAXS grid shape: {result.maxs.shape}")
    print(f"  Max sustained wind: {float(result.maxs.max()):.2f} m/s")
    print(f"  PDDI grid shape: {result.pddi.shape}")
    print(f"  Max PDDI: {float(result.pddi.max()):.2e} m^3/s^2")
    print(f"  Grid cells with wind > 0: {int((result.maxs > 0).sum())}")
    print()
    print("LICRICE standalone is working correctly!")

    return result


if __name__ == "__main__":
    main()

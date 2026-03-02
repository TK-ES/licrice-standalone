#!/usr/bin/env python
"""Command-line script to run the LICRICE wind field model on IBTrACS data.

Usage examples
--------------
# Run two domains, auto-preprocessing the raw IBTrACS netCDF first:
python run_licrice.py \\
    --input /path/to/IBTrACS.ALL.v04r01.nc \\
    --domain south_atlantic western_pacific_south \\
    --outdir /path/to/output/

# Run all built-in domains:
python run_licrice.py \\
    --input /path/to/IBTrACS.ALL.v04r01.nc \\
    --domain all \\
    --outdir /path/to/output/

# Skip re-preprocessing if the zarr already exists:
python run_licrice.py \\
    --input /path/to/IBTrACS.ALL.v04r01.nc \\
    --domain south_atlantic \\
    --outdir /path/to/output/ \\
    --no-overwrite-preproc

# Use an already-preprocessed zarr directly:
python run_licrice.py \\
    --input /path/to/ibtracs_preprocessed.zarr \\
    --domain south_atlantic \\
    --outdir /path/to/output/
"""

import argparse
import json
import pathlib
import sys

# ---------------------------------------------------------------------------
# Built-in domain definitions
# These match the bounds used to produce the licrice_result_data zarr files.
# xlim = [lon_min, lon_max], ylim = [lat_min, lat_max]
# ---------------------------------------------------------------------------
DOMAINS = {
    "south_atlantic":          {"xlim": [-60, -38],  "ylim": [-35, -15]},
    "western_pacific_south":   {"xlim": [98,  174],  "ylim": [0,   25]},
    "north_atlantic":          {"xlim": [-100, -60], "ylim": [10,  50]},
    "eastern_pacific":         {"xlim": [-120, -80], "ylim": [10,  35]},
    "western_pacific_north":   {"xlim": [100,  180], "ylim": [5,   45]},
    "north_indian":            {"xlim": [50,   100], "ylim": [5,   30]},
    "south_indian_west":       {"xlim": [30,   90],  "ylim": [-35,  0]},
    "south_indian_east":       {"xlim": [90,   135], "ylim": [-35,  0]},
    "australia":               {"xlim": [110,  160], "ylim": [-35, -10]},
    "south_pacific":           {"xlim": [155,  210], "ylim": [-35,  -5]},
}


def load_params(params_path=None):
    if params_path is None:
        params_path = pathlib.Path(__file__).parent / "params" / "licrice" / "v1.1.json"
    with open(params_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run LICRICE wind field model on IBTrACS tracks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required="--list-domains" not in sys.argv,
        help=(
            "Path to input track file. Either a raw IBTrACS netCDF (.nc) or a "
            "pre-processed zarr directory (.zarr). If .nc is given, the file is "
            "preprocessed automatically before running LICRICE."
        ),
    )
    parser.add_argument(
        "--domain", "-d", nargs="+", required="--list-domains" not in sys.argv,
        metavar="DOMAIN",
        help=(
            "One or more domain names to process, or 'all' to run every built-in "
            f"domain. Built-in domains: {', '.join(DOMAINS)}."
        ),
    )
    parser.add_argument(
        "--outdir", "-o", required="--list-domains" not in sys.argv,
        help="Output directory. One zarr file per domain is written here.",
    )
    parser.add_argument(
        "--preproc-zarr", default=None,
        metavar="PATH",
        help=(
            "Where to save/load the preprocessed IBTrACS zarr when --input is a "
            ".nc file. Defaults to <outdir>/ibtracs_preprocessed.zarr."
        ),
    )
    parser.add_argument(
        "--no-overwrite-preproc", action="store_true",
        help="Skip preprocessing if the preprocessed zarr already exists.",
    )
    parser.add_argument(
        "--no-overwrite-output", action="store_true",
        help="Skip LICRICE run if the output zarr already exists.",
    )
    parser.add_argument(
        "--params", default=None,
        metavar="PATH",
        help="Path to LICRICE params JSON. Defaults to params/licrice/v1.1.json.",
    )
    parser.add_argument(
        "--storm-chunksize", type=int, default=25,
        help="Number of storms per processing chunk (default 25).",
    )
    parser.add_argument(
        "--list-domains", action="store_true",
        help="Print all built-in domain names and bounds, then exit.",
    )

    args = parser.parse_args()

    if args.list_domains:
        print("Built-in domains:")
        for name, bounds in DOMAINS.items():
            print(f"  {name:30s}  xlim={bounds['xlim']}  ylim={bounds['ylim']}")
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # Resolve domains
    # ------------------------------------------------------------------ #
    if len(args.domain) == 1 and args.domain[0].lower() == "all":
        selected_domains = list(DOMAINS.keys())
    else:
        selected_domains = args.domain
        unknown = [d for d in selected_domains if d not in DOMAINS]
        if unknown:
            parser.error(
                f"Unknown domain(s): {unknown}. "
                f"Use --list-domains to see available domains."
            )

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    params = load_params(args.params)

    # ------------------------------------------------------------------ #
    # Resolve track input path (preprocess .nc → zarr if needed)
    # ------------------------------------------------------------------ #
    input_path = pathlib.Path(args.input)

    if input_path.suffix == ".nc":
        preproc_zarr = (
            pathlib.Path(args.preproc_zarr)
            if args.preproc_zarr
            else outdir / "ibtracs_preprocessed.zarr"
        )
        overwrite_preproc = not args.no_overwrite_preproc

        from licrice.io.ibtracs import preprocess_ibtracs
        preprocess_ibtracs(  # NEW
            nc_path=input_path,  # NEW
            zarr_outpath=preproc_zarr,  # NEW
            overwrite=overwrite_preproc,  # NEW
        )  # NEW
        track_zarr = preproc_zarr
    elif input_path.suffix == ".zarr" or input_path.is_dir():
        track_zarr = input_path
    else:
        parser.error(f"--input must be a .nc file or a .zarr directory, got: {input_path}")

    # ------------------------------------------------------------------ #
    # Find valid tracks for each domain, then run LICRICE
    # ------------------------------------------------------------------ #
    from licrice.licrice.preprocess import find_valid_tracks
    from licrice.licrice.run import run_licrice_on_trackset

    print(f"\nScanning tracks for valid storms across {len(selected_domains)} domain(s)...")
    bboxes = {d: DOMAINS[d] for d in selected_domains}
    valid_by_domain = find_valid_tracks(str(track_zarr), params, bboxes)

    if not valid_by_domain:
        print("No storms found intersecting any of the requested domains. Exiting.")
        sys.exit(0)

    for domain in selected_domains:
        if domain not in valid_by_domain:
            print(f"\n[{domain}] No storms intersect this domain — skipping.")
            continue

        info = valid_by_domain[domain]
        n_storms = len(info["valid_tracks"])
        print(f"\n[{domain}] {n_storms} storm(s) found.")

        outpath = outdir / f"hazard_wind_licrice_hist_{domain}.zarr"
        tmppath = outdir / f"_tmp_{domain}.zarr"
        checkfile = outdir / f"_check_{domain}.txt"

        if args.no_overwrite_output and outpath.exists():
            print(f"  Output already exists at {outpath} — skipping.")
            continue

        attr_dict = {
            "licrice_domain": domain,
            "method": (
                f"`licrice` for the {domain} domain "
                f"(xlim:{DOMAINS[domain]['xlim']}, ylim:{DOMAINS[domain]['ylim']})"
            ),
            "notes": "Historical storm track wind field summary statistics",
        }

        print(f"  Running LICRICE → {outpath}")
        result = run_licrice_on_trackset(
            ds_path=track_zarr,
            valid_storms=info["valid_tracks"],
            start_dates=info["start_dates"],
            params=params,
            xlim=DOMAINS[domain]["xlim"],
            ylim=DOMAINS[domain]["ylim"],
            outpath=outpath,
            tmppath=tmppath,
            checkfile_path=checkfile,
            attr_dict=attr_dict,
            storm_chunksize=args.storm_chunksize,
            trackset_type="ibtracs",
            overwrite=not args.no_overwrite_output,
        )

        if result == 0:
            print(f"  No storms produced non-zero wind in this domain.")
        else:
            print(f"  Done. {result} storm(s) written to {outpath}")

    print("\nAll domains complete.")


if __name__ == "__main__":
    main()

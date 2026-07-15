#!/usr/bin/env python3
"""
Download benchmark datasets for table-synthesizers to a local folder.

Two groups:

1. SYNTHONY_* dicts (11 datasets): the exact datasets used by the SYNTHONY
   benchmark suite (see ~/Projects/Synthony/dataset/input_data). Each was
   matched by inspecting row/column counts and column names, not just
   dataset name, and downloads from its canonical source:
     - 8 via the UCI Machine Learning Repository's official `ucimlrepo`
       client (no API key needed).
     - 1 (wilt) via UCI's static per-dataset zip, since it predates
       ucimlrepo's API and isn't importable through the client.
     - 2 (Titanic, insurance) originated on Kaggle and are pulled from the
       same GitHub raw mirrors this repo's own
       tests/integration/setup_sandbox_datasets.py already trusts, so no
       Kaggle account or API token is needed.
   Row counts match the downloaded source exactly except IndianLiverPatient
   (SYNTHONY: 579 vs UCI raw: 583) -- that's SYNTHONY's own light cleaning
   (dropped rows with missing values), not a different dataset.

2. EXTRA_* dicts (8 datasets): additional datasets recommended to extend
   the SYNTHONY set, covering gaps it doesn't have -- scale (Covertype:
   581K rows), genuine high-cardinality categoricals (Diabetes130US: 700+
   distinct ICD9 codes), the standard differential-privacy synthetic-data
   benchmark (Adult, used in the AIM paper itself), organic/structured
   missingness (CommunitiesCrime, HousePrices), extreme class imbalance
   (CreditCardFraud: ~577:1), and genuine nested-JSON / long free-text
   columns (TMDBMovies, WineReviews) -- none of which any SYNTHONY dataset
   has, but which this repo's own test_edge_cases.py::TestNestedJSON is
   built to exercise.
     - 5 via ucimlrepo (Adult, Covertype, Diabetes130US, BankMarketing,
       CommunitiesCrime).
     - 3 via `kagglehub` (TMDBMovies, WineReviews, CreditCardFraud) --
       verified to download anonymously, no Kaggle login required for
       these specific public datasets.

Usage:
    pip install ucimlrepo kagglehub
    python scripts/download_synthony_datasets.py                  # SYNTHONY set only
    python scripts/download_synthony_datasets.py --extra           # + the 8 recommended additions
    python scripts/download_synthony_datasets.py --extra-only      # only the 8 additions
    python scripts/download_synthony_datasets.py --out ~/downloaded/dataset --force
    python scripts/download_synthony_datasets.py --list

The target directory is intentionally *not* meant to be committed to this
repo -- point it somewhere outside the working tree (the default,
~/downloaded/dataset, already is) so there's no risk of accidentally
tracking tens to hundreds of megabytes of raw data. Covertype alone is
~75 MB; the full --extra set is a few hundred MB.
"""
import argparse
import io
import os
import sys
import urllib.request
import zipfile

import pandas as pd

# ---------------------------------------------------------------------------
# SYNTHONY benchmark suite (11 datasets)
# ---------------------------------------------------------------------------

# name -> (UCI dataset id, one-line description)
# id maps to https://archive.ics.uci.edu/dataset/<id>
SYNTHONY_UCI_DATASETS = {
    "Bean": (602, "Dry Bean Dataset -- 13,611 images' shape features, 7-class classification"),
    "HTRU2": (372, "HTRU2 -- 17,898 pulsar candidate stats, binary classification, ~9:1 imbalance"),
    "IndianLiverPatient": (225, "ILPD -- 583 liver patient records, binary classification"),
    "News": (332, "Online News Popularity -- 39,797 articles, 58 features, regression (shares)"),
    "Obesity": (544, "Estimation of Obesity Levels -- 2,111 records, 7-class classification"),
    "Shoppers": (468, "Online Shoppers Purchasing Intention -- 12,330 sessions, binary classification"),
    "abalone": (1, "Abalone -- 4,177 records, age (rings) regression/multiclass"),
    "faults": (198, "Steel Plates Faults -- 1,941 records, 7-class defect classification"),
}

# UCI datasets whose id exists but isn't importable through the ucimlrepo
# Python client (older-format datasets not yet migrated to the new API) --
# fetched from UCI's static per-dataset zip instead, and their train/test
# split files concatenated back into a single CSV to match SYNTHONY's shape.
# id maps to https://archive.ics.uci.edu/dataset/<id>
SYNTHONY_UCI_STATIC_ZIP_DATASETS = {
    "wilt": (
        285,
        "https://archive.ics.uci.edu/static/public/285/wilt.zip",
        ["training.csv", "testing.csv"],
        "Wilt -- 4,839 records (train+test), binary classification, imbalanced",
    ),
}

# name -> (URL, one-line description)
# Both originated on Kaggle; pulled from the same GitHub mirrors this repo
# already uses in tests/integration/setup_sandbox_datasets.py.
SYNTHONY_KAGGLE_MIRROR_DATASETS = {
    "Titanic": (
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "Kaggle Titanic - Machine Learning from Disaster -- 891 passengers, binary classification",
    ),
    "insurance": (
        "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv",
        "Kaggle Medical Cost Personal Datasets -- 1,338 records, regression (charges)",
    ),
}

# ---------------------------------------------------------------------------
# Recommended additions (8 datasets) -- fill gaps SYNTHONY doesn't cover
# ---------------------------------------------------------------------------

# name -> (UCI dataset id, one-line description)
EXTRA_UCI_DATASETS = {
    "Adult": (2, "Census Income -- 48,842 records, ~7% missing, standard DP synthetic-data benchmark (used in the AIM paper)"),
    "Covertype": (31, "Forest Covertype -- 581,012 records, 54 features, 7-class -- scale stress test"),
    "Diabetes130US": (296, "Diabetes 130-US hospitals -- 101,766 records, ICD9 diagnosis codes = genuine high-cardinality categoricals"),
    "BankMarketing": (222, "Bank Marketing -- 45,211 records, binary classification, ~11% positive imbalance"),
    "CommunitiesCrime": (183, "Communities and Crime -- 1,994 records, 128 features, ~23% organically missing values"),
}

# name -> (kaggle dataset slug, filename inside the download, one-line description)
EXTRA_KAGGLEHUB_DATASETS = {
    "TMDBMovies": (
        "tmdb/tmdb-movie-metadata",
        "tmdb_5000_movies.csv",
        "TMDB 5000 Movies -- 4,803 records: nested-JSON columns (genres/keywords/production_companies), "
        "long free text (overview/tagline), mixed numeric+categorical, plus a datetime column (release_date)",
    ),
    "WineReviews": (
        "zynicide/wine-reviews",
        "winemag-data-130k-v2.csv",
        "Wine Reviews -- 129,971 records: long free-text tasting notes (description) + numeric (points/price) "
        "+ categorical (country/variety/winery)",
    ),
    "CreditCardFraud": (
        "mlg-ulb/creditcardfraud",
        "creditcard.csv",
        "Credit Card Fraud Detection -- 284,807 records, ~0.17% fraud rate (~577:1 imbalance)",
    ),
}


def download_uci(name, dataset_id, out_dir, force):
    dest = os.path.join(out_dir, f"{name}.csv")
    if os.path.exists(dest) and not force:
        print(f"  skip {name} (already exists at {dest})")
        return
    from ucimlrepo import fetch_ucirepo

    print(f"  fetching UCI dataset id={dataset_id} ...")
    ds = fetch_ucirepo(id=dataset_id)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.to_csv(dest, index=False)
    print(f"  saved {dest} ({len(df)} rows, {df.shape[1]} cols)")


def download_mirror(name, url, out_dir, force):
    dest = os.path.join(out_dir, f"{name}.csv")
    if os.path.exists(dest) and not force:
        print(f"  skip {name} (already exists at {dest})")
        return
    print(f"  downloading {url} ...")
    df = pd.read_csv(url)
    df.to_csv(dest, index=False)
    print(f"  saved {dest} ({len(df)} rows, {df.shape[1]} cols)")


def download_static_zip(name, zip_url, member_files, out_dir, force):
    dest = os.path.join(out_dir, f"{name}.csv")
    if os.path.exists(dest) and not force:
        print(f"  skip {name} (already exists at {dest})")
        return
    print(f"  downloading {zip_url} ...")
    with urllib.request.urlopen(zip_url) as resp:
        zip_bytes = resp.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        parts = [pd.read_csv(zf.open(member)) for member in member_files]
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(dest, index=False)
    print(f"  saved {dest} ({len(df)} rows, {df.shape[1]} cols, from {len(member_files)} files in zip)")


def download_kagglehub(name, kaggle_slug, member_filename, out_dir, force):
    dest = os.path.join(out_dir, f"{name}.csv")
    if os.path.exists(dest) and not force:
        print(f"  skip {name} (already exists at {dest})")
        return
    import kagglehub

    print(f"  downloading kaggle:{kaggle_slug} ...")
    download_path = kagglehub.dataset_download(kaggle_slug)
    src = os.path.join(download_path, member_filename)
    df = pd.read_csv(src)
    df.to_csv(dest, index=False)
    print(f"  saved {dest} ({len(df)} rows, {df.shape[1]} cols)")


def _print_group(title, entries, describe):
    print(title)
    for name, entry in entries.items():
        print(describe(name, entry))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", default="~/downloaded/dataset", help="Output directory (default: ~/downloaded/dataset)"
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if the file already exists")
    parser.add_argument("--list", action="store_true", help="List the datasets and sources, then exit")
    parser.add_argument(
        "--extra", action="store_true", help="Also download the 8 recommended additional datasets"
    )
    parser.add_argument(
        "--extra-only", action="store_true", help="Download only the 8 recommended additional datasets"
    )
    args = parser.parse_args()

    do_synthony = not args.extra_only
    do_extra = args.extra or args.extra_only

    if args.list:
        _print_group(
            "SYNTHONY set -- UCI ML Repository datasets (via ucimlrepo):",
            SYNTHONY_UCI_DATASETS,
            lambda n, e: f"  {n:20s} id={e[0]:<4d} {e[1]}",
        )
        _print_group(
            "\nSYNTHONY set -- UCI ML Repository datasets (via static zip):",
            SYNTHONY_UCI_STATIC_ZIP_DATASETS,
            lambda n, e: f"  {n:20s} id={e[0]:<4d} {e[3]}\n{'':25s}{e[1]}",
        )
        _print_group(
            "\nSYNTHONY set -- Kaggle-origin datasets (via GitHub mirror):",
            SYNTHONY_KAGGLE_MIRROR_DATASETS,
            lambda n, e: f"  {n:20s} {e[1]}\n{'':25s}{e[0]}",
        )
        _print_group(
            "\nRecommended additions -- UCI ML Repository datasets (via ucimlrepo):",
            EXTRA_UCI_DATASETS,
            lambda n, e: f"  {n:20s} id={e[0]:<4d} {e[1]}",
        )
        _print_group(
            "\nRecommended additions -- Kaggle datasets (via kagglehub, no login needed):",
            EXTRA_KAGGLEHUB_DATASETS,
            lambda n, e: f"  {n:20s} kaggle:{e[0]}\n{'':25s}{e[2]}",
        )
        return

    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading to {out_dir}\n")

    total = 0

    if do_synthony:
        try:
            import ucimlrepo  # noqa: F401
        except ImportError:
            print("ucimlrepo is required for the UCI datasets: pip install ucimlrepo", file=sys.stderr)
            sys.exit(1)

        print("SYNTHONY set -- UCI ML Repository datasets:")
        for name, (dataset_id, _desc) in SYNTHONY_UCI_DATASETS.items():
            download_uci(name, dataset_id, out_dir, args.force)

        print("\nSYNTHONY set -- UCI ML Repository datasets (static zip):")
        for name, (_dataset_id, url, members, _desc) in SYNTHONY_UCI_STATIC_ZIP_DATASETS.items():
            download_static_zip(name, url, members, out_dir, args.force)

        print("\nSYNTHONY set -- Kaggle-origin datasets (via GitHub mirror, no API key needed):")
        for name, (url, _desc) in SYNTHONY_KAGGLE_MIRROR_DATASETS.items():
            download_mirror(name, url, out_dir, args.force)

        total += (
            len(SYNTHONY_UCI_DATASETS)
            + len(SYNTHONY_UCI_STATIC_ZIP_DATASETS)
            + len(SYNTHONY_KAGGLE_MIRROR_DATASETS)
        )

    if do_extra:
        try:
            import ucimlrepo  # noqa: F401
        except ImportError:
            print("ucimlrepo is required for the extra UCI datasets: pip install ucimlrepo", file=sys.stderr)
            sys.exit(1)
        try:
            import kagglehub  # noqa: F401
        except ImportError:
            print("kagglehub is required for the extra Kaggle datasets: pip install kagglehub", file=sys.stderr)
            sys.exit(1)

        print("\nRecommended additions -- UCI ML Repository datasets:")
        for name, (dataset_id, _desc) in EXTRA_UCI_DATASETS.items():
            download_uci(name, dataset_id, out_dir, args.force)

        print("\nRecommended additions -- Kaggle datasets (via kagglehub, no login needed):")
        for name, (kaggle_slug, member_filename, _desc) in EXTRA_KAGGLEHUB_DATASETS.items():
            download_kagglehub(name, kaggle_slug, member_filename, out_dir, args.force)

        total += len(EXTRA_UCI_DATASETS) + len(EXTRA_KAGGLEHUB_DATASETS)

    print(f"\nDone. {total} datasets in {out_dir}")


if __name__ == "__main__":
    main()

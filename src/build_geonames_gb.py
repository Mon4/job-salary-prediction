"""Generate data/external/geonames_gb.csv - GB locations with population.

Source: GeoNames dump (https://download.geonames.org/export/dump/).

By default the script downloads GB.zip (GeoNames entries for the United
Kingdom only, a few MB) and filters feature classes:
  P - populated places
  A - administrative divisions
Renaming PCLI (country-level row) to 'UK' keeps compatibility with the
preprocessing pipeline.

If you already have the full allCountries.txt dump, you can replicate the
original pipeline exactly by passing its path:

    python src/build_geonames_gb.py path/to/allCountries.txt
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = ROOT / "data" / "external"
GEONAMES_URL = "https://download.geonames.org/export/dump/"

COLUMNS = [
    "geonameid", "name", "asciiname", "alternatenames", "lat", "lon",
    "feature_class", "feature_code", "country_code", "cc2", "admin1",
    "admin2", "admin3", "admin4", "population", "elevation", "dem", "tz", "moddate",
]
USECOLS = [
    "asciiname", "alternatenames", "country_code", "feature_code",
    "feature_class", "admin1", "admin2", "admin3", "lon", "lat", "population",
]


def download_gb_dump() -> Path:
    """Download and extract GB.txt (entries for the United Kingdom)."""
    zip_path = EXTERNAL_DIR / "GB.zip"
    txt_path = EXTERNAL_DIR / "GB.txt"

    if not txt_path.exists():
        print(f"Downloading {GEONAMES_URL}GB.zip ...")
        urllib.request.urlretrieve(GEONAMES_URL + "GB.zip", zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("GB.txt", EXTERNAL_DIR)
        print("Downloaded and extracted.")

    return txt_path


def main() -> None:
    import pandas as pd  # imported here so the download messages come first

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    source = sys.argv[1] if len(sys.argv) > 1 else download_gb_dump()
    print(f"Reading {source} ...")
    geonames = pd.read_csv(
        source, sep="\t", names=COLUMNS, usecols=USECOLS, dtype=str, header=None
    )

    # GB entries only, populated places (P) and administrative areas (A)
    geonames_gb = geonames[geonames["country_code"] == "GB"].copy().reset_index(drop=True)
    geonames_gb = geonames_gb[geonames_gb["feature_class"].isin(["P", "A"])].reset_index(drop=True)

    # country-level row used later as the 'UK' fallback population
    geonames_gb.loc[geonames_gb["feature_code"] == "PCLI", "asciiname"] = "UK"

    out_path = EXTERNAL_DIR / "geonames_gb.csv"
    geonames_gb.to_csv(out_path, index=False)
    print(f"Saved {len(geonames_gb)} rows to {out_path}")


if __name__ == "__main__":
    main()

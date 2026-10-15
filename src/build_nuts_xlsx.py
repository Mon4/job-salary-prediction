"""Generate data/external/NUTS.xlsx - NUTS level 1 (January 2018) regions.

Data source: ONS Open Geography Portal,
'NUTS, level 1 (January 2018) Boundaries UK BSC'
(https://www.data.gov.uk/dataset/ba66b4fc-34a0-4323-ae9a-118ab8929552/
nuts-level-1-january-2018-boundaries-uk-bsc).

The script first tries to fetch the region centroids from the ONS ArcGIS
REST API. If the API is unavailable it falls back to the embedded snapshot
(exact values retrieved on 2026-10-15), so the file can always be built
offline.

Output columns (expected by preprocessing): NUTS118CD, NUTS118NM, LONG, LAT.
"""
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = ROOT / "data" / "external"

ONS_QUERY_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "NUTS1_Jan_2018_SGCB_in_the_UK_2022/FeatureServer/0/query"
    "?where=1%3D1&outFields=nuts118cd,nuts118nm,long,lat&outSR=4326"
    "&returnGeometry=false&f=json"
)

# Offline snapshot: NUTS118CD, NUTS118NM, LONG, LAT (ONS, Jan 2018 BSC centroids)
FALLBACK = [
    ("UKC", "North East (England)", -1.72889996, 55.2970314),
    ("UKD", "North West (England)", -2.7723701, 54.44945145),
    ("UKE", "Yorkshire and The Humber", -1.28711998, 53.93264008),
    ("UKF", "East Midlands (England)", -0.84966999, 52.79571915),
    ("UKG", "West Midlands (England)", -2.2035799, 52.55696869),
    ("UKH", "East of England", 0.50414598, 52.24066925),
    ("UKI", "London", -0.30864, 51.49227142),
    ("UKJ", "South East (England)", -0.99311, 51.4509697),
    ("UKK", "South West (England)", -3.63343, 50.81119156),
    ("UKL", "Wales", -3.99415994, 52.06740952),
    ("UKM", "Scotland", -3.97091007, 56.1774292),
    ("UKN", "Northern Ireland", -6.85481024, 54.61494064),
]


def fetch_from_ons():
    """Try to download region centroids from the ONS ArcGIS REST API."""
    try:
        with urllib.request.urlopen(ONS_QUERY_URL, timeout=30) as resp:
            data = json.load(resp)
        rows = [
            (
                f["attributes"]["nuts118cd"],
                f["attributes"]["nuts118nm"],
                f["attributes"]["long"],
                f["attributes"]["lat"],
            )
            for f in data["features"]
        ]
        if len(rows) == 12:
            print("Fetched NUTS1 centroids from the ONS API.")
            return rows
        print(f"Unexpected ONS API response ({len(rows)} rows); using snapshot.")
    except Exception as exc:
        print(f"ONS API unavailable ({exc}); using embedded snapshot.")
    return None


def main() -> None:
    import pandas as pd

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    rows = fetch_from_ons()
    if rows is None:
        rows = FALLBACK

    nuts = pd.DataFrame(rows, columns=["NUTS118CD", "NUTS118NM", "LONG", "LAT"])
    out_path = EXTERNAL_DIR / "NUTS.xlsx"
    nuts.to_excel(out_path, index=False)
    print(f"Saved {len(nuts)} regions to {out_path}")


if __name__ == "__main__":
    main()

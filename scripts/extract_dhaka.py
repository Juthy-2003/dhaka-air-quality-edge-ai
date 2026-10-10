"""Download the Bangladesh AQI dataset (303 MB, CC BY 4.0) and keep only the
four locations inside Dhaka's bounding box, as data/raw/dhaka_aqi_hourly.parquet.

Source: Mendeley Data doi:10.17632/9j447cynb9.1 (file "AQI Bangladesh.csv").
"""

import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import requests

URL = (
    "https://data.mendeley.com/public-files/datasets/9j447cynb9/files/"
    "b78e7cc6-8e21-4b06-996e-6b5f9d0b6f69/file_downloaded"
)
SHA256 = "b2c459e54e6707a3bd9974a3e2c0edf9e896720abe63d9a65fd6e128c97c25c0"
OUT = Path("data/raw/dhaka_aqi_hourly.parquet")
BBOX = {"lat": (23.66, 23.90), "lon": (90.30, 90.50)}


def main() -> None:
    if OUT.exists():
        print(f"{OUT} exists")
        return
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "aqi.csv"
        h = hashlib.sha256()
        with requests.get(URL, stream=True, timeout=300) as r, open(csv, "wb") as f:
            r.raise_for_status()
            for block in r.iter_content(1 << 20):
                h.update(block)
                f.write(block)
        if h.hexdigest() != SHA256:
            raise SystemExit(f"checksum mismatch: {h.hexdigest()}")
        parts = [
            c[c.lat.between(*BBOX["lat"]) & c.lon.between(*BBOX["lon"])] for c in pd.read_csv(csv, chunksize=500_000)
        ]
        df = pd.concat(parts)
        df["datetime"] = pd.to_datetime(df["datetime"])
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
        print(f"wrote {OUT}: {len(df)} rows, locations {sorted(df.city_name.unique())}")


if __name__ == "__main__":
    main()

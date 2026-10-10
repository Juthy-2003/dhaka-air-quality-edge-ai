"""Historical Dhaka readings, replayed as if they were a live feed.

Source: "Bangladesh Air Quality Index (AQI) Dataset (2000–2025)", Mendeley Data
v1, doi:10.17632/9j447cynb9.1, CC BY 4.0. The full file (303 MB, 3.19 M rows,
103 locations) is reduced to the four locations inside Dhaka's bounding box
by `scripts/extract_dhaka.py`.

Caveat, recorded honestly: the publisher does not say where the values come
from. The column names match Open-Meteo's air-quality API (CAMS model
output), and Dhaka's series has no missing hour from 2000 to 2025, which
real monitors do not achieve — so these are most likely *modelled* values.
They are used for training and replay, never presented as sensor readings.
"""

from datetime import UTC, timedelta

import pandas as pd

from aqedge import config

COLUMNS = {"pm2_5": "pm25", "pm10": "pm10", "nitrogen_dioxide": "no2", "sulphur_dioxide": "so2", "ozone": "o3",
           "carbon_monoxide": "co"}
SOURCE = "replay-mendeley"
# "Live" replay: the reading for hour H is the dataset's value at H − 730 days,
# re-stamped to H, with the original time kept in `replayed_from`.
REPLAY_LAG = timedelta(days=730)


def load(path=config.REPLAY_PARQUET) -> pd.DataFrame:
    """Wide table: one row per (location, hour). Timestamps are treated as UTC
    (the dataset's ISO strings carry no offset; Open-Meteo returns GMT by default)."""
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(UTC)
    df = df.rename(columns=COLUMNS)
    df["location_id"] = df["city_id"].astype(str)
    df["location"] = df["city_name"]
    return df.sort_values(["location_id", "datetime"]).reset_index(drop=True)


def series(df: pd.DataFrame, location: str = "Dhaka", parameter: str = "pm25") -> pd.Series:
    """One location's hourly values for one pollutant (used by training and the simulator)."""
    s = df[df["location"] == location].set_index("datetime")[parameter].astype(float)
    return s[~s.index.duplicated()].asfreq("h")


def replay_as_live(df: pd.DataFrame, start, end) -> list[dict]:
    """Rows for live hours start < H <= end, taken from two years earlier.

    One dict per (location, hour), ready for db.save_readings()."""
    lo, hi = pd.Timestamp(start) - REPLAY_LAG, pd.Timestamp(end) - REPLAY_LAG
    window = df[(df["datetime"] > lo) & (df["datetime"] <= hi)]
    rows = []
    for r in window.itertuples():
        old_ts = r.datetime.to_pydatetime()
        row = {"location_id": r.location_id, "location": r.location, "ts": old_ts + REPLAY_LAG,
               "source": SOURCE, "replayed_from": old_ts}
        for p in config.POLLUTANTS:
            value = getattr(r, p)
            row[p] = None if pd.isna(value) else float(value)
        rows.append(row)
    return rows

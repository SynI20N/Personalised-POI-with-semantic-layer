"""
Attach Foursquare category paths to the TSMC2014 check-ins.

TSMC2014 stores Foursquare's legacy 24-hex category IDs. The Foursquare Open Source
Places categories table uses the same IDs, so the join is on ``category_id``. Each row
of that table carries the full breadcrumb ("Dining and Drinking > Bar > Sake Bar"),
its depth (1-6) and the name of its ancestor at every level.

Assumption: the category of a check-in is the most specific (leaf) level of its path.

Run as a script to write ``data/processed/checkins_<city>_with_paths.parquet``:

    python src/construct_category_path.py --city NYC
"""
import argparse
from pathlib import Path

import pandas as pd

BASE_PATH = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_PATH / "data" / "raw"
PROCESSED_DIR = BASE_PATH / "data" / "processed"

CATEGORY_FILE = RAW_DIR / "categories_000000.parquet"
CHECKIN_FILES = {
    "NYC": RAW_DIR / "dataset_TSMC2014_NYC.txt",
    "TKY": RAW_DIR / "dataset_TSMC2014_TKY.csv",  # Kaggle version: comma-separated, header, UTF-8
}

# Column order of the raw TSMC2014 files (tab-separated, no header)
CHECKIN_COLUMNS = [
    "userId",
    "venueId",
    "categoryId",
    "category_name",
    "lat",
    "lon",
    "timezoneOffset",  # minutes from UTC, differs per row (daylight saving time)
    "utcTimestamp",
]
UTC_FORMAT = "%a %b %d %H:%M:%S %z %Y"  # e.g. "Tue Apr 03 18:00:09 +0000 2012"

MAX_DEPTH = 6
PATH_SEP = " > "
LEVEL_NAME_COLUMNS = [f"level{level}_category_name" for level in range(1, MAX_DEPTH + 1)]


def load_category_taxonomy(path=CATEGORY_FILE):
    """
    Load the Foursquare category tree: one row per category.

    Columns: category_id, category_level (depth 1-6), category_name,
    full_category_path (the breadcrumb), level<k>_category_id / _name for k = 1..6.
    """
    columns = ["category_id", "category_level", "category_name", "category_label"]
    for level in range(1, MAX_DEPTH + 1):
        columns += [f"level{level}_category_id", f"level{level}_category_name"]
    taxonomy = pd.read_parquet(path, columns=columns)
    taxonomy = taxonomy.rename(columns={"category_label": "full_category_path"})
    return taxonomy.drop_duplicates("category_id").reset_index(drop=True)


def load_checkins(city="NYC", path=None):
    """
    Load the raw TSMC2014 check-ins of one city and parse the timestamps.

    Adds ``utc_time`` (timezone-aware UTC) and ``local_time`` (naive local clock time:
    UTC plus the row's own offset, so daylight saving time is handled per row).

    Addtion: we directly compute slot column for TLDA
    """
    path = Path(path) if path is not None else CHECKIN_FILES[city]
    if not path.exists():
        raise FileNotFoundError(f"No check-in file for {city}: {path}")
    if path.suffix == ".csv":
        # Kaggle CSV: same 8 columns in the same order, but comma-separated, UTF-8 and with a
        # header row using other names (venueCategoryId, latitude, ...); header=0 replaces it
        checkins = pd.read_csv(path, sep=",", header=0, names=CHECKIN_COLUMNS, encoding="utf-8")
    else:
        # original TSMC2014 .txt: tab-separated, no header, latin-1
        checkins = pd.read_csv(
            path, sep="\t", header=None, names=CHECKIN_COLUMNS, encoding="latin-1"
        )
    checkins["utc_time"] = pd.to_datetime(
        checkins["utcTimestamp"], format=UTC_FORMAT, utc=True
    )
    offset = pd.to_timedelta(checkins["timezoneOffset"], unit="m")
    checkins["local_time"] = (checkins["utc_time"] + offset).dt.tz_localize(None)
    checkins['slot'] = checkins['local_time'].dt.hour + 24 * (checkins['local_time'].dt.day_of_week >= 5)
    
    return checkins


def attach_category_paths(frame, taxonomy, id_column="categoryId"):
    """
    Add the category path of ``frame[id_column]`` to any frame (check-ins or venues).

    Adds full_category_path, category_depth and level1..level6 category names.
    IDs missing from the taxonomy get NaN in all added columns.
    """
    lookup = taxonomy.set_index("category_id")[
        ["full_category_path", "category_level"] + LEVEL_NAME_COLUMNS
    ].rename(columns={"category_level": "category_depth"})
    return frame.join(lookup, on=id_column)


def category_nodes_at_level(taxonomy, level):
    """
    Map every category to its path truncated at ``level``.

    Categories shallower than ``level`` keep their full path, so ``level=6`` gives the
    leaf categories and ``level=1`` the 11 top-level categories.
    Returns a Series indexed by category_id.
    """
    names = taxonomy.set_index("category_id")[LEVEL_NAME_COLUMNS[:level]]
    return names.apply(lambda row: PATH_SEP.join(row.dropna()), axis=1)


def build_venue_table(checkins):
    """
    One row per venue.

    categoryId: the venue's most frequent category ID (some venues carry several);
                ties go to the smallest ID so the result is deterministic.
    lat, lon:   median of the venue's check-in coordinates (they vary slightly).
    n_checkins, n_users: popularity of the venue.
    """
    category_counts = (
        checkins.groupby(["venueId", "categoryId"]).size().rename("n").reset_index()
    )
    modal_category = (
        category_counts.sort_values(["venueId", "n", "categoryId"], ascending=[True, False, True])
        .drop_duplicates("venueId")
        .set_index("venueId")["categoryId"]
    )
    venues = checkins.groupby("venueId").agg(
        lat=("lat", "median"),
        lon=("lon", "median"),
        n_checkins=("userId", "size"),
        n_users=("userId", "nunique"),
    )
    venues["categoryId"] = modal_category
    return venues.reset_index()


def build_checkins_with_paths(city="NYC"):
    """Load a city's check-ins and attach the category path of every check-in."""
    return attach_category_paths(load_checkins(city), load_category_taxonomy())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--city", default="NYC", choices=sorted(CHECKIN_FILES))
    args = parser.parse_args()

    checkins = build_checkins_with_paths(args.city)
    matched = checkins["full_category_path"].notna()
    print(f"{args.city}: {len(checkins):,} check-ins, "
          f"{matched.mean():.4%} matched to a category path, "
          f"{checkins.loc[~matched, 'categoryId'].nunique()} unmatched category IDs")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"checkins_{args.city.lower()}_with_paths.parquet"
    checkins.to_parquet(out_path, index=False)
    print("saved", out_path)


if __name__ == "__main__":
    main()

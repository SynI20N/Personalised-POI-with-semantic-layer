"""
Exploratory data analysis for Semantic TEMN on the Foursquare TSMC2014 data.

Scoped to the decisions the project has to make:

1. Integrity      - can the raw data be trusted (category join, venues with several
                    categories or coordinates, time zones)?
2. Sparsity       - how sparse is the user-POI matrix (TEMN filters, max_p_num,
                    long tail of venues)?
3. Split          - with the paper's chronological 70/15/15 split, how many held-out
                    POIs are cold, and how much does knowing the category help?
4. Time           - is the data dense enough for TLDA time slots (hour x day type)?
5. Category paths - how deep and how coarse are the paths, which categories dominate,
                    which are rare (RQ3 buckets), how ambiguous are the names?
6. Embeddings     - do the frozen text embeddings respect the taxonomy, and does their
                    similarity agree with how users actually combine categories?

Every analysis is a function that returns a table, so it can be reused in a notebook.
``run_eda`` runs all of them and writes ``reports/eda/<city>/eda_summary.md`` plus figures.

    python src/eda.py --city NYC
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_PATH = Path(__file__).resolve().parent.parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

from src.construct_category_path import (  # noqa: E402
    MAX_DEPTH,
    attach_category_paths,
    build_venue_table,
    category_nodes_at_level,
    load_category_taxonomy,
    load_checkins,
)
from src.path_to_embedding import EMBEDDING_FILE, load_category_embeddings  # noqa: E402

REPORT_DIR = BASE_PATH / "reports" / "eda"
EARTH_RADIUS_KM = 6371.0
SEED = 2026

# Paper settings used by the split diagnostics
MIN_USER_POIS = 10
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15

# Category frequency buckets (number of venues per leaf category), for RQ3
RARE_MAX_VENUES = 10      # < 10 venues  -> rare
FREQUENT_MIN_VENUES = 100  # >= 100 venues -> frequent

EXAMPLE_CATEGORIES = ["Sake Bar", "Subway", "Home (private)", "Gym", "Coffee Shop", "Bridge"]


# --------------------------------------------------------------------------- helpers

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; works element-wise on arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def gini(values):
    """Gini coefficient of non-negative values (0 = equal, 1 = all in one item)."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    ranks = np.arange(1, n + 1)
    return float(2 * np.sum(ranks * x) / (n * x.sum()) - (n + 1) / n)


def effective_number(counts):
    """exp(entropy) of a count distribution: how many equally likely items it is worth."""
    p = np.asarray(counts, dtype=float)
    p = p[p > 0] / p.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


# --------------------------------------------------------------- 1. data integrity

def dataset_overview(checkins):
    """Size, time span, duplicates and density of the raw check-ins."""
    n_users = checkins["userId"].nunique()
    n_venues = checkins["venueId"].nunique()
    n_pairs = len(checkins[["userId", "venueId"]].drop_duplicates())
    return pd.Series({
        "check-ins": len(checkins),
        "users": n_users,
        "venues": n_venues,
        "category IDs": checkins["categoryId"].nunique(),
        "category names": checkins["category_name"].nunique(),
        "first check-in (local)": checkins["local_time"].min(),
        "last check-in (local)": checkins["local_time"].max(),
        "duplicate (user, venue, timestamp) rows": int(checkins.duplicated(
            ["userId", "venueId", "utcTimestamp"]).sum()),
        "distinct user-venue pairs": n_pairs,
        "user-venue density (%)": round(100 * n_pairs / (n_users * n_venues), 4),
        "repeat check-ins (%)": round(100 * (1 - n_pairs / len(checkins)), 2),
    })


def category_join_coverage(checkins):
    """Category IDs that have no path in the taxonomy, with their check-in counts."""
    unmatched = checkins[checkins["full_category_path"].isna()]
    table = (unmatched.groupby(["categoryId", "category_name"]).size()
             .rename("check-ins").sort_values(ascending=False).reset_index())
    summary = pd.Series({
        "check-ins with a path (%)": round(100 * checkins["full_category_path"].notna().mean(), 3),
        "category IDs without a path": unmatched["categoryId"].nunique(),
        "check-ins without a path": len(unmatched),
    })
    return summary, table


def venue_category_consistency(checkins, n_examples=10):
    """Venues whose check-ins carry more than one category ID."""
    ids_per_venue = checkins.groupby("venueId")["categoryId"].nunique()
    multi = ids_per_venue[ids_per_venue > 1].index
    rows = checkins[checkins["venueId"].isin(multi)]
    summary = pd.Series({
        "venues with > 1 category ID": len(multi),
        "share of venues (%)": round(100 * len(multi) / len(ids_per_venue), 2),
        "share of check-ins (%)": round(100 * len(rows) / len(checkins), 2),
    })
    examples = (rows.groupby("venueId")["category_name"]
                .agg(lambda names: " | ".join(sorted(set(names))))
                .rename("category names").head(n_examples).reset_index())
    return summary, examples


def venue_coordinate_spread(checkins):
    """How far a venue's check-in coordinates lie from its median coordinate (metres)."""
    median = checkins.groupby("venueId")[["lat", "lon"]].transform("median")
    metres = 1000 * haversine_km(checkins["lat"], checkins["lon"], median["lat"], median["lon"])
    spread = metres.groupby(checkins["venueId"]).max()
    moved = spread[spread > 0]
    return pd.Series({
        "venues with > 1 coordinate": len(moved),
        "median spread of those (m)": round(moved.median(), 1),
        "p90 spread (m)": round(moved.quantile(0.9), 1),
        "max spread (m)": round(moved.max(), 1),
        "venues with spread > 100 m": int((spread > 100).sum()),
        "venues with spread > 1 km": int((spread > 1000).sum()),
    })


def timezone_offsets(checkins):
    """Distribution of the per-row UTC offsets (minutes)."""
    counts = checkins["timezoneOffset"].value_counts()
    return pd.DataFrame({"check-ins": counts,
                         "share (%)": (100 * counts / counts.sum()).round(3)})


# ------------------------------------------------------------ 2. interaction sparsity

def user_activity(checkins, max_p_num_values=(100, 200)):
    """Per-user activity, its distribution, and how many users the TEMN settings affect."""
    per_user = checkins.groupby("userId").agg(
        checkins=("venueId", "size"), distinct_pois=("venueId", "nunique"))
    distribution = per_user.describe(percentiles=[0.1, 0.5, 0.9, 0.99]).T.round(1)
    thresholds = {f"users with < {MIN_USER_POIS} distinct POIs (dropped by paper filter)":
                  int((per_user["distinct_pois"] < MIN_USER_POIS).sum())}
    for cap in max_p_num_values:
        thresholds[f"users with > {cap} distinct POIs (history cut at max_p_num={cap})"] = \
            int((per_user["distinct_pois"] > cap).sum())
    return per_user, distribution, pd.Series(thresholds)


def venue_popularity(checkins):
    """Long tail of venue popularity."""
    per_venue = checkins.groupby("venueId").agg(
        checkins=("userId", "size"), users=("userId", "nunique"))
    counts = per_venue["checkins"].sort_values(ascending=False)

    def top_share(fraction):
        k = max(1, int(np.ceil(fraction * len(counts))))
        return round(100 * counts.iloc[:k].sum() / counts.sum(), 1)

    summary = pd.Series({
        "venues with 1 check-in (%)": round(100 * (per_venue["checkins"] == 1).mean(), 1),
        "venues with 1 user (%)": round(100 * (per_venue["users"] == 1).mean(), 1),
        "venues with >= 5 users (%)": round(100 * (per_venue["users"] >= 5).mean(), 1),
        "check-ins at top 1% of venues (%)": top_share(0.01),
        "check-ins at top 10% of venues (%)": top_share(0.10),
        "Gini of check-ins per venue": round(gini(per_venue["checkins"]), 3),
    })
    return per_venue, summary


# --------------------------------------------------------- 3. chronological split

def first_visit_sequences(checkins):
    """Each user's distinct POIs ordered by first visit (the TEMN input H_u)."""
    first = (checkins.groupby(["userId", "venueId"], as_index=False)["utc_time"].min()
             .sort_values(["userId", "utc_time", "venueId"]))
    first["position"] = first.groupby("userId").cumcount()
    first["n_user_pois"] = first.groupby("userId")["venueId"].transform("size")
    return first.reset_index(drop=True)


def chronological_split(sequences, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC,
                        min_pois=MIN_USER_POIS):
    """
    Paper split per user: first 70% of first visits = train, next 15% = val, rest = test.
    Users with fewer than ``min_pois`` distinct POIs are dropped (paper, appendix A.1).
    """
    seq = sequences[sequences["n_user_pois"] >= min_pois].copy()
    n = seq["n_user_pois"]
    n_train = np.floor(train_frac * n + 0.5).astype(int)
    n_val = np.floor(val_frac * n + 0.5).astype(int)
    seq["split"] = np.where(seq["position"] < n_train, "train",
                            np.where(seq["position"] < n_train + n_val, "val", "test"))
    return seq


def split_cold_start_rates(split):
    """Share of held-out POIs that no user visited in train (the natural cold-start rate)."""
    train_pois = set(split.loc[split["split"] == "train", "venueId"])
    rows = []
    for name in ("val", "test"):
        held_out = split[split["split"] == name]
        cold = ~held_out["venueId"].isin(train_pois)
        rows.append({
            "split": name,
            "held-out user-POI pairs": len(held_out),
            "distinct POIs": held_out["venueId"].nunique(),
            "pairs with a cold POI (%)": round(100 * cold.mean(), 1),
            "distinct cold POIs": held_out.loc[cold, "venueId"].nunique(),
        })
    return pd.DataFrame(rows).set_index("split")


def category_headroom(split, venues, taxonomy, levels=(1, 2, 3, MAX_DEPTH), seed=SEED):
    """
    How often a test POI's category (at several path levels) is already in the user's
    train history, against the same rate for a random POI the user never visited.

    A high rate with a large lift means category preference alone is a strong signal
    (and a strong baseline); a rate near the random one means it is not.
    Also reports how many test POIs have a category no train POI has (RQ3 'unseen').
    """
    rng = np.random.default_rng(seed)
    venue_category = venues.set_index("venueId")["categoryId"]
    train = split[split["split"] == "train"]
    test = split[split["split"] == "test"]
    all_venues = venues["venueId"].to_numpy()
    train_by_user = train.groupby("userId")["venueId"].agg(set)

    # one random never-visited venue per test pair
    random_venues = []
    for user, n_rows in test.groupby("userId").size().items():
        visited = train_by_user.get(user, set())
        while n_rows:
            candidate = all_venues[rng.integers(len(all_venues))]
            if candidate not in visited:
                random_venues.append(candidate)
                n_rows -= 1
    test = test.sort_values("userId").assign(random_venue=random_venues)

    rows = []
    for level in levels:
        node = venue_category.map(category_nodes_at_level(taxonomy, level))
        user_nodes = train.assign(node=train["venueId"].map(node)).groupby("userId")["node"].agg(set)
        known = user_nodes.reindex(test["userId"]).to_numpy()
        hit = np.array([n in s for n, s in zip(test["venueId"].map(node), known)])
        hit_random = np.array([n in s for n, s in zip(test["random_venue"].map(node), known)])
        label = "leaf" if level == MAX_DEPTH else f"level {level}"
        rows.append({
            "path level": label,
            "distinct nodes": node.nunique(),
            "test POI category in user's train history (%)": round(100 * hit.mean(), 1),
            "random unvisited POI (%)": round(100 * hit_random.mean(), 1),
            "lift": round(hit.mean() / max(hit_random.mean(), 1e-9), 2),
        })

    train_categories = set(train["venueId"].map(venue_category))
    unseen = ~test["venueId"].map(venue_category).isin(train_categories)
    unseen_summary = pd.Series({
        "test pairs whose leaf category has no train POI (%)": round(100 * unseen.mean(), 2),
        "distinct unseen leaf categories in test": test.loc[unseen, "venueId"]
        .map(venue_category).nunique(),
    })
    return pd.DataFrame(rows).set_index("path level"), unseen_summary


# ------------------------------------------------------------------- 4. time

def temporal_profile(checkins):
    """Check-ins per local hour for weekdays and weekends, and TLDA slot sparsity."""
    local = checkins["local_time"]
    day_type = np.where(local.dt.dayofweek >= 5, "weekend", "weekday")
    counts = pd.crosstab(local.dt.hour.rename("hour"), pd.Series(day_type, name="day type",
                                                                 index=checkins.index))
    shares = (100 * counts / counts.sum()).round(2)
    slots = counts.stack()
    slot_summary = pd.Series({
        "slots (24 hours x weekday/weekend)": len(slots),
        "min check-ins per slot": int(slots.min()),
        "median check-ins per slot": int(slots.median()),
        "max check-ins per slot": int(slots.max()),
        "weekend share of check-ins (%)": round(100 * (day_type == "weekend").mean(), 1),
    })
    return counts, shares, slot_summary


# ---------------------------------------------------------------- 5. category paths

def taxonomy_coverage(taxonomy, venues):
    """Per depth: categories in the taxonomy, used by venues, and their check-in share."""
    in_taxonomy = taxonomy.groupby("category_level").size().rename("categories in taxonomy")
    matched = venues.dropna(subset=["category_depth"])
    used = matched.groupby("category_depth").agg(
        **{"categories used": ("categoryId", "nunique"),
           "venues": ("venueId", "size"),
           "check-ins": ("n_checkins", "sum")})
    used.index = used.index.astype(int)
    table = pd.concat([in_taxonomy, used], axis=1).fillna(0).astype(int)
    table["check-ins (%)"] = (100 * table["check-ins"] / table["check-ins"].sum()).round(1)
    table.index.name = "depth"
    return table


def granularity_by_level(taxonomy, venues, max_level=MAX_DEPTH):
    """
    How coarse the paths are: cut every path at level L and count what is left.

    distinct nodes:      number of different (truncated) categories
    effective number:    exp(entropy) of check-ins over those nodes; far below the
                         distinct count means a few nodes dominate
    nodes with < 10 venues: nodes too rare to learn from
    """
    rows = []
    for level in range(1, max_level + 1):
        node = venues["categoryId"].map(category_nodes_at_level(taxonomy, level))
        per_node = venues.assign(node=node).dropna(subset=["node"]).groupby("node").agg(
            venues=("venueId", "size"), checkins=("n_checkins", "sum"))
        rows.append({
            "level": level,
            "distinct nodes": len(per_node),
            "effective number": round(effective_number(per_node["checkins"]), 1),
            "top node share of check-ins (%)": round(
                100 * per_node["checkins"].max() / per_node["checkins"].sum(), 1),
            "median venues per node": int(per_node["venues"].median()),
            "nodes with < 10 venues": int((per_node["venues"] < RARE_MAX_VENUES).sum()),
        })
    return pd.DataFrame(rows).set_index("level")


def name_id_ambiguity(checkins, n_examples=15):
    """Category names that map to several category IDs, and to how many distinct paths."""
    per_name = checkins.groupby("category_name").agg(
        ids=("categoryId", "nunique"),
        paths=("full_category_path", "nunique"),
        checkins=("categoryId", "size"),
        path_examples=("full_category_path",
                       lambda p: " | ".join(sorted(set(p.dropna()))[:3])),
    )
    ambiguous = per_name[per_name["ids"] > 1].sort_values(["ids", "checkins"], ascending=False)
    summary = pd.Series({
        "category names": len(per_name),
        "category IDs": checkins["categoryId"].nunique(),
        "names with > 1 ID": len(ambiguous),
        "check-ins under those names (%)": round(
            100 * ambiguous["checkins"].sum() / len(checkins), 1),
    })
    return summary, ambiguous.head(n_examples)


def category_frequency(venues, checkins, top_k=20):
    """
    Leaf-category popularity and the RQ3 frequency buckets.

    Buckets by number of venues: rare (< 10), medium (10-99), frequent (>= 100).
    """
    per_category = venues.dropna(subset=["full_category_path"]).groupby(
        ["categoryId", "full_category_path"]).agg(
        venues=("venueId", "size"), checkins=("n_checkins", "sum")).reset_index()
    users = checkins.groupby("categoryId")["userId"].nunique().rename("users")
    per_category = per_category.join(users, on="categoryId")
    per_category["check-ins (%)"] = (
        100 * per_category["checkins"] / per_category["checkins"].sum()).round(2)
    per_category["bucket"] = pd.cut(
        per_category["venues"], [0, RARE_MAX_VENUES - 1, FREQUENT_MIN_VENUES - 1, np.inf],
        labels=["rare (< 10 venues)", "medium (10-99)", "frequent (>= 100)"])

    buckets = per_category.groupby("bucket", observed=False).agg(
        categories=("categoryId", "size"), venues=("venues", "sum"),
        checkins=("checkins", "sum"))
    buckets["check-ins (%)"] = (100 * buckets["checkins"] / buckets["checkins"].sum()).round(1)

    top = (per_category.sort_values("checkins", ascending=False).head(top_k)
           [["full_category_path", "venues", "users", "checkins", "check-ins (%)"]]
           .reset_index(drop=True))
    return per_category, buckets, top


def level1_distribution(venues):
    """Share of venues and check-ins per top-level category."""
    table = venues.dropna(subset=["level1_category_name"]).groupby("level1_category_name").agg(
        categories=("categoryId", "nunique"), venues=("venueId", "size"),
        checkins=("n_checkins", "sum"))
    table["venues (%)"] = (100 * table["venues"] / table["venues"].sum()).round(1)
    table["check-ins (%)"] = (100 * table["checkins"] / table["checkins"].sum()).round(1)
    return table.sort_values("checkins", ascending=False)


def user_category_diversity(checkins):
    """
    How concentrated each user's check-ins are over leaf and top-level categories.

    normalised entropy: 0 = all check-ins in one category, 1 = spread evenly over the
    categories the user visits.
    """
    matched = checkins.dropna(subset=["full_category_path"])

    def per_user(group):
        counts = group["categoryId"].value_counts()
        k = len(counts)
        p = counts / counts.sum()
        entropy = -(p * np.log(p)).sum()
        return pd.Series({
            "distinct leaf categories": k,
            "distinct level-1 categories": group["level1_category_name"].nunique(),
            "top category share (%)": 100 * p.iloc[0],
            "normalised entropy": entropy / np.log(k) if k > 1 else 0.0,
        })

    table = matched.groupby("userId")[["categoryId", "level1_category_name"]].apply(per_user)
    return table.describe(percentiles=[0.1, 0.5, 0.9]).T.round(2)


# -------------------------------------------------------------------- 6. embeddings

def used_category_embeddings(venues, taxonomy, embedding_path=EMBEDDING_FILE):
    """Embeddings of the categories used by at least one venue, with their level-1 name."""
    table, matrix = load_category_embeddings(embedding_path)
    used_ids = set(venues["categoryId"])
    keep = table["category_id"].isin(used_ids).to_numpy()
    table = table[keep].reset_index(drop=True)
    matrix = matrix[keep]
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    level1 = taxonomy.set_index("category_id")["level1_category_name"]
    table["level1"] = table["category_id"].map(level1).to_numpy()
    coverage = pd.Series({
        "categories used by venues": len(used_ids),
        "of those with an embedding": int(keep.sum()),
        "embedding dimension": matrix.shape[1],
    })
    return table, matrix, coverage


def level1_separation(table, matrix):
    """Mean cosine similarity within the same top-level category vs across them."""
    cos = matrix @ matrix.T
    same = table["level1"].to_numpy()[:, None] == table["level1"].to_numpy()[None, :]
    off_diag = ~np.eye(len(table), dtype=bool)
    rows = []
    for name, idx in table.groupby("level1").indices.items():
        if len(idx) < 2:
            continue
        others = np.setdiff1d(np.arange(len(table)), idx)
        within = cos[np.ix_(idx, idx)][~np.eye(len(idx), dtype=bool)].mean()
        rows.append({"level-1 category": name, "categories": len(idx),
                     "mean cos within": round(float(within), 3),
                     "mean cos to others": round(float(cos[np.ix_(idx, others)].mean()), 3)})
    per_level1 = pd.DataFrame(rows).set_index("level-1 category")
    per_level1["gap"] = per_level1["mean cos within"] - per_level1["mean cos to others"]
    overall = pd.Series({
        "mean cos, same level-1": round(float(cos[same & off_diag].mean()), 3),
        "mean cos, different level-1": round(float(cos[~same].mean()), 3),
    })
    return per_level1.sort_values("gap"), overall


def nearest_neighbours(table, matrix, names=EXAMPLE_CATEGORIES, k=5):
    """Top-k most similar used categories for a few example categories (qualitative check)."""
    cos = matrix @ matrix.T
    rows = []
    for name in names:
        idx = np.flatnonzero(table["category_name"].to_numpy() == name)
        if len(idx) == 0:
            continue
        i = idx[0]
        order = [j for j in np.argsort(-cos[i]) if j != i][:k]
        for rank, j in enumerate(order, 1):
            rows.append({"query": table.at[i, "full_category_path"], "rank": rank,
                         "neighbour": table.at[j, "full_category_path"],
                         "cosine": round(float(cos[i, j]), 3)})
    return pd.DataFrame(rows)


def near_duplicate_pairs(table, matrix, threshold=0.95):
    """Pairs of different categories whose embeddings are almost identical."""
    cos = matrix @ matrix.T
    i, j = np.where(np.triu(cos, k=1) >= threshold)
    pairs = pd.DataFrame({"category A": table["full_category_path"].to_numpy()[i],
                          "category B": table["full_category_path"].to_numpy()[j],
                          "cosine": cos[i, j].round(3)})
    return pairs.sort_values("cosine", ascending=False).reset_index(drop=True)


def behaviour_alignment(checkins, table, matrix, min_users=20):
    """
    Does text similarity agree with behaviour?

    Behavioural similarity of two categories = cosine between their user-visit vectors
    (1 if the user checked in at that category at least once). Categories visited by
    fewer than ``min_users`` users are left out as too noisy. Reported as Spearman
    correlations over all category pairs:

    - embedding cosine vs behaviour: does the embedding carry behavioural signal?
    - same-level-1 indicator vs behaviour: what the taxonomy alone already explains
    - embedding cosine vs behaviour within / across level-1 pairs: does the embedding
      add information beyond the top-level taxonomy?

    Caveat: very common categories (e.g. Home) co-occur with everything; the binary
    cosine reduces but does not remove that popularity effect.
    """
    from scipy.stats import spearmanr

    visits = pd.crosstab(checkins["userId"], checkins["categoryId"]) > 0
    visits = visits.loc[:, visits.sum() >= min_users]
    position = pd.Series(np.arange(len(table)), index=table["category_id"])
    shared = [c for c in visits.columns if c in position.index]
    visits = visits[shared].to_numpy(dtype=float)
    emb = matrix[position[shared].to_numpy()]
    level1 = table.set_index("category_id").loc[shared, "level1"].to_numpy()

    norms = np.linalg.norm(visits, axis=0)
    behaviour = (visits.T @ visits) / np.outer(norms, norms)
    text = emb @ emb.T
    i, j = np.triu_indices(len(shared), k=1)
    beh, txt = behaviour[i, j], text[i, j]
    same = (level1[i] == level1[j]).astype(float)

    def rho(a, b):
        return round(float(spearmanr(a, b).statistic), 3)

    return pd.Series({
        "categories compared": len(shared),
        "category pairs": len(i),
        "rho(embedding cosine, behaviour)": rho(txt, beh),
        "rho(same level-1, behaviour)": rho(same, beh),
        "rho(embedding, behaviour) within same level-1": rho(txt[same == 1], beh[same == 1]),
        "rho(embedding, behaviour) across level-1": rho(txt[same == 0], beh[same == 0]),
    })


# ------------------------------------------------------------------------ figures

COLORS = {
    "series_1": "#2a78d6",
    "series_2": "#eb6834",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "grid": "#e5e4e0",
    "surface": "#fcfcfb",
}


def _new_axes(width=7.0, height=4.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height), facecolor=COLORS["surface"])
    ax.set_facecolor(COLORS["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["grid"])
    ax.tick_params(colors=COLORS["text_secondary"], labelsize=9)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _finish(fig, ax, title, xlabel, ylabel, path):
    ax.set_title(title, loc="left", color=COLORS["text"], fontsize=11)
    ax.set_xlabel(xlabel, color=COLORS["text_secondary"], fontsize=9)
    ax.set_ylabel(ylabel, color=COLORS["text_secondary"], fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_user_activity(per_user, path, caps=(MIN_USER_POIS, 100, 200)):
    fig, ax = _new_axes()
    values = per_user["distinct_pois"]
    bins = np.logspace(np.log10(max(values.min(), 1)), np.log10(values.max()), 40)
    ax.hist(values, bins=bins, color=COLORS["series_1"], edgecolor=COLORS["surface"], linewidth=1)
    ax.set_xscale("log")
    from matplotlib.ticker import ScalarFormatter

    ax.xaxis.set_major_formatter(ScalarFormatter())
    top = ax.get_ylim()[1]
    for cap in caps:
        ax.axvline(cap, color=COLORS["text_secondary"], linewidth=1, linestyle="--")
        ax.text(cap * 1.03, top * 0.95, str(cap), color=COLORS["text_secondary"], fontsize=8,
                va="top")
    _finish(fig, ax, "Distinct POIs per user (dashed: min filter and max_p_num options)",
            "distinct POIs (log scale)", "users", path)


def plot_venue_popularity(per_venue, path):
    fig, ax = _new_axes()
    counts = np.sort(per_venue["checkins"].to_numpy())
    share_at_least = 1 - np.arange(len(counts)) / len(counts)
    ax.plot(counts, share_at_least, color=COLORS["series_1"], linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _finish(fig, ax, "Venue popularity: share of venues with at least x check-ins",
            "check-ins per venue (log)", "share of venues (log)", path)


def plot_temporal_profile(shares, path):
    fig, ax = _new_axes()
    for column, color in (("weekday", COLORS["series_1"]), ("weekend", COLORS["series_2"])):
        if column not in shares:
            continue
        ax.plot(shares.index, shares[column], color=color, linewidth=2, marker="o",
                markersize=3, label=column)
        ax.text(shares.index[-1] + 0.3, shares[column].iloc[-1], column, color=COLORS["text"],
                fontsize=8, va="center")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False, fontsize=8, labelcolor=COLORS["text"])
    _finish(fig, ax, "Check-ins by local hour (share within weekdays / weekends)",
            "local hour", "share of check-ins (%)", path)


def plot_granularity(granularity, path):
    fig, ax = _new_axes()
    levels = granularity.index.to_numpy()
    width = 0.38
    bars = (("distinct nodes", COLORS["series_1"], -width / 2),
            ("effective number", COLORS["series_2"], width / 2))
    for column, color, shift in bars:
        ax.bar(levels + shift, granularity[column], width=width - 0.04, color=color, label=column)
    for level, value in zip(levels, granularity["distinct nodes"]):
        ax.text(level - width / 2, value, f"{value}", ha="center", va="bottom", fontsize=8,
                color=COLORS["text_secondary"])
    ax.set_xticks(levels)
    ax.legend(frameon=False, fontsize=8, labelcolor=COLORS["text"])
    _finish(fig, ax, "Categories left when paths are cut at each level",
            "path level (6 = leaf categories)", "categories", path)


def plot_top_categories(top, path, k=15):
    fig, ax = _new_axes(width=8, height=5)
    data = top.head(k).iloc[::-1]
    # leaf name with its top-level category: the middle levels are what truncation would cut
    parts = data["full_category_path"].str.split(" > ")
    labels = [f"{p[-1]}  ({p[0]})" if len(p) > 1 else p[0] for p in parts]
    ax.barh(labels, data["check-ins (%)"], color=COLORS["series_1"], height=0.7)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    for y, value in enumerate(data["check-ins (%)"]):
        ax.text(value, y, f" {value:.1f}", va="center", fontsize=8, color=COLORS["text_secondary"])
    _finish(fig, ax, f"Top {k} leaf categories by share of check-ins", "share of check-ins (%)",
            "", path)


# ------------------------------------------------------------------------- report

class EDAReport:
    """Collects sections, prints them, and saves them as one markdown file."""

    def __init__(self, title):
        self.lines = [f"# {title}", ""]

    def add(self, heading, content, note=None):
        text = content.to_string() if isinstance(content, (pd.DataFrame, pd.Series)) else str(content)
        print(f"\n=== {heading}\n{text}")
        self.lines += [f"### {heading}", ""]
        if note:
            self.lines += [note, ""]
        self.lines += ["```", text, "```", ""]

    def section(self, title):
        print(f"\n\n######## {title}")
        self.lines += [f"## {title}", ""]

    def save(self, path):
        Path(path).write_text("\n".join(self.lines), encoding="utf-8")


def load_eda_inputs(city="NYC"):
    """Taxonomy, check-ins with paths, and the venue table with paths."""
    taxonomy = load_category_taxonomy()
    checkins = attach_category_paths(load_checkins(city), taxonomy)
    venues = attach_category_paths(build_venue_table(checkins), taxonomy)
    return taxonomy, checkins, venues


def run_eda(city="NYC", out_dir=None, embedding_path=EMBEDDING_FILE):
    """Run every analysis for one city; write the summary and figures to out_dir."""
    out_dir = Path(out_dir) if out_dir else REPORT_DIR / city.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    taxonomy, checkins, venues = load_eda_inputs(city)
    report = EDAReport(f"EDA: Foursquare TSMC2014 {city}")

    report.section("1. Data integrity")
    report.add("Overview", dataset_overview(checkins))
    join_summary, unmatched = category_join_coverage(checkins)
    report.add("Category join coverage", join_summary)
    report.add("Category IDs without a path", unmatched)
    consistency, examples = venue_category_consistency(checkins)
    report.add("Venues with several category IDs", consistency,
               note="The venue table keeps each venue's most frequent category ID.")
    report.add("Examples", examples)
    report.add("Venue coordinate spread", venue_coordinate_spread(checkins),
               note="The venue table uses the median coordinate.")
    report.add("UTC offsets (minutes)", timezone_offsets(checkins))

    report.section("2. Interaction sparsity")
    per_user, user_distribution, user_thresholds = user_activity(checkins)
    report.add("Per-user activity", user_distribution)
    report.add("Users affected by TEMN settings", user_thresholds)
    per_venue, popularity = venue_popularity(checkins)
    report.add("Venue popularity", popularity)
    plot_user_activity(per_user, out_dir / "user_activity.png")
    plot_venue_popularity(per_venue, out_dir / "venue_popularity.png")

    report.section("3. Chronological split (paper protocol)")
    split = chronological_split(first_visit_sequences(checkins))
    report.add("Users and pairs per split",
               split.groupby("split").agg(users=("userId", "nunique"), pairs=("venueId", "size")))
    report.add("Cold POIs among held-out POIs", split_cold_start_rates(split))
    headroom, unseen = category_headroom(split, venues, taxonomy)
    report.add("Category headroom: is the test POI's category already in the user's history?",
               headroom)
    report.add("Unseen categories in test", unseen)

    report.section("4. Time")
    _, hour_shares, slot_summary = temporal_profile(checkins)
    report.add("TLDA time slots", slot_summary)
    plot_temporal_profile(hour_shares, out_dir / "temporal_profile.png")

    report.section("5. Category paths")
    report.add("Depth: taxonomy vs data", taxonomy_coverage(taxonomy, venues))
    granularity = granularity_by_level(taxonomy, venues)
    report.add("Granularity when cutting paths at each level", granularity)
    plot_granularity(granularity, out_dir / "granularity_by_level.png")
    ambiguity_summary, ambiguous = name_id_ambiguity(checkins)
    report.add("Category names vs IDs", ambiguity_summary)
    report.add("Names with several IDs", ambiguous)
    _, buckets, top = category_frequency(venues, checkins)
    report.add("Frequency buckets (RQ3)", buckets)
    report.add("Top leaf categories", top)
    plot_top_categories(top, out_dir / "top_categories.png")
    report.add("Top-level categories", level1_distribution(venues))
    report.add("Per-user category diversity", user_category_diversity(checkins))

    report.section("6. Category embeddings")
    if Path(embedding_path).exists():
        table, matrix, coverage = used_category_embeddings(venues, taxonomy, embedding_path)
        report.add("Coverage", coverage)
        per_level1, overall = level1_separation(table, matrix)
        report.add("Taxonomy separation (cosine)", overall)
        report.add("Per top-level category", per_level1)
        report.add("Nearest neighbours", nearest_neighbours(table, matrix))
        report.add("Near-duplicate embeddings (cosine >= 0.95)", near_duplicate_pairs(table, matrix))
        report.add("Embedding similarity vs behaviour", behaviour_alignment(checkins, table, matrix))
    else:
        report.add("Embeddings", f"not found at {embedding_path}; run src/path_to_embedding.py")

    report.save(out_dir / "eda_summary.md")
    print(f"\nsaved summary and figures to {out_dir}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--city", default="NYC")
    parser.add_argument("--out", default=None, help="output directory")
    args = parser.parse_args()
    run_eda(args.city, args.out)


if __name__ == "__main__":
    main()

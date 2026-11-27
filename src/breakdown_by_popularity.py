"""
Compare two logged runs pair by pair, broken down by how popular the held-out POI is
(read-only; prints numbers for the report).

    python src/breakdown_by_popularity.py <reference_run_id> <compared_run_id>
    e.g. python src/breakdown_by_popularity.py temn_20261126-173735 temn_20261126-191602

Popularity of a POI = number of distinct users who visited it in training (column 1 of the data file).
Per bucket: number of pairs, NDCG@10 of both runs, the difference, and the share of pairs where the
compared run ranks the held-out POI higher / lower than the reference run.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BASE_PATH = Path(__file__).resolve().parent.parent
POPULARITY_BUCKETS = [(1, 1, "1"), (2, 4, "2-4"), (5, 19, "5-19"), (20, 10**9, "20+")]

reference_id, compared_id = sys.argv[1], sys.argv[2]
runs = {json.loads(line)["run_id"]: json.loads(line)
        for line in open(BASE_PATH / "results" / "runs.jsonl", encoding="utf-8")}
reference_run, compared_run = runs[reference_id], runs[compared_id]

# both runs must score the same groups (same data file and seed = same test negatives), or they are not paired
for setting in ("data_dir", "data_file", "seed"):
    assert reference_run["settings"][setting] == compared_run["settings"][setting], f"runs differ in {setting}"
data_dir = BASE_PATH / reference_run["settings"]["data_dir"]
data_lines = open(data_dir / reference_run["settings"]["data_file"]).read().splitlines()

# training popularity: column 1 holds each user's training check-ins; count each user once per POI
training_users_per_poi = Counter()
for line in data_lines:
    training_users_per_poi.update({checkin[0] for checkin in json.loads(line.split("\t")[1])})

# held-out positives in the order process_fun builds the groups (only POIs in poi2_xy are evaluated, N7)
evaluated_pois = {int(line.split("\t")[0]) for line in open(data_dir / "poi2_xy")}
held_out_pois = [checkin[0] for line in data_lines for checkin in json.loads(line.split("\t")[2])
                 if checkin[0] in evaluated_pois]

reference_ranks = np.loadtxt(BASE_PATH / reference_run["ranks_file"].replace("\\", "/"))
compared_ranks = np.loadtxt(BASE_PATH / compared_run["ranks_file"].replace("\\", "/"))
assert len(held_out_pois) == len(reference_ranks) == len(compared_ranks), "pairs do not line up"
poi_popularity = np.array([training_users_per_poi[poi] for poi in held_out_pois])


def ndcg_at_10(ranks):
    # one positive per group: 1 / log2(rank + 2) if it is in the top 10, else 0 (rank is 0-based)
    return np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0.0)


print(f"reference: {reference_id} (NDCG@10 {reference_run['metrics']['NDCG@10']})")
print(f"compared:  {compared_id} (NDCG@10 {compared_run['metrics']['NDCG@10']})")
print(f"{'training users':>14s} {'pairs':>6s} {'share':>6s} {'ref':>6s} {'comp':>6s} {'diff':>7s} "
      f"{'comp better':>11s} {'comp worse':>10s}")
for low, high, label in POPULARITY_BUCKETS + [(1, 10**9, "all")]:
    in_bucket = (poi_popularity >= low) & (poi_popularity <= high)
    reference_ndcg = ndcg_at_10(reference_ranks[in_bucket]).mean()
    compared_ndcg = ndcg_at_10(compared_ranks[in_bucket]).mean()
    better = (compared_ranks[in_bucket] < reference_ranks[in_bucket]).mean()  # lower rank = held-out POI higher up
    worse = (compared_ranks[in_bucket] > reference_ranks[in_bucket]).mean()
    print(f"{label:>14s} {in_bucket.sum():6d} {in_bucket.mean():6.1%} {reference_ndcg:6.3f} {compared_ndcg:6.3f} "
          f"{compared_ndcg - reference_ndcg:+7.3f} {better:11.1%} {worse:10.1%}")

# Inference: Here are 101 places user has not visited, rank them in order, most likely first: one of the 101 is possitive, the other 100 are randomly
# sample palces user has not visited

# For each entry: (user, place, distance fromt he users center, history)

# Recomment when it is popular, ignore the user completely.
#
# Usage (same groups as TEMN as long as the file and the seed are the same):
#   python src/popularity_based.py --data_dir data/processed/nyc --data_file data_nyc_val
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BASE_PATH = Path(__file__).resolve().parent.parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

import src.process_TEMN_data as process_TEMN_data  # noqa: E402
import src.run_log as run_log  # noqa: E402


def popularity_scores(train_examples):
    """
    popularity(place) = number of distinct users with that place in their training data.

    process_fun makes one training example (user, place, distance, history, topic) per
    distinct (user, place) pair, so counting the places of the examples counts distinct
    users. Only training data is used: counting the val/test period would leak the future.
    """
    return Counter(example[1] for example in train_examples)


def rank_with_random_ties(scores, rng):
    """
    Rank (0 = first) of the positive, which is scores[0], among all scores.

    Places with a higher score are always above it; among places with the same score the
    order is random, so ties are neither won (optimistic) nor lost (pessimistic) on purpose.
    """
    positive = scores[0]
    others = np.asarray(scores[1:])
    higher = int((others > positive).sum())
    tied = int((others == positive).sum())
    return higher + int(rng.integers(0, tied + 1))


def hit_and_ndcg(ranks, k):
    """One positive per group: HR@k = rank < k, NDCG@k = 1 / log2(rank + 2) if rank < k."""
    ranks = np.asarray(ranks)
    hit = ranks < k
    return hit.mean(), np.where(hit, 1.0 / np.log2(ranks + 2), 0.0).mean()


def main():
    parser = argparse.ArgumentParser(description="Global popularity baseline on TEMN's test groups")
    parser.add_argument("--data_dir", default=str(BASE_PATH / "data/processed"))
    parser.add_argument("--data_file", default="data_nyc_val")
    parser.add_argument("--seed", type=int, default=2026)  # must equal TEMN's seed: same negatives
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    train, _, _, _, groups, _ = process_TEMN_data.process_fun(
        args.data_file, data_dir=data_dir, seed=args.seed)
    popularity = popularity_scores(train)
    rng = np.random.default_rng(args.seed)

    ranks, optimistic, pessimistic = [], [], []
    for group in groups:
        # each entry is (user, place, distance, history); the positive is entry 0
        scores = np.array([popularity[entry[1]] for entry in group])
        ranks.append(rank_with_random_ties(scores, rng))
        optimistic.append(int((scores[1:] > scores[0]).sum()))    # wins every tie
        pessimistic.append(int((scores[1:] >= scores[0]).sum()))  # loses every tie

    out_path = data_dir / f"top_k_label_popularity_{args.data_file}"
    np.savetxt(out_path, ranks, fmt="%d")  # per-pair ranks, same order as TEMN's, for paired tests

    print(f"\npopularity on {args.data_file}: {len(ranks)} held-out pairs, "
          f"each ranked among 1 + {len(groups[0]) - 1}")
    for k in (5, 10):
        hr, ndcg = hit_and_ndcg(ranks, k)
        print(f"HR@{k}: {hr:.4f}   NDCG@{k}: {ndcg:.4f}")
    print("tie check, HR@10 optimistic / random / pessimistic: "
          f"{hit_and_ndcg(optimistic, 10)[0]:.4f} / {hit_and_ndcg(ranks, 10)[0]:.4f} / "
          f"{hit_and_ndcg(pessimistic, 10)[0]:.4f}")

    run_log.log_run(run_log.new_run_id("popularity"), "popularity",
                    {**vars(args), "ties": "random", "n_candidates": len(groups[0])}, ranks,
                    extra={"HR@10_optimistic_ties": round(hit_and_ndcg(optimistic, 10)[0], 4),
                           "HR@10_pessimistic_ties": round(hit_and_ndcg(pessimistic, 10)[0], 4)})


if __name__ == "__main__":
    main()

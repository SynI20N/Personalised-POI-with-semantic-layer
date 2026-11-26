#encoding:utf-8
"""
xz331@cam.ac.uk
abnerzxzhao@tencent.com
"""
# Trains TEMN on one data file and evaluates it on the held-out POIs of that file.
#   1. process_fun reads the data file -> training examples + test groups (1 positive + 100 negatives)
#   2. build the TEMN graph (src/TEMN.py) and train it with early stopping (--max_epochs, --eval_every,
#      --patience, --min_delta), fresh negatives every epoch; the best epoch's weights are used for testing
#   3. score every test group, write the rank of the positive, print HR@5/10 and NDCG@5/10
#   4. append the run (settings, metrics, per-pair ranks) to results/runs.jsonl
#
# [CHANGED C4] upstream: `import TEMN, random, tensorflow as tf, process_TEMN_data` with a fixed data
# file "sample_data" in the working directory, no command-line options, no seeds.

import sys
from pathlib import Path

# Project root (the folder that contains "src"), independent of the current working directory
BASE_PATH = Path(__file__).resolve().parent.parent
SRC_PATH = BASE_PATH / "src"
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

import src.TEMN as TEMN
import argparse
import random
import numpy as np
import tensorflow.compat.v1 as tf  # [CHANGED C1] upstream: import tensorflow as tf (TF 1.12)
tf.disable_v2_behavior()
import src.process_TEMN_data as process_TEMN_data
import src.run_log as run_log  # [CHANGED C13] new: run log in results/runs.jsonl

# Example for NYC (the folder must also hold poi2_xy and tlda_model_file):
#   python src/main_TEMN.py --data_dir data/processed/nyc --data_file data_nyc_val
# Any hyperparameter of `class args` below can be overridden, e.g. --eta 0.2 --ratio2 0.1
parser = argparse.ArgumentParser(description="Train TEMN and rank each held-out POI among 100 negatives")
parser.add_argument("--data_dir", default=str(SRC_PATH))
parser.add_argument("--data_file", default="sample_data")

# [CHANGED C17] early stopping on the groups of --data_file instead of a fixed number of epochs
# (upstream: 10 epochs, fixed). Run it on the val file: on the test file it would stop on test data.
parser.add_argument("--max_epochs",type=int, default=150)
parser.add_argument("--eval_every", type=int,default=5)
parser.add_argument("--patience", type=int,default=3) # stop after n evaluation in a row with no improvement
parser.add_argument("--min_delta",type=float, default=0.001) # smaller gain than this does not count

# [CHNAGE C22] Naive uniform sampling of negatives during training causes pairing of easy negatvive POI, the harder ones
# are those who are popular. 
parser.add_argument("--neg_alpha", type=float, default=0.5) # 0.5 best NYC val run; 0 = uniform as in the paper

# [NOTE] the seed also fixes the 100 test negatives of every group (process_fun). Pass the same --seed
# to every model you compare (popularity_based.py has the same default), or they are not paired.
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--distance", default="log1p", choices=["log1p", "km"])  # [CHANGED C7]
cli, hyperparameter_overrides = parser.parse_known_args()  # the rest is checked after `class args`
random.seed(cli.seed)          # order of the training examples
tf.set_random_seed(cli.seed)   # parameter initialisation
DATA_DIR = Path(cli.data_dir)
process_TEMN_data.DISTANCE = cli.distance

# [CHANGED C10] upstream: process_fun("sample_data") returned 5 values and the data were shuffled once
# here (random.shuffle(all_data)); the training negatives were fixed inside the data. Now the 6th value
# is the NegativeSampler, used once per epoch below.
all_data, user_num, poi_num, topic_k, all_test_data, neg_sampler = process_TEMN_data.process_fun(
    cli.data_file, data_dir=DATA_DIR, seed=cli.seed
)

neg_sampler.set_popularity_weights(cli.neg_alpha) # CHNAGED C22

num_users = user_num
num_items = poi_num
topic_num = topic_k
    # paper: A.5 args:
"""
lr = 0.005
l2 reg = 1e-4
margin = 0.2
TLDA topics: 10
Gibss iterations: not stated, intil it converges
tpoic weight: 0.2
ϵ, η, d, h = 0.1,0.4,50,10
validation split and tunning on NDCGL yes
Metrics: HR@k, NDCG@k
"""

# [CHANGED C14] hyperparameters moved from the values in the authors' script to the paper's A.5 values.
# "upstream" = value in the authors' main_TEMN.py. [NOTE N4]: std, rnn_type, margin and dropout are not
# used by the model (lamb_m / lamb_d are the margins actually used).
class args:
    std = 0.1                 # unused
    num_mem = 10              # h, number of memory slots (paper: 10)
    embedding_size = 50       # d (paper: 50)
    constraint = True         # clip embeddings to the unit ball (see [CHANGED C20]/[CHANGED C21] in TEMN.py)
    rnn_type = set(['PAIR'])  # unused
    margin = 0.2              # unused; upstream 0.1
    topic_num = topic_num     # from the TLDA file
    l2_reg = 0.0001           # [CHANGED C14] upstream 0.00001; paper 0.0001
    opt = 'SGD'
    clip_norm = 2             # gradient clipping (value then norm), see ClipIfNotNone in TEMN.py
    dropout = 0.7             # unused (no dropout op in the graph)

    learn_rate = 0.01       # 0.01 best fozen setting
    max_p_num = 200           # [CHANGED C14] upstream 100; longest history used for p_u (6 NYC / 22 Tokyo users have more)
    stddev = 0.1              # range of the uniform initialiser
    # memory and geographic margin
    lamb_m = 0.2              # [CHANGED C14] upstream 0.1; paper lambda_m = 0.2
    lamb_d = 0.2              # [CHANGED C14] upstream 0.1; paper lambda_sigma = 0.2
    # topic weight
    ratio1 = 0.2              # [CHANGED C14] upstream 0.1; paper varsigma = 0.2
    # geographic loss weight
    ratio2 = 0.1              # paper epsilon = 0.1 (same as upstream)
    init_method = "uniform"   # [CHANGED C14] upstream "normal"; paper: uniform initialisation
    eta = 0.4 # paper uses 0.4  / geographic module weight parameter   [CHANGED C5/C14] new: weight of s_geo in Eq. 19


ar = args()

# [CHANGED C14] new: command-line overrides.
# Command-line overrides: one --name option per setting of `class args`, typed like its default
# (topic_num is not a choice: it comes from the TLDA file). A misspelled name is an error.
override_parser = argparse.ArgumentParser()
for name, value in vars(args).items():
    if name.startswith("_") or name == "topic_num" or not isinstance(value, (int, float, str)):
        continue
    kind = (lambda s: s.lower() in ("1", "true", "yes")) if isinstance(value, bool) else type(value)
    override_parser.add_argument(f"--{name}", type=kind, default=value)
for name, value in vars(override_parser.parse_args(hyperparameter_overrides)).items():
    setattr(ar, name, value)
hyperparameters = {name: getattr(ar, name) for name in dir(ar) if not name.startswith("_")}
print("settings:", vars(cli), hyperparameters)

user_num = num_users
item_num = num_items

model = TEMN.TEMN(user_num,item_num,ar)
saver = tf.train.get_or_create_global_step()  # (upstream name; it is a step counter, not a Saver)
init = tf.global_variables_initializer()
# [CHANGED C17] keeps the weights of the best epoch. Created after the model, because a Saver only
# covers the variables that exist when it is created.
best_saver = tf.train.Saver(max_to_keep=1)

# Launch the graph.
sess = tf.Session()
sess.run(init)

print("build finish")


def get_topk(scores):
    # 0-based rank of the positive (scores[0]) among the group, highest score first.
    # [NOTE N8] .index() returns the first equal score, so a tie is counted as a win for the positive.
    # Irrelevant for TEMN's float scores; popularity_based.py breaks ties at random instead.
    f = scores[0]
    sort_scores = sorted(scores, reverse=True)
    cur_index = sort_scores.index(f)
    return cur_index


# [CHANGED C17] rank of the positive in every group, used by early stopping
def evaluate(groups):
    ranks = []
    for group in groups:
        scores = sess.run(model.predict_op, model.get_list_feed_dict(group, ""))
        ranks.append(get_topk(scores))
    return np.array(ranks)


n_sample = len(all_data)
n_sample_test = len(all_test_data)
batch_size = 64
batch_num = int((n_sample + batch_size - 1)/batch_size)
batch_num_test = int((n_sample_test + batch_size - 1)/batch_size)

print("batch_num", batch_num)
print("batch_num_test", batch_num_test)

run_id = run_log.new_run_id("temn")
epoch_costs = []  # [CHANGED C13] mean total cost per epoch: shows whether training had converged

# [CHANGED C17] early-stopping state; the best weights are saved in results/checkpoints/<run_id>/
checkpoint_dir = run_log.RESULTS_DIR / "checkpoints" / run_id
checkpoint_dir.mkdir(parents=True, exist_ok=True)  # the Saver does not create folders
checkpoint_path = str(checkpoint_dir / "best")
val_curve = []  # metrics at every evaluation
best_ndcg, best_epoch, evals_without_gain = -1.0, 0, 0

for epoch in range(1, cli.max_epochs + 1):  # [CHANGED C17] upstream: for it in range(10)
    # [CHANGED C10] upstream trained on `all_data` (shuffled once, each example with its fixed negative).
    # Now each epoch: draw a new negative for every example (uniform, never a POI the user visited in
    # training), then reshuffle.
    # fresh negatives (never a POI the user visited) and a new order every epoch
    epoch_data = neg_sampler.add_negatives(all_data)
    random.shuffle(epoch_data)
    batch_costs = []
    for i in range(batch_num):
        beg = i*batch_size
        end = min((i + 1)*batch_size, n_sample)
        cur_train_data = epoch_data[beg:end]
        feed_dict = model.get_list_feed_dict(cur_train_data)
        cost = sess.run([model.cost, model.dist_cost, model.mem_cost, model.topic_cost, model.train_op], feed_dict)
        batch_costs.append(float(cost[0]))
        if i % 50 == 0:
            print(epoch, i, cost[0])
    epoch_costs.append(round(float(np.mean(batch_costs)), 4))
    print(f"epoch {epoch}: mean cost {epoch_costs[-1]}")

    # [CHANGED C17] early stopping: every eval_every epochs (and after the last one) score the groups,
    # save the weights if NDCG@10 improved by more than min_delta, stop after `patience` evaluations
    # in a row without such a gain
    if epoch % cli.eval_every == 0 or epoch == cli.max_epochs:
        val_metrics = run_log.ranking_metrics(evaluate(all_test_data))
        val_curve.append({"epoch": epoch, **val_metrics})
        if val_metrics["NDCG@10"] > best_ndcg + cli.min_delta:
            best_ndcg, best_epoch, evals_without_gain = val_metrics["NDCG@10"], epoch, 0
            best_saver.save(sess, checkpoint_path)
        else:
            evals_without_gain += 1
        print(f"epoch {epoch}: {val_metrics} | best NDCG@10 {best_ndcg} at epoch {best_epoch}")
        if evals_without_gain >= cli.patience:
            print(f"early stop at epoch {epoch}")
            break

# [CHANGED C17] test with the weights of the best epoch, not those of the last epoch
best_saver.restore(sess, checkpoint_path)
print(f"restored the weights of epoch {best_epoch}")


# Test
print("start testing")
# [CHANGED C4] upstream wrote "top_k_label" and "test_scores" in the working directory; now named
# after the data file, next to it (a later run on the same file overwrites them).
f_out = open(DATA_DIR / f"top_k_label_{cli.data_file}", "w")
f_o = open(DATA_DIR / f"test_scores_{cli.data_file}", "w")
ii_x = 0
ranks = []  # 0-based rank of the true POI among its group (1 positive + negatives)

for t_data in all_test_data:
    cur_test_data = t_data
    feed_dict_test = model.get_list_feed_dict(cur_test_data, "")
    scores = sess.run(model.predict_op, feed_dict_test)
    cur_pos = get_topk(scores)
    ranks.append(cur_pos)
    f_out.write(str(cur_pos) + "\n")
    for j in range(len(cur_test_data)):
        cur_u = cur_test_data[j][0]
        cur_i = cur_test_data[j][1]
        cur_s = scores[j]
        r_line = str(cur_u) + "\t" + str(cur_i) + "\t" + str(cur_s)
        f_o.write(r_line + "\n")
    ii_x += 1
    if ii_x % 1000 == 0:
        print(ii_x)

f_o.close()
f_out.close()

# [CHANGED C13] new: upstream only wrote the ranks to a file and computed no metric.
# One positive per group: HR@k = rank < k, NDCG@k = 1 / log2(rank + 2) if rank < k else 0
ranks = np.array(ranks)
print(f"\n{cli.data_file}: {len(ranks)} held-out pairs, each ranked among 1 + {len(all_test_data[0]) - 1}")
for k in (5, 10):
    hit = ranks < k
    ndcg = np.where(hit, 1.0 / np.log2(ranks + 2), 0.0)
    print(f"HR@{k}: {hit.mean():.4f}   NDCG@{k}: {ndcg.mean():.4f}")

run_log.log_run(run_id, "TEMN", {**vars(cli), "hyperparameters": hyperparameters,
                                 "n_candidates": len(all_test_data[0])},
                ranks, extra={"epoch_costs": epoch_costs, "val_curve": val_curve, "best_epoch": best_epoch})

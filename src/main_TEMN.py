#encoding:utf-8
"""
xz331@cam.ac.uk
abnerzxzhao@tencent.com
"""

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
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import src.process_TEMN_data as process_TEMN_data
import src.run_log as run_log

# Example for NYC (the folder must also hold poi2_xy and tlda_model_file):
#   python src/main_TEMN.py --data_dir data/processed/nyc --data_file data_nyc_val
# Any hyperparameter of `class args` below can be overridden, e.g. --eta 0.2 --ratio2 0.1
parser = argparse.ArgumentParser(description="Train TEMN and rank each held-out POI among 100 negatives")
parser.add_argument("--data_dir", default=str(SRC_PATH))
parser.add_argument("--data_file", default="sample_data")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--distance", default="log1p", choices=["log1p", "km"])
cli, hyperparameter_overrides = parser.parse_known_args()  # the rest is checked after `class args`
random.seed(cli.seed)
tf.set_random_seed(cli.seed)
DATA_DIR = Path(cli.data_dir)
process_TEMN_data.DISTANCE = cli.distance

all_data, user_num, poi_num, topic_k, all_test_data, neg_sampler = process_TEMN_data.process_fun(
    cli.data_file, data_dir=DATA_DIR, seed=cli.seed
)

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

class args:
    std = 0.1
    num_mem = 10
    embedding_size = 50
    constraint = True
    rnn_type = set(['PAIR'])
    margin = 0.2
    topic_num = topic_num
    l2_reg = 0.0001
    opt = 'SGD'
    clip_norm = 2
    dropout = 0.7

    learn_rate = 0.005
    max_p_num = 200
    stddev = 0.1
    # memory and geographic margin
    lamb_m = 0.2
    lamb_d = 0.2
    # topic weight
    ratio1 = 0.2
    # geographic loss weight
    ratio2 = 0.1
    init_method = "uniform"
    eta = 0.4 # paper uses 0.4  / geographic module weight parameter


ar = args()

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
saver = tf.train.get_or_create_global_step()
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

print("build finish")

n_sample = len(all_data)
n_sample_test = len(all_test_data)
batch_size = 64
batch_num = int((n_sample + batch_size - 1)/batch_size)
batch_num_test = int((n_sample_test + batch_size - 1)/batch_size)

print("batch_num", batch_num)
print("batch_num_test", batch_num_test)

Iter = cli.epochs
run_id = run_log.new_run_id("temn")
epoch_costs = []  # mean total cost per epoch: shows whether training had converged

for it in range(Iter):
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
            print(it, i, cost[0])
    epoch_costs.append(round(float(np.mean(batch_costs)), 4))
    print(f"epoch {it}: mean cost {epoch_costs[-1]}")


# Test
print("start testing")
def get_topk(scores):
    f = scores[0]
    sort_scores = sorted(scores, reverse=True)
    cur_index = sort_scores.index(f)
    return cur_index


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

# One positive per group: HR@k = rank < k, NDCG@k = 1 / log2(rank + 2) if rank < k else 0
ranks = np.array(ranks)
print(f"\n{cli.data_file}: {len(ranks)} held-out pairs, each ranked among 1 + {len(all_test_data[0]) - 1}")
for k in (5, 10):
    hit = ranks < k
    ndcg = np.where(hit, 1.0 / np.log2(ranks + 2), 0.0)
    print(f"HR@{k}: {hit.mean():.4f}   NDCG@{k}: {ndcg.mean():.4f}")

run_log.log_run(run_id, "TEMN", {**vars(cli), "hyperparameters": hyperparameters,
                                 "n_candidates": len(all_test_data[0])},
                ranks, extra={"epoch_costs": epoch_costs})
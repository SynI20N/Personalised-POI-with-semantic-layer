#encoding:utf-8
"""
xz331@cam.ac.uk
abnerzxzhao@tencent.com
"""
# Trains TLDA (src/TLDA.py) on the training check-ins and saves each user's topic mixture.
# Document = one user; word = one check-in (POI id), with its time slot as the temporal context.
# The output (tlda_model_file) gives the labels of TEMN's topic loss (Eq. 12-14).
#
# Semantic TEMN, change 2: with --word_map the POI id is replaced by the POI's k-means cluster id when
# building cur_item below, so topics become mixtures of KINDS of places that exist in every city.
#
# [CHANGED C4] upstream: fixed input "sample_data" and output "tlda_model_file", no options.

import sys
from pathlib import Path

# Project root (the folder that contains "src"), independent of the current working directory
BASE_PATH = Path(__file__).resolve().parent.parent
SRC_PATH = BASE_PATH / "src"
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

import src.TLDA as TLDA
import argparse
import json
import random

import numpy as np

# Example for NYC (writes tlda_model_file next to the data, where process_fun looks for it):
#   python src/main_tlda.py --data data/processed/nyc/data_nyc_val --out data/processed/nyc/tlda_model_file
parser = argparse.ArgumentParser(description="Train TLDA on column 1 (training items) of a data file")
parser.add_argument("--data", default=str(SRC_PATH / "sample_data"))
parser.add_argument("--out", default=str(SRC_PATH / "tlda_model_file"))
# [CHANGED C15] upstream: topic_k = 7 (fixed). Paper A.5: 10 patterns.
parser.add_argument("--topics", type=int, default=10)       # paper: 10 (tuned among 3, 5, 10, 20)
# [CHANGED C15] upstream: 10 sweeps, hard-coded in the tlda(...) call. No convergence check exists yet.
parser.add_argument("--iterations", type=int, default=10)   # Gibbs sweeps; set after a convergence run
parser.add_argument("--seed", type=int, default=2026)       # [CHANGED C15] new: TLDA was unseeded
# [CHANGED C24] Semantic TEMN, change 2: with a POI -> cluster table (e.g. poi_cluster_K50.npy, row = poi_id),
# each check-in's word is its POI's cluster, so topics become mixtures of KINDS of places.
parser.add_argument("--word_map", default=None)




cli = parser.parse_args()
random.seed(cli.seed)      # TLDA samples with Python's random module
np.random.seed(cli.seed)

# ids are used directly as array rows, so each count = largest id + 1
doc_num = 0
poi_num = 0
time_num = 0  # Foursquare files: 48 slots = 24 hours x (weekday, weekend) in local time; upstream sample: 7

topic_k = cli.topics

input_data = []
doc_index = []
doc_len = []

word_of_poi = np.load(cli.word_map) if cli.word_map else None  # [CHANGED C24] poi_id -> cluster id
f_i = open(cli.data)
readlines = f_i.readlines()
f_i.close()

for line in readlines:
    new_line = line.strip("\n").split("\t")
    ii = int(new_line[0])
    if ii > doc_num:
        doc_num = ii 
    json_data = json.loads(new_line[1])
    
    cur_data = []
    for item in json_data:  # item = [poi_id, time_slot, 0]; repeats kept, TLDA counts every check-in
        cur_poi_index = item[0] if word_of_poi is None else int(word_of_poi[item[0]])  # [CHANGED C24]
        cur_time_index = item[1]
        if cur_poi_index > poi_num:
            poi_num = cur_poi_index
        if cur_time_index > time_num:
            time_num = cur_time_index
        cur_item = (ii, cur_poi_index, item[1])  # [CHANGED C24] upstream: (ii, item[0], item[1])
        cur_data.append(cur_item)
    doc_len.append(len(cur_data))
    doc_index.append(ii)
    input_data.append(cur_data)
doc_num = doc_num + 1
poi_num = poi_num + 1
time_num = time_num + 1

print("doc_num:",doc_num)
print("poi_num:",poi_num)
print("time_num:",time_num)
print(topic_k)
# last arg is Gibbs iterations ([CHANGED C15] upstream: 10)
tlda = TLDA.tlda(input_data,doc_index,doc_len,doc_num,poi_num,time_num,topic_k,cli.iterations)
tlda.train_model()
tlda.save_model_data(cli.out)
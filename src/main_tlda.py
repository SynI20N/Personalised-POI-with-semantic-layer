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
parser.add_argument("--topics", type=int, default=10)       # paper: 10 (tuned among 3, 5, 10, 20)
parser.add_argument("--iterations", type=int, default=10)   # Gibbs sweeps; set after a convergence run
parser.add_argument("--seed", type=int, default=2026)
cli = parser.parse_args()
random.seed(cli.seed)      # TLDA samples with Python's random module
np.random.seed(cli.seed)

doc_num = 0
poi_num = 0
time_num = 0

topic_k = cli.topics

input_data = []
doc_index = []
doc_len = []

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
    for item in json_data:
        cur_poi_index = item[0]
        cur_time_index = item[1]
        if cur_poi_index > poi_num:
            poi_num = cur_poi_index
        if cur_time_index > time_num:
            time_num = cur_time_index
        cur_item = (ii,item[0],item[1])
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
# last arg is Gibs iterations
tlda = TLDA.tlda(input_data,doc_index,doc_len,doc_num,poi_num,time_num,topic_k,cli.iterations)
tlda.train_model()
tlda.save_model_data(cli.out)
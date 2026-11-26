#encoding:utf-8
'''
xz331@cam.ac.uk
abnerzxzhao@tencent.com
'''
# Random toy data in the authors' file format, only to check that the code runs (not used for results).
# Real data: src/preprocess.ipynb writes data/processed/<city>/data_<city>_val|test and poi2_xy.
# Each item is [poi_id, time_slot, neg_id]; the neg_id column and the 4th column (neg_data) were the
# authors' FIXED negatives, which TEMN no longer reads (see [CHANGED C10]/[CHANGED C11] in process_TEMN_data.py).

import json
import random
from pathlib import Path

# [CHANGED C4] new: write next to this file, not into the working directory
# Project root (the folder that contains "src"), independent of the current working directory
BASE_PATH = Path(__file__).resolve().parent.parent
SRC_PATH = BASE_PATH / "src"

doc_num = 5000
# [CHANGED C16] upstream: word_num = 10. With 10 POIs a user has visited all of them, so the new
# sampler (which only draws unvisited POIs) would find no negative.
word_num = 200  # must exceed what users visit (~0-40), otherwise no valid negatives exist
time_num = 7
train_k = 100
test_k = 3
neg_k = 10

f_o = open(SRC_PATH / "sample_data","w")
f_poi = open(SRC_PATH / "poi2_xy","w")

for i in range(doc_num):
    train_data = []
    test_data = []
    neg_data = []
    for j in range(train_k):
        pos_id = abs(int(random.gauss(3,10)))%word_num
        time_id = abs(int(random.gauss(3,3)))%time_num
        neg_id = abs(int(random.gauss(7,10)))%word_num
        cur_item = [pos_id,time_id,neg_id]
        train_data.append([pos_id,time_id,neg_id])
    for k in range(test_k):
        pos_id = abs(int(random.gauss(3,10)))%word_num
        time_id = abs(int(random.gauss(3,3)))%time_num
        neg_id = abs(int(random.gauss(7,10)))%word_num
        cur_item = [pos_id,time_id,neg_id]
        test_data.append([pos_id,time_id,neg_id])
    for n in range(neg_k):
        neg_id = abs(int(random.gauss(7,10)))%word_num
        neg_data.append(neg_id)
    r_line = str(i) + "\t" + json.dumps(train_data) + "\t" + json.dumps(test_data) + "\t" + json.dumps(neg_data) + "\n"
    f_o.write(r_line)
f_o.close()

for poi in range(word_num):
    cur_x = random.uniform(115,117)
    cur_y = random.uniform(39,41)
    r_line = str(poi) + "\t" +  str(cur_x) +  "\t" + str(cur_y) + "\n"
    f_poi.write(r_line)
f_poi.close()
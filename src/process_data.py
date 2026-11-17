
import json
from pathlib import Path

# Project root (the folder that contains "src"), independent of the current working directory
BASE_PATH = Path(__file__).resolve().parent.parent
SRC_PATH = BASE_PATH / "src"

f_i = open(SRC_PATH / 'sample_data')
readlines = f_i.readlines()
f_i.close()
ret = []
user_num = 0
poi_num = 0
ret_test = []
for line in readlines:
    new_line = line.strip().split("\t")
    cur_doc_id = int(new_line[0])
    cur_data = json.loads(new_line[1])
    test_data = json.loads(new_line[2])
    neg_datas = json.loads(new_line[3])
    print(line)
    break

# encoding:utf-8
"""
xz331@cam.ac.uk
abnerzxzhao@tencent.com
"""
import json
import math
import pickle
from pathlib import Path
import numpy as np
# Default location of the data files (poi2_xy, tlda_model_file, sample_data)
DATA_DIR = Path(__file__).resolve().parent


# Distance fed to the geography module: "log1p" = log(1 + km) (default), "km" = plain km.
# main_TEMN sets it from --distance, so every run records which one it used.
DISTANCE = "log1p"


def get_dis(node1, node2):

    # Haversine: convert degrees to radians
    # comput a
    # COmpute angular distance c
    # Cpnvert to kilometer

    # IMPORTANT: We treat km as log(1+  km) since these are very big and dominate 
    # the loss function otherwise

    R = 6371.0 # Earth radius km
    phi1,phi2 = np.radians(node1[1]), np.radians(node2[1])
    dphi = np.radians(node1[1] - node2[1])
    dlon = np.radians(node1[0] - node2[0])
    
    a = ( np.sin(dphi/2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon/2.0) ** 2)
    c = 2.0 * np.arcsin(np.sqrt(a))
    km = R * c
    return np.log1p(km) if DISTANCE == "log1p" else km

    # x = node1[0] - node2[0]
    # y = node1[1] - node2[1]
    # return math.sqrt(x * x + y * y)


def get_poi_xy(data_dir=DATA_DIR):
    poi2xy = {}
    f_i = open(Path(data_dir) / "poi2_xy")
    readlines = f_i.readlines()
    f_i.close()
    for line in readlines:
        new_line = line.strip().split("\t")
        poi = int(new_line[0])
        x = float(new_line[1])
        y = float(new_line[2])
        poi2xy[poi] = [x, y]
    return poi2xy


class NegativeSampler:
    """
    Draws negative POIs for a user: POIs that have coordinates and that the user
    did NOT visit in training (paper Eq. 7/18: j not in V_u^+).
    Negatives are drawn uniformly by rejection sampling, so call add_negatives()
    once per epoch to get fresh negatives instead of the fixed ones in the data file.
    """

    def __init__(self, poi2xy, user_center, user_visited, seed=None):
        self.poi2xy = poi2xy
        self.pois = np.array(sorted(poi2xy))
        self.poi_set = set(self.pois.tolist())
        self.user_center = user_center    # user -> [x, y] centre of training POIs
        self.user_visited = user_visited  # user -> set of training POIs
        self.rng = np.random.default_rng(seed)

    def n_available(self, u, extra_exclude=()):
        excluded = (self.user_visited[u] | set(extra_exclude)) & self.poi_set
        return len(self.pois) - len(excluded)

    def sample(self, u, n=1, extra_exclude=(), rng=None):
        """Return up to n distinct POIs not visited by u (and not in extra_exclude)."""
        rng = self.rng if rng is None else rng
        exclude = self.user_visited[u] | set(extra_exclude)
        n = min(n, self.n_available(u, extra_exclude))
        chosen = []
        chosen_set = set()
        while len(chosen) < n:
            j = int(self.pois[rng.integers(len(self.pois))])
            if j in exclude or j in chosen_set:
                continue
            chosen.append(j)
            chosen_set.add(j)
        return chosen

    def distance(self, u, poi):
        return get_dis(self.user_center[u], self.poi2xy[poi])

    def add_negatives(self, examples):
        """
        (u, i, dist, history, topic) -> (u, i, dist, history, neg, topic, neg_dist),
        the tuple layout TEMN.get_list_feed_dict expects in training mode.
        """
        out = []
        for u, i, dist, history, topic in examples:
            neg = self.sample(u)[0]
            out.append((u, i, dist, history, neg, topic, self.distance(u, neg)))
        return out


def process_fun(file_name, data_dir=DATA_DIR, num_test_neg=100, seed=2026):
    """
    Returns
      train:   list of (u, i, dist, history, topic); negatives are added per epoch
               with sampler.add_negatives(train)
      user_num, poi_num, topic_k
      test:    list of groups [positive, neg_1, ..., neg_n], each (u, i, dist, history);
               the n = num_test_neg negatives are fixed (seeded) and never visited by u
      sampler: the NegativeSampler
    """
    data_dir = Path(data_dir)
    poi2xy = get_poi_xy(data_dir)
    test_rng = np.random.default_rng(seed + 1)  # separate stream: test negatives stay fixed

    with open(data_dir / "tlda_model_file", "rb") as f_model:
        document_topic, time_topic, topic_word = pickle.load(f_model)
    print("document_topic,", document_topic.shape)
    topic_k = document_topic.shape[1]
    f_i = open(data_dir / file_name)
    readlines = f_i.readlines()
    f_i.close()
    ret = []
    user_num = 0
    poi_num = max(poi2xy)
    ret_test = []
    user_center = {}
    user_visited = {}
    sampler = NegativeSampler(poi2xy, user_center, user_visited, seed=seed)
    skipped_users = 0
    for line in readlines:
        new_line = line.strip().split("\t")
        cur_doc_id = int(new_line[0])
        cur_data = json.loads(new_line[1])
        test_data = json.loads(new_line[2])
        # column 4: POIs that must never be drawn as negatives for this user (in the test file:
        # the user's val POIs, which they did visit). In the old sample_data format it holds
        # the old fixed negatives; excluding those is harmless.
        excluded_pois = set(json.loads(new_line[3])) if len(new_line) > 3 else set()
        # Column 1 holds every training check-in (TLDA needs the repeats), but TEMN uses each
        # user-POI pair once: V_u+ is the set of distinct training POIs (paper Eq. 2).
        all_i = {item[0] for item in cur_data if item[0] in poi2xy}
        if len(all_i) < 1:
            continue
        # user centre c_u: mean over distinct POIs, not weighted by repeat visits (paper Eq. 15)
        poi_x = sum(poi2xy[p][0] for p in all_i) / len(all_i)
        poi_y = sum(poi2xy[p][1] for p in all_i) / len(all_i)
        if cur_doc_id > user_num:
            user_num = cur_doc_id
        user_center[cur_doc_id] = [poi_x, poi_y]
        user_visited[cur_doc_id] = all_i
        if sampler.n_available(cur_doc_id) == 0:
            # user visited every POI: no valid negative exists, so no ranking signal
            skipped_users += 1
            continue
        for cur_i in sorted(all_i):  # one training example per distinct POI
            if cur_i > poi_num:
                poi_num = cur_i
            cur_u = cur_doc_id
            cur_x = poi2xy[cur_i][0]
            cur_y = poi2xy[cur_i][1]
            opt_dis = get_dis([poi_x, poi_y], [cur_x, cur_y])
            cur_topic = document_topic[cur_u]
            cur_all_item = []
            for p in all_i:
                if p == cur_i:
                    continue
                cur_all_item.append(p)
            if len(cur_all_item) > 0:
                xx = cur_u, cur_i, opt_dis, cur_all_item, cur_topic
                ret.append(xx)

        # test negatives: sampled once per user, never a POI the user visited in
        # training, any of the user's own held-out POIs, or a POI listed in column 4
        test_pois = {item[0] for item in test_data}
        neg_datas = sampler.sample(cur_doc_id, n=num_test_neg,
                                   extra_exclude=test_pois | excluded_pois, rng=test_rng)
        for item in test_data:
            cur_test_data = []
            if item[0] > poi_num:
                poi_num = item[0]
            if item[0] not in poi2xy:
                continue
            cur_u = cur_doc_id
            cur_i = item[0]
            cur_x = poi2xy[cur_i][0]
            cur_y = poi2xy[cur_i][1]
            opt_dis = get_dis([poi_x, poi_y], [cur_x, cur_y])
            cur_all_item = []
            for p in all_i:
                if p == cur_i:
                    continue
                cur_all_item.append(p)
            if len(cur_all_item) > 0:
                xx = cur_u, cur_i, opt_dis, cur_all_item
                cur_test_data.append(xx)
                for neg_i in neg_datas:
                    neg_x = poi2xy[neg_i][0]
                    neg_y = poi2xy[neg_i][1]
                    opt_dis = get_dis([poi_x, poi_y], [neg_x, neg_y])
                    xx = cur_u, neg_i, opt_dis, cur_all_item
                    cur_test_data.append(xx)
            ret_test.append(cur_test_data)

    if skipped_users:
        print("skipped", skipped_users, "users with no unvisited POI to use as negative")
    return ret, user_num + 1, poi_num + 1, topic_k, ret_test, sampler


if __name__ == '__main__':
    train_data = process_fun("sample_data")
    print(len(train_data[0]))

# encoding:utf-8
"""
xz331@cam.ac.uk
abnerzxzhao@tencent.com
"""
# Turns a data file into TEMN's training examples and test groups.
#
# Input files in data_dir (written by src/preprocess.ipynb for NYC and Tokyo):
#   data_<city>_val / data_<city>_test   one line per user: user_id \t col1 \t col2 \t col4
#       col1  every training check-in as [poi_id, time_slot, 0], repeats included (TLDA needs them)
#       col2  the held-out POIs (val or test) as [poi_id, slot, 0]
#       col4  POIs never to be drawn as negatives ([] in the val file; the user's val POIs in the test file)
#       (upstream: the 3rd number of each item and col4 held FIXED negatives; both are now unused, see C10/C11)
#   poi2_xy          poi_id \t lon \t lat, for the TRAINING POIs only
#   tlda_model_file  pickle written by main_tlda.py; row u of its first array = user u's topic mixture
#
# Output of process_fun:
#   train  one tuple (u, POI, dist, history, topics) per distinct training (user, POI) pair;
#          a negative is added every epoch by NegativeSampler.add_negatives
#   test   one group per held-out (user, POI): [positive, 100 negatives], each (u, POI, dist, history)
import json
import math
import pickle
from pathlib import Path
import numpy as np
from collections import Counter
# [CHANGED C4] upstream read "poi2_xy", "tlda_model_file" and the data file from the working directory.
# Default location of the data files (poi2_xy, tlda_model_file, sample_data)
DATA_DIR = Path(__file__).resolve().parent


# [CHANGED C7] new setting.
# Distance fed to the geography module: "log1p" = log(1 + km) (default), "km" = plain km.
# main_TEMN sets it from --distance, so every run records which one it used.
DISTANCE = "log1p"


def get_dis(node1, node2):
    # [CHANGED C7] upstream: Euclidean distance between (lon, lat) in DEGREES (code kept below,
    # commented out). Paper Eq. 16 is also a plain Euclidean distance. Now: great-circle (haversine)
    # distance in km, because a degree of longitude is shorter than a degree of latitude
    # (NYC ~85 vs ~111 km), and with the log1p option log(1 + km) so a few far POIs (airports,
    # trips, up to hundreds of km) do not dominate the geography loss.
    #
    # Haversine: a = sin^2(dlat/2) + cos(lat1) cos(lat2) sin^2(dlon/2);  c = 2 asin(sqrt(a));  km = R c
    # node = [lon, lat], the column order of poi2_xy.

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
    # poi_id -> [lon, lat]. Only POIs listed here can be trained on, drawn as negatives or evaluated.
    poi2xy = {}
    f_i = open(Path(data_dir) / "poi2_xy")  # [CHANGED C4] upstream: open("poi2_xy")
    readlines = f_i.readlines()
    f_i.close()
    for line in readlines:
        new_line = line.strip().split("\t")
        poi = int(new_line[0])
        x = float(new_line[1])
        y = float(new_line[2])
        poi2xy[poi] = [x, y]
    return poi2xy


# [CHANGED C10] new class: how negative POIs are chosen (training and test).
#
# Upstream had no sampler. Every training item in the data file carried a FIXED negative
# (its 3rd number, made by gen_data.py as abs(int(gauss(7, 10))) % n_pois), reused in every
# epoch and never checked against the user's visits, so a "negative" could be a POI the user
# visited, even the positive itself. Test groups used the first 10 ids of col4 (not 100).
#
# Now: a negative for user u is drawn UNIFORMLY from the POIs in poi2_xy (= POIs with training
# check-ins), rejecting the POIs u visited in training (the paper's  j not in V_u^+ , Eq. 7/18).
# Training: one fresh negative per example per epoch (add_negatives, called in main_TEMN).
# Test: 100 negatives per user, drawn once from a separate seeded stream (see process_fun).
#
# Keep in mind for the next steps: the pool is "every POI in poi2_xy". When cold POIs get
# coordinates for evaluation, the TRAINING pool must stay the training POIs, otherwise cold POIs
# are trained as negatives and stop being cold.
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
        self.cdf = None # uniform

    

    def n_available(self, u, extra_exclude=()):
        excluded = (self.user_visited[u] | set(extra_exclude)) & self.poi_set
        return len(self.pois) - len(excluded)

    def sample(self, u, n=1, extra_exclude=(), rng=None, weighted=False):
        """Return up to n distinct POIs not visited by u (and not in extra_exclude)."""
        # Rejection sampling: draw a random POI, throw it away if u visited it or it is already
        # chosen. Fast because a user visits ~0.3% of the POIs (NYC: 84 of 30,358 on average).
        rng = self.rng if rng is None else rng
        exclude = self.user_visited[u] | set(extra_exclude)
        n = min(n, self.n_available(u, extra_exclude))
        chosen = []
        chosen_set = set()
        while len(chosen) < n:

            if weighted and self.cdf is not None: 
            #cumulative weights in [0,1]; a POI's share of that range equals its probability, so a uniform random number lands more often on popular POIs
                j = int(self.pois[np.searchsorted(self.cdf, rng.random())])
            else: 
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
            neg = self.sample(u,weighted=True)[0]
            out.append((u, i, dist, history, neg, topic, self.distance(u, neg)))
        return out

    def set_popularity_weights(self,alpha):
        if alpha == 0:
            self.cdf = None
            return
        counts = Counter(p for visited in self.user_visited.values() for p in visited)

        w = np.array([counts[p] for p in self.pois],dtype=float) ** alpha # power to alpha makes higher popuaurity values grow faster than smaller ones as alpha goes to 1
        cdf = np.cumsum(w)
        self.cdf = cdf / cdf[-1]


# [CHANGED C4/C10/C11] upstream signature: process_fun(file_name), returning 5 values (no sampler)
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
    # [CHANGED C11] test negatives come from their own random stream (seed + 1), so they are the
    # same in every run with the same seed, whatever the training draws. Different seed =
    # different test groups: compare two models only on runs with the SAME --seed.
    test_rng = np.random.default_rng(seed + 1)  # separate stream: test negatives stay fixed

    with open(data_dir / "tlda_model_file", "rb") as f_model:
        # [NOTE] TLDA.save_model_data writes [theta, phi, vt] = (user-topic, topic-POI, time-topic),
        # so the last two names below are swapped. Harmless: only document_topic (theta) is used.
        document_topic, time_topic, topic_word = pickle.load(f_model)
    print("document_topic,", document_topic.shape)
    topic_k = document_topic.shape[1]
    f_i = open(data_dir / file_name)
    readlines = f_i.readlines()
    f_i.close()
    ret = []
    user_num = 0
    # [CHANGED C10] upstream: poi_num = 0, then grown from the POI ids seen in the data (positives
    # and fixed negatives). Now every POI the sampler can draw has an embedding row.
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
        # [CHANGED C11] upstream: neg_datas = json.loads(new_line[3]) -- the fixed test negatives.
        # column 4: POIs that must never be drawn as negatives for this user (in the test file:
        # the user's val POIs, which they did visit). In the old sample_data format it holds
        # the old fixed negatives; excluding those is harmless.
        excluded_pois = set(json.loads(new_line[3])) if len(new_line) > 3 else set()
        # [CHANGED C9] upstream built all_i while looping over every check-in (repeats included).
        # Column 1 holds every training check-in (TLDA needs the repeats), but TEMN uses each
        # user-POI pair once: V_u+ is the set of distinct training POIs (paper Eq. 2).
        all_i = {item[0] for item in cur_data if item[0] in poi2xy}
        if len(all_i) < 1:
            continue
        # [CHANGED C8] upstream summed the coordinates of every CHECK-IN, so c_u leaned towards
        # POIs visited often (e.g. home, office).
        # user centre c_u: mean over distinct POIs, not weighted by repeat visits (paper Eq. 15)
        poi_x = sum(poi2xy[p][0] for p in all_i) / len(all_i)
        poi_y = sum(poi2xy[p][1] for p in all_i) / len(all_i)
        if cur_doc_id > user_num:
            user_num = cur_doc_id
        user_center[cur_doc_id] = [poi_x, poi_y]
        user_visited[cur_doc_id] = all_i
        if sampler.n_available(cur_doc_id) == 0:
            # [CHANGED C12] new check.
            # user visited every POI: no valid negative exists, so no ranking signal
            skipped_users += 1
            continue
        # [CHANGED C9] upstream: `for item in cur_data:` -- one example per CHECK-IN, so a POI visited
        # 50 times gave 50 examples (same POI, each with its fixed negative item[2]). Now one example
        # per distinct POI, as in the paper's loss (Eq. 7 sums over v in V_u^+), and no negative yet:
        # the negative is drawn per epoch by NegativeSampler.add_negatives ([CHANGED C10]).
        for cur_i in sorted(all_i):  # one training example per distinct POI
            if cur_i > poi_num:
                poi_num = cur_i
            cur_u = cur_doc_id
            cur_x = poi2xy[cur_i][0]
            cur_y = poi2xy[cur_i][1]
            opt_dis = get_dis([poi_x, poi_y], [cur_x, cur_y])  # d(c_u, v) for the geography module
            cur_topic = document_topic[cur_u]                   # TLDA label for the topic loss
            # history = the user's other training POIs (leave the positive out, as upstream)
            cur_all_item = []
            for p in all_i:
                if p == cur_i:
                    continue
                cur_all_item.append(p)
            if len(cur_all_item) > 0:
                # [CHANGED C10] upstream: (cur_u, cur_i, opt_dis, cur_all_item, neg_i, cur_topic, neg_dis)
                xx = cur_u, cur_i, opt_dis, cur_all_item, cur_topic
                ret.append(xx)

        # [CHANGED C11] upstream: each test group = positive + neg_datas[0:10], i.e. 10 FIXED negatives
        # from column 4, not checked against the user's visits (the paper reports 100 random unvisited POIs).
        # Now: num_test_neg = 100 negatives per user, uniform over the training POIs, drawn once and shared
        # by all of the user's held-out POIs.
        # test negatives: sampled once per user, never a POI the user visited in
        # training, any of the user's own held-out POIs, or a POI listed in column 4
        test_pois = {item[0] for item in test_data}
        neg_datas = sampler.sample(cur_doc_id, n=num_test_neg,
                                   extra_exclude=test_pois | excluded_pois, rng=test_rng)
        for item in test_data:
            cur_test_data = []
            if item[0] > poi_num:
                poi_num = item[0]
            # [NOTE N7] poi2_xy holds training POIs only, so every held-out POI that nobody visited in
            # training (a COLD POI) is silently skipped here: 32.5% of NYC and 22.9% of Tokyo val pairs.
            # All current results are therefore on warm POIs only. RQ1 needs these pairs back.
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
                cur_test_data.append(xx)  # entry 0 of the group is always the positive
                # [CHANGED C11] upstream: `for neg_i in neg_datas[0:10]:` plus a skip of ids missing from poi2_xy
                for neg_i in neg_datas:
                    neg_x = poi2xy[neg_i][0]
                    neg_y = poi2xy[neg_i][1]
                    opt_dis = get_dis([poi_x, poi_y], [neg_x, neg_y])
                    xx = cur_u, neg_i, opt_dis, cur_all_item
                    cur_test_data.append(xx)
            ret_test.append(cur_test_data)

    if skipped_users:
        print("skipped", skipped_users, "users with no unvisited POI to use as negative")
    # [CHANGED C10] upstream returned 5 values; the sampler is new (main_TEMN draws training negatives with it)
    return ret, user_num + 1, poi_num + 1, topic_k, ret_test, sampler




if __name__ == '__main__':
    train_data = process_fun("sample_data")
    print(len(train_data[0]))

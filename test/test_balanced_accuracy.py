# -*- coding: utf-8 -*-

import pickle
import pprint
import sys
import time

import h5py
import numpy as np

sys.path.append('/Users/Ryan/code/python/hnsw-python')
from hnsw import HNSW


fr = open('glove-25-angular-balanced-128.ind','rb')
hnsw_n = pickle.load(fr)

# add_point_time = time.time()
# idx = hnsw_n.search(np.float32(np.random.random((1, 25))), 10)
# search_time = time.time()
# pprint.pprint(idx)
# pprint.pprint("searchtime: %f"  % (search_time - add_point_time))

f = h5py.File('glove-25-angular.hdf5','r')
distances = f['distances']
neighbors = f['neighbors']
test = f['test']
train = f['train']

pprint.pprint(len(hnsw_n._graphs))

for index, i in enumerate(train[0:10]):
    idx = hnsw_n.search(i, 5)
    pprint.pprint(idx)

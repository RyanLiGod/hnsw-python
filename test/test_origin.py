# -*- coding: utf-8 -*-

import pickle
import pprint
import sys
import time

import h5py
import numpy as np
from pandas import DataFrame

sys.path.append('/Users/Ryan/code/python/hnsw-python')
from hnsw import HNSW


fr = open('glove-25-angular-origin-128.ind','rb')
hnsw_n = pickle.load(fr)

# add_point_time = time.time()
# idx = hnsw_n.search(np.float32(np.random.random((1, 25))), 10)
# search_time = time.time()
# pprint.pprint(idx)
# pprint.pprint("searchtime: %f" % (search_time - add_point_time))

f = h5py.File('glove-25-angular.hdf5','r')
distances = f['distances']
neighbors = f['neighbors']
test = f['test']
train = f['train']

variance_record = []
mean_record = []

for j in range(20):
    print(j)
    time_record = []
    for index, i in enumerate(test):
        search_begin = time.time()
        idx = hnsw_n.search(i, 10)
        # pprint.pprint(idx)
        search_end = time.time()
        search_time = search_end - search_begin
        time_record.append(search_time * 1000)

    variance_n = np.var(time_record)
    mean_n = np.mean(time_record)
    pprint.pprint('variance: %f' % variance_n)
    pprint.pprint('mean: %f' % mean_n)
    variance_record.append(variance_n)
    mean_record.append(mean_n)

data = {
    'mean_origin': mean_record,
    'variance_origin': variance_record
}

df = DataFrame(data)
df.to_excel('variance_result_origin_8.xlsx')

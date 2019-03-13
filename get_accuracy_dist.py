# -*- coding: utf-8 -*-

import h5py
import pprint

f = h5py.File('glove-25-angular.hdf5','r')
distances = f['distances']
neighbors = f['neighbors']
test = f['test']
train = f['train']

pprint.pprint(distances[2])
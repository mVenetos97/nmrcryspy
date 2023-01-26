import os
import sys
from pathlib import Path

path_root = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(path_root)
sys.path.append(str(path_root))
# print(sys.path)

import copy
import logging
import math
import os
import shutil
import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from e3nn.io import CartesianTensor
from loguru import logger
from monty.serialization import dumpfn, loadfn
from numpy.linalg import pinv
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import (
    find_in_coord_list_pbc,
    in_coord_list,
    in_coord_list_pbc,
    pbc_shortest_vectors,
)

from nmrcryspy import Gauss_Newton_Solver
from nmrcryspy.geometric import Distance_Function
from nmrcryspy.utils import get_unique_indicies, make_distance_data
from nmrcryspy.utils import coords_with_pbc, dist_from_coords


file_path = '/Users/mvenetos/Box Sync/All Manuscripts/zeolite refinements/ZSM-12_calcined.cif' #
s = CifParser(file_path).get_structures(False)[0]

# data = {'Bond_Distances': make_distance_data(s)}
# data['Bond_Distances'][0]['pairs'].append({'atom 1': 0, 'atom 2': 16, 'true_pair': [0, 16], 'min_image_construction': False},)
# data['Bond_Distances'][0]['pairs'].append({'atom 1': 8, 'atom 2': 24, 'true_pair': [8, 24], 'min_image_construction': False},)
# data['Bond_Distances'][0]['pairs'].append({'atom 1': 32, 'atom 2': 40, 'true_pair': [32, 40], 'min_image_construction': False},)

data = {
    'Bond_Distances': [
    {'bond': 'SiSi', 'pairs': [ #11 total
        {'atom 1': 0, 'atom 2': 8, 'true_pair': [0, 13]}, #new
        {'atom 1': 0, 'atom 2': 8, 'true_pair': [0, 15]},
        {'atom 1': 0, 'atom 2': 16, 'true_pair': [0, 16]},
        # {'atom 1': 0, 'atom 2': 16, 'true_pair': [0, 16]},
        {'atom 1': 0, 'atom 2': 16, 'true_pair': [0, 16], 'min_image_construction': False},
        {'atom 1': 8, 'atom 2': 24, 'true_pair': [8, 24]},
        # {'atom 1': 8, 'atom 2': 24, 'true_pair': [8, 24]},
        {'atom 1': 8, 'atom 2': 24, 'true_pair': [8, 24], 'min_image_construction': False},
        {'atom 1': 16, 'atom 2': 32, 'true_pair': [16, 32]},
        {'atom 1': 16, 'atom 2': 48, 'true_pair': [16, 48]},
        {'atom 1': 24, 'atom 2': 32, 'true_pair': [24, 37]},
        {'atom 1': 24, 'atom 2': 40, 'true_pair': [24, 40]},
        {'atom 1': 32, 'atom 2': 40, 'true_pair': [32, 40]},
        # {'atom 1': 32, 'atom 2': 40, 'true_pair': [32, 40]},
        {'atom 1': 32, 'atom 2': 40, 'true_pair': [32, 40], 'min_image_construction': False},
        {'atom 1': 40, 'atom 2': 48, 'true_pair': [40, 54]},
        {'atom 1': 48, 'atom 2': 48, 'true_pair': [48, 54]}, #new
    ]},
    {'bond': 'SiO', 'pairs': [ #28 total
        {'atom 1': 0, 'atom 2': 56, 'true_pair': [0, 56]},
        {'atom 1': 0, 'atom 2': 64, 'true_pair': [0, 64]},
        {'atom 1': 0, 'atom 2': 72, 'true_pair': [0, 72]},
        {'atom 1': 0, 'atom 2': 144, 'true_pair': [0, 144]},
        {'atom 1': 8, 'atom 2': 88, 'true_pair': [8, 88]},
        {'atom 1': 8, 'atom 2': 64, 'true_pair': [8, 71]},
        {'atom 1': 8, 'atom 2': 56, 'true_pair': [8, 61]},
        {'atom 1': 8, 'atom 2': 152, 'true_pair': [8, 152]},
        {'atom 1': 16, 'atom 2': 144, 'true_pair': [16, 144]},
        {'atom 1': 16, 'atom 2': 72, 'true_pair': [16, 72]},
        {'atom 1': 16, 'atom 2': 96, 'true_pair': [16, 96]},
        {'atom 1': 16, 'atom 2': 80, 'true_pair': [16, 80]},
        {'atom 1': 24, 'atom 2': 88, 'true_pair': [24, 88]},
        {'atom 1': 24, 'atom 2': 120, 'true_pair': [24, 120]},
        {'atom 1': 24, 'atom 2': 128, 'true_pair': [24, 128]},
        {'atom 1': 24, 'atom 2': 152, 'true_pair': [24, 152]},
        {'atom 1': 32, 'atom 2': 112, 'true_pair': [32, 112]},
        {'atom 1': 32, 'atom 2': 160, 'true_pair': [32, 160]},
        {'atom 1': 32, 'atom 2': 80, 'true_pair': [32, 80]},
        {'atom 1': 32, 'atom 2': 128, 'true_pair': [32, 133]},
        {'atom 1': 40, 'atom 2': 112, 'true_pair': [40, 112]},
        {'atom 1': 40, 'atom 2': 120, 'true_pair': [40, 120]},
        {'atom 1': 40, 'atom 2': 160, 'true_pair': [40, 160]},
        {'atom 1': 40, 'atom 2': 136, 'true_pair': [40, 142]},
        {'atom 1': 48, 'atom 2': 104, 'true_pair': [48, 104]},
        {'atom 1': 48, 'atom 2': 136, 'true_pair': [48, 136]},
        {'atom 1': 48, 'atom 2': 96, 'true_pair': [48, 96]},
        {'atom 1': 48, 'atom 2': 104, 'true_pair': [48, 110]} #new
    ]},
    {'bond': 'OO', 'pairs': [ #42 total
        {'atom 1': 56, 'atom 2': 64, 'true_pair': [56, 64]},
        {'atom 1': 56, 'atom 2': 72, 'true_pair': [56, 72]},
        {'atom 1': 56, 'atom 2': 144, 'true_pair': [56, 144]},
        {'atom 1': 64, 'atom 2': 72, 'true_pair': [64, 72]},
        {'atom 1': 64, 'atom 2': 144, 'true_pair': [64, 144]},
        {'atom 1': 72, 'atom 2': 144, 'true_pair': [72, 144]},

        {'atom 1': 56, 'atom 2': 64, 'true_pair': [61, 71]}, #grad here is questionable
        {'atom 1': 56, 'atom 2': 88, 'true_pair': [61, 88]}, #grad here is questionable
        {'atom 1': 56, 'atom 2': 152, 'true_pair': [61, 152]}, #grad here is questionable
        {'atom 1': 64, 'atom 2': 88, 'true_pair': [71, 88]}, #grad here is questionable
        {'atom 1': 64, 'atom 2': 152, 'true_pair': [71, 152]}, #grad here is questionable
        {'atom 1': 88, 'atom 2': 152, 'true_pair': [88, 152]},

        {'atom 1': 72, 'atom 2': 80, 'true_pair': [72, 80]},
        {'atom 1': 72, 'atom 2': 96, 'true_pair': [72, 96]},
        {'atom 1': 72, 'atom 2': 144, 'true_pair': [72, 144],},# 'min_image_construction': False},
        {'atom 1': 80, 'atom 2': 96, 'true_pair': [80, 96]},
        {'atom 1': 80, 'atom 2': 144, 'true_pair': [80, 144]},
        {'atom 1': 96, 'atom 2': 144, 'true_pair': [96, 144]},

        {'atom 1': 88, 'atom 2': 120, 'true_pair': [88, 120]},
        {'atom 1': 88, 'atom 2': 128, 'true_pair': [88, 128]},
        {'atom 1': 88, 'atom 2': 152, 'true_pair': [88, 152],},# 'min_image_construction': False},
        {'atom 1': 120, 'atom 2': 128, 'true_pair': [120, 128]},
        {'atom 1': 120, 'atom 2': 152, 'true_pair': [120, 152]},
        {'atom 1': 128, 'atom 2': 152, 'true_pair': [128, 152]},

        {'atom 1': 80, 'atom 2': 112, 'true_pair': [80, 112]},
        {'atom 1': 80, 'atom 2': 128, 'true_pair': [80, 133]},
        {'atom 1': 80, 'atom 2': 160, 'true_pair': [80, 160]},
        {'atom 1': 112, 'atom 2': 128, 'true_pair': [112, 133]},
        {'atom 1': 112, 'atom 2': 160, 'true_pair': [112, 160],},# 'min_image_construction': False},
        {'atom 1': 128, 'atom 2': 160, 'true_pair': [133, 160]}, #grad here is questionable

        {'atom 1': 112, 'atom 2': 120, 'true_pair': [112, 120]},
        {'atom 1': 112, 'atom 2': 136, 'true_pair': [112, 142]},
        {'atom 1': 112, 'atom 2': 160, 'true_pair': [112, 160]},
        {'atom 1': 120, 'atom 2': 136, 'true_pair': [120, 142]},
        {'atom 1': 120, 'atom 2': 160, 'true_pair': [120, 160],},# 'min_image_construction': False},
        {'atom 1': 136, 'atom 2': 160, 'true_pair': [142, 160]}, #grad here is questionable

        {'atom 1': 96, 'atom 2': 104, 'true_pair': [96, 104]},
        {'atom 1': 96, 'atom 2': 104, 'true_pair': [96, 110]}, #new
        {'atom 1': 96, 'atom 2': 136, 'true_pair': [96, 136]},
        {'atom 1': 104, 'atom 2': 136, 'true_pair': [104, 136]},
        {'atom 1': 104, 'atom 2': 104, 'true_pair': [104, 110]}, #new
        {'atom 1': 104, 'atom 2': 136, 'true_pair': [110, 136]}, # new #grad here is questionable
    ]},
    ]
}

# distance_dict = {
#       'SiO': {'mu' : 1.595, 'sigma' : 0.011},
#     'OO': {'mu' : 2.604, 'sigma' : 0.025},
#     'SiSi': { 'mu' : 3.101, 'sigma' : 0.041}
# }
distance_dict = {
      'SiO': {'mu' : 1.6, 'sigma' : 0.01},
    'OO': {'mu' : 2.61, 'sigma' : 0.02},
    'SiSi': { 'mu' : 3.1, 'sigma' : 0.05}
}
zeolite_dists = Distance_Function(distance_dict)

print(f'There are {3*len(get_unique_indicies(s))} degrees of freedom')

gn = Gauss_Newton_Solver(
    fit_function=[zeolite_dists],
    structure=s,
    data_dictionary=data,
    max_iter=20,
    tolerance_difference=1e-10
    )

test = pd.DataFrame(gn.fit()).sort_values(by = 'chi', ascending=True)
dist_test_dict = data['Bond_Distances']
distributions = []

s = test.iloc[0]['structure']
# CifWriter(s).write_file('/Users/mvenetos/Desktop/dls_zsm12.cif')

for idx, layer in enumerate(['SiSi', 'SiO', 'OO']):
    temp = []
    print(layer)
    for i in dist_test_dict[idx]['pairs']:
        if 'min_image_construction' in i.keys():
            coord1 = s[i['true_pair'][0]].coords
            coord2 = s[i['true_pair'][1]].coords
            temp.append(dist_from_coords(coord1, coord2))
        else:
            temp.append(s.get_distance(i['true_pair'][0], i['true_pair'][1]))
        # if layer == 'OO':
        # n1 = i['atom 1']
        # n2 = i['atom 2']
        # print(f'[{n1}, {n2}]', i['true_pair'][0], i['true_pair'][1], ': ', s.get_distance(i['true_pair'][0], i['true_pair'][1]))
        # print(temp)
    # print()
    distributions.append({
        'bond': layer,
        'data': temp
    })
df = pd.DataFrame(distributions)

import matplotlib.pyplot as plt

# test_sisi = set([3.06937,3.19093,3.03871,3.03954,3.19093,3.01823,3.07921,3.06937,3.03871,3.03954,2.98193,3.18579,3.01278,3.07921,3.01823,3.09163,3.09163,3.03826,3.06379,3.18579,3.06379,3.03826,3.16698,3.01278,3.14347,3.14347,2.98193,3.16698])
# test_oo = set([2.51211,2.59857,2.5277,2.57471,2.60264,2.54026,2.65011,2.59706,2.63257,2.54535,2.60308,2.61316,2.59833,2.58189,2.53496,2.65789,2.63407,2.56883,2.69748,2.62299,2.60694,2.60378,2.6816,2.60115,2.58855,2.71212,2.63388,2.53635,2.5665,2.64128,2.60616,2.64497,2.60681,2.64678,2.54671,2.7033,2.65493,2.6136,2.59944,2.58219,2.61786,2.5906])

# print(f'Sisi is len{len(test_sisi)}')
# print(f'OO is len{len(test_oo)}')
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
# ax[2].hist(test_sisi, np.array(range(147, 163))/50, color = 'orange', edgecolor = 'black', alpha = 0.5)
ax[2].hist(df.iloc[0]['data'], np.array(range(147, 163))/50, edgecolor = 'black', alpha = 0.5)
ax[2].set_title("SiSi")
ax[2].set_xlim([2.94, 3.26])
ax[2].set_ylim([0, 4])
ax[2].set_xticks([3, 3.08, 3.16, 3.24])

ax[0].hist(df.iloc[1]['data'], np.array(range(2*149, 2*171))/200, edgecolor = 'black')
ax[0].set_title("SiO")
ax[0].set_xlim([1.49, 1.7])
ax[0].set_ylim([0, 22])
ax[0].set_xticks([1.5, 1.54, 1.58, 1.62, 1.66, 1.7])
ax[1].hist(df.iloc[2]['data'], np.array(range(248, 271))/100, edgecolor = 'black', alpha = 0.5)
# ax[1].hist(test_oo, np.array(range(248, 271))/100, color = 'orange', edgecolor = 'black', alpha = 0.5)
ax[1].set_title("OO")
ax[1].set_xlim([2.48, 2.7])
ax[1].set_ylim([0, 28])
ax[1].set_xticks([2.5, 2.54, 2.58, 2.62, 2.66, 2.70])
plt.tight_layout()
plt.show()

print(len(df.iloc[0]['data']))
print(len(df.iloc[1]['data']))
print(len(df.iloc[2]['data']))


print('completed')


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
from nmrcryspy.utils import get_unique_indicies

file_path = '/Users/mvenetos/Box Sync/All Manuscripts/zeolite refinements/ZSM-12_calcined.cif'
s = CifParser(file_path).get_structures(False)[0]

data = {
    'Bond_Distances': [
    {'bond': 'SiSi', 'pairs': [ #11 total
        {'atom 1': 0, 'atom 2': 8, 'true_pair': [0, 12]}, #new
        {'atom 1': 0, 'atom 2': 8, 'true_pair': [0, 14]},
        {'atom 1': 0, 'atom 2': 16, 'true_pair': [0, 16]},
        {'atom 1': 8, 'atom 2': 24, 'true_pair': [8, 24]},
        {'atom 1': 16, 'atom 2': 32, 'true_pair': [16, 32]},
        {'atom 1': 16, 'atom 2': 48, 'true_pair': [16, 48]},
        {'atom 1': 24, 'atom 2': 32, 'true_pair': [24, 38]},
        {'atom 1': 24, 'atom 2': 40, 'true_pair': [24, 40]},
        {'atom 1': 32, 'atom 2': 40, 'true_pair': [32, 40]},
        {'atom 1': 40, 'atom 2': 48, 'true_pair': [40, 53]},
        {'atom 1': 48, 'atom 2': 48, 'true_pair': [48, 53]}, #new
    ]},
    {'bond': 'SiO', 'pairs': [ #28 total
        {'atom 1': 0, 'atom 2': 56, 'true_pair': [0, 56]},
        {'atom 1': 0, 'atom 2': 64, 'true_pair': [0, 64]},
        {'atom 1': 0, 'atom 2': 72, 'true_pair': [0, 72]},
        {'atom 1': 0, 'atom 2': 144, 'true_pair': [0, 144]},
        {'atom 1': 8, 'atom 2': 88, 'true_pair': [8, 88]},
        {'atom 1': 8, 'atom 2': 64, 'true_pair': [8, 68]},
        {'atom 1': 8, 'atom 2': 56, 'true_pair': [8, 62]},
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
        {'atom 1': 32, 'atom 2': 128, 'true_pair': [32, 134]},
        {'atom 1': 40, 'atom 2': 112, 'true_pair': [40, 112]},
        {'atom 1': 40, 'atom 2': 120, 'true_pair': [40, 120]},
        {'atom 1': 40, 'atom 2': 160, 'true_pair': [40, 160]},
        {'atom 1': 40, 'atom 2': 136, 'true_pair': [40, 141]},
        {'atom 1': 48, 'atom 2': 104, 'true_pair': [48, 104]},
        {'atom 1': 48, 'atom 2': 136, 'true_pair': [48, 136]},
        {'atom 1': 48, 'atom 2': 96, 'true_pair': [48, 96]},
        {'atom 1': 48, 'atom 2': 104, 'true_pair': [48, 109]} #new
    ]},
    {'bond': 'OO', 'pairs': [ #42 total
        {'atom 1': 56, 'atom 2': 64, 'true_pair': [56, 64]},
        {'atom 1': 56, 'atom 2': 72, 'true_pair': [56, 72]},
        {'atom 1': 56, 'atom 2': 144, 'true_pair': [56, 144]},
        {'atom 1': 64, 'atom 2': 72, 'true_pair': [64, 72]},
        {'atom 1': 64, 'atom 2': 144, 'true_pair': [64, 144]},
        {'atom 1': 72, 'atom 2': 144, 'true_pair': [72, 144]},
        {'atom 1': 56, 'atom 2': 64, 'true_pair': [62, 68]},
        {'atom 1': 56, 'atom 2': 88, 'true_pair': [62, 88]},
        {'atom 1': 56, 'atom 2': 152, 'true_pair': [62, 152]},
        {'atom 1': 64, 'atom 2': 88, 'true_pair': [68, 88]},
        {'atom 1': 64, 'atom 2': 152, 'true_pair': [68, 152]},
        {'atom 1': 88, 'atom 2': 152, 'true_pair': [88, 152]},
        {'atom 1': 72, 'atom 2': 80, 'true_pair': [72, 80]},
        {'atom 1': 72, 'atom 2': 96, 'true_pair': [72, 96]},
        {'atom 1': 72, 'atom 2': 144, 'true_pair': [72, 144]},
        {'atom 1': 80, 'atom 2': 96, 'true_pair': [80, 96]},
        {'atom 1': 80, 'atom 2': 144, 'true_pair': [80, 144]},
        {'atom 1': 96, 'atom 2': 144, 'true_pair': [96, 144]},
        {'atom 1': 88, 'atom 2': 120, 'true_pair': [88, 120]},
        {'atom 1': 88, 'atom 2': 128, 'true_pair': [88, 128]},
        {'atom 1': 88, 'atom 2': 152, 'true_pair': [88, 152]},
        {'atom 1': 120, 'atom 2': 128, 'true_pair': [120, 128]},
        {'atom 1': 120, 'atom 2': 152, 'true_pair': [120, 152]},
        {'atom 1': 128, 'atom 2': 152, 'true_pair': [128, 152]},
        {'atom 1': 80, 'atom 2': 112, 'true_pair': [80, 112]},
        {'atom 1': 80, 'atom 2': 128, 'true_pair': [80, 134]},
        {'atom 1': 80, 'atom 2': 160, 'true_pair': [80, 160]},
        {'atom 1': 112, 'atom 2': 128, 'true_pair': [112, 134]},
        {'atom 1': 112, 'atom 2': 160, 'true_pair': [112, 160]},
        {'atom 1': 128, 'atom 2': 160, 'true_pair': [134, 160]},
        {'atom 1': 112, 'atom 2': 120, 'true_pair': [112, 120]},
        {'atom 1': 112, 'atom 2': 136, 'true_pair': [112, 141]},
        {'atom 1': 112, 'atom 2': 160, 'true_pair': [112, 160]},
        {'atom 1': 120, 'atom 2': 136, 'true_pair': [120, 141]},
        {'atom 1': 120, 'atom 2': 160, 'true_pair': [120, 160]},
        {'atom 1': 136, 'atom 2': 160, 'true_pair': [141, 160]},
        {'atom 1': 96, 'atom 2': 104, 'true_pair': [96, 104]},
        {'atom 1': 96, 'atom 2': 104, 'true_pair': [96, 109]}, #new
        {'atom 1': 96, 'atom 2': 136, 'true_pair': [96, 136]},
        {'atom 1': 104, 'atom 2': 136, 'true_pair': [104, 136]},
        {'atom 1': 104, 'atom 2': 104, 'true_pair': [104, 109]}, #new
        {'atom 1': 104, 'atom 2': 136, 'true_pair': [109, 136]}, # new
    ]},
    ]
}

distance_dict = {
      'SiO': {'mu' : 1.595, 'sigma' : 0.011},
    'OO': {'mu' : 2.604, 'sigma' : 0.025},
    'SiSi': { 'mu' : 3.101, 'sigma' : 0.041}
}
zeolite_dists = Distance_Function(distance_dict)

print(f'There are {3*len(get_unique_indicies(s))} degrees of freedom')

gn = Gauss_Newton_Solver(
    fit_function=[zeolite_dists],
    structure=s,
    data_dictionary=data,
    max_iter=20,
    tolerance_difference=1e-6 #-10
    )

test = pd.DataFrame(gn.fit()).sort_values(by = 'chi', ascending=True)
dist_test_dict = data['Bond_Distances']
distributions = []

s = test.iloc[0]['structure']
for idx, layer in enumerate(['SiSi', 'SiO', 'OO']):
    temp = []
    print(layer)
    for i in dist_test_dict[idx]['pairs']:
        # print(i['true_pair'][0], i['true_pair'][1], ': ', s.get_distance(i['true_pair'][0], i['true_pair'][1]))
        temp.append(s.get_distance(i['true_pair'][0], i['true_pair'][1]))
    # print()
    distributions.append({
        'bond': layer,
        'data': temp
    })
df = pd.DataFrame(distributions)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
ax[2].hist(df.iloc[0]['data'], np.array(range(147, 163))/50, edgecolor = 'black')
ax[2].set_title("SiSi")
ax[2].set_xlim([2.94, 3.26])
ax[2].set_ylim([0, 4])
ax[2].set_xticks([3, 3.08, 3.16])

ax[0].hist(df.iloc[1]['data'], np.array(range(2*149, 2*170))/200, edgecolor = 'black')
ax[0].set_title("SiO")
ax[0].set_xlim([1.49, 1.7])
ax[0].set_ylim([0, 22])
ax[0].set_xticks([1.5, 1.54, 1.58, 1.62, 1.66, 1.7])
ax[1].hist(df.iloc[2]['data'], np.array(range(248, 270))/100, edgecolor = 'black')
ax[1].set_title("OO")
ax[1].set_xlim([2.48, 2.7])
ax[1].set_ylim([0, 28])
ax[1].set_xticks([2.5, 2.54, 2.58, 2.62, 2.66, 2.70])
plt.tight_layout()
plt.show()

print('completed')

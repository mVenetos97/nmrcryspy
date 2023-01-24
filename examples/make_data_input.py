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
import yaml
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

from itertools import combinations
from pymatgen.analysis.local_env import CrystalNN

file_path = '/Users/mvenetos/Box Sync/All Manuscripts/zeolite refinements/ZSM-12_calcined.cif'
s = CifParser(file_path).get_structures(False)[0]

# def remove_repeat_entries(data_list):
#     temp = []
#     already_exists = []
#     print('data list', data_list)
#     for datum in data_list:
#         check = datum['true_pair']
#         if check not in already_exists:

#             temp.append(datum)
#             already_exists.append(datum['true_pair'])
#     return temp

# def second_coordination_distance(index, equiv_indicies, nn_function, structure):
#     pairs = []
#     for atom in index:
#         nn = nn_function.get_nn_info(structure, atom[0])
#         for neighbor in nn:
#             sites = nn_function.get_nn_info(structure, neighbor['site_index'])
#             true_pairs = np.sort([sites[0]['site_index'], sites[1]['site_index']])
            
#             for idx in equiv_indicies:
#                 if true_pairs[0] in idx:
#                     ind1 = idx[0]
#                 if true_pairs[1] in idx:
#                     ind2 = idx[0]

            
#             pairs.append({
#                 'atom 1' : ind1,
#                 'atom 2' : ind2,
#                 'true_pair': list(true_pairs)
#             })
#     return remove_repeat_entries(pairs)

# def first_coordination_distance(index, equiv_indicies, nn_function, structure):
#     pairs = []
#     for atom in index:
#         nn = nn_function.get_nn_info(structure, atom[0])
#         for neighbor in nn:
#             for idx in equiv_indicies:
#                 if neighbor['site_index'] in idx:
#                     ind_neighbor = idx[0]
            
#             pairs.append({
#                 'atom 1' : atom[0],
#                 'atom 2' : neighbor['site_index'],
#                 'true_pair': [atom[0], ind_neighbor]
#             })
#     return pairs

# def first_coordination_vertex_vertex(index, equiv_indicies, nn_function, structure):
#     pairs = []
#     for atom in index:
#         nn = nn_function.get_nn_info(structure, atom[0])
#         ind_list = [i['site_index'] for i in nn]

#         for pair in list(combinations(ind_list, 2)):
#             for i in equiv_indicies:
#                 if pair[0] in i:
#                     ind1 = (i[0], pair[0])
#                 if pair[1] in i:
#                     ind2 = (i[0], pair[1])
#             unsorted = [ind1, ind2]
#             indicies = sorted(unsorted, key=lambda tup: tup[1])
#             pairs.append({
#                 'atom 1': indicies[0][0],
#                 'atom 2': indicies[1][0],
#                 'true_pair': [indicies[0][1], indicies[1][1]]
#             })
#     return pairs

# def make_distance_data(structure):
#     distance_data = []

#     species = [i.symbol for i in structure.species]
#     indicies, symmetry_equiv = get_unique_indicies(structure, full_list=True)
#     indicies = [(i, species[i]) for i in indicies if species[i] == 'Si']
#     nn = CrystalNN()
    
#     distance_data.append({
#         'bond': 'SiSi',
#         'pairs': second_coordination_distance(indicies, symmetry_equiv, nn, structure)
#     })
#     distance_data.append({
#         'bond': 'SiO',
#         'pairs': first_coordination_distance(indicies, symmetry_equiv, nn, structure)
#     })
#     distance_data.append({
#         'bond': 'OO',
#         'pairs': first_coordination_vertex_vertex(indicies, symmetry_equiv, nn, structure)
#     })

#     return distance_data


test_data = make_distance_data(s)

for i in test_data:
    print(i['bond'])
    for j in i['pairs']:
        print(j)
    print()

# {'bond': 'SiSi', 'pairs': [ #11 total
#         {'atom 1': 0, 'atom 2': 8, 'true_pair': [0, 12]}, #new
#         {'atom 1': 0, 'atom 2': 8, 'true_pair': [0, 14]},
#         {'atom 1': 0, 'atom 2': 16, 'true_pair': [0, 16]},
#         {'atom 1': 8, 'atom 2': 24, 'true_pair': [8, 24]},
#         {'atom 1': 16, 'atom 2': 32, 'true_pair': [16, 32]},
#         {'atom 1': 16, 'atom 2': 48, 'true_pair': [16, 48]},
#         {'atom 1': 24, 'atom 2': 32, 'true_pair': [24, 38]},
#         {'atom 1': 24, 'atom 2': 40, 'true_pair': [24, 40]},
#         {'atom 1': 32, 'atom 2': 40, 'true_pair': [32, 40]},
#         {'atom 1': 40, 'atom 2': 48, 'true_pair': [40, 53]},
#         {'atom 1': 48, 'atom 2': 48, 'true_pair': [48, 53]}, #new
#     ]},
#     {'bond': 'SiO', 'pairs': [ #28 total
#         {'atom 1': 0, 'atom 2': 56, 'true_pair': [0, 56]},
#         {'atom 1': 0, 'atom 2': 64, 'true_pair': [0, 64]},
#         {'atom 1': 0, 'atom 2': 72, 'true_pair': [0, 72]},
#         {'atom 1': 0, 'atom 2': 144, 'true_pair': [0, 144]},
#         {'atom 1': 8, 'atom 2': 88, 'true_pair': [8, 88]},
#         {'atom 1': 8, 'atom 2': 64, 'true_pair': [8, 68]},
#         {'atom 1': 8, 'atom 2': 56, 'true_pair': [8, 62]},
#         {'atom 1': 8, 'atom 2': 152, 'true_pair': [8, 152]},
#         {'atom 1': 16, 'atom 2': 144, 'true_pair': [16, 144]},
#         {'atom 1': 16, 'atom 2': 72, 'true_pair': [16, 72]},
#         {'atom 1': 16, 'atom 2': 96, 'true_pair': [16, 96]},
#         {'atom 1': 16, 'atom 2': 80, 'true_pair': [16, 80]},
#         {'atom 1': 24, 'atom 2': 88, 'true_pair': [24, 88]},
#         {'atom 1': 24, 'atom 2': 120, 'true_pair': [24, 120]},
#         {'atom 1': 24, 'atom 2': 128, 'true_pair': [24, 128]},
#         {'atom 1': 24, 'atom 2': 152, 'true_pair': [24, 152]},
#         {'atom 1': 32, 'atom 2': 112, 'true_pair': [32, 112]},
#         {'atom 1': 32, 'atom 2': 160, 'true_pair': [32, 160]},
#         {'atom 1': 32, 'atom 2': 80, 'true_pair': [32, 80]},
#         {'atom 1': 32, 'atom 2': 128, 'true_pair': [32, 134]},
#         {'atom 1': 40, 'atom 2': 112, 'true_pair': [40, 112]},
#         {'atom 1': 40, 'atom 2': 120, 'true_pair': [40, 120]},
#         {'atom 1': 40, 'atom 2': 160, 'true_pair': [40, 160]},
#         {'atom 1': 40, 'atom 2': 136, 'true_pair': [40, 141]},
#         {'atom 1': 48, 'atom 2': 104, 'true_pair': [48, 104]},
#         {'atom 1': 48, 'atom 2': 136, 'true_pair': [48, 136]},
#         {'atom 1': 48, 'atom 2': 96, 'true_pair': [48, 96]},
#         {'atom 1': 48, 'atom 2': 104, 'true_pair': [48, 109]} #new
#     ]},
#     {'bond': 'OO', 'pairs': [ #42 total
#         {'atom 1': 56, 'atom 2': 64, 'true_pair': [56, 64]},
#         {'atom 1': 56, 'atom 2': 72, 'true_pair': [56, 72]},
#         {'atom 1': 56, 'atom 2': 144, 'true_pair': [56, 144]},
#         {'atom 1': 64, 'atom 2': 72, 'true_pair': [64, 72]},
#         {'atom 1': 64, 'atom 2': 144, 'true_pair': [64, 144]},
#         {'atom 1': 72, 'atom 2': 144, 'true_pair': [72, 144]},
#         {'atom 1': 56, 'atom 2': 64, 'true_pair': [62, 68]},
#         {'atom 1': 56, 'atom 2': 88, 'true_pair': [62, 88]},
#         {'atom 1': 56, 'atom 2': 152, 'true_pair': [62, 152]},
#         {'atom 1': 64, 'atom 2': 88, 'true_pair': [68, 88]},
#         {'atom 1': 64, 'atom 2': 152, 'true_pair': [68, 152]},
#         {'atom 1': 88, 'atom 2': 152, 'true_pair': [88, 152]},
#         {'atom 1': 72, 'atom 2': 80, 'true_pair': [72, 80]},
#         {'atom 1': 72, 'atom 2': 96, 'true_pair': [72, 96]},
#         {'atom 1': 72, 'atom 2': 144, 'true_pair': [72, 144]},
#         {'atom 1': 80, 'atom 2': 96, 'true_pair': [80, 96]},
#         {'atom 1': 80, 'atom 2': 144, 'true_pair': [80, 144]},
#         {'atom 1': 96, 'atom 2': 144, 'true_pair': [96, 144]},
#         {'atom 1': 88, 'atom 2': 120, 'true_pair': [88, 120]},
#         {'atom 1': 88, 'atom 2': 128, 'true_pair': [88, 128]},
#         {'atom 1': 88, 'atom 2': 152, 'true_pair': [88, 152]},
#         {'atom 1': 120, 'atom 2': 128, 'true_pair': [120, 128]},
#         {'atom 1': 120, 'atom 2': 152, 'true_pair': [120, 152]},
#         {'atom 1': 128, 'atom 2': 152, 'true_pair': [128, 152]},
#         {'atom 1': 80, 'atom 2': 112, 'true_pair': [80, 112]},
#         {'atom 1': 80, 'atom 2': 128, 'true_pair': [80, 134]},
#         {'atom 1': 80, 'atom 2': 160, 'true_pair': [80, 160]},
#         {'atom 1': 112, 'atom 2': 128, 'true_pair': [112, 134]},
#         {'atom 1': 112, 'atom 2': 160, 'true_pair': [112, 160]},
#         {'atom 1': 128, 'atom 2': 160, 'true_pair': [134, 160]},
#         {'atom 1': 112, 'atom 2': 120, 'true_pair': [112, 120]},
#         {'atom 1': 112, 'atom 2': 136, 'true_pair': [112, 141]},
#         {'atom 1': 112, 'atom 2': 160, 'true_pair': [112, 160]},
#         {'atom 1': 120, 'atom 2': 136, 'true_pair': [120, 141]},
#         {'atom 1': 120, 'atom 2': 160, 'true_pair': [120, 160]},
#         {'atom 1': 136, 'atom 2': 160, 'true_pair': [141, 160]},
#         {'atom 1': 96, 'atom 2': 104, 'true_pair': [96, 104]},
#         {'atom 1': 96, 'atom 2': 104, 'true_pair': [96, 109]}, #new
#         {'atom 1': 96, 'atom 2': 136, 'true_pair': [96, 136]},
#         {'atom 1': 104, 'atom 2': 136, 'true_pair': [104, 136]},
#         {'atom 1': 104, 'atom 2': 104, 'true_pair': [104, 109]}, #new
#         {'atom 1': 104, 'atom 2': 136, 'true_pair': [109, 136]}, # new
#     ]},
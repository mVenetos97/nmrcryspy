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
from eigenn.cli import EigennCLI, SaveConfigCallback
from eigenn.data.datamodule import BaseDataModule
from eigenn.dataset.LSDI import SiNMRDataMoldule
from eigenn.model_factory.atomic_tensor_model import AtomicTensorModel
from eigenn.utils import to_path
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


def coords_with_pbc(idx_1, idx_2, stucture):
    lattice = stucture.lattice
    frac_coords1 = stucture[idx_1].frac_coords
    frac_coords2 = stucture[idx_2].frac_coords

    v, d2 = pbc_shortest_vectors(lattice, frac_coords1, frac_coords2, return_d2=True)
    fc = lattice.get_fractional_coords(v[0][0]) + frac_coords1 - frac_coords2
    fc = np.array(np.round(fc), dtype=int)

    jimage = np.array(fc)
    mapped_vec = lattice.get_cartesian_coords(jimage + frac_coords2 - frac_coords1)
    cart_coord_1 = lattice.get_cartesian_coords(frac_coords1)
    cart_coord_2 = lattice.get_cartesian_coords(jimage + frac_coords2)

    return cart_coord_1, cart_coord_2

def dist_from_coords(coord1, coord2):

    squared_dist = np.sum((coord1 - coord2) ** 2, axis=0)
    return np.sqrt(squared_dist)

def get_unique_indicies(structure):
    sga = SpacegroupAnalyzer(structure)
    symmeterized_struc = sga.get_symmetrized_structure()
    unique_indicies = [i[0] for i in symmeterized_struc.equivalent_indices]
    return unique_indicies

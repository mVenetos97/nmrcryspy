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
from pydantic import BaseModel
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import (
    find_in_coord_list_pbc,
    in_coord_list,
    in_coord_list_pbc,
    pbc_shortest_vectors,
)

from nmrcryspy.utils import coords_with_pbc, dist_from_coords, get_unique_indicies


class Distance_Function:
    """
    Function to calculate the distance and gradients between points.
    """

    def __init__(self,
                 distance_measures: dict = None
                 ):
        """
        :param distance_measures: Dictionary containing mean bond distance and standard
        deviation of relevant bond distance types.
        Example: {'SiO':{'mu': 1.595, 'sigma': 0.011}}
        """
        self.distance_measures = distance_measures

    def calculate_residual_and_gradients(self, idx_1, idx_2, structure, bond_type):
        """
        Calculates the residual and gradients between two points in a pymatgen structure
        :param idx_1: index of atom 1
        :param idx_2: index of atom 2
        :param structure: pymatgen structure
        :param bond_type: string denoting the type of bond
        """
        mu = self.distance_measures[bond_type]['mu']
        sigma = self.distance_measures[bond_type]['sigma']

        new_coords = coords_with_pbc(idx_1, idx_2, structure)

        coord_1 = new_coords[0]
        coord_2 = new_coords[1]

        d = dist_from_coords(coord_1, coord_2)

        res = (d-mu)/sigma

        ddx = (coord_2[0] - coord_1[0])/(d)#*sigma)
        ddy = (coord_2[1] - coord_1[1])/(d)#*sigma)
        ddz = (coord_2[2] - coord_1[2])/(d)#*sigma)

        # ddx = (coord_2[0] - coord_1[0])/(d*sigma)
        # ddy = (coord_2[1] - coord_1[1])/(d*sigma)
        # ddz = (coord_2[2] - coord_1[2])/(d*sigma)

        return res, ddx, ddy, ddz

    def assemble_residual_and_grad(self, structure, data_dictionary):
        """
        Assembles a residual vector and jacobian matrix from all of the observables in the
        data_dictionary
        :param structure: pymatgen structure
        :param data_dictionary: dictionary containing all of the relevant observables
        """
        unique_ind = get_unique_indicies(structure)
        num_atoms = len(unique_ind)
        geometric_dict = data_dictionary['Bond_Distances']
        sub_Jacobian = 0
        residuals = []
        for bond_type in geometric_dict:
            J_row = np.zeros([len(bond_type['pairs']), 3*num_atoms])
            for i, neighbors in enumerate(bond_type['pairs']):
                distances, grad_x, grad_y, grad_z = self.calculate_residual_and_gradients(
                    neighbors['true_pair'][0],
                    neighbors['true_pair'][1],
                    structure,
                    bond_type = bond_type['bond']
                    )
                residuals.append(distances)
                pos_1 = unique_ind.index(neighbors['atom 1'])
                pos_2 = unique_ind.index(neighbors['atom 2'])
                J_row[i, 3*pos_2] = grad_x
                J_row[i, 3*pos_2+1] = grad_y
                J_row[i, 3*pos_2+2] = grad_z
                J_row[i, 3*pos_1] = -grad_x
                J_row[i, 3*pos_1+1] = -grad_y
                J_row[i, 3*pos_1+2] = -grad_z
            if type(sub_Jacobian) == int:
                sub_Jacobian = J_row
            else:
                sub_Jacobian = np.vstack((sub_Jacobian, J_row))
        residuals = np.hstack(residuals)
        return sub_Jacobian, residuals

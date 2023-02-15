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

    def perturb_structure(self, initial_structure, sym_dict, x_prime, epsilon=1e-5):
        structure = copy.deepcopy(initial_structure)
        perturbations = np.reshape(epsilon*x_prime, (int(len(x_prime)/3), 3))
        for atom in sym_dict:
            base_idx = atom['base_idx']
            atom_idx = atom['atom']
            perturbation_opt = atom['sym_op'].apply_rotation_only(perturbations[base_idx])
            structure.translate_sites(atom_idx, perturbation_opt, frac_coords=False)
        return structure

    def calculate_gradients(self, idx_1, idx_2, structure, structure_e, epsilon, bond_type, mic = True):
        """
        Calculates the residual and gradients between two points in a pymatgen structure
        :param idx_1: index of atom 1
        :param idx_2: index of atom 2
        :param structure: pymatgen structure
        :param bond_type: string denoting the type of bond
        """
        mu = self.distance_measures[bond_type]['mu']
        sigma = self.distance_measures[bond_type]['sigma']

        new_coords = coords_with_pbc(idx_1, idx_2, structure, mic)
        coord_1 = new_coords[0]
        coord_2 = new_coords[1]
        d = dist_from_coords(coord_1, coord_2)

        new_coords_e = coords_with_pbc(idx_1, idx_2, structure_e, mic)
        coord_1_e = new_coords_e[0]
        coord_2_e = new_coords_e[1]
        d_e = dist_from_coords(coord_1_e, coord_2_e)


        derivative = (d_e - d)/(epsilon*sigma)

        return derivative

    def calculate_residual(self, idx_1, idx_2, structure, bond_type, mic = True):
        """
        Calculates the residual and gradients between two points in a pymatgen structure
        :param idx_1: index of atom 1
        :param idx_2: index of atom 2
        :param structure: pymatgen structure
        :param bond_type: string denoting the type of bond
        """
        mu = self.distance_measures[bond_type]['mu']
        sigma = self.distance_measures[bond_type]['sigma']

        new_coords = coords_with_pbc(idx_1, idx_2, structure, mic)
        coord_1 = new_coords[0]
        coord_2 = new_coords[1]
        d = dist_from_coords(coord_1, coord_2)

        res = (d-mu)/sigma

        return res

    def assemble_residual_and_grad(self, structure, data_dictionary):
        """
        Assembles a residual vector and jacobian matrix from all of the observables in the
        data_dictionary
        :param structure: pymatgen structure
        :param data_dictionary: dictionary containing all of the relevant observables
        """
        unique_ind = get_unique_indicies(structure)
        num_atoms = len(unique_ind)
        sym_dict = self.make_symmetry_dictionary(structure)
        observations = 0
        geometric_dict = data_dictionary['Bond_Distances']
        for bond_type in geometric_dict:
            observations += len(bond_type['pairs'])
        Jacobian = np.zeros([observations, 3*num_atoms])
        residuals = np.zeros(observations)
        perturbations = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        for atom_idx, atom in enumerate(unique_ind):
            for idx, vector in enumerate(perturbations):
                pert_vector = np.zeros(3*num_atoms)
                pert_vector[3*atom_idx + idx] = 1
                perturbed_structure = self.perturb_structure(structure, sym_dict, pert_vector)
                J_index = 0
                for bond_type in geometric_dict:
                    for neighbors in bond_type['pairs']:
                        if 'min_image_construction' in neighbors.keys():
                            mic = neighbors['min_image_construction']
                        else:
                            mic = True
                            
                        grad = self.calculate_gradients(
                            neighbors['true_pair'][0],
                            neighbors['true_pair'][1],
                            structure,
                            perturbed_structure,
                            epsilon = 1e-5,
                            bond_type = bond_type['bond'],
                            mic= mic
                            )
        
                        Jacobian[J_index, 3*atom_idx+idx] = grad
                        J_index +=1

        res_index = 0
        for bond_type in geometric_dict:
            for neighbors in bond_type['pairs']:
                        if 'min_image_construction' in neighbors.keys():
                            mic = neighbors['min_image_construction']
                        else:
                            mic = True
                            
                        res = self.calculate_residual(
                            neighbors['true_pair'][0],
                            neighbors['true_pair'][1],
                            structure,
                            bond_type = bond_type['bond'],
                            mic= mic
                            )
                        residuals[res_index] = res
                        res_index +=1

        
        return Jacobian, residuals

    def make_symmetry_dictionary(self, structure):
        sga = SpacegroupAnalyzer(structure)
        symmeterized_struc = sga.get_symmetrized_structure()
        sym_ops = sga.get_space_group_operations()
        atom_list = []

        for idx, equiv_group in enumerate(symmeterized_struc.equivalent_indices):
            base = structure[equiv_group[0]].frac_coords
            for atom in equiv_group:
                temp_coords = structure[atom].frac_coords
                for op in sym_ops:
                    coord = op.operate(base)
                    coord = np.array([i - math.floor(i) for i in coord])
                    if np.allclose(temp_coords, coord):
                        d = {
                            'atom': atom,
                            'base_atom': equiv_group[0],
                            'base_idx': idx,
                            'sym_op': op
                        }
                        atom_list.append(d)
                        break
        return atom_list

    # def calculate_residual_and_gradients(self, idx_1, idx_2, structure, bond_type, mic = True):
    #     """
    #     Calculates the residual and gradients between two points in a pymatgen structure
    #     :param idx_1: index of atom 1
    #     :param idx_2: index of atom 2
    #     :param structure: pymatgen structure
    #     :param bond_type: string denoting the type of bond
    #     """
    #     mu = self.distance_measures[bond_type]['mu']
    #     sigma = self.distance_measures[bond_type]['sigma']

    #     new_coords = coords_with_pbc(idx_1, idx_2, structure, mic)

    #     coord_1 = new_coords[0]
    #     coord_2 = new_coords[1]

    #     d = dist_from_coords(coord_1, coord_2)

    #     res = (d-mu)/sigma

    #     # ddx = (coord_2[0] - coord_1[0])/(d)#*sigma)
    #     # ddy = (coord_2[1] - coord_1[1])/(d)#*sigma)
    #     # ddz = (coord_2[2] - coord_1[2])/(d)#*sigma)

    #     ddx = (coord_2[0] - coord_1[0])/(d*sigma)
    #     ddy = (coord_2[1] - coord_1[1])/(d*sigma)
    #     ddz = (coord_2[2] - coord_1[2])/(d*sigma)

    #     return res, ddx, ddy, ddz


    # def assemble_residual_and_grad(self, structure, data_dictionary, epsilon = 1e-5):
    #     """
    #     Assembles a residual vector and jacobian matrix from all of the observables in the
    #     data_dictionary
    #     :param structure: pymatgen structure
    #     :param data_dictionary: dictionary containing all of the relevant observables
    #     """
    #     unique_ind = get_unique_indicies(structure)
    #     num_atoms = len(unique_ind)
    #     geometric_dict = data_dictionary['Bond_Distances']
    #     sub_Jacobian = 0
    #     residuals = []
    #     for bond_type in geometric_dict:
    #         J_row = np.zeros([len(bond_type['pairs']), 3*num_atoms])
    #         for i, neighbors in enumerate(bond_type['pairs']):
    #             if 'min_image_construction' in neighbors.keys():
    #                 mic = neighbors['min_image_construction']
    #             else:
    #                 mic = True
                    
    #             distances, grad_x, grad_y, grad_z = self.calculate_residual_and_gradients(
    #                 neighbors['true_pair'][0],
    #                 neighbors['true_pair'][1],
    #                 structure,
    #                 bond_type = bond_type['bond'],
    #                 mic= mic
    #                 )
    #             residuals.append(distances)
    #             pos_1 = unique_ind.index(neighbors['atom 1'])
    #             pos_2 = unique_ind.index(neighbors['atom 2'])
    #             J_row[i, 3*pos_2] = grad_x
    #             J_row[i, 3*pos_2+1] = grad_y
    #             J_row[i, 3*pos_2+2] = grad_z
    #             J_row[i, 3*pos_1] = -grad_x
    #             J_row[i, 3*pos_1+1] = -grad_y
    #             J_row[i, 3*pos_1+2] = -grad_z
    #         if type(sub_Jacobian) == int:
    #             sub_Jacobian = J_row
    #         else:
    #             sub_Jacobian = np.vstack((sub_Jacobian, J_row))
    #     residuals = np.hstack(residuals)
    #     return sub_Jacobian, residuals

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


# simple_line_search(self.fit_function[0], self.data_dictionary, self.structure, perturbations, sym_dict,)
def simple_line_search(function, data_dictionary, initial_struct, x_prime, sym_dict, chi = np.inf):
    prev_rev = chi
    alpha_residuals = []
    for idx, alpha in enumerate([1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]):
    # for idx, alpha in enumerate([0.001, 0.0001, 0.00001, 0.000001]):
        struct = copy.deepcopy(initial_struct)
        # file_path = '/Users/mvenetos/Box Sync/All Manuscripts/zeolite refinements/sigma_2_singlextal.cif'
        # struct = CifParser(file_path).get_structures(False)[0]

        perturbations = np.reshape(x_prime*alpha, (int(len(x_prime)/3), 3)) #perturbations*alpha
        for atom in sym_dict:
            base_idx = atom['base_idx']
            atom_idx = atom['atom']
            perturbation_opt = atom['sym_op'].apply_rotation_only(perturbations[base_idx])
            struct.translate_sites(atom_idx, -perturbation_opt, frac_coords=False)

        append_counter = f'_a{idx}'

        J, res = function.assemble_residual_and_grad(struct, data_dictionary)
        # J, res = pluck_distance_gradients(dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND)
        alpha_residuals.append(
            {
                'alpha': alpha,
                'chi': np.sum(res**2),
                'structure': struct
            }
        )
    return pd.DataFrame(alpha_residuals).sort_values(by = 'chi', ascending=True)

def line_search(initial_struct, x_prime, sym_dict, filepath = '/Users/mvenetos/Box Sync/All Manuscripts/zeolite refinements/sigma_2_temp/', cs_name = 'sigma_2_CS.json', J_name = 'sigma_2_J.json', chi = np.inf):

    temp_path = filepath + 'tmp'
    os.makedirs(temp_path, exist_ok=True)
    cspath = filepath + cs_name
    Jpath = filepath + J_name
    shutil.copy2(cspath, temp_path)
    shutil.copy2(Jpath, temp_path)
    prev_rev = chi
    alpha_residuals = []
    for idx, alpha in enumerate([1, 0.1, 0.01, 0.001, 0.0001,]): # 0.00001, 0.000001, 0.0000001]):
    # for idx, alpha in enumerate([0.001, 0.0001, 0.00001, 0.000001]):
        struct = copy.deepcopy(initial_struct)
        # file_path = '/Users/mvenetos/Box Sync/All Manuscripts/zeolite refinements/sigma_2_singlextal.cif'
        # struct = CifParser(file_path).get_structures(False)[0]

        perturbations = np.reshape(x_prime*alpha, (int(len(x_prime)/3), 3)) #perturbations*alpha
        for atom in sym_dict:
            base_idx = atom['base_idx']
            atom_idx = atom['atom']
            perturbation_opt = atom['sym_op'].apply_rotation_only(perturbations[base_idx])
            struct.translate_sites(atom_idx, -perturbation_opt, frac_coords=False)

        append_counter = f'_a{idx}'

        temp_cs = make_new_data(temp_path + '/' + cs_name, struct, append_counter)
        testing_cs = 'tmp/' + temp_cs.split('/')[-1]
        temp_J = make_new_data(temp_path + '/' + J_name, struct, append_counter)
        testing_J = 'tmp/' + temp_J.split('/')[-1]
        res, J = get_residuals_and_jacobian(test_dict, dist_test_dict, s, NUM_ATOMS, UNIQUE_IND, CS_data = testing_cs, J_data = testing_J)
        alpha_residuals.append(
            {
                'CS': temp_cs,
                'J': temp_J,
                'chi': np.sum(res**2)
            }
        )
        # if np.sum(res**2) < prev_rev:
        #     return pd.DataFrame(alpha_residuals).sort_values(by = 'chi', ascending=True)
    return pd.DataFrame(alpha_residuals).sort_values(by = 'chi', ascending=True)

def remove_tmp(filenames):
    root = filenames[0].split('tmp')[0]
    temp = []
    for path in filenames:
        filename = path.split('tmp/')[-1]
        temp.append(filename)
        shutil.move(path, os.path.join(root, filename))
    for file_items in os.listdir(os.path.join(root, 'tmp')):
        os.remove(os.path.join(root, 'tmp', file_items))
    os.rmdir(os.path.join(root, 'tmp'))
    return temp

#################################
## wolfe search
#################################

def update_chi2(function, alpha, x_prime, sym_dict, data_dictionary, initial_struct, NUM_ATOMS, UNIQUE_IND):
    struct= copy.deepcopy(initial_struct)
    perturbations = np.reshape(x_prime*alpha, (int(len(x_prime)/3), 3)) #perturbations*alpha
    for atom in sym_dict:
        base_idx = atom['base_idx']
        atom_idx = atom['atom']
        perturbation_opt = atom['sym_op'].apply_rotation_only(perturbations[base_idx])
        struct.translate_sites(atom_idx, -perturbation_opt, frac_coords=False)
    J, res = function.assemble_residual_and_grad(struct, data_dictionary)
    return np.sum(res**2)

def get_derivative(function, phi, alpha, x_prime, sym_dict, data_dictionary, initial_struct, NUM_ATOMS, UNIQUE_IND, epsilon = 0.01):
    phi_e = update_chi2(function, alpha + epsilon, x_prime, sym_dict, data_dictionary, initial_struct, NUM_ATOMS, UNIQUE_IND)
    return (phi_e - phi)/epsilon

# def quadratic_interpolation(alpha_high, phi_high, phi_low, dphi_low,):
#     alpha = -dphi_low*(alpha_high**2)/(2*(phi_high - phi_low - alpha_high*dphi_low))
#     return alpha

# def cubic_interpolation(alpha_high, phi_high, phi_low, dphi_low, alpha_int, phi_int):
#     mat1 = [[alpha_high**2, -(alpha_int**2)], [-(alpha_high**3), alpha_int**3]]
#     mat2 = [phi_int - phi_low - alpha_int*dphi_low, phi_high - phi_low - alpha_high*dphi_low]
#     a, b = np.matmul(mat1, mat2)/(alpha_high**2 * alpha_int**2 *(alpha_int-alpha_high))
#     alpha =  (-b + np.sqrt(b**2 - 3*a*dphi_low))/(3*a)
#     return alpha

def cubic_interpolation(alpha_prev, phi_alpha_prev, d_alpha_prev, alpha, phi_alpha, d_alpha):
    if alpha_prev < 0 or alpha < 0:
        print('Error in cubic interpolation: alpha is not > 0')
    d1 = d_alpha_prev + d_alpha - 3*(phi_alpha_prev - phi_alpha)/(alpha_prev - alpha)
    sign = np.sign(alpha - alpha_prev)
    d2 = sign*np.sqrt(d1**2 - d_alpha_prev*d_alpha)

    fraction = (d_alpha + d2 - d1)/(d_alpha - d_alpha_prev + 2*d2)
    alpha_next = alpha - (alpha-alpha_prev)*fraction

    return alpha_next



def zoom(function, phi_0, dphi_0, alpha_low, alpha_high, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND, epsilon = 0.01, c1 = 0.0001, c2 = 0.9):
    if alpha_low > alpha_high:
        print(f'Error: alpha_lo={alpha_low} > alpha_hi={alpha_high}')

    for i in range(15):
        phi_low = update_chi2(function, alpha_low, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND)
        dphi_low = get_derivative(function, phi_low, alpha_low, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND)

        phi_high = update_chi2(function, alpha_high, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND)
        dphi_high = get_derivative(function, phi_high, alpha_high, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND)


        alpha_j = cubic_interpolation(alpha_low, phi_low, dphi_low, alpha_high, phi_high, dphi_high)
        phi_j = update_chi2(function, alpha_j, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND)



        if phi_j > phi_0 + c1*alpha_j*dphi_0 or phi_j >= phi_low:
            # print(f'aplha high ({alpha_high}) is now alpha j ({alpha_j})')
            alpha_high = alpha_j
        else:
            dphi_j = get_derivative(function, phi_j, alpha_j, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND)

            if np.abs(dphi_j) <= -c2*dphi_0:
                return alpha_j, phi_j
            if dphi_j*(alpha_high - alpha_low) >= 0:
                # print(f'aplha high ({alpha_high}) is now alpha low ({alpha_low})')
                alpha_high = alpha_low
            # print(f'aplha low ({alpha_low}) is now alpha j ({alpha_j})')
            alpha_low = alpha_j
    print('Max iterations excedded on Zoom function')
    return alpha_j, phi_j

# def zoom(function, phi_0, dphi_0, alpha_low, alpha_high, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND, epsilon = 0.01, c1 = 0.0001, c2 = 0.9):
#     alpha_j = None
#     phi_j = None
#     for i in range(15):
#         phi_low = update_chi2(function, alpha_low, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND)
#         phi_lowe = update_chi2(function, alpha_low + epsilon, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND)
#         dphi_low = (phi_lowe - phi_low)/epsilon
#         phi_high = update_chi2(function, alpha_high, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND)
#         alpha_j = quadratic_interpolation(alpha_high, phi_high, phi_low, dphi_low)
#         phi_j = update_chi2(function, alpha_j, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND)

#         if phi_j > phi_low + c1*alpha_high*dphi_low:
#             alpha_j = cubic_interpolation(alpha_high, phi_high, phi_low, dphi_low, alpha_j, phi_j)
#             phi_j = update_chi2(function, alpha_j, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND)

#         if phi_j > phi_0 + c1*alpha_j*dphi_0 or phi_j >= phi_low:
#             alpha_high = alpha_j
#         else:
#             phi_j_e = update_chi2(function, alpha_j + epsilon, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND)
#             dphi_j = (phi_j_e - phi_j)/epsilon
#             if np.abs(dphi_j) <= -c2*dphi_0:
#                 return alpha_j, phi_j
#             if dphi_j*(alpha_high - alpha_low) >= 0:
#                 alpha_high = alpha_low
#             alpha_low = alpha_j
#     print('Max iterations excedded on Zoom function')
#     return alpha_j, phi_j

def wolfe_line_search(function, phi_0, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND, epsilon = 0.01, max_iter = 10, c1 = 0.0001, c2 = 0.9):
    alpha_max = 1#0.75
    dphi_0 = get_derivative(function, phi_0, 0, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND)
    # dphi_0 = (phi_e - phi_0)/epsilon
    alpha_prev = 0
    phi_alpha_prev = phi_0
    alpha = np.random.uniform(0, alpha_max) #1
    for i in range(max_iter):
        phi_alpha = update_chi2(function, alpha, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND)

        if phi_alpha > phi_0 + c1*alpha*dphi_0  or (i > 0 and phi_alpha > phi_alpha_prev):
            return zoom(function, phi_0, dphi_0, alpha_prev, alpha, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND)
        dphi_alpha = get_derivative(function, phi_alpha, 0, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND)
        # dphi_alpha = (phi_alpha_e - phi_alpha)/epsilon
        if np.abs(dphi_alpha) > -c2*dphi_0:
            return alpha, phi_alpha
        if dphi_alpha >= 0:
            return zoom(function, phi_0, dphi_0, alpha, alpha_prev, x_prime, sym_dict, dist_test_dict, struct, NUM_ATOMS, UNIQUE_IND)
        alpha_prev = alpha#, phi_alpha
        phi_alpha_prev = phi_alpha
        alpha = np.random.uniform(alpha, alpha_max)
    print('Max iterations excedded on Wolfe line search')
    return alpha_prev, phi_alpha_prev

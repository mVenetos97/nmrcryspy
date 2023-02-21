"""
Gauss-Newton Optimization module for Python
====================================================

nmrcryspy is an basic NMR crystallography package, capable of
refining the structures of solids against NMR and geometric data.


It aims to provide simple and efficient solution to NMR crystallographic
structure refinements. It includes tools for users to perform distance least
squares optimizations and use the power of Machine Learning to perform quick
refinements against NMR data.
"""
# version has to be specified at the start.
__author__ = "Maxwell C. Venetos"
__email__ = "mvenetos@berkeley.edu"
__license__ = "BSD License"
__maintainer__ = "Maxwell C. Venetos"
__status__ = "Beta"
__version__ = "0.1"

import copy
import math
from typing import Callable, List

import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from nmrcryspy.linesearch import wolfe_line_search
from nmrcryspy.utils import get_unique_indicies

# logger = logging.getLogger(__name__)


class Gauss_Newton_Solver:
    """The Gauss-Newton solver class.
    Given data dictionary, the crystal structure and a list of
    fit functions, minimize sum(residual^2) where
    residual = f(structure) - y.

    Attributes
    ----------

    fit_function: List of callable functions that need to be fitted;

    structure: a pymatgen structure containing the initial structure to be optimized.

    data_dictionary: a Dict object containing the observations for the fit_functions.

    max_iter: An int representing maximum number of iterations for optimization.

    tolerance_difference: A float. Terminate iteration if RMSE difference between
            iterations smaller than tolerance.

    tolerance: a float. Terminate iteration if RMSE is smaller than tolerance.
    """

    def __init__(
        self,
        fit_function: List[Callable],
        structure: Structure,
        data_dictionary: dict,
        max_iter: int = 1000,
        tolerance_difference: float = 10 ** (-10),
        tolerance: float = 10 ** (-9),
    ):
        self.fit_function = fit_function
        self.initial_structure = copy.deepcopy(structure)
        self.structure = structure
        self.data_dictionary = data_dictionary
        self.max_iter = max_iter
        self.tolerance_difference = tolerance_difference
        self.tolerance = tolerance

    def get_residuals_and_jacobian(self, data_dictionary, structure):
        jacobians = []
        residuals = []

        for function in self.fit_function:
            temp_Jacobian, temp_res = function.assemble_residual_and_grad(
                structure, data_dictionary
            )
            jacobians.append(temp_Jacobian)
            residuals.append(temp_res)
        Jacobian = np.vstack(jacobians)
        residual = np.hstack(residuals)

        return residual, Jacobian

    def fit(self):
        """
        Optimize the atom positions in structure by minimizing RMSE.

        """
        minimization_steps = []

        UNIQUE_IND = get_unique_indicies(self.structure)
        NUM_ATOMS = len(UNIQUE_IND)
        sym_dict = self.make_symmetry_dictionary()

        chi2_prev = np.inf
        res, J = self.get_residuals_and_jacobian(self.data_dictionary, self.structure)
        print(f"Initial: chi2 {np.sum(res**2)/81}")

        for k in range(self.max_iter):

            res, J = self.get_residuals_and_jacobian(
                self.data_dictionary, self.structure
            )

            j_pinv = self._calculate_pseudoinverse(J)
            perturbations = j_pinv @ res
            phi_0 = np.sum(res**2)

            # df = simple_line_search(self.fit_function[0], self.data_dictionary,
            # self.structure, perturbations, sym_dict,)
            # # print('chis: ', df.iloc[:]['chi'])
            # rmse = df.iloc[0]['chi']
            # alpha = df.iloc[0]['alpha']
            # phi = rmse
            # minimization_steps.append(df.iloc[0].to_dict())

            alpha, phi = wolfe_line_search(
                self.fit_function,  # TODO: make not first element
                phi_0,
                perturbations,
                sym_dict,
                self.data_dictionary,
                self.structure,
                NUM_ATOMS,
                UNIQUE_IND,
            )

            print(f"Round {k}: chi2 {phi/(3*NUM_ATOMS)}")  # with alpha {alpha}")
            if self.tolerance_difference is not None:
                diff = np.abs(chi2_prev - phi)
                if diff < self.tolerance_difference:
                    print(
                        "RMSE difference between iterations smaller than tolerance."
                        " Fit terminated."
                    )
                    return minimization_steps
            if phi < self.tolerance:
                print("RMSE error smaller than tolerance. Fit terminated.")
                return minimization_steps
            chi2_prev = phi
            self.updata_structure(self.structure, sym_dict, perturbations, alpha)
            minimization_steps.append(
                {
                    "step": k,
                    "alpha": alpha,
                    "chi": np.sum(phi**2),
                    "structure": self.structure,
                }
            )
        print("Max number of iterations reached. Fit didn't converge.")

        return minimization_steps

    def updata_structure(self, structure, sym_dict, x_prime, alpha):
        """description

        Attributes
        ----------
        """

        perturbations = np.reshape(x_prime * alpha, (int(len(x_prime) / 3), 3))
        for atom in sym_dict:
            base_idx = atom["base_idx"]
            atom_idx = atom["atom"]
            perturbation_opt = atom["sym_op"].apply_rotation_only(
                perturbations[base_idx]
            )
            structure.translate_sites(atom_idx, -perturbation_opt, frac_coords=False)
        self.structure = structure

    def make_symmetry_dictionary(self):
        """description

        Attributes
        ----------
        """
        sga = SpacegroupAnalyzer(self.structure)
        symmeterized_struc = sga.get_symmetrized_structure()
        sym_ops = sga.get_space_group_operations()
        atom_list = []

        for idx, equiv_group in enumerate(symmeterized_struc.equivalent_indices):
            base = self.structure[equiv_group[0]].frac_coords
            for atom in equiv_group:
                temp_coords = self.structure[atom].frac_coords
                for op in sym_ops:
                    coord = op.operate(base)
                    coord = np.array([i - math.floor(i) for i in coord])
                    if np.allclose(temp_coords, coord):
                        d = {
                            "atom": atom,
                            "base_atom": equiv_group[0],
                            "base_idx": idx,
                            "sym_op": op,
                        }
                        atom_list.append(d)
                        break
        return atom_list

    @staticmethod
    def _calculate_pseudoinverse(x: np.ndarray) -> np.ndarray:
        """Calculate the Moore-Penrose inverse of an array.

        Attributes
        ----------

        x: an np.ndarray. The input matrix.
        """
        return np.linalg.pinv(x.T @ x) @ x.T

import os
import random
import string
from typing import Callable

import numpy as np
import pandas as pd
import torch
from e3nn.io import CartesianTensor
from eigenn.dataset.LSDI import SiNMRDataMoldule
from eigenn.model_factory.atomic_tensor_model import AtomicTensorModel
from monty.serialization import dumpfn
from monty.serialization import loadfn

from nmrcryspy.utils import get_unique_indicies


def shielding_regr(sigma):
    """A regression function correlating the shielding tensor
    principle component, sigma_ii, to the chemical shift tensor
    principal component, delta_ii

    Args:
        sigma: float representing the shielding tensor principal
            component.

    Returns: float
    """
    return (0.8292 * sigma) - 437.69


def J_regr(J):
    """A regression function correlating the DFT-calculated
    J-coupling value to an experimentally observed J-coupling.

    Args:
        J: float representing the DFT-calculated J-coupling.

    Returns: float
    """
    return (1.4217 * J) + 3.7953


def randomword(length):
    """Returns a random lowercase alphabetical string of length, length.

    Args:
        length: a float of length of the string to return.

    Returns: string
    """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


class ML_function:
    """Base class for the machine learning functions. Contains useful
    functions for the creation of the data files used by the ML function.

    Attributes
    ----------

    root: string filepath to data used for ML model

    data_file: string name of initial data file for ML model
    """

    def __init__(
        self,
        root: str = None,
        data_file: str = None,
    ):
        """ """
        self.root = root
        self.data_file = data_file

    def make_file_from_structure(self, structure):
        """Takes a structure and creates the corresponding ML data file for
        that structure

        Args:
            structure: pymatgen.Structure object corresponding to the data file.

        Return: string, string
        """
        original_file = "".join([self.root, self.data_file])
        temp = pd.DataFrame(loadfn(original_file))
        idxs = list(temp.index.values)

        for i in idxs:
            temp.at[i, "structure"] = structure

        tmp_filepath = os.path.join(self.root, "tmp")
        new_name = randomword(10) + ".json"
        os.makedirs(tmp_filepath, exist_ok=True)

        data = temp.to_dict("records")
        dumpfn(data, os.path.join(tmp_filepath, new_name))

        return tmp_filepath, new_name

    def remove_tmp_folder(self):
        """
        File to cleanup the generated scratch files.
        """
        root = self.root
        for file_items in os.listdir(os.path.join(root, "tmp")):
            os.remove(os.path.join(root, "tmp", file_items))
        os.rmdir(os.path.join(root, "tmp"))

    def get_unique_atoms(self, structure):
        """Finds the symmetrically unique atoms of a structure object.

        Args:
            structure: pymatgen.Structure object to find symmetrically unique atoms.

        Returns: int
        """
        unique_ind = get_unique_indicies(structure)
        return len(unique_ind)


class ShieldingTensor_Function(ML_function):
    """ShieldingTensor_Function class for the prediction and residuals
    of the shielding tensor ML model.

    Attributes
    ----------

    sigma_errors: Dictionary containing the standard
        deviation of sigma_11, sigma_22, sigma_33 where the order
        of the sigma_ii's follow the standard convention of
        sigma_11 > sigma_22 > sigma_33.
        Example: {'sigma_11': 0.4, 'sigma_22': 2.5, 'sigma_33': 0.7}

    regr_func: Callable Regression function used to convert shielding
        values to shift values

    r_cut: float representing cutoff radius used to define the
        local neighborhood in the GNN

    checkpoint: string containing name of the checkpoint file containing
        the GNN model

    root: string filepath to data used for ML model

    data_file: string name of initial data file for ML model
    """

    def __init__(
        self,
        sigma_errors: dict = None,
        regr_func: Callable = shielding_regr,
        r_cut: float = 5,
        checkpoint: str = None,
        root: str = None,
        data_file: str = None,
    ):
        super().__init__(root, data_file)
        self.sigma_errors = sigma_errors
        self.regr_func = regr_func
        self.r_cut = r_cut
        self.checkpoint = checkpoint
        # self.root = root
        # self.data_file = data_file

    def predict(self):
        """Function to predict the shielding tensor. This function does not apply
            the regression calibration.

        Args:
            root: string containing the data file location for the calculation of the
                gradient and residual.
            data_file: string containing the name of the data file to be used for the
                gradient and residual.

        Returns: list
        """
        data_file = self.data_file
        root = self.root
        converter = CartesianTensor(formula="ij=ji")
        model = AtomicTensorModel.load_from_checkpoint(
            self.checkpoint, strict=True, verbose=False
        )

        dm = SiNMRDataMoldule(
            trainset_filename=data_file,
            valset_filename=data_file,
            testset_filename=data_file,
            r_cut=self.r_cut,
            symmetric=True,
            root=root,
        )

        dm.prepare_data()
        dm.setup()

        loader = dm.val_dataloader()

        return_data = []

        for data_point in loader:
            graphs, labels = model(data_point)

            pred = graphs["tensor_output"][0]

            pred = converter.to_cartesian(pred)
    
            return_data.append(pred)
        return return_data

    def calculate_grad_and_residual(self, root, data_file):
        """Function to calculate the gradient and residual for processing by
        the assemble_residual_and_grad function.

        Args:
            root: string containing the data file location for the calculation of the
                gradient and residual.

            data_file: string contaning the name of the data file to be used for the
                gradient and residual.

        Returns: list
        """
        converter = CartesianTensor(formula="ij=ji")
        model = AtomicTensorModel.load_from_checkpoint(
            self.checkpoint, strict=True, verbose=False
        )

        dm = SiNMRDataMoldule(
            trainset_filename=data_file,
            valset_filename=data_file,
            testset_filename=data_file,
            r_cut=self.r_cut,
            symmetric=True,
            root=root,
        )

        dm.prepare_data()
        dm.setup()

        loader = dm.val_dataloader()

        return_data = []

        for data_point in loader:
            graphs, labels = model(data_point)

            true = labels["tensor_output"][0]
            pred = graphs["tensor_output"][0]

            true = torch.sort(torch.linalg.eigvals(true).real, descending=True).values
            pred = converter.to_cartesian(pred)
            pred = torch.sort(torch.linalg.eigvals(pred).real, descending=True)

            grads_11 = torch.autograd.grad(
                outputs=self.regr_func(pred.values[0]),
                inputs=data_point.pos,
                retain_graph=True,
            )
            grads_22 = torch.autograd.grad(
                outputs=self.regr_func(pred.values[1]),
                inputs=data_point.pos,
                retain_graph=True,
            )
            grads_33 = torch.autograd.grad(
                outputs=self.regr_func(pred.values[2]),
                inputs=data_point.pos,
                retain_graph=True,
            )
            residual = self.regr_func(pred.values) - true

            residual[0] = residual[0] / self.sigma_errors["sigma_11"]
            residual[1] = residual[1] / self.sigma_errors["sigma_22"]
            residual[2] = residual[2] / self.sigma_errors["sigma_33"]

            d = {
                "index": data_point.y["node_masks"].tolist().index(True),
                "residual": residual,
                "grad_11": grads_11,
                "grad_22": grads_22,
                "grad_33": grads_33,
            }
            return_data.append(d)
        return return_data

    def assemble_residual_and_grad(
        self,
        structure,
        data_dictionary,
    ):
        """Function to package the Jacobian matrix and residual vector for the
        Gauss_Newton_Solver class.

        Args:
            structure: pymatgen.Structure object used to calculate the residual
                and Jacobian.
            data_dictionary: Dict of the data_dictionary attribute from the
                Gauss_Newton_Solver which contains the ML data.

        Returns: np.ndarray, np.array
        """

        root, data_file = self.make_file_from_structure(structure)
        num_atoms = self.get_unique_atoms(structure)
        data_dictionary = data_dictionary["Shielding_Tensor"]

        sub_Jacobian = 0
        residuals = np.array([])

        shielding_data = sorted(
            self.calculate_grad_and_residual(root, data_file), key=lambda d: d["index"]
        )

        for preds in shielding_data:
            temp_res = preds["residual"].detach().numpy()
            residuals = (
                np.hstack([residuals, temp_res])
                if residuals.size
                else np.array(temp_res)
            )
            # residuals.append(preds["residual"].detach().numpy())
            dict_row = list(
                filter(lambda atom: atom["index"] == preds["index"], data_dictionary)
            )[0]
            J_row = np.zeros([3, 3 * num_atoms])
            for i, neighbors in enumerate(dict_row["neighbor_idx"]):
                J_row[0, 3 * i] = (
                    preds["grad_11"][0][neighbors][0].item()
                    / self.sigma_errors["sigma_11"]
                )
                J_row[0, 3 * i + 1] = (
                    preds["grad_11"][0][neighbors][1].item()
                    / self.sigma_errors["sigma_11"]
                )
                J_row[0, 3 * i + 2] = (
                    preds["grad_11"][0][neighbors][2].item()
                    / self.sigma_errors["sigma_11"]
                )
                J_row[1, 3 * i] = (
                    preds["grad_22"][0][neighbors][0].item()
                    / self.sigma_errors["sigma_22"]
                )
                J_row[1, 3 * i + 1] = (
                    preds["grad_22"][0][neighbors][1].item()
                    / self.sigma_errors["sigma_22"]
                )
                J_row[1, 3 * i + 2] = (
                    preds["grad_22"][0][neighbors][2].item()
                    / self.sigma_errors["sigma_22"]
                )
                J_row[2, 3 * i] = (
                    preds["grad_33"][0][neighbors][0].item()
                    / self.sigma_errors["sigma_33"]
                )
                J_row[2, 3 * i + 1] = (
                    preds["grad_33"][0][neighbors][1].item()
                    / self.sigma_errors["sigma_33"]
                )
                J_row[2, 3 * i + 2] = (
                    preds["grad_33"][0][neighbors][2].item()
                    / self.sigma_errors["sigma_33"]
                )
            if type(sub_Jacobian) == int:
                sub_Jacobian = J_row
            else:
                sub_Jacobian = np.vstack((sub_Jacobian, J_row))

        # self.remove_tmp_folder()

        return sub_Jacobian, residuals


class JTensor_Function(ML_function):
    """JTensor_Function class for the prediction and residuals
    of the J tensor ML model.

    Attributes
    ----------

    J_error: The standard deviation of J coupling

    regr_func: Callable Regression function used to convert shielding
        values to shift values

    r_cut: float representing cutoff radius used to define the
        local neighborhood in the GNN

    checkpoint: string containing name of the checkpoint file containing
        the GNN model

    root: string filepath to data used for ML model

    data_file: string name of initial data file for ML model
    """

    def __init__(
        self,
        J_error: float = None,
        regr_func: Callable = J_regr,
        r_cut: float = 6,
        checkpoint: str = None,
        root: str = None,
        data_file: str = None,
    ):
        super().__init__(root, data_file)
        self.J_error = J_error
        self.regr_func = regr_func
        self.r_cut = r_cut
        self.checkpoint = checkpoint
        # self.root = root
        # self.data_file = data_file

    def predict(self, root, data_file):
        """Function to predict the J coupling tensor. This function does not apply
            the regression calibration function.

        Args:
            root: string containing the data file location for the calculation of the
                gradient and residual.
            data_file: string containing the name of the data file to be used for the
                gradient and residual.

        Returns: list
        """
        data_file = self.data_file
        root = self.root
        model = AtomicTensorModel.load_from_checkpoint(
            self.checkpoint, strict=True, verbose=False
        )

        dm = SiNMRDataMoldule(
            trainset_filename=data_file,
            valset_filename=data_file,
            testset_filename=data_file,
            r_cut=self.r_cut,
            symmetric=True,
            root=root,
        )

        dm.prepare_data()
        dm.setup()

        loader = dm.val_dataloader()

        return_data = []

        for data_point in loader:
            graphs, labels = model(data_point)

            pred = graphs["tensor_output"][0]

            return_data.append(pred)

        return return_data

    def calculate_grad_and_residual(self, root, data_file):
        """Function to calculate the gradient and residual for processing by
        the assemble_residual_and_grad function.

        Args:
            root: string containing the data file location for the calculation of the
                gradient and residual.
            data_file: string containg the name of the data file to be used for the
                gradient and residual.

        Returns: list
        """
        # converter = CartesianTensor(formula="ij=ji")
        model = AtomicTensorModel.load_from_checkpoint(
            self.checkpoint, strict=True, verbose=False
        )

        dm = SiNMRDataMoldule(
            trainset_filename=data_file,
            valset_filename=data_file,
            testset_filename=data_file,
            r_cut=self.r_cut,
            symmetric=True,
            root=root,
        )

        dm.prepare_data()
        dm.setup()

        loader = dm.val_dataloader()

        return_data = []

        for data_point in loader:
            graphs, labels = model(data_point)

            true = labels["tensor_output"][0]
            pred = graphs["tensor_output"][0]

            true = (true[0][0] + true[1][1] + true[2][2]) / 3
            pred = self.regr_func((pred[0][0] + pred[1][1] + pred[2][2]) / 3)  # /2

            grads = torch.autograd.grad(
                outputs=pred, inputs=data_point.pos, retain_graph=True
            )
            d = {
                "index": data_point.y["node_masks"].tolist().index(True),
                "residual": (pred - true)
                / self.J_error,  # (true-pred)/sigma, #(pred - true)/sigma,
                "grad": grads,
            }
            return_data.append(d)

        return return_data

    def assemble_residual_and_grad(
        self,
        structure,
        data_dictionary,
    ):
        """Function to package the Jacobian matrix and residual vector for the
        Gauss_Newton_Solver class.

        Args:
            structure: pymatgen.Structure object used to calculate the residual
                and Jacobian.
            data_dictionary: Dict of the data_dictionary attribute from the
                Gauss_Newton_Solver which contains the ML data.

        Returns: np.ndarray, np.array
        """

        root, data_file = self.make_file_from_structure(structure)
        num_atoms = self.get_unique_atoms(structure)
        data_dictionary = data_dictionary["J_Tensor"]

        sub_Jacobian = 0
        residuals = []

        J_data = sorted(
            self.calculate_grad_and_residual(root, data_file), key=lambda d: d["index"]
        )

        for preds in J_data:
            residuals.append(preds["residual"].item())
            dict_row = list(
                filter(lambda atom: atom["index"] == preds["index"], data_dictionary)
            )[0]
            J_row = np.zeros(3 * num_atoms)
            for i, neighbors in enumerate(dict_row["neighbor_idx"]):
                J_row[3 * i] = preds["grad"][0][neighbors][0].item() / self.J_error
                J_row[3 * i + 1] = preds["grad"][0][neighbors][1].item() / self.J_error
                J_row[3 * i + 2] = preds["grad"][0][neighbors][2].item() / self.J_error
            if type(sub_Jacobian) == int:
                sub_Jacobian = J_row
            else:
                sub_Jacobian = np.vstack((sub_Jacobian, J_row))

        # self.remove_tmp_folder()

        return sub_Jacobian, np.array(residuals)

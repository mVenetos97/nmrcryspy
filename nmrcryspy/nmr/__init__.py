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
    return (0.8292 * sigma) - 437.69


def J_regr(J):
    return (1.4217 * J) + 3.7953


def randomword(length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


class ML_function:
    def __init__(
        self,
        root: str = None,
        data_file: str = None,
    ):
        """
        :param root: filepath to data used for ML model
        :param data_file: name of initial data file for ML model
        """
        self.root = root
        self.data_file = data_file

    def make_file_from_structure(self, structure):
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
        root = self.root
        for file_items in os.listdir(os.path.join(root, "tmp")):
            os.remove(os.path.join(root, "tmp", file_items))
        os.rmdir(os.path.join(root, "tmp"))

    def get_unique_atoms(self, structure):
        unique_ind = get_unique_indicies(structure)
        return len(unique_ind)


class ShieldingTensor_Function(ML_function):
    """
    Function to calculate the distance and gradients between points.
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
        """
        :param sigma_errors: Dictionary containing the standard
        deviation of sigma_11, sigma_22, sigma_33 where the order
        of the sigma_ii's follow the standard convention of
        sigma_11 > sigma_22 > sigma_33.
        Example: {'sigma_11': 0.4, 'sigma_22': 2.5, 'sigma_33': 0.7}
        :param regr_func: Regression function used to convert shielding
        values to shift values
        :param r_cut: cutoff radius used to define the local neighborhood
        in the GNN
        :param checkpoint: name of the checkpoint file containing the GNN model
        :param root: filepath to data used for ML model
        :param data_file: name of initial data file for ML model
        """
        super().__init__(root, data_file)
        self.sigma_errors = sigma_errors
        self.regr_func = regr_func
        self.r_cut = r_cut
        self.checkpoint = checkpoint
        # self.root = root
        # self.data_file = data_file

    def calculate_grad_and_residual(self, root, data_file):
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
                J_row[0, 3 * i] = preds["grad_11"][0][neighbors][0].item()
                J_row[0, 3 * i + 1] = preds["grad_11"][0][neighbors][1].item()
                J_row[0, 3 * i + 2] = preds["grad_11"][0][neighbors][2].item()
                J_row[1, 3 * i] = preds["grad_22"][0][neighbors][0].item()
                J_row[1, 3 * i + 1] = preds["grad_22"][0][neighbors][1].item()
                J_row[1, 3 * i + 2] = preds["grad_22"][0][neighbors][2].item()
                J_row[2, 3 * i] = preds["grad_33"][0][neighbors][0].item()
                J_row[2, 3 * i + 1] = preds["grad_33"][0][neighbors][1].item()
                J_row[2, 3 * i + 2] = preds["grad_33"][0][neighbors][2].item()
            if type(sub_Jacobian) == int:
                sub_Jacobian = J_row
            else:
                sub_Jacobian = np.vstack((sub_Jacobian, J_row))

        # self.remove_tmp_folder()

        return sub_Jacobian, residuals


class JTensor_Function(ML_function):
    """
    Function to calculate the distance and gradients between points.
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
        """
        :param sigma_errors: The standard deviation of J coupling
        :param regr_func: Regression function used to ab initio J coupling
        values to experimental J coupling values
        :param r_cut: cutoff radius used to define the local neighborhood
        in the GNN
        :param checkpoint: name of the checkpoint file containing the GNN model
        :param root: filepath to data used for ML model
        :param data_file: name of initial data file for ML model
        """
        super().__init__(root, data_file)
        self.J_error = J_error
        self.regr_func = regr_func
        self.r_cut = r_cut
        self.checkpoint = checkpoint
        # self.root = root
        # self.data_file = data_file

    def calculate_grad_and_residual(self, root, data_file):
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

        root, data_file = self.make_file_from_structure(structure)
        num_atoms = self.get_unique_atoms(structure)
        data_dictionary = data_dictionary["J_Tensor"]

        sub_Jacobian = 0
        residuals = []

        J_data = sorted(
            self.calculate_grad_and_residual(root, data_file), key=lambda d: d["index"]
        )

        for preds in J_data:
            residuals.append(preds["residual"].detach().numpy())
            dict_row = list(
                filter(lambda atom: atom["index"] == preds["index"], data_dictionary)
            )[0]
            J_row = np.zeros(3 * num_atoms)
            for i, neighbors in enumerate(dict_row["neighbor_idx"]):
                J_row[3 * i] = preds["grad"][0][neighbors][0].item()
                J_row[3 * i + 1] = preds["grad"][0][neighbors][1].item()
                J_row[3 * i + 2] = preds["grad"][0][neighbors][2].item()
            if type(sub_Jacobian) == int:
                sub_Jacobian = J_row
            else:
                sub_Jacobian = np.vstack((sub_Jacobian, J_row))

        # self.remove_tmp_folder()

        return sub_Jacobian, residuals

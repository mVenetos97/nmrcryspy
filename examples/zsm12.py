import os
import sys
from pathlib import Path
import time
from monty.serialization import loadfn, dumpfn

path_root = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(path_root)
sys.path.append(str(path_root))
# print(sys.path)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pymatgen.io.cif import CifParser, CifWriter

from nmrcryspy import Gauss_Newton_Solver
from nmrcryspy.geometric import Distance_Function
from nmrcryspy.nmr import ShieldingTensor_Function, JTensor_Function
from nmrcryspy.utils import get_unique_indicies
from nmrcryspy.utils import dist_from_coords


file_path = (
    "/Users/mvenetos/Box Sync/All Manuscripts/"
    "zeolite refinements/ZSM-12_DLS.cif"
)
checkpoint_file_path = (
    "/Users/mvenetos/Documents/" "Jupyter Testing Grounds/eigenn_testing"
)
shielding_chk = "".join([checkpoint_file_path, "/EigennBestTensor.ckpt"])
j_chk = "".join([checkpoint_file_path, "/j_coupling.ckpt"])

s = CifParser(file_path).get_structures(False)[0]


data = {
    "Bond_Distances": [
        {
            "bond": "SiSi",
            "pairs": [  # 11 total
                {"atom 1": 0, "atom 2": 8, "true_pair": [0, 13]},  # new
                {"atom 1": 0, "atom 2": 8, "true_pair": [0, 15]},
                {"atom 1": 0, "atom 2": 16, "true_pair": [0, 16]},
                # {'atom 1': 0, 'atom 2': 16, 'true_pair': [0, 16]},
                {
                    "atom 1": 0,
                    "atom 2": 16,
                    "true_pair": [0, 16],
                    "min_image_construction": False,
                },
                {"atom 1": 8, "atom 2": 24, "true_pair": [8, 24]},
                # {'atom 1': 8, 'atom 2': 24, 'true_pair': [8, 24]},
                {
                    "atom 1": 8,
                    "atom 2": 24,
                    "true_pair": [8, 24],
                    "min_image_construction": False,
                },
                {"atom 1": 16, "atom 2": 32, "true_pair": [16, 32]},
                {"atom 1": 16, "atom 2": 48, "true_pair": [16, 48]},
                {"atom 1": 24, "atom 2": 32, "true_pair": [24, 37]},
                {"atom 1": 24, "atom 2": 40, "true_pair": [24, 40]},
                {"atom 1": 32, "atom 2": 40, "true_pair": [32, 40]},
                # {'atom 1': 32, 'atom 2': 40, 'true_pair': [32, 40]},
                {
                    "atom 1": 32,
                    "atom 2": 40,
                    "true_pair": [32, 40],
                    "min_image_construction": False,
                },
                {"atom 1": 40, "atom 2": 48, "true_pair": [40, 54]},
                {"atom 1": 48, "atom 2": 48, "true_pair": [48, 54]},  # new
            ],
        },
        {
            "bond": "SiO",
            "pairs": [  # 28 total
                {"atom 1": 0, "atom 2": 56, "true_pair": [0, 56]},
                {"atom 1": 0, "atom 2": 64, "true_pair": [0, 64]},
                {"atom 1": 0, "atom 2": 72, "true_pair": [0, 72]},
                {"atom 1": 0, "atom 2": 144, "true_pair": [0, 144]},
                {"atom 1": 8, "atom 2": 88, "true_pair": [8, 88]},
                {"atom 1": 8, "atom 2": 64, "true_pair": [8, 71]},
                {"atom 1": 8, "atom 2": 56, "true_pair": [8, 61]},
                {"atom 1": 8, "atom 2": 152, "true_pair": [8, 152]},
                {"atom 1": 16, "atom 2": 144, "true_pair": [16, 144]},
                {"atom 1": 16, "atom 2": 72, "true_pair": [16, 72]},
                {"atom 1": 16, "atom 2": 96, "true_pair": [16, 96]},
                {"atom 1": 16, "atom 2": 80, "true_pair": [16, 80]},
                {"atom 1": 24, "atom 2": 88, "true_pair": [24, 88]},
                {"atom 1": 24, "atom 2": 120, "true_pair": [24, 120]},
                {"atom 1": 24, "atom 2": 128, "true_pair": [24, 128]},
                {"atom 1": 24, "atom 2": 152, "true_pair": [24, 152]},
                {"atom 1": 32, "atom 2": 112, "true_pair": [32, 112]},
                {"atom 1": 32, "atom 2": 160, "true_pair": [32, 160]},
                {"atom 1": 32, "atom 2": 80, "true_pair": [32, 80]},
                {"atom 1": 32, "atom 2": 128, "true_pair": [32, 133]},
                {"atom 1": 40, "atom 2": 112, "true_pair": [40, 112]},
                {"atom 1": 40, "atom 2": 120, "true_pair": [40, 120]},
                {"atom 1": 40, "atom 2": 160, "true_pair": [40, 160]},
                {"atom 1": 40, "atom 2": 136, "true_pair": [40, 142]},
                {"atom 1": 48, "atom 2": 104, "true_pair": [48, 104]},
                {"atom 1": 48, "atom 2": 136, "true_pair": [48, 136]},
                {"atom 1": 48, "atom 2": 96, "true_pair": [48, 96]},
                {"atom 1": 48, "atom 2": 104, "true_pair": [48, 110]},  # new
            ],
        },
        {
            "bond": "OO",
            "pairs": [  # 42 total
                {"atom 1": 56, "atom 2": 64, "true_pair": [56, 64]},
                {"atom 1": 56, "atom 2": 72, "true_pair": [56, 72]},
                {"atom 1": 56, "atom 2": 144, "true_pair": [56, 144]},
                {"atom 1": 64, "atom 2": 72, "true_pair": [64, 72]},
                {"atom 1": 64, "atom 2": 144, "true_pair": [64, 144]},
                {"atom 1": 72, "atom 2": 144, "true_pair": [72, 144]},
                {
                    "atom 1": 56,
                    "atom 2": 64,
                    "true_pair": [61, 71],
                },  # grad here is questionable
                {
                    "atom 1": 56,
                    "atom 2": 88,
                    "true_pair": [61, 88],
                },  # grad here is questionable
                {
                    "atom 1": 56,
                    "atom 2": 152,
                    "true_pair": [61, 152],
                },  # grad here is questionable
                {
                    "atom 1": 64,
                    "atom 2": 88,
                    "true_pair": [71, 88],
                },  # grad here is questionable
                {
                    "atom 1": 64,
                    "atom 2": 152,
                    "true_pair": [71, 152],
                },  # grad here is questionable
                {"atom 1": 88, "atom 2": 152, "true_pair": [88, 152]},
                {"atom 1": 72, "atom 2": 80, "true_pair": [72, 80]},
                {"atom 1": 72, "atom 2": 96, "true_pair": [72, 96]},
                {
                    "atom 1": 72,
                    "atom 2": 144,
                    "true_pair": [72, 144],
                },  # 'min_image_construction': False},
                {"atom 1": 80, "atom 2": 96, "true_pair": [80, 96]},
                {"atom 1": 80, "atom 2": 144, "true_pair": [80, 144]},
                {"atom 1": 96, "atom 2": 144, "true_pair": [96, 144]},
                {"atom 1": 88, "atom 2": 120, "true_pair": [88, 120]},
                {"atom 1": 88, "atom 2": 128, "true_pair": [88, 128]},
                {
                    "atom 1": 88,
                    "atom 2": 152,
                    "true_pair": [88, 152],
                },  # 'min_image_construction': False},
                {"atom 1": 120, "atom 2": 128, "true_pair": [120, 128]},
                {"atom 1": 120, "atom 2": 152, "true_pair": [120, 152]},
                {"atom 1": 128, "atom 2": 152, "true_pair": [128, 152]},
                {"atom 1": 80, "atom 2": 112, "true_pair": [80, 112]},
                {"atom 1": 80, "atom 2": 128, "true_pair": [80, 133]},
                {"atom 1": 80, "atom 2": 160, "true_pair": [80, 160]},
                {"atom 1": 112, "atom 2": 128, "true_pair": [112, 133]},
                {
                    "atom 1": 112,
                    "atom 2": 160,
                    "true_pair": [112, 160],
                },  # 'min_image_construction': False},
                {
                    "atom 1": 128,
                    "atom 2": 160,
                    "true_pair": [133, 160],
                },  # grad here is questionable
                {"atom 1": 112, "atom 2": 120, "true_pair": [112, 120]},
                {"atom 1": 112, "atom 2": 136, "true_pair": [112, 142]},
                {"atom 1": 112, "atom 2": 160, "true_pair": [112, 160]},
                {"atom 1": 120, "atom 2": 136, "true_pair": [120, 142]},
                {
                    "atom 1": 120,
                    "atom 2": 160,
                    "true_pair": [120, 160],
                },  # 'min_image_construction': False},
                {
                    "atom 1": 136,
                    "atom 2": 160,
                    "true_pair": [142, 160],
                },  # grad here is questionable
                {"atom 1": 96, "atom 2": 104, "true_pair": [96, 104]},
                {"atom 1": 96, "atom 2": 104, "true_pair": [96, 110]},  # new
                {"atom 1": 96, "atom 2": 136, "true_pair": [96, 136]},
                {"atom 1": 104, "atom 2": 136, "true_pair": [104, 136]},
                {"atom 1": 104, "atom 2": 104, "true_pair": [104, 110]},  # new
                {
                    "atom 1": 104,
                    "atom 2": 136,
                    "true_pair": [110, 136],
                },  # new #grad here is questionable
            ],
        },
    ],
    "Shielding_Tensor": [
        {
            "index": 0,
            "target": "shielding",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
        {
            "index": 8,
            "target": "shielding",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
        {
            "index": 16,
            "target": "shielding",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
        {
            "index": 24,
            "target": "shielding",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
        {
            "index": 32,
            "target": "shielding",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
        {
            "index": 40,
            "target": "shielding",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
        {
            "index": 48,
            "target": "shielding",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
    ],
    "J_Tensor": [
        {
            "index": 56,
            "target": "J",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
        {
            "index": 64,
            "target": "J",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
        {
            "index": 80,
            "target": "J",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
        {
            "index": 88,
            "target": "J",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
        {
            "index": 96,
            "target": "J",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
        {
            "index": 112,
            "target": "J",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
        {
            "index": 120,
            "target": "J",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
        {
            "index": 128,
            "target": "J",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
        {
            "index": 136,
            "target": "J",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
        {
            "index": 144,
            "target": "J",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
        {
            "index": 152,
            "target": "J",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
        {
            "index": 160,
            "target": "J",
            "neighbor_idx": [
                0,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
            ],
        },
    ],
}

distance_dict = {
      'SiO': {'mu' : 1.595, 'sigma' : 0.011},
    'OO': {'mu' : 2.604, 'sigma' : 0.025},
    'SiSi': { 'mu' : 3.101, 'sigma' : 0.041}
}
# distance_dict = {
#     "SiO": {"mu": 1.6, "sigma": 0.01},
#     "OO": {"mu": 2.61, "sigma": 0.02},
#     "SiSi": {"mu": 3.1, "sigma": 0.05},
# }
zeolite_dists = Distance_Function(distance_dict)

shielding_dict = {"sigma_11": 0.4, "sigma_22": 2.5, "sigma_33": 0.7}
zeolite_shieldings = ShieldingTensor_Function(
    sigma_errors=shielding_dict,
    checkpoint=shielding_chk,
    root="/Users/mvenetos/Box Sync/All Manuscripts/zeolite " "refinements/ZSM12_temp/",
    data_file="ZSM12_CS.json",
)

zeolite_j = JTensor_Function(
    J_error=1.5,
    checkpoint=j_chk,
    root="/Users/mvenetos/Box Sync/All Manuscripts/zeolite " "refinements/ZSM12_temp/",
    data_file="ZSM12_J_INADEQUATE.json",
)

print(f"There are {3*len(get_unique_indicies(s))} degrees of freedom")

gn = Gauss_Newton_Solver(
    fit_function=[zeolite_shieldings, zeolite_j, zeolite_dists],
    structure=s,
    data_dictionary=data,
    max_iter=20,
    tolerance_difference=1e-8,
)



start = time.time()

test = pd.DataFrame(gn.fit()).sort_values(by="chi", ascending=True)

end = time.time()
print('Time taken = ', end - start)

dist_test_dict = data["Bond_Distances"]
distributions = []

s = test.iloc[0]["structure"]
CifWriter(s).write_file("/Users/mvenetos/Desktop/INADEQUATE_zsm12.cif")

for idx, layer in enumerate(["SiSi", "SiO", "OO"]):
    temp = []
    print(layer)
    for i in dist_test_dict[idx]["pairs"]:
        if "min_image_construction" in i.keys():
            coord1 = s[i["true_pair"][0]].coords
            coord2 = s[i["true_pair"][1]].coords
            temp.append(dist_from_coords(coord1, coord2))
        else:
            temp.append(s.get_distance(i["true_pair"][0], i["true_pair"][1]))
        # if layer == 'OO':
        # n1 = i['atom 1']
        # n2 = i['atom 2']
        # print(temp)
    # print()
    distributions.append({"bond": layer, "data": temp})
df = pd.DataFrame(distributions)
data = df.to_dict("records")
dumpfn(data, '/Users/mvenetos/Box Sync/All Manuscripts/zeolite refinements/ZSM12_temp/ZSM12_opt_data.json')


# print(f'Sisi is len{len(test_sisi)}')
# print(f'OO is len{len(test_oo)}')
# fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
# ax[2].hist(
#     df.iloc[0]["data"], np.array(range(147, 163)) / 50, edgecolor="black", alpha=0.5
# )
# ax[2].set_title("SiSi")
# ax[2].set_xlim([2.94, 3.26])
# ax[2].set_ylim([0, 4])
# ax[2].set_xticks([3, 3.08, 3.16, 3.24])

# ax[0].hist(
#     df.iloc[1]["data"], np.array(range(2 * 149, 2 * 171)) / 200, edgecolor="black"
# )
# ax[0].set_title("SiO")
# ax[0].set_xlim([1.49, 1.7])
# ax[0].set_ylim([0, 22])
# ax[0].set_xticks([1.5, 1.54, 1.58, 1.62, 1.66, 1.7])
# ax[1].hist(
#     df.iloc[2]["data"], np.array(range(248, 271)) / 100, edgecolor="black", alpha=0.5
# )
# ax[1].set_title("OO")
# ax[1].set_xlim([2.48, 2.7])
# ax[1].set_ylim([0, 28])
# ax[1].set_xticks([2.5, 2.54, 2.58, 2.62, 2.66, 2.70])
# plt.tight_layout()
# plt.show()

# print(len(df.iloc[0]["data"]))
# print(len(df.iloc[1]["data"]))
# print(len(df.iloc[2]["data"]))

print('Time taken = ', end - start)
print("completed")

# Initial: chi2 206.23983627103604
# chi2 20.787865048363095

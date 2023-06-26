from itertools import combinations

import numpy as np
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import pbc_shortest_vectors


def coords_with_pbc(idx_1, idx_2, stucture, mic=True):
    """returns the cartesian coordinates between two sites in a structure
    while applying periodic boundary conditions.

    Args:
        idx_1: integer index for position 1
        idx_2: integer index for position 2
        structure: pymatgen.Structure object

    Returns: np.array, np.array
    """
    if mic:
        lattice = stucture.lattice
        frac_coords1 = stucture[idx_1].frac_coords
        frac_coords2 = stucture[idx_2].frac_coords

        v, d2 = pbc_shortest_vectors(
            lattice, frac_coords1, frac_coords2, return_d2=True
        )
        fc = lattice.get_fractional_coords(v[0][0]) + frac_coords1 - frac_coords2
        fc = np.array(np.round(fc), dtype=int)

        jimage = np.array(fc)
        # mapped_vec = lattice.get_cartesian_coords(jimage + frac_coords2-frac_coords1)
        cart_coord_1 = lattice.get_cartesian_coords(frac_coords1)
        cart_coord_2 = lattice.get_cartesian_coords(jimage + frac_coords2)
    else:
        cart_coord_1 = stucture[idx_1].coords
        cart_coord_2 = stucture[idx_2].coords

    return cart_coord_1, cart_coord_2


def dist_from_coords(coord1, coord2):
    """Helper function to calculate the distance between two sets
    of Caretsian coordinates

    Args:
        coord1: np.ndarray coordinates of site 1
        coord2: np.ndarray coordinates of site 2

    Returns: float
    """
    squared_dist = np.sum((coord1 - coord2) ** 2, axis=0)
    return np.sqrt(squared_dist)


def get_unique_indicies(structure, full_list=False):
    """Returns a list of the symmetrically equivalent atoms in a structure.

    Args:
        structure: pymatgen.Structure object
        full_list: boolena flag for whether or not to return the full list of
            lists if True or just the first member of each list if False.

    Returns: list
    """
    sga = SpacegroupAnalyzer(structure)
    symmeterized_struc = sga.get_symmetrized_structure()
    full_ind_list = symmeterized_struc.equivalent_indices
    unique_indicies = [i[0] for i in full_ind_list]
    if full_list is False:
        return unique_indicies
    else:
        return unique_indicies, full_ind_list


def remove_repeat_entries(data_list):
    """Helper function to remove repeated bonding pairs in the bonding data
    generation function

    Args:
        data_list: list of data to parse over and remove repeated entries.

    Returns: list
    """
    temp = []
    already_exists = []
    print("data list", data_list)
    for datum in data_list:
        check = datum["true_pair"]
        if check not in already_exists:

            temp.append(datum)
            already_exists.append(datum["true_pair"])
    return temp


# def second_coordination_distance(index, equiv_indicies, nn_function, structure):
#     pairs = []
#     double_count = []
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

#             new_coords = coords_with_pbc(true_pairs[0], true_pairs[1], structure)
#             coord_1 = new_coords[0]
#             coord_2 = new_coords[1]
#             d = dist_from_coords(coord_1, coord_2)
#             pairs.append({
#                 'vals': {
#                     'atom 1' : ind1,
#                     'atom 2' : ind2,
#                     'true_pair': list(true_pairs),
#                     'min_image_construction': True
#                 },
#                 'dists': d
#             })
#             print(true_pairs, d)
#         print()
#             # if ind1 not in double_count:
#             #     pairs.append({
#             #         'atom 1' : ind1,
#             #         'atom 2' : ind2,
#             #         'true_pair': list(true_pairs)
#             #     })
#     distances = []
#     reduced_pairs = []
#     for bond_pair in pairs:
#         if bond_pair['dists'] not in distances:
#             reduced_pairs.append(bond_pair['vals'])
#             distances.append(bond_pair['dists'])
#         # elif bond_pair['dists'] in distances:
#         #     mol_pair = bond_pair['vals']['true_pair']
#         #     molecule_coords_1 = structure[mol_pair[0]].coords
#         #     molecule_coords_2 = structure[mol_pair[1]].coords
#         #     d_test = dist_from_coords(molecule_coords_1, molecule_coords_2)
#         #     if d_test != bond_pair['dists'] and
# (np.abs(d_test - bond_pair['dists']) < 0.01)
# and (np.abs(d_test - bond_pair['dists']) > 0.0001):
#         #         bond_pair['vals']['min_image_construction'] = False
#         #         reduced_pairs.append(bond_pair['vals'])
#         #         distances.append(d_test)

#     return reduced_pairs #remove_repeat_entries(pairs)


def second_coordination_distance(index, equiv_indicies, nn_function, structure):
    """Function to find atom pairs between a central atom and atoms in the
    second coordination sphere

    Args:
        index: integer index of central atom
        equiv_indicies: list of symmetrically equivalent atom indicies
        nn_function: pymatgen.analysis.local_env function
        structure: pymatgen.Structure object

    Returns: list
    """
    pairs = []
    for atom in index:
        nn = nn_function.get_nn_info(structure, atom[0])
        true_pairs = np.sort([nn[0]["site_index"], nn[1]["site_index"]])

        for idx in equiv_indicies:
            if true_pairs[0] in idx:
                ind1 = idx[0]
            if true_pairs[1] in idx:
                ind2 = idx[0]
        pairs.append(
            {
                "vals": {
                    "atom 1": ind1,
                    "atom 2": ind2,
                    "true_pair": list(true_pairs),
                    "min_image_construction": True,
                },
                "dists": structure.get_distance(true_pairs[0], true_pairs[1]),
            }
        )
        print(list(true_pairs), structure.get_distance(true_pairs[0], true_pairs[1]))
    print()
    print(f"pairs is {len(pairs)}")
    distances = []
    reduced_pairs = []
    for bond_pair in pairs:
        if bond_pair["dists"] not in distances:
            reduced_pairs.append(bond_pair["vals"])
            distances.append(bond_pair["dists"])
        elif bond_pair["dists"] in distances:
            mol_pair = bond_pair["vals"]["true_pair"]
            molecule_coords_1 = structure[mol_pair[0]].coords
            molecule_coords_2 = structure[mol_pair[1]].coords
            d_test = dist_from_coords(molecule_coords_1, molecule_coords_2)
            if d_test != bond_pair["dists"] and (
                np.abs(d_test - bond_pair["dists"]) < 0.1
            ):
                bond_pair["vals"]["min_image_construction"] = False
                reduced_pairs.append(bond_pair["vals"])
                # distances.append(d_test)

    return reduced_pairs  # remove_repeat_entries(pairs)


def first_coordination_distance(index, equiv_indicies, nn_function, structure):
    """Function to find atom pairs between a central atom and atoms in the
    first coordination sphere

    Args:
        index: integer index of central atom
        equiv_indicies: list of symmetrically equivalent atom indicies
        nn_function: pymatgen.analysis.local_env function
        structure: pymatgen.Structure object

    Returns: list
    """
    pairs = []
    for atom in index:
        nn = nn_function.get_nn_info(structure, atom[0])
        for neighbor in nn:
            for idx in equiv_indicies:
                if neighbor["site_index"] in idx:
                    ind_neighbor = idx[0]

            pairs.append(
                {
                    "atom 1": atom[0],
                    "atom 2": ind_neighbor,
                    "true_pair": [atom[0], neighbor["site_index"]],
                    "min_image_construction": True,
                }
            )
    return remove_repeat_entries(pairs)


def first_coordination_vertex_vertex(index, equiv_indicies, nn_function, structure):
    """Function to find of all the atoms in the first coordination sphere of a central atom

    Args:
        index: integer index of central atom
        equiv_indicies: list of symmetrically equivalent atom indicies
        nn_function: pymatgen.analysis.local_env function
        structure: pymatgen.Structure object

    Returns: list
    """
    pairs = []
    for atom in index:
        nn = nn_function.get_nn_info(structure, atom[0])
        ind_list = [i["site_index"] for i in nn]

        for pair in list(combinations(ind_list, 2)):
            for i in equiv_indicies:
                if pair[0] in i:
                    ind1 = (i[0], pair[0])
                if pair[1] in i:
                    ind2 = (i[0], pair[1])
            unsorted = [ind1, ind2]
            indicies = sorted(unsorted, key=lambda tup: tup[1])
            pairs.append(
                {
                    "atom 1": indicies[0][0],
                    "atom 2": indicies[1][0],
                    "true_pair": [indicies[0][1], indicies[1][1]],
                    "min_image_construction": True,
                }
            )
    return pairs  # remove_repeat_entries(pairs)


def make_distance_data(structure):
    """Helper function to generate the distance data dictionary

    Args:
        structure: pymatgen.Structure object

    Returns: list
    """
    distance_data = []

    species = [i.symbol for i in structure.species]
    indicies, symmetry_equiv = get_unique_indicies(structure, full_list=True)
    silicons = [(i, species[i]) for i in indicies if species[i] == "Si"]
    oxygens = [(i, species[i]) for i in indicies if species[i] == "O"]
    nn = CrystalNN()

    distance_data.append(
        {
            "bond": "SiSi",
            "pairs": second_coordination_distance(
                oxygens, symmetry_equiv, nn, structure
            ),
        }
    )
    for i in distance_data[0]["pairs"]:
        print(i)
    distance_data.append(
        {
            "bond": "SiO",
            "pairs": first_coordination_distance(
                silicons, symmetry_equiv, nn, structure
            ),
        }
    )
    distance_data.append(
        {
            "bond": "OO",
            "pairs": first_coordination_vertex_vertex(
                silicons, symmetry_equiv, nn, structure
            ),
        }
    )

    return distance_data

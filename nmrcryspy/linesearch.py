import copy

import numpy as np


def update_chi2(
    functions,
    alpha,
    x_prime,
    sym_dict,
    data_dictionary,
    initial_struct,
    NUM_ATOMS,
    UNIQUE_IND,
):
    struct = copy.deepcopy(initial_struct)
    perturbations = np.reshape(
        x_prime * alpha, (int(len(x_prime) / 3), 3)
    )  # perturbations*alpha
    for atom in sym_dict:
        base_idx = atom["base_idx"]
        atom_idx = atom["atom"]
        perturbation_opt = atom["sym_op"].apply_rotation_only(perturbations[base_idx])
        struct.translate_sites(atom_idx, -perturbation_opt, frac_coords=False)
    chi2 = 0
    for function in functions:
        J, res = function.assemble_residual_and_grad(struct, data_dictionary)
        chi2 += np.sum(res**2)

    return chi2  # np.sum(res**2)


def get_derivative(
    function,
    phi,
    alpha,
    x_prime,
    sym_dict,
    data_dictionary,
    initial_struct,
    NUM_ATOMS,
    UNIQUE_IND,
    epsilon=0.01,
):
    phi_e = update_chi2(
        function,
        alpha + epsilon,
        x_prime,
        sym_dict,
        data_dictionary,
        initial_struct,
        NUM_ATOMS,
        UNIQUE_IND,
    )
    return (phi_e - phi) / epsilon


def quadratic_interpolation(alpha_low, phi_low, d_phi_low, alpha_high, phi_high):
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        try:
            D = phi_low
            C = d_phi_low
            db = alpha_high - alpha_low
            B = (phi_high - D - C * db) / (db * db)
            alpha_min = alpha_low - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(alpha_min):
        return None
    return alpha_min


def cubic_interpolation(
    alpha_low,
    phi_alpha_low,
    d_alpha_low,
    alpha_high,
    phi_alpha_high,
    alpha_test,
    phi_alpha_test,
):
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        try:
            C = d_alpha_low
            db = alpha_high - alpha_low
            dc = alpha_test - alpha_low
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc**2
            d1[0, 1] = -(db**2)
            d1[1, 0] = -(dc**3)
            d1[1, 1] = db**3
            [A, B] = np.dot(
                d1,
                np.asarray(
                    [
                        phi_alpha_high - phi_alpha_low - C * db,
                        phi_alpha_test - phi_alpha_low - C * dc,
                    ]
                ).flatten(),
            )
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            alpha_min = alpha_low + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(alpha_min):
        return None
    return alpha_min


def zoom(
    function,
    phi_0,
    dphi_0,
    alpha_low,
    alpha_high,
    x_prime,
    sym_dict,
    dist_test_dict,
    struct,
    NUM_ATOMS,
    UNIQUE_IND,
    epsilon=0.01,
    c1=0.0001,
    c2=0.9,
):
    max_iter = 15
    delta1 = 0.2  # cubic_interpolation check
    delta2 = 0.1  # quadratic_interpolation check
    phi_test = phi_0
    alpha_test = 0

    phi_low = update_chi2(
        function,
        alpha_low,
        x_prime,
        sym_dict,
        dist_test_dict,
        struct,
        NUM_ATOMS,
        UNIQUE_IND,
    )
    dphi_low = get_derivative(
        function,
        phi_low,
        alpha_low,
        x_prime,
        sym_dict,
        dist_test_dict,
        struct,
        NUM_ATOMS,
        UNIQUE_IND,
    )

    phi_high = update_chi2(
        function,
        alpha_high,
        x_prime,
        sym_dict,
        dist_test_dict,
        struct,
        NUM_ATOMS,
        UNIQUE_IND,
    )
    # dphi_high = get_derivative(
    #     function,
    #     phi_high,
    #     alpha_high,
    #     x_prime,
    #     sym_dict,
    #     dist_test_dict,
    #     struct,
    #     NUM_ATOMS,
    #     UNIQUE_IND,
    # )

    for i in range(max_iter):

        d_alpha = alpha_high - alpha_low
        if d_alpha < 0:
            a, b = alpha_high, alpha_low
        else:
            a, b = alpha_low, alpha_high

        if i > 0:
            cube_check = delta1 * d_alpha
            alpha_j = cubic_interpolation(
                alpha_low, phi_low, dphi_low, alpha_high, phi_high, alpha_test, phi_test
            )
        if (
            i == 0
            or alpha_j is None
            or (alpha_j > (b - cube_check))
            or (alpha_j < (a + cube_check))
        ):
            quad_check = delta2 * d_alpha
            alpha_j = quadratic_interpolation(
                alpha_low, phi_low, dphi_low, alpha_high, phi_high
            )
            if (
                alpha_j is None
                or (alpha_j > (b - quad_check))
                or (alpha_j < (a + quad_check))
            ):
                alpha_j = alpha_low + 0.5 * d_alpha  # alpha_high

        phi_j = update_chi2(
            function,
            alpha_j,
            x_prime,
            sym_dict,
            dist_test_dict,
            struct,
            NUM_ATOMS,
            UNIQUE_IND,
        )
        if phi_j > phi_0 + c1 * alpha_j * dphi_0 or phi_j >= phi_low:
            phi_test = phi_high
            alpha_test = phi_high
            alpha_high = alpha_j
            phi_high = phi_j
        else:
            dphi_j = get_derivative(
                function,
                phi_j,
                0,
                x_prime,
                sym_dict,
                dist_test_dict,
                struct,
                NUM_ATOMS,
                UNIQUE_IND,
            )
            if np.abs(dphi_j) <= -c2 * dphi_0:
                return alpha_j, phi_j

            if dphi_j * (alpha_high - alpha_low) >= 0:
                phi_test = phi_high
                alpha_test = alpha_high
                alpha_high = alpha_low
                phi_high = phi_low
            else:
                phi_test = phi_low
                alpha_test = alpha_low
            alpha_low = alpha_j
            phi_low = phi_j
            dphi_low = dphi_j
    print("Zoom exceeded max iter")
    return alpha_low, phi_low


def wolfe_line_search(
    function,
    phi_0,
    x_prime,
    sym_dict,
    dist_test_dict,
    struct,
    NUM_ATOMS,
    UNIQUE_IND,
    epsilon=0.01,
    max_iter=10,
    c1=0.0001,
    c2=0.9,
):
    dphi_0 = get_derivative(
        function,
        phi_0,
        0,
        x_prime,
        sym_dict,
        dist_test_dict,
        struct,
        NUM_ATOMS,
        UNIQUE_IND,
    )

    alpha_max = 1  # 0.75

    alpha_prev = 0
    phi_alpha_prev = phi_0
    # dalpha_prev = dphi_0

    alpha = 1  # np.random.uniform(0, alpha_max) #1
    phi_alpha = update_chi2(
        function,
        alpha,
        x_prime,
        sym_dict,
        dist_test_dict,
        struct,
        NUM_ATOMS,
        UNIQUE_IND,
    )

    for i in range(max_iter):

        if phi_alpha > phi_0 + c1 * alpha * dphi_0 or (
            i > 0 and phi_alpha >= phi_alpha_prev
        ):
            return zoom(
                function,
                phi_0,
                dphi_0,
                alpha_prev,
                alpha,
                x_prime,
                sym_dict,
                dist_test_dict,
                struct,
                NUM_ATOMS,
                UNIQUE_IND,
            )

        dphi_alpha = get_derivative(
            function,
            phi_alpha,
            0,
            x_prime,
            sym_dict,
            dist_test_dict,
            struct,
            NUM_ATOMS,
            UNIQUE_IND,
        )
        # dphi_alpha = (phi_alpha_e - phi_alpha)/epsilon
        if np.abs(dphi_alpha) <= -c2 * dphi_0:
            return alpha, phi_alpha

        if dphi_alpha >= 0:
            return zoom(
                function,
                phi_0,
                dphi_0,
                alpha,
                alpha_prev,
                x_prime,
                sym_dict,
                dist_test_dict,
                struct,
                NUM_ATOMS,
                UNIQUE_IND,
            )

        alpha_next = min(
            alpha_max, 2 * alpha
        )  # np.random.uniform(alpha, alpha_max) #  #np.random.uniform(alpha, alpha_max)
        alpha_prev = alpha
        alpha = alpha_next

        phi_alpha_prev = phi_alpha
        phi_alpha = update_chi2(
            function,
            alpha,
            x_prime,
            sym_dict,
            dist_test_dict,
            struct,
            NUM_ATOMS,
            UNIQUE_IND,
        )
        # dalpha_prev = dphi_alpha

    print("Max iterations excedded on Wolfe line search")
    return alpha, phi_alpha

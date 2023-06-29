.. _getting_started:

===========
Basic Usage
===========

Overview
--------

At the heart of the **nmrcryspy** workflow is the definition of
a :ref:`gauss_newton_api` object describing the minimization framework to be 
used in the minimization. Each :ref:`gauss_newton_api` object holds a list 
of :ref:`distance_api` and :ref:`nmr_api` objects which are used to calculate 
geometric or NMR properties which are used during the refinement.

The following examples take parts from the script in **examples/zsm12.py** 
to describe the components that make up this library.

Target Functions
----------------

When performing a minimization we will need to choose the information we want 
to optimize against (our target) and some way to predict that target. Currently, 
we offer simple pairwise atom-atom distance based :ref:`distance_api` targets as 
well as machine learning based :ref:`nmr_api:` targets. These functions provide 
methods to construct residual vectors and Jacobian matricies.

Consider the example below of a :ref:`distance_api` object for a silicon-oxygen 
distance created in Python.

.. code-block:: python

    # Import objects for the Distance Function
    from nmrcryspy.geometric import Distance_Function

    # Create a dictionary of data
    distance_dict = {
        "SiO": {"mu": 1.6, "sigma": 0.01}
    }

    #Pass this data to the Distance_Function() class
    zeolite_dists = Distance_Function(distance_dict)

This target function uses known Si-O distance distribution data to calculate the following 
residual: 

.. math::
    :label: eq_1

    \chi^2 = \sum_{i} \frac{(\mu_{i} - f(\text{structure})_{i})^2}{\sigma}

where :math:`\mu` is the distributions mean distance (in angstroms) and 
:math:`\sigma` is the distributions standard deviation (in angstroms).

The NMR fit functions are slightly more complicated as they require the 
checkpoint files (.chk) for creating the machine learning (ML) functions. 
We will also need to point to a data file for the ML function and provide 
some additional data for use in the refinement. 

Consider the example below of a :ref:`nmr_api` object for a silicon-oxygen 
distance created in Python.

.. code-block:: python

    # Import objects for the Shielding Function
    from nmrcryspy.nmr import ShieldingTensor_Function

    # Point to the location of the relevant checkpoint file
    checkpoint_file_path = (
    "/Users/mvenetos/Documents/Jupyter Testing Grounds/eigenn_testing/EigennBestTensor.ckpt"
    )

    # Create a dictionary of standard deviation data for each eigenvalue
    shielding_dict = {"sigma_11": 0.4, "sigma_22": 2.5, "sigma_33": 0.7}

    zeolite_shieldings = ShieldingTensor_Function(
        sigma_errors=shielding_dict,
        checkpoint=shielding_chk,
        root="/Users/mvenetos/Box Sync/All Manuscripts/zeolite " "refinements/ZSM12_temp/",
        data_file="ZSM12_CS.json", #data_file together with root give the location of the ML data
    )

.. note::
  We parameterize a shielding eigenvalues using the standard convention convention 
  with parameters ``sigma_11`` :math:`\geq` ``sigma_22`` :math:`\geq` ``sigma_33``.

For more information on how to use the ML functions (particularly the construction 
for **data_file** objects) see the code for `matTEN <https://github.com/mjwen/matten>`__ 

Minimization Framework
----------------------

Once we have the target functions identified we can now put them into a minimization 
framework. Currently, **nmrcryspy** offers a Gauss-Newton (:ref:`gauss_newton_api`) 
optimizer. In addition to the target functions, the optimizer also takes a 
**pymatgen.Structure** object and some hyper-parameters for the optimization itself.

Consider the example below of a :ref:`gauss_newton_api` object for a zeolite
refinement created in Python.

.. code-block:: python

    # Import objects for the Gauss_Newton_Solver
    from nmrcryspy import Gauss_Newton_Solver

    # Import the structure of the zeolite from a .cif file
    file_path = (
        "/Users/mvenetos/Box Sync/All Manuscripts/"
        "zeolite refinements/ZSM-12_calcined.cif"
    )
    s = CifParser(file_path).get_structures(False)[0]

    # Create a dictionary of data
    # Note this is a fraction of the data and utility functions exist to create this
    data = {
        "Bond_Distances": [
            {
                "bond": "SiO",
                "pairs": [
                    {"atom 1": 0, "atom 2": 56, "true_pair": [0, 56]},
                ],
            },
        ]
        "Shielding_Tensor": [
            {
                "index": 0,
                "target": "shielding",
                "neighbor_idx": [
                    0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,
                ],
            },
        ]
    }

    #Pass this data to the Gauss_Newton_Solver() class
    gn = Gauss_Newton_Solver(
        fit_function=[zeolite_shieldings, zeolite_dists],
        structure=s,
        data_dictionary=data,
        max_iter=2,
        tolerance_difference=1e-8,
    )
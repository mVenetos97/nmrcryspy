# nmrcryspy
A library for NMR crystallography refinements to solids

## Install

This below installing guide should get you started on Mac, without using GPUs.

- [matten](https://github.com/mjwen/matten)

  Create a conda environment based on the matten environment file.

  ```bash
  git clone https://github.com/mjwen/matten.git
  cd matten
  conda env create -f environment.yml -n myenv
  ```

  Then, follow the Installation guide for matten to install the matten code along with all of the relevant dependencies.

- This repo

  ```bash
  git clone https://github.com/mVenetos97/nmrcryspy.git
  cd nmrcryspy
  pip install -e .
  ```

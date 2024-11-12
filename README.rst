pygrog: A PyTorch-based Package for Implicit GROG Interpolation
===============================================================

**pygrog** is a PyTorch library for implementing implicit GRAPPA operator gridding (GROG) with field modeling. This package enables efficient interpolation of non-Cartesian MRI data onto Cartesian grids using deep learning. With `pygrog`, you can set up and train neural networks that replace conventional interpolation with a multi-source model, improving the quality of MRI reconstructions.

Features
--------

- **Calibration**: Extract low-resolution k-space data or synthesize calibration data.
- **Trajectory Pre-processing**: Generate and upsample displacement vectors from non-Cartesian trajectories.
- **GRAPPA Training**: Set up and train neural networks for GROG interpolation.
- **Interpolation**: Apply trained models to new non-Cartesian datasets for interpolation.
- **Sparse-to-Dense Gridding**: Convert interpolated sparse data to dense Cartesian grids.

Installation
------------

To install the package, clone the repository and run:

.. code-block:: bash

    git clone https://github.com/your-username/pygrog.git
    cd pygrog
    pip install -e .

Requirements
------------

- Python >= 3.7
- PyTorch >= 1.8
- numpy
- finufft / cufinufft / torchkbnufft / sigpy (for NUFFT operations)

Quick Start
-----------

Here’s a simple example showing how to use **pygrog**:

.. code-block:: python

    import torch
    from pygrog.calibration.calibration_data import CalibrationData

    # Step 1: Calibration
    kspace_data = torch.rand(64, 64)
    trajectory = torch.rand(64, 2)  # Example 2D non-Cartesian trajectory
    calibration = CalibrationData(kspace_data, trajectory)
    low_res_data = calibration.extract_low_res((16, 16))

    print("Low-resolution k-space data shape:", low_res_data.shape)

Documentation
-------------

Documentation for **pygrog** follows the `Diátaxis framework <https://diataxis.fr/>`_:

- **Tutorials**: Step-by-step guides for getting started.
- **How-to Guides**: Instructions for specific use cases.
- **Explanations**: Background on implicit GROG and interpolation concepts.
- **References**: API reference for all classes and functions.

To build the documentation locally, navigate to the `docs/` directory and run:

.. code-block:: bash

    make html

Testing
-------

Tests for **pygrog** are located in the `tests/` directory and use `pytest`. To run the tests, execute:

.. code-block:: bash

    pytest tests/

Project Structure
-----------------

The project follows a `src` layout:

.. code-block:: text

    pygrog/
    ├── src/
    │   └── pygrog/
    │       ├── calibration/
    │       ├── trajectory_preprocessing/
    │       ├── grappa_training/
    │       ├── interpolation/
    │       └── grid_reconstruction/
    ├── tests/
    ├── examples/
    └── docs/

Contributing
------------

We welcome contributions! Please read our `CONTRIBUTING.rst` for guidelines on how to contribute to **pygrog**.

License
-------

**pygrog** is licensed under the MIT License. See the `LICENSE` file for more details.

Contact
-------

For questions or support, please open an issue on the project's GitHub page or contact the authors.



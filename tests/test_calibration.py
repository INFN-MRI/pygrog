import torch
import pytest
from pygrog.calibration.calibration_data import CalibrationData


def test_extract_low_res():
    """Test low-resolution region extraction."""
    kspace_data = torch.rand(32, 32)
    calib = CalibrationData(kspace_data)
    low_res_data = calib.extract_low_res((16, 16))
    assert low_res_data.shape == (16, 16), "Low-res extraction shape mismatch"

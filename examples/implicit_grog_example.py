import torch
from pygrog.calibration.calibration_data import CalibrationData

# Step 1: Calibration
kspace_data = torch.rand(64, 64)
trajectory = torch.rand(64, 2)  # Example 2D non-Cartesian trajectory
calibration = CalibrationData(kspace_data, trajectory)
low_res_data = calibration.extract_low_res((16, 16))

print("Low-resolution k-space data shape:", low_res_data.shape)

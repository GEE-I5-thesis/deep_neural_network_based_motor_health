import numpy as np
import pandas as pd

# Parameters for data generation
n_samples = 5000  # Number of samples
random_state = 42  # For reproducibility

# Set random seed
np.random.seed(random_state)

# Generate synthetic data
accel_x = np.random.uniform(-1, 1, n_samples)  # Acceleration x (-1g to 1g)
accel_y = np.random.uniform(-1, 1, n_samples)  # Acceleration y (-1g to 1g)
accel_z = np.random.uniform(-1, 1, n_samples)  # Acceleration z (-1g to 1g)
temperature = np.random.uniform(20, 80, n_samples)  # Temperature (20°C to 80°C)
cos_phi = np.random.uniform(0.5, 1.0, n_samples)  # Power factor (0.5 to 1.0)

# Generate class labels based on synthetic rules
# Example: Assign classes based on logical rules for simplicity
# Class 0: Low accel and low temp
# Class 1: High accel and moderate temp
# Class 2: Low accel and high cos(phi)
# Class 3: High accel, high temp, and low cos(phi)
class_labels = []
for x, y, z, temp, phi in zip(accel_x, accel_y, accel_z, temperature, cos_phi):
    accel_magnitude = np.sqrt(x**2 + y**2 + z**2)
    if accel_magnitude < 0.5 and temp < 50:
        class_labels.append(0)
    elif accel_magnitude >= 0.5 and 50 <= temp <= 65:
        class_labels.append(1)
    elif accel_magnitude < 0.5 and phi > 0.8:
        class_labels.append(2)
    else:
        class_labels.append(3)

# Combine features and labels into a DataFrame
dataset = pd.DataFrame({
    "accel_x": accel_x,
    "accel_y": accel_y,
    "accel_z": accel_z,
    "temperature": temperature,
    "cos_phi": cos_phi,
    "class": class_labels
})

# Save the synthetic dataset as a CSV file
dataset.to_csv("realistic_motor_health_dataset.csv", index=False)
print("Synthetic dataset saved as 'realistic_motor_health_dataset.csv'")

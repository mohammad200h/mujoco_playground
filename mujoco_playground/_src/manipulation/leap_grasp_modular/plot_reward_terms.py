import numpy as np
import matplotlib.pyplot as plt

# Define parameters
k = 1.0
dq_values = np.linspace(0, 2 , 20)  # Range of dq values
joint_torque_1 = 0.5
joint_torque_2 = 2.0

# Compute results for both torque values
result_1 = np.exp(-k * dq_values**2) * joint_torque_1**2
result_2 = np.exp(-k * dq_values**2) * joint_torque_2**2

# Plot both results
plt.figure(figsize=(8, 5))
plt.plot(dq_values, result_1, label=f'Joint Torque = {joint_torque_1}', color='b')
plt.plot(dq_values, result_2, label=f'Joint Torque = {joint_torque_2}', color='r')
plt.xlabel('dq')
plt.ylabel('Function Value')
plt.title(r'$\exp(-k dq^2) \cdot \text{Torque}^2$')
plt.legend()
plt.grid(True)
plt.show()

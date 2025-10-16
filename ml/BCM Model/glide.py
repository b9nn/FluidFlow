import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sys

# load
modelRF = joblib.load('models/RF_BCM.pkl')
x_scaler = joblib.load('models/x_scaler_BCM.pkl')
df2 = pd.read_parquet("./data_binary.parquet")

# evaluate random forest model
def evalua_rf(model, x):
    x2 = np.array(x).reshape(1, -1)
    y = model.predict(x2)
    return y[0]

# calculate the jacobian
def compute_jacobian(predicted_outputs, data, func, epsilon):
    jacobian = np.zeros((2, 3))
    for j in range(3):
        perturb = np.zeros(3)
        perturb[j] = epsilon
        fmas = func(data + perturb)
        fmenos = func(data - perturb)
        jacobian[:, j] = (fmas - fmenos) / (2 * epsilon)
    return jacobian

# Function to update the data
def actualizar_datos(func_evalua, newData, stepRef, epsilon, alpha, gamma, i):
    predictedOutputs = func_evalua(newData)
    error = stepRef[i, :] - predictedOutputs

    # Calculate Jacobian
    jacobian = compute_jacobian(predictedOutputs, newData, func_evalua, epsilon)

    # Calculate the pseudoinverse J_inv
    J_inv = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T - gamma**2 * np.eye(jacobian.shape[0]))

    # Calculate the corrected error term with Jacobian
    error_Jac = J_inv @ error.T

    # Update inputs
    newData += alpha * error_Jac.T
    newData[2] = np.clip(newData[2], 0, 1)  # Normalize Ps
    newData[1] = np.clip(newData[1], 0, 1)  # aTA
    newData[0] = np.clip(newData[0], 0, 1)  # aCT

    return newData, predictedOutputs, error

# Main tracking function
def realizar_seguimiento(func_evalua, Fs, stepRef, x_scaler, alpha=0.1, gamma=0.1, epsilon=0.02, start=[0.1, 0.1, 0.1], num_initial_points=50):
    
    # Concatenate 50 points with the first value of stepRef
    initial_value = stepRef[0]  # Take the first value of the reference
    constant_part = np.tile(initial_value, (num_initial_points, 1))  # Create 50 points with that value
    new_stepRef = np.concatenate((constant_part, stepRef), axis=0)  # Concatenate the 50 initial points to the reference
    
    # Calculate the new number of steps
    numSteps2 = len(new_stepRef)

    # Time and history matrices
    dt = 1 / Fs
    t = np.arange(0, len(stepRef)) * dt
    newData = np.array(start)  # Use the initial point provided as parameter
    
    inputHistory = np.zeros((numSteps2, 3))
    outputHistory = np.zeros((numSteps2, 2))
    errorHistory = np.zeros((numSteps2, 2))

    a = 100 / numSteps2
    for i in range(numSteps2):
        # Calculate progress percentage
        percent = (i + 1) * a

        # Display progress bar
        sys.stdout.write(f"\rProgress: {i + 1}/{numSteps2} ({percent:.2f}%)")
        sys.stdout.flush()

        # Update data using the provided evaluation function
        newData, predictedOutputs, error = actualizar_datos(func_evalua, newData, new_stepRef, epsilon, alpha, gamma, i)

        inputHistory[i, :] = newData
        outputHistory[i, :] = predictedOutputs
        errorHistory[i, :] = error

    # Inverse of normalization
    inputHistory = x_scaler.inverse_transform(inputHistory)

    # Trim the first 50 added steps
    inputHistory = inputHistory[num_initial_points:, :]
    outputHistory = outputHistory[num_initial_points:, :]
    errorHistory = errorHistory[num_initial_points:, :]

    return inputHistory, outputHistory, errorHistory, t

# implement glide!

# define params
Fs = 1000
alpha = 0.1
gamma = 0.1
epsilon = 0.02

nStep = 500

dt = 1 / Fs
t = np.arange(0, nStep) * dt

# fo reference
x = np.linspace(0, 1, nStep)  
fo_ref = 150 + (300 - 150) * x**2

# spl reference
spl_ref = np.full(nStep, 70) # create tensor that is nStepx70

stepRef = np.column_stack([fo_ref, spl_ref]) # combine arrays

inputHistory_RF, outputHistory_RF, errorHistory_RF, t_RF = realizar_seguimiento(lambda x: evalua_rf(modelRF, x), Fs, stepRef, x_scaler)

ges = 'fo'
# Plot resultados de las entradas
plt.figure(figsize=(7, 10))
plt.subplot(5, 1, 1)
plt.plot(t, outputHistory_RF[:, 0], 'b', linewidth=2)
plt.plot(t, stepRef[:, 0], 'k--', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('$f_o$ (Hz)', fontsize=14)
plt.ylim([100, 300])
plt.legend(['RF'])
plt.grid(True)

plt.subplot(5, 1, 2)
plt.plot(t, outputHistory_RF[:, 1], 'b', linewidth=2)
plt.plot(t, stepRef[:, 1], 'k--', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('$SPL$ (dB)', fontsize=14)
plt.ylim([60, 100])
plt.legend(['RF'])
plt.grid(True)

plt.subplot(5, 1, 3)
plt.plot(t, inputHistory_RF[:, 2], 'b', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('$P_S$ (Pa)', fontsize=14)
plt.ylim([0, 2000])
plt.legend(['RF'])
plt.grid(True)

plt.subplot(5, 1, 4)
plt.plot(t, inputHistory_RF[:, 1], 'b', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('$a_{TA}$ (--)', fontsize=14)
plt.ylim([0, 1])
plt.grid(True)

plt.subplot(5, 1, 5)
plt.plot(t, inputHistory_RF[:, 0], 'b', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('$a_{CT}$ (--)', fontsize=14)
plt.ylim([0, 1])
plt.grid(True)

# Ajustar los subplots para evitar superposición
plt.tight_layout()

# Show the plot
output_filename = 'Figs/Glide_'+ges+'_low.svg'
plt.savefig(output_filename, format='svg')


# SPL Glide:

Fs = 1000
numSteps = 500
alpha = 0.1
gamma = 0.1
epsilon = 0.02

nStep = 500

dt = 1 / Fs
t = np.arange(0, nStep) * dt

stepRef = np.column_stack([np.full(numSteps, 200), np.linspace(65, 85, numSteps)])

#inital point
ini = [0.1,0.1,0.1]

inputHistory_RF, outputHistory_RF, errorHistory_RF, t_RF = realizar_seguimiento(lambda x: evalua_rf(modelRF, x), Fs, stepRef, x_scaler, start=ini)

ges = 'spl'
# Plot resultados de las entradas
plt.figure(figsize=(14, 20))
plt.subplot(5, 1, 1)
plt.plot(t, outputHistory_RF[:, 0], 'b', linewidth=2)
plt.plot(t, stepRef[:, 0], 'k--', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('$f_o$ (Hz)', fontsize=14)
plt.ylim([100, 300])
plt.legend(['RF'])
plt.grid(True)

plt.subplot(5, 1, 2)
plt.plot(t, outputHistory_RF[:, 1], 'b', linewidth=2)
plt.plot(t, stepRef[:, 1], 'k--', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('$SPL$ (dB)', fontsize=14)
plt.ylim([60, 100])
plt.legend(['RF'])
plt.grid(True)

plt.subplot(5, 1, 3)
plt.plot(t, inputHistory_RF[:, 2], 'b', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('$P_S$ (Pa)', fontsize=14)
plt.ylim([0, 2000])
plt.legend(['RF'])
plt.grid(True)

plt.subplot(5, 1, 4)
plt.plot(t, inputHistory_RF[:, 1], 'b', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('$a_{TA}$ (--)', fontsize=14)
plt.ylim([0, 1])
plt.grid(True)

plt.subplot(5, 1, 5)
plt.plot(t, inputHistory_RF[:, 0], 'b', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('$a_{CT}$ (--)', fontsize=14)
plt.ylim([0, 1])
plt.grid(True)

# Ajustar los subplots para evitar superposición
plt.tight_layout()

# Show the plot
output_filename = 'Figs/Glide_'+ges+'_high.svg'
plt.savefig(output_filename, format='svg')

plt.show()
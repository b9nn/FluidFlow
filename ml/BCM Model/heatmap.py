import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# --- HeatMaps ---

# --- load ---
rf = joblib.load('models/RF_BCM.pkl')
x_scaler = joblib.load('models/x_scaler_BCM.pkl')
df2 = pd.read_parquet("./data_binary.parquet")

# pre process
df2.dropna()
df2['PS'].unique()
x_scaler.transform([[0,0,710]])

# BOOLMASK LOGIC, TO BE FIXED
most_common_ps = df2['PS'].value_counts().idxmax() # find most common ps
print(f"\nUsing Ps = {most_common_ps} (most common value)")
df2 = df2[df2['PS']==most_common_ps] # apply boolmask on df

variables = ['a_CT', 'a_TA', 'PS']
covar = ['F0', 'SPL']

X = df2[variables] # shrink
X_norm = x_scaler.transform(X)

y = np.array(df2[covar]) # convert to an array
np.arange(0, 1.0, 0.02) # like linspace, but define steps as last arg (0.02)

# input 
ps = 0.5
ct = df2['a_CT'].unique()
ta = df2['a_TA'].unique()

# Generate grid of input values (renamed to avoid conflict with data variable X)
X_mesh, Y_mesh = np.meshgrid(ta, ct)

# Prepare input data for prediction
n_rows, n_cols = X_mesh.shape
input_data = np.zeros((n_rows * n_cols, 3))
input_data[:, 0] = X_mesh.flatten()
input_data[:, 1] = Y_mesh.flatten()
input_data[:, 2] = ps  # Set constant input

# Predict outputs using the Linear model
Z1_l = y[:,0].reshape((n_rows, n_cols))
Z2_l = y[:,1].reshape((n_rows, n_cols))

reg = 'RF'
# Predict outputs using the random forest model
predictions = rf.predict(input_data)
Z1 = predictions[:, 0].reshape((n_rows, n_cols))
Z2 = predictions[:, 1].reshape((n_rows, n_cols))

# Define specific levels for contours
levels_f0 = [100, 200, 300, 400, 500, 600]  # For F0
levels_spl = np.arange(65, 90, 5)  # For SPL (70, 75, 80,..., 100)

# Plot results
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Contour lines for data F0
im1 = axs[0, 0].imshow(Z1_l.T, interpolation='bilinear', origin='lower', extent=(ta[0], ta[-1], ct[0], ct[-1]), vmin=100, vmax=700)
CS1 = axs[0, 0].contour(Y_mesh, X_mesh, Z1_l, levels=levels_f0, colors='k')
axs[0, 0].clabel(CS1, inline=True, fontsize=10)
axs[0, 0].set_title('Contour lines for data $f_0$')
axs[0, 0].set_xlabel('$a_{TA}$', fontsize=20)
axs[0, 0].set_ylabel('$a_{CT}$', fontsize=20)
plt.colorbar(im1, ax=axs[0, 0])  # Colorbar for subplot 1

# Plot 2: Contour lines for predicted F0 (NN)
im2 = axs[0, 1].imshow(Z1.T, interpolation='bilinear', origin='lower', extent=(ta[0], ta[-1], ct[0], ct[-1]), vmin=100, vmax=700)
CS2 = axs[0, 1].contour(Y_mesh, X_mesh, Z1, levels=levels_f0, colors='k')
axs[0, 1].clabel(CS2, inline=True, fontsize=10)
axs[0, 1].set_title('Contour lines for predicted $f_0$ '+reg)
axs[0, 1].set_xlabel('$a_{TA}$', fontsize=20)
axs[0, 1].set_ylabel('$a_{CT}$', fontsize=20)
plt.colorbar(im2, ax=axs[0, 1])  # Colorbar for subplot 2

# Plot 3: Absolute error between F0 values
error_f0 = np.abs(Z1_l.T - Z1.T)  # Absolute error
im3 = axs[0, 2].imshow(error_f0, interpolation='bilinear', origin='lower', extent=(ta[0], ta[-1], ct[0], ct[-1]))
axs[0, 2].set_title('$f_0$ absolute error (Hz)')
axs[0, 2].set_xlabel('$a_{TA}$', fontsize=20)
axs[0, 2].set_ylabel('$a_{CT}$', fontsize=20)
plt.colorbar(im3, ax=axs[0, 2])  # Colorbar for F0 error

# Plot 4: Contour lines for data SPL
im4 = axs[1, 0].imshow(Z2_l.T, interpolation='bilinear', origin='lower', extent=(ta[0], ta[-1], ct[0], ct[-1]), vmin=65, vmax=85)
CS4 = axs[1, 0].contour(Y_mesh, X_mesh, Z2_l, levels=levels_spl, colors='k')
axs[1, 0].clabel(CS4, inline=True, fontsize=10)
axs[1, 0].set_title('Contour lines for data SPL')
axs[1, 0].set_xlabel('$a_{TA}$', fontsize=20)
axs[1, 0].set_ylabel('$a_{CT}$', fontsize=20)
plt.colorbar(im4, ax=axs[1, 0])  # Colorbar for subplot 3

# Plot 5: Contour lines for predicted SPL (NN)
im5 = axs[1, 1].imshow(Z2.T, interpolation='bilinear', origin='lower', extent=(ta[0], ta[-1], ct[0], ct[-1]), vmin=65, vmax=85)
CS5 = axs[1, 1].contour(Y_mesh, X_mesh, Z2, levels=levels_spl, colors='k')
axs[1, 1].clabel(CS5, inline=True, fontsize=10)
axs[1, 1].set_title('Contour lines for predicted SPL '+reg)
axs[1, 1].set_xlabel('$a_{TA}$', fontsize=20)
axs[1, 1].set_ylabel('$a_{CT}$', fontsize=20)
plt.colorbar(im5, ax=axs[1, 1])  # Colorbar for subplot 4

# Plot 6: Absolute error between SPL values
error_spl = np.abs(Z2_l.T - Z2.T)  # Absolute error
im6 = axs[1, 2].imshow(error_spl, interpolation='bilinear', origin='lower', extent=(ta[0], ta[-1], ct[0], ct[-1]))
axs[1, 2].set_title('SPL absolute error (dB)')
axs[1, 2].set_xlabel('$a_{TA}$', fontsize=20)
axs[1, 2].set_ylabel('$a_{CT}$', fontsize=20)
plt.colorbar(im6, ax=axs[1, 2])  # Colorbar for SPL error

# Adjust spaces between subplots
plt.tight_layout()
    
output_filename = 'Figs/heatmap_BCM_'+reg+'.svg'
plt.savefig(output_filename, format='svg')
plt.show()
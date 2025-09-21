import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

left = pd.read_csv('glottal_area\VF_Right_data.csv')
right = pd.read_csv('glottal_area\VF_Left_data.csv')

init_left = left['Points:0'] # left fold init pos
init_right = right['Points:0'] # right fold init pos
time = left['TimeStep']

deltax_left = left['displacements:0'] # displacement column for left fold
deltax_right = right['displacements:0'] # displacement column for right fold

pos_left = -(init_left + deltax_left) # consult nicholas about that logic here
pos_right = init_right + deltax_right

glottal_area = np.abs(pos_right - pos_left) # for a given timestep

lowerbound = 0 
upperbound = 50

boolmask = ((time >= lowerbound) and (time <= upperbound)) # filter as per timesteps
time_filtered = time[boolmask] # apply mask
area_filtered = glottal_area[boolmask]

area_trapz = np.trapz(area_filtered, time_filtered)

plt.plot(time_filtered, area_filtered, 'b-', linewidth=2, label='Glottal Area')
plt.set_xlabel('Time')
plt.set_ylabel('Glottal Area')
plt.set_title('Glottal Area vs Time')
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()

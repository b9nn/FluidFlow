import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

left = pd.read_csv('glottal_area/VF_Left_data.csv')
right = pd.read_csv('glottal_area/VF_Right_data.csv')

init_left = left.groupby('TimeStep')['Points:0'].first() # left fold init pos
init_right = right.groupby('TimeStep')['Points:0'].first() # right fold init pos
time = left.groupby('TimeStep')['Time'].first()

deltax_left = left.groupby('TimeStep')['displacements:0'].mean() # displacement column for left fold
deltax_right = right.groupby('TimeStep')['displacements:0'].mean() # displacement column for right fold

pos_left = -(init_left + deltax_left) # consult nicholas about that logic here
pos_right = init_right + deltax_right

glottal_area = np.abs(pos_right - pos_left) # for a given timestep

lowerbound = 0 
upperbound = 50

boolmask = ((time >= lowerbound) & (time <= upperbound)) # filter as per timesteps
time_filtered = time[boolmask] # apply mask
area_filtered = glottal_area[boolmask]

area_trapz = np.trapezoid(area_filtered, time_filtered)

plt.plot(time_filtered, area_filtered, 'b-', linewidth=2, label='Glottal Area')
plt.xlabel('Time (sec)')
plt.ylabel('Glottal Area (mm)')
plt.title('Glottal Area vs Time')
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()

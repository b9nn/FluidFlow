import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

left = pd.read_csv('glottal_area/VF_Left_data.csv')
right = pd.read_csv('glottal_area/VF_Right_data.csv')


#print(left["displacements:0"].head(15))
print(right["Time"].head(15))

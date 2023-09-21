from rdfpy import rdf
import matplotlib.pyplot as plt
import numpy as np

t=5

def get_dimensions(lst):
    if isinstance(lst, list):
        return [len(lst)] + get_dimensions(lst[0])
    else:
        return []

NUM_OF_ATOMS=256


coords_real = np.empty((NUM_OF_ATOMS,3))

for i in range(NUM_OF_ATOMS-1):
    pos = np.load(f'md_dataset/lj_data_to_test/data_0_{t}.npz')['pos'][i]
    coords_real[i,]=pos

print(coords_real.T.shape)


g_r_real, radii_real = rdf(coords_real, dr=4)
"""

pos = np.load(f'md_dataset/lj_data_tested/data_test1_{t}.npz')['pos']
coords_real.append(pos)
transposed_coords_model = [list(x) for x in zip(*coords_model)]

g_r_real, radii_real = rdf(coords_real, dr=0.1)
g_r_model, radii_model = rdf(coords_model, dr=0.1)

plt.plot(g_r_real, radii_real)
"""
from rdfpy import rdf
import matplotlib.pyplot as plt
import numpy as np

t=999

NUM_OF_ATOMS=258

coords_real = np.empty((NUM_OF_ATOMS,3))
coords_model = np.empty((NUM_OF_ATOMS,3))

#put the coordinates of all atoms at time t in a numpy array for the simulation ran by OpenMM

for i in range(NUM_OF_ATOMS):
    pos = np.load(f'md_dataset/lj_data_to_test/data_0_{t}.npz')['pos'][i]
    coords_real[i,]=pos

#make a radial graph
g_r_real, radii_real = rdf(coords_real, dr=0.1)

#put the coordinates of all atoms at time t in a numpy array ran by our model

for i in range(NUM_OF_ATOMS):
    pos = np.load(f'md_dataset/lj_data_tested/data_test1_{t}.npz')['pos'][i]
    coords_model[i,]=pos

#make rdf graph
g_r_model, radii_model = rdf(coords_model, dr=0.1)

print(coords_model)


#see the graph
plt.plot(radii_real, g_r_real)
plt.savefig('rdf_graph_real.png')

plt.plot(radii_model, g_r_model)
plt.savefig('rdf_graph_model.png')
from rdfpy import rdf
import matplotlib.pyplot as plt
import numpy as np
from train_endtoend_autoencoder_nice import BOX_SIZE
import MDAnalysis

time=999

NUM_OF_ATOMS=258

coords_real = np.empty((NUM_OF_ATOMS*7,3))
coords_model = np.empty((NUM_OF_ATOMS*7,3))

#put the coordinates of all atoms at time t in a numpy array for the simulation ran by OpenMM

for i in range(NUM_OF_ATOMS):
    pos = np.load(f'md_dataset/lj_data_to_test/data_0_{time}.npz')['pos'][i]
    coords_real[i,]=pos

### periodic boundary conditions
def periodic_boundary(coords_real):
    for i in range(NUM_OF_ATOMS,2*NUM_OF_ATOMS):
        coords_real[i,]=coords_real[i-NUM_OF_ATOMS,]
        coords_real[i,0]=coords_reag[i,0]+BOX_SIZE
    for i in range(NUM_OF_ATOMS*2,3*NUM_OF_ATOMS):
        coords_real[i,]=coords_real[i-NUM_OF_ATOMS*2,]
        coords_real[i,0]=coords_reag[i,0]-BOX_SIZE
    for i in range(NUM_OF_ATOMS*3,4*NUM_OF_ATOMS):
        coords_real[i,]=coords_real[i-NUM_OF_ATOMS*3,]
        coords_real[i,1]=coords_reag[i,1]+BOX_SIZE
    for i in range(NUM_OF_ATOMS*4,5*NUM_OF_ATOMS):
        coords_real[i,]=coords_real[i-NUM_OF_ATOMS*4,]
        coords_real[i,1]=coords_reag[i,1]-BOX_SIZE
    for i in range(NUM_OF_ATOMS*5,6*NUM_OF_ATOMS):
        coords_real[i,]=coords_real[i-NUM_OF_ATOMS*5,]
        coords_real[i,2]=coords_reag[i,2]+BOX_SIZE
    for i in range(NUM_OF_ATOMS*6,7*NUM_OF_ATOMS):
        coords_real[i,]=coords_real[i-NUM_OF_ATOMS*6,]
        coords_real[i,2]=coords_reag[i,2]-BOX_SIZE
    return coords_real

#make a radial graph
g_r_real, radii_real = rdf(periodic_boundary(coords_real), dr=0.1)

#put the coordinates of all atoms at time t in a numpy array ran by our model

for i in range(500,NUM_OF_ATOMS):
    pos = np.load(f'md_dataset/lj_data_tested/data_test1_{t}.npz')['pos'][i]
    coords_model[i,]=pos

#make rdf graph
g_r_model, radii_model = rdf(periodic_boundary(coords_model), dr=0.1)

print(coords_model)


#see the graph
plt.plot(radii_real, g_r_real)
plt.savefig('rdf_graph_real.png')

plt.plot(radii_model, g_r_model)
plt.savefig('rdf_graph_model.png')
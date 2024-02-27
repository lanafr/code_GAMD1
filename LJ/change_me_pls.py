import numpy as np

def numpy_to_xyz(numpy_file, output_file):
    # Load NumPy file
    data = np.load(numpy_file)

    # Extract atomic coordinates
    coordinates = data

    # Open output file for writing
    with open(output_file, 'w') as f:
        # Write each frame in XYZ format
        for frame in coordinates:
            # Write number of atoms
            num_atoms = len(frame)

            f.write(f"{num_atoms}\n")
            f.write('\n')
            # Write atom coordinates
            for atom in frame:
                # Write argon symbol and coordinates
                f.write(f"Ar {atom[0]} {atom[1]} {atom[2]}\n")

# Example usage
numpy_file = '/home/guests/lana_frkin/GAMDplus/code/LJ/results_all/results_EVERYTHING_20s_justnode_epoch=5/trajectory_data_t=0-20'
output_file = 'results_all/results_EVERYTHING_20s_justnode_epoch=5/trajectory_data_t=0-20.xyz'
numpy_to_xyz(numpy_file, output_file)

import numpy as np

# Load the .npz file
file_path = 'md_dataset/lj_data/data_0_4.npz'  # Replace with the actual path to your .npz file
data = np.load(file_path)

# Print the keys of the stored arrays in the .npz file
print("Keys in the .npz file:", data.keys())

# Access and print the contents of specific arrays
for key in data.keys():
    print(f"\nContents of '{key}':")
    print(data[key])
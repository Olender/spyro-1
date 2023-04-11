import h5py
import numpy as np

import matplotlib.pyplot as plt

filename = "velocity_models/overthrust_3D_true_model.h5"

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[2]

    # Get the data
    data = list(f[a_group_key])

    data = np.array(data)
    # convert to m/s
    data = (1 / (data ** (1 / 2))) * 1000.0
    # print(data)
    print(data.shape)
    print(np.amin(data))
    print(np.amax(data))

# 4 km by 6 km by 6 km
data_reduced = data[:, 201:501, 201:501]

print(data_reduced.shape)

plt.pcolor(data_reduced[:, :, 199])
plt.show()

data_reduced.astype("int32").tofile("overthrust_3D_exact_model_reduced_v5.bin")

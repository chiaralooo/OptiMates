import numpy as np
import matplotlib.pyplot as plt
import zarr

file = r'/localscratch/DL4MIA_2024/project/OptiMates/scripts/snapshots/snapshot_1521.zarr'
zarrfile = zarr.open(file, mode='r')
prefix = 'volumes/pred_cell_indicator'
test = zarrfile[prefix]
print(test.shape)
print(set(list(test[0, 0].flatten())))

plt.imshow(test[0, 0, 0])
plt.show()


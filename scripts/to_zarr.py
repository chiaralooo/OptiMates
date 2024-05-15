import h5py
import zarr
import numpy as np

file = h5py.File("/group/dl4miacourse/projects/OptiMates/JR_22-10-18/fusedStack.corrected.h5")
zarr_file = zarr.open("/group/dl4miacourse/projects/OptiMates/JR_22-10-18/fusedStack.corrected.zarr", mode = "w")
image_shape = (100,*file["t00199"]["s00"]["0"]["cells"].shape)
dataset = zarr_file.create_dataset(name = "view_0_tp_200-300", shape = image_shape, dtype = file["t00199"]["s00"]["0"]["cells"].dtype, dimension_separator = "/")
for i,time in zip(range(100),range(200,300)):
    print(time)
    dataset[i] = file[f"t{time:05d}" ]["s00"]["0"]["cells"]
print("done")
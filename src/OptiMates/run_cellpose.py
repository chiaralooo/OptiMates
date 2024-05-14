import zarr
from pathlib import Path
from cellpose import models

def run_cellpose(zarr_path: Path, raw_group: str, seg_group: str):
    model = models.Cellpose(gpu=False, model_type='nuclei')
    channels = [0, 0]


    # Data loading - sub in loading your images
    store = zarr.NestedDirectoryStore(zarr_path)
    zarr_array = zarr.open(store=store, mode='a', path=raw_group)
    mask_array = zarr.open(store=store, mode='a', path=seg_group, shape=zarr_array.shape, dtype="uint16")
    
    
    for t in range(zarr_array.shape[0]): # this had time, you should loop over images instead
        masks, _, _, _ = model.eval(zarr_array[t], diameter=None, channels=channels)
        mask_array[t] = masks
        
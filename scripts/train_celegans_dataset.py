
from appdirs import AppDirs
from pathlib import Path

from funlib.learn.torch.models import UNet
from OptiMates.train_linajea.train import run_training, get_pipeline
#from OptiMates.unet import UNet
import gunpowder as gp
import matplotlib.pyplot as plt
import logging
import zarr
from tifffile import imread
import os
import torch
import torch.optim

import numpy as np

logger = logging.basicConfig(level=logging.INFO)

def celegans_config(zarrStore):
    
    return {
        "raw_channel": "raw/GCamp6s",  
        "raw_data_path": zarrStore,
        "csv_path": "/group/dl4miacourse/projects/OptiMates/TRNs_calcium/batch1/tracks/GN692_125kPa002_20231017-142851.csv",
        "ndims": 3,
        "voxel_size": (1, 1, 1),
        "dtype": np.uint16
    }

def get_model():
    # these are default values
    # You will need to change them for your training
    unet = UNet(
            in_channels=1,
            num_fmaps=16,
            fmap_inc_factor=2,
            downsample_factors=[(1,2,2),(1,2,2),(1,2,2)],
            activation='ReLU',
            voxel_size=(1, 1, 1),
            num_heads=1,
            constant_upsample=True,
            padding='same')
    conv = torch.nn.Conv3d(16,1,1)
    sigm = torch.nn.Sigmoid()
    return torch.nn.Sequential(unet, conv, sigm)


if __name__ == "__main__":
    dataPath = r'/group/dl4miacourse/projects/OptiMates/TRNs_calcium/batch1'
    fname = r'GN692_125kPa002_20231017-142851.zarr'
    fnameZarr = os.path.join(dataPath, fname)
    data_config = celegans_config(fnameZarr)
    model = get_model() 
    optimizer = torch.optim.Adam(model.parameters())

    # construct a request that will determine the inputs and
    # outputs that we get
    input_size =(3, 256, 256)
    output_size = (3, 256, 256)

    pipeline, request = get_pipeline(data_config, model, torch.nn.MSELoss(), optimizer, input_size, output_size, radius=(0,5,5))



    raw_key = gp.ArrayKey("RAW")
    points_key = gp.GraphKey("POINTS")
    cell_indicator = gp.ArrayKey('CELL_INDICATOR')
    pred_cell_indicator = gp.ArrayKey('PRED_CELL_INDICATOR')
    debug_request = gp.BatchRequest()
    debug_request.add(raw_key, gp.Coordinate(input_size))
    debug_request.add(points_key, gp.Coordinate(input_size))
    debug_request.add(cell_indicator, gp.Coordinate(input_size))
    debug_request.add(pred_cell_indicator, gp.Coordinate(output_size))

    debug = False
    with gp.build(pipeline):
        for i in range(3):
            print(i)
            if debug:
                root = zarr.open("test_training.zarr", 'w')
                batch = pipeline.request_batch(debug_request)
                iteration = root.create_group(i)
                iteration["raw"] = batch[raw_key].data
                iteration["cell_indicator"] = batch[cell_indicator].data
                iteration["pred_cell_indicator"] = batch[pred_cell_indicator].data
            else:
                max_iterations = 200000  # you will definitely want more iterations
                print("Starting training...")
                for i in range(max_iterations):
                    pipeline.request_batch(request)

        
    # run_training(data_config, model)


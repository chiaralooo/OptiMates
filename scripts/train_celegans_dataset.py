
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
        "voxel_size": (1, 1, 1)
    }

def get_model():
    # these are default values
    # You will need to change them for your training
    unet= UNet(
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
    return torch.nn.Sequential(unet, conv)


class WeightedMSELoss(torch.nn.Module):
    """Wraps a set of torch losses and tensorboard summaries used to train
    the tracking model of Linajea
    """
    def __init__(self):
        super().__init__()


    def forward(inputs, target, mask):
        ws = mask.sum() * inputs.size()[0] / mask.size()[0]
        if abs(ws) <= 0.001:
            ws = 1
        return (mask * ((inputs - target) ** 2)).sum() / ws

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


    loss = WeightedMSELoss()

    pipeline, request = get_pipeline(data_config, model, torch.nn.MSE(), optimizer, input_size, output_size, name="weighted_loss", radius=(0,5,5))
    debug = False
    if debug:


        raw_key = gp.ArrayKey("RAW")
        points_key = gp.GraphKey("POINTS")
        cell_indicator = gp.ArrayKey('CELL_INDICATOR')
        pred_cell_indicator = gp.ArrayKey('PRED_CELL_INDICATOR')
        request = gp.BatchRequest()
        request.add(raw_key, gp.Coordinate(input_size))
        request.add(points_key, gp.Coordinate(input_size))
        request.add(cell_indicator, gp.Coordinate(input_size))
        request.add(pred_cell_indicator, gp.Coordinate(output_size))
        with gp.build(pipeline):
            root = zarr.open("test_training.zarr", 'w')
            for i in range(10):
                print(i)

                batch = pipeline.request_batch(request)
                iteration = root.create_group(i)
                iteration["raw"] = batch[raw_key].data
                iteration["cell_indicator"] = batch[cell_indicator].data
                iteration["pred_cell_indicator"] = batch[pred_cell_indicator].data
    else:
        # Training 3: no sigmoid, MSE loss
        # Training 4: training 3 with normalize factor 1/4000

        get_pipeline(data_config, model, loss, optimizer, input_size, output_size, name="training3", radius=(0,5,5))

        with gp.build(pipeline):
            max_iterations = 101  # you will definitely want more iterations
            print("Starting training...")
            for i in range(max_iterations):
                pipeline.request_batch(request)

import torch
import os
import gunpowder as gp

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

if __name__ == "__main__":
    predictdataPath = r'/group/dl4miacourse/projects/OptiMates/TRNs_calcium/batch1'
    predictFname = r'GN692_125kPa003_20231017-142851.zarr'
    fnameZarr = os.path.join(predictdataPath, predictFname)
    data_config = celegans_config(fnameZarr)
    model = get_model() 
    model.load_state_dict(torch.load('/localscratch/DL4MIA_2024/project/OptiMates/scripts/nosigmoid_checkpoint_1100')['model_state_dict'])
    
    # we need the raw data data from point source but not the csv data since you don't need gt for pred
    points_source, _ = get_sources(
        config["raw_data_path"],
        config["raw_channel"],
        config["csv_path"],
        raw_key,
        points_key,
        voxel_size,
        config['ndims'])
    
    # construct a request that will determine the inputs and
    # outputs that we get
    input_size =(3, 256, 256)
    output_size = (3, 256, 256)

    reference_request = gp.BatchRequest()
    reference_request.add(raw_key, input_size)
    reference_request.add(pred_key, output_size)

    with gp.build(points_source):
        full_roi = points_source.spec[raw_key].roi
        voxel_size = points_source.spec[raw_key].voxel_size

    pipeline = (
        points_source
        + gp.Normalize(raw_key, factor=1/4000)
        + gp.Predict(model, array_specs = {pred_key: gp.ArraySpec(full_roi, voxel_size)})
        + gp.ZarrWrite(output_zarr, {pred_key: pred_dataset})
        + gp.Scan(reference_request=reference_request, num_workers=1)
        )

    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())
    # pipeline, request = get_pipeline(data_config, model, torch.nn.MSELoss(), optimizer, input_size, output_size, name="nosigmoid", radius=(0,5,5))
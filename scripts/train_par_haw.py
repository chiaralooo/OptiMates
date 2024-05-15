
from appdirs import AppDirs
from pathlib import Path
import zarr
import torch
from funlib.learn.torch.models import UNet
from OptiMates.train_linajea.train import run_training, get_pipeline
import gunpowder as gp
import matplotlib.pyplot as plt

def hela_config():
    appdir = AppDirs("motile-plugin")
    return {
        "raw_channel": "view_0_tp_200-300",  
        "raw_data_path": "/group/dl4miacourse/projects/OptiMates/JR_22-10-18/fusedStack.corrected.zarr",
        "csv_path": "/group/dl4miacourse/projects/OptiMates/JR_22-10-18/karkinos5.csv",
        "ndims": 4,
        "voxel_size": (1, 5, 1, 1)
    }

def get_model():
    # these are default values
    # You will need to change them for your training
    return UNet(
            in_channels=1,
            num_fmaps=16,
            fmap_inc_factor=2,
            downsample_factors=[(1,2,2),(1,2,2),(2,2,2)],
            activation='ReLU',
            voxel_size=(5, 1, 1),
            num_heads=1,
            constant_upsample=True,
            padding='same')




if __name__ == "__main__":
    data_config = hela_config()
    model = get_model()
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # construct a request that will determine the inputs and
    # outputs that we get

    input_size =(2, 15, 100, 100)
    output_size = (2, 15, 100, 100)
    raw_key = gp.ArrayKey("RAW")
    points_key = gp.GraphKey("POINTS")
    cell_indicator = gp.ArrayKey('CELL_INDICATOR')
    pred_cell_indicator = gp.ArrayKey('PRED_CELL_INDICATOR')
    request = gp.BatchRequest()
    request.add(raw_key, gp.Coordinate(input_size))
    request.add(points_key, gp.Coordinate(input_size))
    request.add(cell_indicator, gp.Coordinate(input_size))
    pipeline, request = get_pipeline(
        data_config,
        model,
        loss,
        optimizer,
        input_size,
        output_size,
        # augment_only=False
        )

    with gp.build(pipeline):
        root =zarr.open("test.zarr", 'w')
        for i in range(3):

            batch = pipeline.request_batch(request)
            iteration = root.create_group(i)
            iteration["raw"] = batch[raw_key].data
            iteration["cell_indicator"] = batch[cell_indicator].data
            iteration["pred_indicator"] = batch[pred_cell_indicator].data
    # run_training(data_config, model)
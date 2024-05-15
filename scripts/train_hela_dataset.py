
from appdirs import AppDirs
from pathlib import Path

from funlib.learn.torch.models import UNet
from OptiMates.train_linajea.train import run_training, get_pipeline
import gunpowder as gp
import matplotlib.pyplot as plt

def hela_config():
    appdir = AppDirs("motile-plugin")
    return {
        "raw_channel": "01",  
        "raw_data_path": Path(appdir.user_data_dir) / "Fluo-N2DL-HeLa.zarr",
        "csv_path": "/Users/malinmayorc/code/OptiMates/tests/05152024_091740_test_tracks.csv",
        "ndims": 3,
        "voxel_size": (1, 1, 1)
    }

def get_model():
    # these are default values
    # You will need to change them for your training
    return UNet(
            in_channels=1,
            num_fmaps=16,
            fmap_inc_factor=2,
            downsample_factors=[(2,2,2),(2,2,2),(2,2,2)],
            activation='ReLU',
            voxel_size=(1, 1, 1),
            num_heads=1,
            constant_upsample=True,
            padding='same')


if __name__ == "__main__":
    data_config = hela_config()
    model = get_model() 
    pipeline = get_pipeline(data_config, model, augment_only=True)


    # construct a request that will determine the inputs and
    # outputs that we get
    input_size =(16, 256, 256)
    output_size = (16, 256, 256)

    raw_key = gp.ArrayKey("RAW")
    points_key = gp.GraphKey("POINTS")
    cell_indicator = gp.ArrayKey('CELL_INDICATOR')
    request = gp.BatchRequest()
    request.add(raw_key, gp.Coordinate(input_size))
    request.add(points_key, gp.Coordinate(input_size))
    request.add(cell_indicator, gp.Coordinate(input_size))
    with gp.build(pipeline):
        for i in range(5):
            batch = pipeline.request_batch(request)
            print(f"Point: {batch.points_key}")
            plt.imsave(f"test_raw_{i}.png", batch[raw_key]) 
            plt.imsave(f"test_indicator_{i}.png", batch[cell_indicator]) 
        
    # run_training(data_config, model)


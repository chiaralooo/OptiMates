from re import T
from funlib.learn.torch.models import UNet

import gunpowder as gp
import time

config = {
    "raw_channel": "t00000/s00/0",  #TODO: rewrite our target time points into a zarr
    "raw_data_path": "/group/dl4miacourse/....",
    "csv_path": "...",
    "frames": [200, 300],
}

def run_training(config):
    pipeline, request = get_pipeline(config)

        # finalize pipeline and start training
    with gp.build(pipeline):
        max_iterations = 100
        print("Starting training...")
        for i in max_iterations:
            start = time.time()
            pipeline.request_batch(request)
            time_of_iteration = time.time() - start

            print(
                "Batch: iteration=%d, time=%f",
                i, time_of_iteration)

def get_sources(raw_data_path, raw_channel, csv_path, raw_key,points_key, voxel_size):
    """Get gunpowder nodes to do sources"""
    raw_spec = {
        raw_key: gp.ArraySpec(
            interpolatable=True,
            voxel_size=voxel_size)
    }
    raw_source = gp.ZarrSource(
            raw_data_path,
            datasets={raw_key: raw_channel},
            array_specs=raw_spec)
    csv_source = gp.CsvPointsSource(
        csv_path,
        points_key,
        ndims=4,  # first 4 coordinates in csv will be the location
    )
    return raw_source, csv_source


def get_pipeline(config, augment_only=False):
    voxel_size = gp.Coordinate((1,5,1,1))

    raw_key = gp.ArrayKey("RAW")
    points_key = gp.Graphkey("POINTS")
    cell_indicator = gp.ArrayKey('CELL_INDICATOR')
    pred_cell_indicator = gp.ArrayKey('PRED_CELL_INDICATOR')

    points_source, csv_source = get_sources(**config, voxel_size=voxel_size)

    rasterize_graph = gp.RasterizeGraph(
                points_key,
                cell_indicator,
                array_spec=gp.ArraySpec(voxel_size=voxel_size),
                settings=gp.RasterizationSettings(
                    radius=20,  # set this based on data
                    mode='peak'))
    
    simple_augment = gp.SimpleAugment(
        mirror_only=[1, 2, 3],
        transpose_only=[2, 3])  # this should be x and y, but I am not sure if
        # z (which we want to exclude) is dim 1 or dim 3

    augmentation_pipeline = (
            (points_source, csv_source) +
            gp.RandomProvider() +
            rasterize_graph +
            simple_augment) 
    if augment_only:
        return augmentation_pipeline, None
    
    model = UNet()  # TODO: parameters
    loss = ...
    opt = ...

    input_size = ...
    output_size = ...
    train_node = gp.torch.Train(
        model=model,
        loss=loss,
        optimizer=opt,
        checkpoint_basename="train_linajea",
        inputs={ 'raw': raw_key, },
        outputs={"pred_indicator": cell_indicator},
        loss_inputs={"points": points_key},
        log_dir="train_logs",
        save_every=100
    )
    request = gp.BatchRequest()
    request.add(raw_key, input_size)
    request.add(points_key, input_size)
    request.add(cell_indicator, input_size)
    request.add(pred_cell_indicator, output_size)
    snapshot_request = gp.BatchRequest({
        raw_key: request[raw_key],
        cell_indicator: request[cell_indicator],
        pred_cell_indicator: request[pred_cell_indicator],
    })
    snapshot_datasets = {
        raw_key: 'volumes/raw',
        cell_indicator: 'volumes/cell_indicator',

        pred_cell_indicator: 'volumes/pred_cell_indicator',
    }
        # visualize
    snapshot_node = gp.Snapshot(snapshot_datasets,
                    output_dir='snapshots',
                    output_filename='snapshot_{iteration}.zarr',
                    additional_request=snapshot_request,
                    every=config.train.snapshot_stride,
                    )
    print_profiling = gp.PrintProfilingStats(every=10)
    train_pipeline = (
        augmentation_pipeline +
        train_node +
        snapshot_node +
        print_profiling
    )
    
    return train_pipeline, request



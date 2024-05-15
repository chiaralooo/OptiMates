from re import T
from funlib.learn.torch.models import UNet
import gunpowder as gp
import time



def run_training(data_config, model):
    pipeline, request = get_pipeline(data_config, model)

    with gp.build(pipeline):
        max_iterations = 100  # you will definitely want more iterations
        print("Starting training...")
        for i in max_iterations:
            start = time.time()
            pipeline.request_batch(request)
            time_of_iteration = time.time() - start

            print(
                "Batch: iteration=%d, time=%f",
                i, time_of_iteration)

def get_sources(raw_data_path, raw_channel, csv_path, raw_key, points_key, voxel_size, ndims):
    """Get gunpowder nodes to read the data"""
    raw_spec = {
        raw_key: gp.ArraySpec(
            interpolatable=True,
            voxel_size=voxel_size)
    }
    raw_source = (gp.ZarrSource(
            raw_data_path,
            datasets={raw_key: raw_channel},
            array_specs=raw_spec) + gp.Normalize(raw_key))
    csv_source = gp.CsvPointsSource(
        csv_path,
        points_key,
        ndims=ndims,  
        sep= ","
    )
    return raw_source, csv_source


def get_pipeline(
    config,
    model,
    loss,
    optimizer,
    input_size,
    output_size,
    radius,
    augment_only=False
):
    voxel_size = config['voxel_size']

    raw_key = gp.ArrayKey("RAW")
    points_key = gp.GraphKey("POINTS")
    cell_indicator = gp.ArrayKey('CELL_INDICATOR')
    pred_cell_indicator = gp.ArrayKey('PRED_CELL_INDICATOR')

    points_source, csv_source = get_sources(
        config["raw_data_path"],
        config["raw_channel"],
        config["csv_path"],
        raw_key,
        points_key,
        voxel_size,
        config['ndims'])

    rasterize_graph = gp.RasterizeGraph(
                points_key,
                cell_indicator,
                array_spec=gp.ArraySpec(voxel_size=voxel_size),
                settings=gp.RasterizationSettings(
                    radius=radius,  # set this based on data
                    mode='peak'))
    
    # simple_augment = gp.SimpleAugment(
    #     mirror_only=[1, 2],
    #     transpose_only=[1, 2])  # this should be x and y, but I am not sure if
        # z (which we want to exclude) is dim 1 or dim 3

    augmentation_pipeline = (
            (points_source, csv_source) + gp.MergeProvider() +
            rasterize_graph + 
            #gp.IterateLocations(points_key))
            gp.RandomLocation()) 
    if augment_only:
        return augmentation_pipeline
    
    train_node = gp.torch.Train(
        model=model,
        loss=loss,
        optimizer=optimizer,
        checkpoint_basename="train_linajea",
        inputs={ "x": raw_key, }, # argment name of unet forward function parameters
        outputs={0: pred_cell_indicator}, # output layer name of network (we didn't name our layers)
        loss_inputs={0: cell_indicator, 1: pred_cell_indicator},  # index into the loss forward function parameters
        log_dir="train_logs",
        save_every=100
    )

    request = gp.BatchRequest()
    request.add(raw_key, gp.Coordinate(input_size))
    request.add(points_key, gp.Coordinate(input_size))
    request.add(cell_indicator, gp.Coordinate(input_size))
    request.add(pred_cell_indicator, gp.Coordinate(output_size))

    # construct snapshots for debugging purposes
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
                    every=20,
                    )
    print_profiling = gp.PrintProfilingStats(every=10)

    # add training, snapshots, and profiling to the pipeline
    train_pipeline = (
        augmentation_pipeline +
        gp.Unsqueeze([raw_key, cell_indicator]) +
        gp.Stack(1) +
        train_node +
        snapshot_node +
        print_profiling
    )
    
    return train_pipeline, request



import os
import sys

import click
from rich import traceback, print

import numpy as np
import torch
import mlflow.pytorch
import torch.nn.functional as F

# from models import UNet3D
from model.model import LitsSegmentator

# for easy result visualization
import mrcfile as mrc
from utils import save_vol, merge_vol, split_vol, monte_carlo_dropout_proc

###############################################
# dist settings
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group("gloo", rank=0, world_size=1)

###############################################


###############################################
###############################################


#WD = os.path.dirname(__file__)


@click.command()
@click.option('-i', '--input', required=True, type=str, help='Path to data file to predict')
@click.option('-m', '--model', default='lits-unet', type=str, help='ID/Path to an already trained model')
@click.option('-c/-nc', '--cuda/--no-cuda', type=bool, default=False, help='Whether to enable cuda or not')
@click.option('-o', '--output', type=str, help='Output path')
def main(input: str, model: str, cuda: bool, output: str):
    """Command-line interface for liver-ct-segmentation-package"""

    print(r"""[bold blue]
        liver-ct-segmentation-package
        """)

    print('[bold blue]Run [green]liver-ct-segmentation-package --help [blue]for an overview of all commands\n')
    if model == 'lits-unet':
        #model = get_pytorch_model(f'{WD}/models/snapshots/lits_unet_3d/model/')
        model = get_pytorch_model(f'model/snapshots/lits_unet_3d/model/')
    else:
        model = get_pytorch_model(model)
    #if cuda:
    #    model.cuda()
    
    print('[bold blue] Parsing data')
    volume_image = read_input_volume(input)
    print('[bold blue] Performing predictions')
    predictions = predict_img(net=model, img=volume_image)
    print(predictions.shape)
    if output:
        print(f'[bold blue]Writing predictions to {output}')
        write_results(predictions, output)


def read_input_volume(input_path: str):
    """
    TODO
    """

    img = torch.load(input_path)

    # save for debugging
    #save_vol('img.mrc', np.transpose(img, axes=[2, 1, 0]))

    return img

def predict_img(net, img, use_gpu=False):
    """
    TODO
    """
    
    net.eval()

    img = np.expand_dims(img, axis=0)

    img = torch.tensor(img)
    img = img.unsqueeze(0)

    if use_gpu:
        img = img.cuda()

    with torch.no_grad():
        output = net(img)
        labels = torch.argmax(output, dim=1).float()

    labels = labels.squeeze(0)

    if use_gpu:
        labels = labels.cpu()

    labels = labels.numpy().astype(np.float32)

    return labels


def write_results(predictions, path_to_write_to) -> None:
    """
    Writes the predictions into a human readable file.
    :param predictions: Predictions as a numpy array
    :param path_to_write_to: Path to write the predictions to
    """

    save_vol(path_to_write_to + "labels.mrc", np.transpose(predictions, axes=[2, 1, 0]))

def get_pytorch_model(path_to_pytorch_model: str):
    """
    Fetches the model of choice and creates a booster from it.
    :param path_to_pytorch_model: Path to the xgboost model1
    """
    model = mlflow.pytorch.load_model(path_to_pytorch_model, map_location=torch.device('cpu')).module
    return model


if __name__ == "__main__":
    traceback.install()
    sys.exit(main())  # pragma: no cover

import os
import sys

import click
from rich import traceback, print

import zipfile

###############################################
###############################################

#WD = os.path.dirname(__file__)
LITS_UNET_URL = 'https://zenodo.org/record/5153279/files/lits_unet_3d_model.zip'

@click.command()
@click.option('-m', '--model', default='lits-unet', type=str, help='ID of trained model.')
@click.option('-o', '--output', default='snapshots/lits_unet_3d/', type=str, help='Where to save the model')
@click.option('-t', '--tmp', default='./model.zip', type=str, help='Path to tmp zip file')
def main(model: str, output: str, tmp: str):
    """Command-line interface to download models for the liver-ct-segmentation-package"""

    print(r"""[bold blue]
        liver-ct-segmentation-package: Package of 3D U-Nets reproducibly trained on the LiTS dataset.
        -->Model download CLI
        """)

    print('[bold blue]Run [green]liver-ct-seg-model-dl --help [blue]for an overview of all commands\n')
    
    print('[bold blue] Downloading model file: ' + model)
    if model == 'lits-unet':
        os.system('wget -O ' + tmp + ' ' + LITS_UNET_URL)

    print('[bold blue] Unzipping file to: ' + output)
    with zipfile.ZipFile(tmp, 'r') as zip_ref:
        zip_ref.extractall(output)
        
    
    print('[bold blue] removing tmp file: ' + tmp)
    os.system('rm ' + tmp)

if __name__ == "__main__":
    traceback.install()
    sys.exit(main())  # pragma: no cover

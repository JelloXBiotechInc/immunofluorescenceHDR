"""Convert fluorescence images to pseudo-IHC images."""


import argparse
import os
import yaml
import numpy as np
import pandas as pd
from skimage.color import rgb_from_hed
from skimage.filters import threshold_li
import cv2
import itertools
from tqdm import tqdm
from utils import read_image, write_image, write_tiledtiff
from utils import od_from_rgb, rgb_from_od
from utils import otsu_threshold_from_histogram, li_threshold_from_histogram
from typing import Dict, Optional, Sequence, Iterable


OPENSLIDE_PATH = r'C:\Users\chyan\openslide-win64-20220811\bin'
with os.add_dll_directory(OPENSLIDE_PATH):
    import openslide
    from openslide.deepzoom import DeepZoomGenerator


NPAR = np.ndarray


def gamma_correction(img: NPAR, gamma: float) -> NPAR:
    """Return image after Gamma correction."""
    lut = np.empty((1, 256), dtype=np.uint8)
    for i in range(256):
        lut[0, i] = np.clip(np.power(i / 255, 1 / gamma) * 255, 0, 255)
    return cv2.LUT(img, lut)


def _histogram(images: Iterable[NPAR]) -> NPAR:
    """Return histogram collected from images."""
    hist = np.zeros(256)
    for img in images:
        hist += np.bincount(img.flatten(), minlength=256)
    hist = hist / hist.sum()
    return hist


def _threshold(images: Iterable[NPAR], method: str) -> int:
    """Return threshold computed from a dynamic method."""
    if method == 'otsu':
        return otsu_threshold_from_histogram(_histogram(images))
    elif method == 'li':
        return li_threshold_from_histogram(_histogram(images))
    else:
        raise ValueError('only "otsu" and "li" are supported for `method`')


def _stain_matrix_custom(
    rgb_hema: Optional[Sequence[int]] = None,
    rgb_dab: Optional[Sequence[int]] = None
    ) -> NPAR:
    """
    Return matrix of transforming HED to RGB representation.
    
    Arguments
    --------_
    rgb_hema: RGB representation of hematoxylin
    rgb_dab: RGB representation of DAB
    """
    _rgb_from_hed = rgb_from_hed
    if rgb_hema is not None:
        rgb_hema = np.array(rgb_hema)
        hema = od_from_rgb(rgb_hema[None, :])
        hema = (hema / np.linalg.norm(hema, axis=-1, keepdims=True))
        _rgb_from_hed[..., 0] = hema
    if rgb_dab is not None:
        rgb_dab = np.array(rgb_dab)
        dab = od_from_rgb(rgb_dab[None, :])
        dab = (dab / np.linalg.norm(dab, axis=-1, keepdims=True))
        _rgb_from_hed[..., 2] = dab
    return _rgb_from_hed


def _pseudoihc_from_if(
    img: NPAR,
    _rgb_from_hed: NPAR,
    thrsh_hema: int = 0,
    gamma_hema: float = 1,
    factor_hema: float = 1,
    offset_hema: float = 0,
    thrsh_dab: int = 0,
    gamma_dab: float = 1,
    factor_dab: float = 1,
    offset_dab: float = 0,
    **kwargs
    ) -> NPAR:
    """
    Return pseudo-IHC image converted from IF image.

    Arguments
    ---------
    img: Image to process
    _rgb_from_hed: Matrix of transforming HED to RGB representation
    thrsh_hema: Threshold of hematoxylin signal
    gamma_hema: Gamma correction parameter of hematoxylin
    factor_hema: Scaling parameter of hematoxylin
    offset_hema: Shifting parameter of hematoxylin
    thrsh_dab: Threshold of DAB signal
    gamma_dab: Gamma correction parameter of DAB
    factor_dab: Scaling parameter of DAB
    offset_dab: Shifting parameter of DAB
    """
    hema, dab = img[..., 0], img[..., 1]
    hema[hema < thrsh_hema] = 0
    dab[dab < thrsh_dab] = 0
    hema = gamma_correction(hema, gamma_hema)
    dab = gamma_correction(dab, gamma_dab)
    img = np.dstack([
        cv2.add(
            cv2.multiply(hema, factor_hema),
            np.full_like(hema, offset_hema)
        ),
        np.zeros_like(img[..., 2]),
        cv2.add(
            cv2.multiply(dab, factor_dab),
            np.full_like(dab, offset_dab)
        )
    ])
    return rgb_from_od(od_from_rgb(255 - img) @ _rgb_from_hed)


def _convert_by_tiles(
    imgpath: str,
    _rgb_from_hed: NPAR,
    params: Dict,
    tilesize: int = 512,
    ) -> NPAR:
    """Return pseudo-IHC image where conversion occurs by tiles."""
    # Prepare image reader
    osr = openslide.OpenSlide(imgpath)
    width, height = osr.dimensions
    depth = 3  # DeepZoomGenerator read tiles in RGB format
    reader = DeepZoomGenerator(osr, tile_size=tilesize, overlap=0)
    level = reader.level_count - 1
    num_tiles_x, num_tiles_y = reader.level_tiles[-1]
    tile_positions = list(itertools.product(
        range(num_tiles_x),
        range(num_tiles_y)
    ))
    # Compute threshold if necessary
    for key, tiles in zip(
        ['thrsh_hema', 'thrsh_dab'],
        [
            (
                np.array(reader.get_tile(level, addr), dtype=np.uint8)[..., 0]
                for addr in tile_positions
            ),
            (
                np.array(reader.get_tile(level, addr), dtype=np.uint8)[..., 1]
                for addr in tile_positions
            ),
        ]
    ):
        if not isinstance(params[key], int):
            params[key] = _threshold(tiles, params[key])
    # Convert tiles to pseudo-IHC
    converted = np.empty(
        (tilesize * num_tiles_y, tilesize * num_tiles_x, depth),
        dtype=np.uint8
    )
    for addr in tile_positions:
        img = np.array(reader.get_tile(level, addr), dtype=np.uint8)
        converted_tile = _pseudoihc_from_if(img, _rgb_from_hed, **params)
        (l, t), _, (w, h) = reader.get_tile_coordinates(level, addr)
        converted[t:t+h, l:l+w] = converted_tile
    return converted[:height, :width, :]


def _convert_whole_image(
    imgpath: str,
    _rgb_from_hed: NPAR,
    params: Dict
    ) -> NPAR:
    """Return pseudo-IHC image where the whole image is converted."""
    img = read_image(imgpath)
    for key, tiles in zip(  # compute threshold if necessary
        ['thrsh_hema', 'thrsh_dab'],
        [(img[..., 0],), (img[..., 1],)]
    ):
        if not isinstance(params[key], int):
            params[key] = _threshold(tiles, params[key])
    converted = _pseudoihc_from_if(img, _rgb_from_hed, **params)
    return converted
    

def convert(
    imgpath: str,
    savepath: str,
    cvtcfg: Dict,
    read_openslide: bool = False,
    save_tiledtiff: bool = False,
    ) -> None:
    """
    Convert IF images to pseudo-IHC and save the converted images.

    Arguments
    ---------
    imgpath: Filepath to IF image 
    savepath: Filepath to save the converted image
    cvtcfg: Conversion configuration
    read_openslide: Whether to read images by tiles with OpenSlide
    save_tiledtiff: Whether to save the converted image as tiled-TIFF
    """
    _rgb_from_hed = _stain_matrix_custom(
        rgb_hema=cvtcfg.get('rgb_hema', None),
        rgb_dab=cvtcfg.get('rgb_dab', None)
    )
    kwargs = {
        'imgpath': imgpath,
        '_rgb_from_hed': _rgb_from_hed,  # matrix of transforming HED to RGB
        'params': cvtcfg,
    }
    if read_openslide:
        converted = _convert_by_tiles(
            tilesize=cvtcfg.get('read_tilesize', 1024),
            **kwargs
        )
    else:
        converted = _convert_whole_image(**kwargs)
    # Output the converted image
    # if save_tiledtiff:
    #     write_tiledtiff(
    #         converted,
    #         savepath,
    #         tilesize=cvtcfg.get('save_tilesize', 512)
    #     )
    # else:
    #     write_image(converted, savepath)
    write_image(converted, savepath)


def main(
    datacfg: Dict,
    cvtcfg: Dict,
    name: str,
    read_openslide: bool = False,
    save_tiledtiff: bool = False,
    ) -> None:
    """Convert IF images to pseudo-IHC."""
    savedir = os.path.join(cvtcfg['savedir'], name)
    os.makedirs(savedir, exist_ok=True)
    # Iterate over images with available metadata
    metadata = pd.read_csv(datacfg['metadata'])
    for imgname in tqdm(metadata.loc[:, datacfg['imagecol']].drop_duplicates()):
        imgpath = os.path.join(datacfg['savedir'], name, f'{imgname}.tif')
        savepath = os.path.join(savedir, f'{imgname}.tif')
        convert(
            imgpath=imgpath,
            savepath=savepath,
            cvtcfg=cvtcfg,
            read_openslide=read_openslide,
            save_tiledtiff=save_tiledtiff,
        )


if __name__ == '__main__':
    # Parse config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        help='experiemnt configuration to run',
        type=str,
        required=True
    )
    parser.add_argument(
        '--openslide',
        default=False,
        action='store_true',
        help='whether to read images by OpenSlide'
    )
    # parser.add_argument(
    #     '--tiledtiff',
    #     default=False,
    #     action='store_true',
    #     help='whether to save images as tiled-TIFF'
    # )
    args = parser.parse_args()
    assert os.path.exists(args.config), 'configuration file not found'
    assert os.path.splitext(args.config)[-1], 'configuration not a .yaml file'
    # Read config file
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    cfgname = os.path.splitext(os.path.basename(args.config))[0]
    # Convert images
    main(
        datacfg=config['data'],
        cvtcfg=config['convert'],
        name=cfgname,
        read_openslide=args.openslide,
        # save_tiledtiff=args.tiledtiff,
    )

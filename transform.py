"""Transform IF TSA images with different exposures into an HDR image."""


import os
import argparse
import yaml
import numpy as np
import pandas as pd
import cv2
import itertools
import math
from tqdm import tqdm
from typing import Dict, Sequence, Tuple


from processing import erode_blur, gamma_contrast_correction
from algorithm import foreground_neighborhood, fixed_step_samples
from algorithm import debevec_response, debevec_irradiance, minmax_scaling
from utils import ColorChannel
from utils import read_image, write_image, write_tiledtiff


OPENSLIDE_PATH = r'C:\Users\chyan\openslide-win64-20220811\bin'
with os.add_dll_directory(OPENSLIDE_PATH):
    import openslide
    from openslide.deepzoom import DeepZoomGenerator


NPAR = np.ndarray


def _preprocess(
    images: NPAR,
    exposures: NPAR,
    channel: int,
    _exposure_offset: float = 1e-2,
    **kwargs
    ) -> Tuple[NPAR, NPAR]:
    """
    Return the preprocessed and original images along with their exposures
    ordered ascendingly.

    Arguments
    ---------
    images: Images of different exposures, stacked at the first dimension
    exposures: Exposure times ordered ascendingly
    channel: Color channel to preprocess
    """
    preprc = images.copy()
    for i in range(images.shape[0]):
        preprc[i, ..., channel] = erode_blur(preprc[i, ..., channel], **kwargs)
    # Concatenate original and preprocessed images
    shape = images.shape[1:]  # shape per image
    images = np.stack([images, preprc], axis=1).reshape(-1, *shape)
    # Concatenate exposures accordingly
    exposures = np.stack([
        exposures,
        exposures + _exposure_offset
    ], axis=1).flatten()
    return images, exposures


def _foreground_pixels(
    images: NPAR,
    channel: int,
    foreground_channel: int,
    **kwargs
    ) -> NPAR:
    """
    Return pixel values in foreground region.

    Arguments
    ---------
    images: Images of different exposures, stacked at the first dimension
    channel: Color channel of sampled pixel values
    foreground_channel: Color channel to determine foreground region
    """
    mask = foreground_neighborhood(
        images[..., channel],
        target=images[0, ..., foreground_channel],
        **kwargs
    )
    return images[..., mask, channel]


def _sample(pixels: NPAR, num_imgs: int, _num_vals: int = 256)-> NPAR:
    """
    Return sampled pixel values.

    Arguments
    ---------
    pixels: Pixel values of different exposures, stacked at the first dimension
    num_imgs: Number of images of different exposures
    """
    num_smpls = math.ceil((_num_vals - 1) / (num_imgs - 1))
    return fixed_step_samples(pixels, num_smpls)


def _fit(pixels: NPAR, exposures: NPAR, **kwargs) -> NPAR:
    """
    Return the response curve fitted by sampled pixel values.

    Arguments
    ---------
    pixels: Pixel values of different exposures, stacked at the first dimension
    exposures: Exposure times ordered ascendingly
    """
    return debevec_response(pixels, exposures, **kwargs)


def _reconstruct_minmax(
    images: NPAR,
    exposures: NPAR,
    response: NPAR,
    channel: int,
    **kwargs
    ) -> Tuple[float, float]:
    """
    Return the minimum and maximum value reconstructed from the response curve.

    Arguments
    ---------
    images: Images of different exposures, stacked at the first dimension
    exposures: Exposure times ordered ascendingly
    response: Response curve
    channel: Color channel to reconstruct image
    """
    merged = debevec_irradiance(
        images[..., channel],
        exposures,
        response,
        **kwargs
    )
    return merged.min(), merged.max()


def _reconstruct_scale(
    images: NPAR,
    exposures: NPAR,
    response: NPAR,
    channel: int,
    trgtmin: float,
    trgtmax: float,
    srcmin: float,
    srcmax: float,
    **kwargs
    ) -> NPAR:
    """
    Return image reconstructed from the response curve and scaled by given
    minimum/maximum pixel values.

    Arguments
    ---------
    images: Images of different exposures, stacked at the first dimension
    exposures: Exposure times ordered ascendingly
    response: Response curve
    channel: Color channel to reconstruct image
    trgtmin: Minimum pixel value to scale to
    trgtmax: Maximum pixel value to scale to
    srcmin: Minimum pixel value to scale from
    srcmax: Maximum pixel value to scale from
    """
    merged = debevec_irradiance(
        images[..., channel],
        exposures,
        response,
        **kwargs
    )
    channels = [
        minmax_scaling(merged, trgtmin, trgtmax, srcmin, srcmax)
        if i == channel else images[0, ..., i]
        for i in range(images.shape[-1])
    ]
    return cv2.merge(channels)


def _reconstruct(
    images: NPAR,
    exposures: NPAR,
    response: NPAR,
    channel: int,
    **kwargs
    ) -> NPAR:
    """
    Return image reconstructed from the response curve.

    Arguments
    ---------
    images: Images of different exposures, stacked at the first dimension
    exposures: Exposure times ordered ascendingly
    response: Response curve
    channel: Color channel to reconstruct image
    """
    merged = debevec_irradiance(
        images[..., channel],
        exposures,
        response,
        **kwargs
    )
    channels = [
        minmax_scaling(merged, images.min(), images.max())
        if i == channel else images[0, ..., i]
        for i in range(images.shape[-1])
    ]
    return cv2.merge(channels)


def _postprocess(
    image: NPAR,
    channel: int,
    foreground_channel: int,
    **kwargs
    ) -> NPAR:
    """
    Return the postprocessed image.

    Arguments
    ---------
    image: Image to be processed
    channel: Color channel to processed
    foreground_channel: Color channel to determine image region of foreground
    """
    channels = [
        gamma_contrast_correction(
            image[..., channel],
            foreground=image[..., foreground_channel],
            **kwargs
        )
        if i == channel else image[..., i]
        for i in range(image.shape[-1])
    ]
    return cv2.merge(channels)


def _transform_by_patches(
    imgpaths: Sequence[str],
    exposures: NPAR,
    nucleus_chn: str,
    antibody_chn: str,
    params: Dict,
    patchsize: int,
    ) -> NPAR:
    """
    Return image transformed via HDR, where the image is processed by patches.

    Arguments
    ---------
    imgpaths: Filepaths to images, ordered ascendingly by exposures
    exposures: Exposure times ordered ascendingly
    nucleus_chn: Channel representing nulcei
    antibody_chn: Channel representing antibody marker
    params: Model parameters
    patchsize: Patch size
    """
    ncl_chn, atb_chn = ColorChannel[nucleus_chn], ColorChannel[antibody_chn]
    # Intialize image readers
    osrs = [openslide.OpenSlide(fp) for fp in imgpaths]
    assert all(osr.dimensions == osrs[0].dimensions for osr in osrs),\
        'all images must have the same size and depth'
    width, height = osrs[0].dimensions
    depth = 3  # DeepZoomGenerator read tiles in RGB format
    image_readers = [
        DeepZoomGenerator(osr, tile_size=patchsize, overlap=0)
        for osr in osrs
    ]
    level = image_readers[0].level_count - 1
    num_tiles_x, num_tiles_y = image_readers[0].level_tiles[-1]
    # 1st pass: preprocess patches, sample pixels, and fit the response curve
    pixels = []
    for addr in itertools.product(range(num_tiles_x), range(num_tiles_y)):
        images = np.stack([
            np.array(reader.get_tile(level, addr), dtype=np.uint8)
            for reader in image_readers
        ], axis=0)
        images, _exposures = _preprocess(
            images,
            exposures,
            channel=atb_chn,
            **params
        )
        pixels.append(
            _foreground_pixels(
                images,
                channel=atb_chn,
                foreground_channel=ncl_chn,
                **params
            )
        )
    pixels = _sample(np.concatenate(pixels, axis=-1), num_imgs=len(image_readers))
    response = _fit(pixels, _exposures, **params)
    # 2nd pass: preprocess patches and get min/max value of raw/merged images
    rawmin, rawmax = 255, 0
    mrgmin, mrgmax = np.inf, -np.inf
    for addr in itertools.product(range(num_tiles_x), range(num_tiles_y)):
        images = np.stack([
            np.array(reader.get_tile(level, addr), dtype=np.uint8)
            for reader in image_readers
        ], axis=0)
        images, _exposures = _preprocess(
            images,
            exposures,
            channel=atb_chn,
            **params
        )
        rawmin, rawmax = min(rawmin, images.min()), max(rawmax, images.max())
        _min, _max = _reconstruct_minmax(
            images,
            _exposures,
            response,
            channel=atb_chn,
            **params
        )
        mrgmin, mrgmax = min(mrgmin, _min), max(mrgmax, _max)
    # 3rd pass: preprocess and reconstruct patches
    merged = np.empty(
        (patchsize * num_tiles_y, patchsize * num_tiles_x, depth),
        dtype=np.uint8
    )
    for addr in itertools.product(range(num_tiles_x), range(num_tiles_y)):
        images = np.stack([
            np.array(reader.get_tile(level, addr), dtype=np.uint8)
            for reader in image_readers
        ], axis=0)
        images, _exposures = _preprocess(
            images,
            exposures,
            channel=atb_chn,
            **params
        )
        merged_patch = _reconstruct_scale(
            images,
            _exposures,
            response,
            channel=atb_chn,
            trgtmin=rawmin,
            trgtmax=rawmax,
            srcmin=mrgmin,
            srcmax=mrgmax,
            # trgtmin=0,
            # trgtmax=255,
            # srcmin=np.exp(response).min() / exposures.max(),
            # srcmax=np.exp(response).max() / exposures.min(),
            **params
        )
        (l, t), _, (w, h) = image_readers[0].get_tile_coordinates(level, addr)
        merged[t:t+h, l:l+w] = merged_patch
    # Postprocess merged image
    merged = merged[:height, :width, :]
    merged = _postprocess(
        merged,
        channel=atb_chn,
        foreground_channel=ncl_chn,
        **params
    )
    return merged


def _transform_whole_image(
    imgpaths: Sequence[str],
    exposures: NPAR,
    nucleus_chn: str,
    antibody_chn: str,
    params: Dict,
    ) -> NPAR:
    """
    Return image transformed via HDR, where the whole image is processed
    synchronously.

    Arguments
    ---------
    imgpaths: Filepaths to images, ordered ascendingly by exposures
    exposures: Exposure times ordered ascendingly
    nucleus_chn: Channel representing nulcei
    antibody_chn: Channel representing antibody marker
    params: Model parameters
    """
    images = np.stack([read_image(fp) for fp in imgpaths], axis=0)
    assert all(img.shape == images[0].shape for img in images),\
        'all images must have the same size and depth'
    ncl_chn, atb_chn = ColorChannel[nucleus_chn], ColorChannel[antibody_chn]
    images, exposures = _preprocess(
        images,
        exposures,
        channel=atb_chn,
        **params
    )
    pixels = _foreground_pixels(
        images,
        channel=atb_chn,
        foreground_channel=ncl_chn,
        **params
    )
    pixels = _sample(pixels, num_imgs=images.shape[0])
    response = _fit(pixels, exposures, **params)
    merged = _reconstruct(
        images,
        exposures,
        response,
        channel=atb_chn,
        **params
    )
    merged = _postprocess(
        merged,
        channel=atb_chn,
        foreground_channel=ncl_chn,
        **params
    )
    return merged


def transform(
    imgpaths: Sequence[str],
    exposures: Sequence[float],
    savepath: str,
    datacfg: Dict,
    modelcfg: Dict,
    read_openslide: bool = False,
    save_tiledtiff: bool = False,
    ) -> None:
    """
    Transform via HDR and save the processed image.

    Arguments
    ---------
    imgpaths: Filepaths to images
    exposures: Exposure times of images
    savepath: Filepath to save the transformed image
    datacfg: Data configuration
    modelcfg: Model configuration
    read_openslide: Whether to read images by patches with OpenSlide
    save_tiledtiff: Whether to save the transformed image as tiled-TIFF
    """
    # Sort images by ascending exposures
    inds = np.argsort(exposures)
    imgpaths = [imgpaths[ind] for ind in inds]
    exposures = np.array(exposures, dtype=modelcfg.get('dtype', 'float32'))[inds]
    # Transform images of different exposures via HDR algorithm
    kwargs = {
        'imgpaths': imgpaths,
        'exposures': exposures,
        'nucleus_chn': datacfg.get('nucleuschn', 'RED').upper(),
        'antibody_chn': datacfg.get('antibodychn', 'GREEN').upper(),
        'params': modelcfg,
    }
    if read_openslide:
        merged = _transform_by_patches(
            patchsize=datacfg.get('read_tilesize', 1024),
            **kwargs
        )
    else:
        merged = _transform_whole_image(**kwargs)
    # Output transformed image
    # if save_tiledtiff:
    #     write_tiledtiff(
    #         merged,
    #         savepath,
    #         tilesize=datacfg.get('save_tilesize', 512)
    #     )
    # else:
    #     write_image(merged, savepath)
    write_image(merged, savepath)


def main(
    datacfg: Dict,
    modelcfg: Dict,
    name: str,
    read_openslide: bool = False,
    save_tiledtiff: bool = False,
    ) -> None:
    """Transform images, each with snapshots of different exposures."""
    savedir = os.path.join(datacfg['savedir'], name)
    os.makedirs(savedir, exist_ok=True)
    # Iterate over images with available metadata
    metadata = pd.read_csv(datacfg['metadata'])
    for imgname, group in tqdm(metadata.groupby(datacfg['imagecol'])):
        print(imgname)
        imgpaths = [
            os.path.join(datacfg['datadir'], imgname, filename)
            for filename in group[datacfg['filenamecol']]
        ]
        savepath = os.path.join(savedir, f'{imgname}.tif')
        transform(
            imgpaths=imgpaths,
            exposures=group[datacfg['exposurecol']],
            savepath=savepath,
            datacfg=datacfg,
            modelcfg=modelcfg,
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
    # Transform images
    main(
        datacfg=config['data'],
        modelcfg=config['model'],
        name=cfgname,
        read_openslide=args.openslide,
        # save_tiledtiff=args.tiledtiff,
    )

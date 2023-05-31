"""Utilities."""


import os
import cv2
import numpy as np
import scipy as sp
from enum import IntEnum
from mergeregular_to_ometiff import TiffWriter
from skimage.color import hed_from_rgb
import itertools
from dataclasses import dataclass, field
from typing import Iterable, Tuple, Any


OPENSLIDE_PATH = r'C:\Users\chyan\openslide-win64-20220811\bin'
with os.add_dll_directory(OPENSLIDE_PATH):
    import openslide
    from openslide.deepzoom import DeepZoomGenerator


NPAR = np.ndarray


def read_image(filepath: str) -> NPAR:
    """Read image via OpenCV."""
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def write_image(img: NPAR, filepath: str) -> None:
    """Write image via OpenCV"""
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


class ColorChannel(IntEnum):
    """Options of supported color channels."""
    RED = 0
    GREEN = 1
    BLUR = 2


def read_tiles(imgpath: str, tilesize: int) -> Iterable[NPAR]:
    """Yield image tiles read by OpenSlide."""
    reader = DeepZoomGenerator(
        openslide.OpenSlide(imgpath),
        tile_size=tilesize,
        overlap=0
    )
    level = reader.level_count - 1
    num_tiles_x, num_tiles_y = reader.level_tiles[-1]
    for addr in itertools.product(range(num_tiles_x), range(num_tiles_y)):
        img = np.array(reader.get_tile(level, addr), dtype=np.uint8)
        yield img


def read_tiles_coor(
        imgpath: str,
        tilesize: int
    ) -> Iterable[Tuple[NPAR, Tuple[int, int]]]:
    """Yield image tiles and their top-left coordinates read by OpenSlide."""
    reader = DeepZoomGenerator(
        openslide.OpenSlide(imgpath),
        tile_size=tilesize,
        overlap=0
    )
    level = reader.level_count - 1
    num_tiles_x, num_tiles_y = reader.level_tiles[-1]
    for addr in itertools.product(range(num_tiles_x), range(num_tiles_y)):
        img = np.array(reader.get_tile(level, addr), dtype=np.uint8)
        (left, top), _, _ = reader.get_tile_coordinates(level, addr)
        yield img, (top, left)


def read_tiles_coor_size(
        imgpath: str,
        tilesize: int
    ) -> Iterable[Tuple[NPAR, Tuple[int, int], Tuple[int, int]]]:
    """
    Yield image tiles, their top-left coordinates, and their sizes read by
    OpenSlide.
    """
    reader = DeepZoomGenerator(
        openslide.OpenSlide(imgpath),
        tile_size=tilesize,
        overlap=0
    )
    level = reader.level_count - 1
    num_tiles_x, num_tiles_y = reader.level_tiles[-1]
    for addr in itertools.product(range(num_tiles_x), range(num_tiles_y)):
        img = np.array(reader.get_tile(level, addr), dtype=np.uint8)
        (left, top), _, (width, height) = reader.get_tile_coordinates(level, addr)
        yield img, (top, left), (height, width)


def image_size(imgpath: str) -> Tuple[int, int]:
    """Return height and width of an image."""
    osr = openslide.OpenSlide(imgpath)
    width, height = osr.dimensions
    return height, width


def write_tiledtiff(img: NPAR, filepath: str, tilesize: int) -> None:
    """Write image to tiled TIFF."""
    assert os.path.splitext(os.path.basename(filepath))[-1] == '.tif',\
        'file to save must have .tif extension'
    writer = TiffWriter(tilesize)
    writer(img, filepath)


def od_from_rgb(img: NPAR) -> NPAR:
    """Return optical density of an RGB image."""
    img = np.clip(img, 1, 255)
    return -np.log10(img / 255)


def rgb_from_od(img: NPAR) -> NPAR:
    """Return inverse outcome of optical density operation."""
    img = np.power(10, -img) * 255
    return np.clip(img, 0, 255).astype(np.uint8)


def hed_ruifrok(img: NPAR) -> NPAR:
    """Return HED representation of RGB image via Ruifrok algorithm."""
    return 255 - rgb_from_od(od_from_rgb(img) @ hed_from_rgb)


def od_above_thrsh(img: NPAR, thrsh: float) -> NPAR:
    """Return pixel OD values larger than a threshold for any channel."""
    pixels = img.reshape(-1, img.shape[-1])
    return pixels[(pixels > thrsh).any(axis=-1), :]


def two_stain_vectors_svd(
        pixels: NPAR,
        alpha: float = 0.01,
        _reduction: float = 0.5
    ) -> NPAR:
    """Return the two major stain vectors reconstructed via SVD."""
    # Find eigenvectors with SVD
    _size = pixels.shape[0]
    success = False
    while not success:
        try:
            _, _, eigvecs = sp.linalg.svd(pixels[:_size], full_matrices=False)
        except ValueError:  # fail to allocate memory for matrices
            _size = int(_size * _reduction)  # reduce sample size
        else:
            success = True
    # Project pixels to the first two leading components
    eigvecs = eigvecs[:2, :]
    eigvecs *= np.sign(eigvecs.sum(axis=-1))[:, None]
    pixels = pixels @ eigvecs.T  # pixel values projection
    # Find stain vectors from extreme percentile of angles on projected plane
    stain = np.percentile(
        np.arctan2(pixels[:, 1], pixels[:, 0]),
        [alpha * 100, (1 - alpha) * 100]
    )
    stain = np.column_stack([np.cos(stain), np.sin(stain)]) @ eigvecs
    return stain


def hematoxylin_first(stains: NPAR) -> NPAR:
    """Return vectors of two stains where hematoxylin is ordered the first."""
    return stains[[1, 0], :] if stains[0, -1] > stains[1, -1] else stains


def inversed_stain_matrix(stains: NPAR) -> NPAR:
    """
    Return the inversed stain matrix with a background component computed from
    the two major stains.
    """
    background = np.cross(stains[0, :], stains[1, :])
    background /= np.linalg.norm(background)
    mat = np.concatenate([stains, background[None, :]], axis=0)
    return mat


@dataclass
class HDXMacenko:
    """Color deconvolution of hematoxylin and DAB by the Macenko algorithm."""

    perc: float = 1e-2
    od_thrsh: float = 0.15
    hdx_from_rgb: NPAR = field(init=False)

    def fit(self, imgs: Iterable[NPAR]) -> Any:
        """Fit stain matrix."""
        samples = []
        for img in imgs:
            samples.append(od_above_thrsh(od_from_rgb(img), self.od_thrsh))
        samples = np.concatenate(samples, axis=0)
        stains = two_stain_vectors_svd(samples, alpha=self.perc)
        stains = hematoxylin_first(stains)
        self.hdx_from_rgb = np.linalg.inv(inversed_stain_matrix(stains))
        return self
    
    def predict(self, img: NPAR) -> NPAR:
        """Return the HDX representation of an image."""
        try:
            return 255 - rgb_from_od(od_from_rgb(img) @ self.hdx_from_rgb)
        except AttributeError as err:
            raise err(str(err) + '; must run fit before predict')


def histogram_mapping(src: NPAR, trgt: NPAR) -> NPAR:
    """Return lookup table that maps a source histogram to a target."""
    src = np.cumsum(src[1:] / src[1:].sum())
    trgt = np.cumsum(trgt[1:] / trgt[1:].sum())
    vals = np.arange(trgt.shape[0])
    lut = 1 + np.interp(src, trgt, vals).round().astype(np.uint8)
    # lut = np.floor(np.interp(src, trgt, vals)).astype(np.uint8)
    return lut


def otsu_threshold_from_histogram(hist: NPAR, _atol: float = 1e-6) -> int:
    """Return Otsu threshold from a histogram."""
    vals = np.arange(hist.shape[0])
    _min, thrsh = np.inf, 0  # initialize minimum cost and threshold
    hist = hist / hist.sum()
    cumhist = hist.cumsum()  # cumulative sum of histogram
    for i in range(1, hist.shape[0] - 1):
        sum_bg, sum_fg = cumhist[i], cumhist[-1] - cumhist[i]
        if sum_bg < _atol or sum_fg < _atol:  # skip if too few counts
            continue
        # Compute cost from mean and variance
        prob_bg, prob_fg = np.hsplit(hist, [i])
        val_bg, val_fg = np.hsplit(vals, [i])
        mean_bg = (prob_bg * val_bg).sum() / sum_bg
        mean_fg = (prob_fg * val_fg).sum() / sum_fg
        var_bg = ((val_bg - mean_bg) ** 2 * prob_bg).sum() / sum_bg
        var_fg = ((val_fg - mean_fg) ** 2 * prob_fg).sum() / sum_fg
        cost = var_bg * sum_bg + var_fg * sum_fg
        # Update threshold with minimum cost
        if cost < _min:
            _min, thrsh = cost, i
    return thrsh


def li_threshold_from_histogram(
        hist: NPAR,
        _atol: float = 1e-6,
        _offset: float = 1e-6
    ) -> int:
    """Return Li threshold from a histogram."""
    vals = np.arange(hist.shape[0])
    _min, thrsh = np.inf, 0  # initialize minimum cost and threshold
    hist = hist / hist.sum()
    cumhist = hist.cumsum()  # cumulative sum of histogram
    for i in range(1, hist.shape[0] - 1):
        sum_bg, sum_fg = cumhist[i], cumhist[-1] - cumhist[i]
        if sum_bg < _atol or sum_fg < _atol:  # skip if too few counts
            continue
        # Compute cost from first moment and cross entropy
        prob_bg, prob_fg = np.hsplit(hist, [i])
        val_bg, val_fg = np.hsplit(vals, [i])
        fstm_bg = (prob_bg * val_bg).sum()
        fstm_fg = (prob_fg * val_fg).sum()
        cost = -fstm_bg * (np.log(fstm_bg + _offset) - np.log(sum_bg))
        cost -= fstm_fg * (np.log(fstm_fg + _offset) - np.log(sum_fg))
        # Update threshold with minimum cost
        if cost < _min:
            _min, thrsh = cost, i
    return thrsh

def saturation(img: NPAR) -> NPAR:
    """Return the saturation channel of an image."""
    _min, _max = img.min(axis=-1), img.max(axis=-1)
    return np.where(
        _max > 0,
        (255 * (1 - _min / (_max + np.spacing(0)))).round().astype(np.uint8),
        0
    )


def main() -> None:
    """Empty main."""


if __name__ == '__main__':
    main()

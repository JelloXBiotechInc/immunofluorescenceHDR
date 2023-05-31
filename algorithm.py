"""Module of HDR algorithms."""


import numpy as np
import itertools
from processing import _neighborhood
from typing import Any, Optional


NPAR = np.ndarray


def _mean_threshold(images: NPAR, thrsh: int) -> NPAR:
    return images.mean(axis=0) > thrsh


def foreground_neighborhood(
    images: NPAR,
    target: NPAR,
    neighborhood_radius: int,
    sample_thrsh: int,
    **kwargs
    ) -> NPAR:
    """
    Return mask of foreground within neighborhood of some given target.

    Arguments
    ---------
    images: Grayscale images of different exposures, stacked at the first
        dimension
    target: Grayscale image as target, from whose neighborhood to sample pixels
    neighborhood_radius: Radius of neighborhood
    sample_thrsh: Threshold of mean pixel values over different exposures
    """
    mask = (
        _neighborhood(target, neighborhood_radius) &
        _mean_threshold(images, sample_thrsh)
    )
    return mask


class TooFewPixelsError(Exception):
    """User-defined exception."""
    pass


def fixed_step_samples(pixels: NPAR, num_smpls: int) -> NPAR:
    """
    Return the sampled pixel values with fixed step sampling.

    Arguments
    ---------
    pixels: Pixel values of different exposures, stacked at the first dimension
    num_smpls: Number of pixels to sample
    """
    num_popl = pixels.shape[-1]  # popoulation size
    if num_popl > num_smpls:
        step = num_popl // num_smpls
        offset = num_popl % num_smpls  # arbitrary offset
        inds = offset + np.arange(num_smpls) * step
        return pixels[..., inds]
    else:
        raise TooFewPixelsError(
            'there are not enough relevant pixels to fit the response curve'
        )


def _fixed_step_shuffling(values: NPAR, num_groups: int) -> NPAR:
    group_size = values.shape[0] // num_groups
    offset = values.shape[0] % num_groups  # arbitrary offset
    num_vals = group_size * num_groups
    remains, values = values[num_vals:, ...], values[:num_vals, ...]
    values = values.reshape(num_groups, group_size, *values.shape[1:])
    values = np.roll(values, offset, axis=1)
    values = values.reshape(-1, *values.shape[2:])
    return np.concatenate([values, remains], axis=0)


def uniform_bin_samples(pixels: NPAR, num_smpls: int) -> NPAR:
    """
    Return pixel values sampled from bins which distribute uniformly.

    Arguments
    ---------
    pixels: Pixel values of different exposures, stacked at the first dimension
    num_smpls: Number of pixels to sample
    """
    num_popl = pixels.shape[-1]  # popoulation size
    if num_popl > num_smpls:
        avgs = pixels.mean(axis=0)
        inds = np.arange(num_popl)
        bins = np.linspace(avgs.min(), avgs.max(), num=num_smpls+1)
        # Shuffle pixels
        avgs = _fixed_step_shuffling(avgs, num_smpls)
        inds = _fixed_step_shuffling(inds, num_smpls)
        # Find first pixel in each bin
        within = (avgs >= bins[:-1, None]) & (avgs < bins[1:, None])
        sampled = within.argmax(axis=-1)
        return pixels[..., inds[sampled]]
    else:
        raise TooFewPixelsError(
            'there are not enough relevant pixels to fit the response curve'
        )


def _weighting_toward_middle(num_vals: int, dtype: Any, offset: int = 1) -> NPAR:
    weight = np.arange(num_vals, dtype=dtype)
    weight = np.minimum(weight, num_vals - 1 - weight) + offset
    return weight


def _response_curve(
    pixels: NPAR,
    log_exp: NPAR,
    weight: NPAR,
    smooth: float,
    num_vals: int = 256,
    ) -> NPAR:
    """
    Return the estimated response curve.

    Arguments
    ---------
    pixels: Pixel values of different exposures, stacked at the first dimension
    log_exp: Logarithmic exposure times
    weight: Weighting on pixel values
    smooth: Parameter of smoothing regularization
    num_vals: Number of possible pixel values
    """
    num_pixels, num_imgs = pixels.shape[1], pixels.shape[0]
    dtype = log_exp.dtype
    # Construct linear equations that minimizes the quadratic objective
    mat = np.zeros((
        num_pixels * num_imgs + num_vals + 1,
        num_vals + num_pixels
    ), dtype=dtype)
    vec = np.zeros((mat.shape[0], 1), dtype=dtype)
    # Objective for reconstruction
    k = 0
    for i, j in itertools.product(range(num_pixels), range(num_imgs)):
        val = pixels[j, i]
        mat[k, val] = weight[val]
        mat[k, i + num_vals] = -weight[val]
        vec[k, 0] = weight[val] * log_exp[j]
        k += 1
    # Constant constraint on middle point
    mat[k, num_vals // 2] = 1
    k += 1
    # Smoothness regularization
    for i in range(num_vals - 2):
        mat[k, i] = smooth * weight[i]
        mat[k, i + 1] = -2 * smooth * weight[i]
        mat[k, i + 2] = smooth * weight[i]
        k += 1
    # Solve linear equations by SVD
    lsng_vecs, sng_vals, rsng_vecs_h = np.linalg.svd(mat, full_matrices=False)
    pseudo_inv = rsng_vecs_h.T @ np.diag(1 / sng_vals) @ lsng_vecs.T
    sol = (pseudo_inv @ vec).squeeze(axis=-1)
    return sol[:num_vals]


def debevec_response(
    pixels: NPAR,
    exposures: NPAR,
    smooth: float = 50,
    weight_offset: int = 1,
    _num_vals: int = 256,
    **kwargs
    ) -> NPAR:
    """
    Return the resposne curve following the Debevec algorithm.

    Arguments
    ---------
    pixels: Pixel values of different exposures, stacked at the first dimension
    exposures: Exposure times ordered ascendingly
    smooth: Parameter of smoothing regularization
    weight_offset: Constant offset of weighting function
    """
    weight = _weighting_toward_middle(_num_vals, exposures.dtype, weight_offset)
    return _response_curve(
        pixels,
        np.log(exposures),
        weight,
        smooth=smooth,
        num_vals=_num_vals
    )


def debevec_irradiance(
    images: NPAR,
    exposures: NPAR,
    response: NPAR,
    weight_offset: int = 1,
    **kwargs
    ) -> NPAR:
    """
    Return irradiance of image estimated by Debevec HDR algorithm.
    
    Arguments
    ---------
    images: Grayscale images of different exposures, stacked at the first
        dimension
    exposures: Exposure times ordered ascendingly
    response: Response curve
    weight_offset: Constant offset of weighting function
    """
    num_vals = response.shape[0]
    weight = _weighting_toward_middle(num_vals, exposures.dtype, weight_offset)
    log_irradiance = (
        weight[images] * (response[images] - np.log(exposures).reshape(-1, 1, 1))
    ).sum(axis=0)
    log_irradiance /= weight[images].sum(axis=0)
    return np.exp(log_irradiance)


def minmax_scaling(
    img: NPAR,
    trgtmin: float,
    trgtmax: float,
    srcmin: Optional[float] = None,
    srcmax: Optional[float] = None,
    ) -> NPAR:
    """
    Return image scaled to target values of minimum and maximum.

    Arguments
    ---------
    img: Image to process
    trgtmin: Minimum pixel value to scale to
    trgtmax: Maximum pixel value to scale to
    srcmin: Minimum pixel value to scale from
    srcmax: Maximum pixel value to scale from

    Note
    ----
    If either `srcmin` or `srcmax` is not specified, they are inferred from
    `img` automatically.
    """
    if srcmin is None or srcmax is None:
        srcmin, srcmax = img.min(), img.max()
    img = trgtmin + (trgtmax - trgtmin) * ((img - srcmin) / (srcmax - srcmin))
    return img.clip(0, 255).astype(np.uint8)


def main() -> None:
    """Empty main."""


if __name__ == '__main__':
    main()

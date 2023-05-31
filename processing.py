"""Module of computer-vision processing."""


import cv2
import numpy as np


NPAR = np.ndarray


def _cv_element(ksize: int, morph: int = cv2.MORPH_RECT) -> NPAR:
    return cv2.getStructuringElement(morph, (ksize, ksize))


def _erode(img: NPAR, ksize: int) -> None:
    """Erode image with rectangular kernel."""
    cv2.erode(img, _cv_element(ksize), dst=img)


def _blur_gaussian(img: NPAR, ksize: int) -> None:
    """Blur image with Gaussian kernel."""
    cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0, dst=img)


def erode_blur(
    image: NPAR,
    erode_ksize: int,
    blur_ksize: int,
    **kwargs
    ) -> None:
    """
    Erode and blur an image.
    
    Arguments
    ---------
    image: Image to process
    erode_ksize: Kernel size of erosion
    blur_ksize: Kernel size of Gaussian blurring
    """
    image = image.copy()
    _erode(image, erode_ksize)
    _blur_gaussian(image, blur_ksize)
    return image


def _neighborhood(target: NPAR, radius: int) -> NPAR:
    """Return mask of neighborhood with dilation and Otsu thresholding."""
    mask = cv2.dilate(target, _cv_element(2 * radius + 1))
    _, mask = cv2.threshold(
        mask,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return mask > 0


def _background_threshold(img: NPAR, fg: NPAR, rad: int, perc: float) -> int:
    """
    Return threshold to filter background.

    Arguments
    ---------
    img: Grayscale image to threshold
    fg: Grayscale image of sure foreground in `img`
    rad: Radius to dilate sure foreground
    perc: Top percentage of background pixel values to determine threshold
    """
    mask = _neighborhood(fg, rad)
    return np.quantile(img[~mask], 1 - perc) if (~mask).any() else 0


def _curved_gamma_correct(
    img: NPAR,
    thrsh: int,
    gamma_upr: float,
    gamma_lwr: float
    ) -> None:
    """
    Apply sfifted gamma correction for pixel values above a threshold, and
    apply complement gamma correction for those below.
    
    Arguments
    ---------
    img: Grayscale image to process
    thrsh: Threshold of pixel value
    gamma_upr: Gamma parameter to correct pixels values above `thrsh`
    gamma_lwr: Gamma parameter to correct pixels values below `thrsh`
    """
    lut = np.empty((1, 256), dtype=np.uint8)
    for i in range(256):
        if i >= thrsh:
            lut[0, i] = thrsh + (255 - thrsh) * (
                (i - thrsh) / (255 - thrsh)
            ) ** (1 / gamma_upr)
        else:
            lut[0, i] = thrsh - thrsh * (
                (thrsh - i) / thrsh
            ) ** (1 / gamma_lwr)
    cv2.LUT(img, lut, dst=img)


def _clahe(img: NPAR, clip_limit: float = 3, grid_size: int = 8) -> None:
    """Apply contrast limited adaptive histogram equalization."""
    clahe = cv2.createCLAHE(clip_limit, (grid_size, grid_size))
    clahe.apply(img, dst=img)


def _enhance_contrast(
    img: NPAR,
    ksize: int,
    num_iters: int,
    morph: int = cv2.MORPH_RECT,
    ) -> None:
    """
    Enhance contrast of image iteratively, with blurring between consecutive
    adjustment.
    
    Arguments
    ---------
    img: Grayscale image to process
    ksize: Kernel size for tophat and blackhat operations
    num_iters: Number of iterations
    morph: Kernel type of tophat and blackhat operations

    Note
    ----
    Contrast is adjusted by enhancing the bright part and lessening the dark
    part in the image. See https://towardsdatascience.com/contrast-enhancement-of-grayscale-images-using-morphological-operators-de6d483545a1.
    """
    element = _cv_element(ksize, morph)
    for i in range(num_iters):
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, element)
        blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, element)
        cv2.subtract(img, blackhat, dst=img)
        cv2.add(img, tophat, dst=img)
        if i < num_iters - 1:  # between consecutive contrast adjustment
            cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0, dst=img)


# def otsu_threshold(hist: NPAR, atol: float = 1e-6) -> float:
#     """Return threshold determined by Otsu method."""
#     vals = np.arange(256)
#     _min, thrsh = np.inf, 0  # initialize minimum cost and threshold
#     cumhist = hist.cumsum()  # cumulative sum of histogram
#     for i in range(1, 255):
#         sum_bg, sum_fg = cumhist[i], cumhist[-1] - cumhist[i]
#         if sum_bg < atol or sum_fg < atol:  # skip if too few counts
#             continue
#         # Compute cost from mean and variance
#         prob_bg, prob_fg = np.hsplit(hist, [i])
#         val_bg, val_fg = np.hsplit(vals, [i])
#         mean_bg = (prob_bg * val_bg).sum() / sum_bg
#         mean_fg = (prob_fg * val_fg).sum() / sum_fg
#         var_bg = ((val_bg - mean_bg) ** 2 * prob_bg).sum() / sum_bg
#         var_fg = ((val_fg - mean_fg) ** 2 * prob_fg).sum() / sum_fg
#         cost = var_bg * sum_bg + var_fg * sum_fg
#         # Update threshold with minimum cost
#         if cost < _min:
#             _min, thrsh = cost, i
#     return thrsh


# def li_threshold(hist: NPAR, atol: float = 1e-6) -> float:
#     """Return threshold determined by Li method."""
#     vals = np.arange(256)
#     _min, thrsh = np.inf, 0  # initialize minimum cost and threshold
#     cumhist = hist.cumsum()  # cumulative sum of histogram
#     for i in range(1, 255):
#         sum_bg, sum_fg = cumhist[i], cumhist[-1] - cumhist[i]
#         if sum_bg < atol or sum_fg < atol:  # skip if too few counts
#             continue
#         # Compute cost from first moment and cross entropy
#         prob_bg, prob_fg = np.hsplit(hist, [i])
#         val_bg, val_fg = np.hsplit(vals, [i])
#         fstm_bg = (prob_bg * val_bg).sum()
#         fstm_fg = (prob_fg * val_fg).sum()
#         cost = -fstm_bg * (np.log(fstm_bg + atol) - np.log(sum_bg))
#         cost -= fstm_fg * (np.log(fstm_fg + atol) - np.log(sum_fg))
#         # Update threshold with minimum cost
#         if cost < _min:
#             _min, thrsh = cost, i
#     return thrsh


def gamma_contrast_correction(
    image: NPAR,
    foreground: NPAR,
    neighborhood_radius: int,
    background_top_percentage: float,
    gamma_upper: float,
    gamma_lower: float,
    contrast_ksize: int,
    contrast_iters: int,
    clahe_limit: int = 3,
    clahe_gsize: int = 8,
    **kwargs
    ) -> NPAR:
    """
    Return image after (1) curved gamma correction, (2) contrast limited
    adaptive histogram equalization, and (3) contrast enhancement.

    Arguments
    ---------
    image: Grayscale image to process
    foreground: Grayscale image of sure foreground
    neighborhood_radius: Radius of neighborhood dilation
    background_top_percentage: Top percentage of background pixel values to
        determine threshold for curved gamma correction
    gamma_upper: Gamma parameter to correct pixels values above a threshold
    gamma_lower: Gamma parameter to correct pixels values below a threshold
    contrast_ksize: Kernel size for contrast enhancement
    contrast_iters: Number of iterations during contrast enhancement
    clahe_limit: Contrast clip limit of CLAHE
    clahe_gsize: Size of grids to apply CLAHE
    """
    image = image.copy()
    thrsh = _background_threshold(
        image,
        foreground,
        neighborhood_radius,
        background_top_percentage
    )
    print(thrsh)
    _curved_gamma_correct(image, thrsh, gamma_upper, gamma_lower)
    _clahe(image, clahe_limit, clahe_gsize)
    _enhance_contrast(
        image,
        contrast_ksize,
        contrast_iters,
        morph=cv2.MORPH_ELLIPSE
    )
    return image


def main() -> None:
    """Empty main."""


if __name__ == '__main__':
    main()

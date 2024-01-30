import numpy as np
from PIL import Image
import random
from typing import Sequence
import mmcv


def convert(img: np.ndarray,
            alpha: int = 1,
            beta: int = 0) -> np.ndarray:
    """Multiple with alpha and add beat with clip.

    Args:
        img (np.ndarray): The input image.
        alpha (int): Image weights, change the contrast/saturation
            of the image. Default: 1
        beta (int): Image bias, change the brightness of the
            image. Default: 0

    Returns:
        np.ndarray: The transformed image.
    """

    img = img.astype(np.float32) * alpha + beta
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def brightness(img: np.ndarray, brightness_delta: int = 32) -> np.ndarray:
    """Brightness distortion.

    Args:
        img (np.ndarray): The input image.
    Returns:
        np.ndarray: Image after brightness change.
    """

    if random.randint(0, 1):
        return convert(
            img,
            beta=random.uniform(-brightness_delta,
                                brightness_delta))
    return img

def contrast(img: np.ndarray, contrast_range: Sequence[float] = (0.5, 1.5)) -> np.ndarray:
    """Contrast distortion.

    Args:
        img (np.ndarray): The input image.
    Returns:
        np.ndarray: Image after contrast change.
    """
    contrast_lower, contrast_upper = contrast_range
    if random.randint(0, 1):
        return convert(
            img,
            alpha=random.uniform(contrast_lower, contrast_upper))
    return img

def saturation(img: np.ndarray, saturation_range: Sequence[float] = (0.5, 1.5)) -> np.ndarray:
    """Saturation distortion.

    Args:
        img (np.ndarray): The input image.
    Returns:
        np.ndarray: Image after saturation change.
    """
    saturation_lower, saturation_upper = saturation_range
    if random.randint(0, 1):
        img = mmcv.bgr2hsv(img)
        img[:, :, 1] = convert(
            img[:, :, 1],
            alpha=random.uniform(saturation_lower,
                                 saturation_upper))
        img = mmcv.hsv2bgr(img)
    return img

def hue(img: np.ndarray, hue_delta: int = 18) -> np.ndarray:
    """Hue distortion.

    Args:
        img (np.ndarray): The input image.
    Returns:
        np.ndarray: Image after hue change.
    """

    if random.randint(0, 1):
        img = mmcv.bgr2hsv(img)
        img[:, :,
            0] = (img[:, :, 0].astype(int) +
                  random.randint(-hue_delta, hue_delta)) % 180
        img = mmcv.hsv2bgr(img)
    return img

def PhotoMetricDistortion(img):
    """Transform function to perform photometric distortion on images.

    Args:
        results (dict): Result dict from loading pipeline.

    Returns:
        dict: Result dict with images distorted.
    """

    # random brightness
    img = brightness(img)

    # mode == 0 --> do random contrast first
    # mode == 1 --> do random contrast last
    mode = random.randint(0, 1)
    if mode == 1:
        img = contrast(img)

    # random saturation
    img = saturation(img)

    # random hue
    img = hue(img)

    # random contrast
    if mode == 0:
        img = contrast(img)

    return img

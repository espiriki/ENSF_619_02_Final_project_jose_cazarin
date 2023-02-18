import torch
from PIL import Image
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
import PIL


def add_margin(pil_img, top, right, bottom, left, color):

    result = np.pad(
        pil_img,
        pad_width=((top, bottom), (left, right), (0, 0)),
        mode='constant')

    return result


class PadToMaintainAR(ImageOnlyTransform):

    def __init__(self, aspect_ratio):
        super().__init__()
        self.aspect_ratio = aspect_ratio

    def apply(self, img, **params):

        size = img.shape

        current_aspect_ratio = size[0] / size[1]
        target_aspect_ratio = self.aspect_ratio
        original_width = size[0]
        original_height = size[1]
        new_img = []

        if current_aspect_ratio == target_aspect_ratio:
            new_img = img
        if current_aspect_ratio < target_aspect_ratio:
            # need to increase width
            target_width = int(target_aspect_ratio * original_height)
            pad_amount_pixels = target_width - original_width
            new_img = add_margin(img, 0, int(pad_amount_pixels/2),
                                 0, int(pad_amount_pixels/2), (0, 0, 0))

        if current_aspect_ratio > target_aspect_ratio:
            # need to increase height
            target_height = int(original_width/target_aspect_ratio)
            pad_amount_pixels = target_height - original_height
            new_img = add_margin(img, int(pad_amount_pixels/2),
                                 0, int(pad_amount_pixels/2), 0, (0, 0, 0))

        return new_img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{str(self.aspect_ratio)}"


class PadToMaintainAR_Album(ImageOnlyTransform):

    def __init__(self, aspect_ratio):
        super().__init__()
        self.aspect_ratio = aspect_ratio

    def apply(self, img, **params):
        return img

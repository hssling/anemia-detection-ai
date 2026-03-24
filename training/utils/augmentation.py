# training/utils/augmentation.py
"""Albumentations pipelines for training and validation."""

try:
    import albumentations as A
except ModuleNotFoundError:  # pragma: no cover
    A = None


class _IdentityCompose:
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, *, image):
        import numpy as np
        from PIL import Image

        pil = Image.fromarray(image).resize((self.image_size, self.image_size))
        return {"image": np.array(pil)}


def get_augmentation_pipeline(image_size: int = 380):
    if A is None:
        return _IdentityCompose(image_size)
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.3),
        ]
    )


def get_val_transforms(image_size: int = 380):
    if A is None:
        return _IdentityCompose(image_size)
    return A.Compose(
        [
            A.Resize(image_size, image_size),
        ]
    )

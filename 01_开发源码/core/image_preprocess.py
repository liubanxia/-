import numpy as np


def to_uint8(image):
    image = np.asarray(image, dtype=np.float32)

    lo, hi = np.percentile(
        image,
        (1, 99),
    )

    if hi <= lo:
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    image = np.clip(image, lo, hi)
    image = (image - lo) / (hi - lo)

    return (image * 255).astype(
        np.uint8
    )


def to_rgb(image):
    image = to_uint8(image)

    if image.ndim == 2:
        image = np.stack(
            [image, image, image],
            axis=-1,
        )

    return image

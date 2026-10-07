"""Image decoding and the single preprocessing pipeline shared by the router and every expert."""

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# ImageNet statistics - part of the model contract, identical for router and experts.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# PIL reports some phone JPEGs (multi-picture format) as "MPO".
ALLOWED_FORMATS = frozenset({"JPEG", "MPO", "PNG"})


class ImageDecodeError(Exception):
    """Upload is not a readable JPEG/PNG image."""


@dataclass(frozen=True)
class DecodedImage:
    image: Image.Image  # RGB, EXIF orientation applied, possibly JPEG-draft downscaled
    original_size: tuple[int, int]  # (width, height) as captured, before any downscale
    format: str


def decode_image(data: bytes, draft_min_side: int) -> DecodedImage:
    """Decode an upload to RGB.

    For JPEGs, PIL's draft mode decodes directly at a reduced DCT scale (1/2, 1/4, 1/8) while
    keeping both sides >= draft_min_side. A 12 MP phone photo then decodes in a few ms instead
    of ~100 ms, which matters on a single vCPU.
    """
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format or ""
        if fmt not in ALLOWED_FORMATS:
            raise ImageDecodeError(f"unsupported format {fmt!r}")
        original_size = img.size
        if fmt in ("JPEG", "MPO"):
            img.draft("RGB", (draft_min_side, draft_min_side))
        img.load()
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except ImageDecodeError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageDecodeError(str(exc)) from exc
    return DecodedImage(image=img, original_size=original_size, format=fmt)


def preprocess_image(img: Image.Image, input_size: tuple[int, int]) -> np.ndarray:
    """RGB PIL image -> normalised (1, 3, H, W) float32 tensor.

    input_size is (height, width), matching manifest.json's router.input_size.
    """
    height, width = input_size
    resized = img.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0  # HWC
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # HWC -> 1CHW
    return np.ascontiguousarray(arr, dtype=np.float32)

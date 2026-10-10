"""Image loading (URL / base64 / bytes), normalisation, hashing and encoding."""
import base64
import io
import ipaddress
import socket
from typing import Tuple
from urllib.parse import urlparse

import httpx
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import settings

Image.MAX_IMAGE_PIXELS = 50_000_000  # protect against decompression bombs


class ImageError(ValueError):
    """Raised for any problem with the supplied image (bad URL, corrupt file, too large...)."""


def _check_url_is_safe(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ImageError("image_url must be a valid http(s) URL")
    host = parsed.hostname
    if settings.allowed_image_hosts and host not in settings.allowed_image_hosts:
        raise ImageError("image_url host is not in the allowed list")
    if settings.allow_private_urls:
        return
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror:
        raise ImageError("image_url host could not be resolved")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ImageError("image_url points to a private address")


def load_bytes_from_url(url: str) -> bytes:
    _check_url_is_safe(url)
    try:
        with httpx.Client(timeout=settings.url_timeout_s, follow_redirects=True) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                chunks, total = [], 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > settings.max_upload_bytes:
                        raise ImageError("image is too large")
                    chunks.append(chunk)
                return b"".join(chunks)
    except ImageError:
        raise
    except httpx.HTTPError as exc:
        raise ImageError(f"could not download image: {exc.__class__.__name__}")


def load_bytes_from_base64(data: str) -> bytes:
    if "," in data and data.strip().lower().startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=False)
    except Exception:
        raise ImageError("image_base64 is not valid base64")
    if len(raw) > settings.max_upload_bytes:
        raise ImageError("image is too large")
    return raw


def open_image(raw: bytes) -> Image.Image:
    """Open image bytes WITHOUT altering them, so EXIF metadata is still readable."""
    if not raw:
        raise ImageError("empty image")
    if len(raw) > settings.max_upload_bytes:
        raise ImageError("image is too large")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ImageError("file is not a valid image")


def prepare_rgb(img: Image.Image) -> np.ndarray:
    """Upright (EXIF orientation), RGB, resized so the longest side <= max_image_side."""
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    longest = max(img.size)
    if longest > settings.max_image_side:
        scale = settings.max_image_side / longest
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8).copy()


def dhash(img: Image.Image, size: int = 8) -> int:
    """64-bit difference hash: robust to resizing / recompression."""
    gray = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    arr = np.asarray(gray, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def encode_jpeg_base64(rgb: np.ndarray, max_side: int = 800, quality: int = 80) -> str:
    img = Image.fromarray(rgb)
    longest = max(img.size)
    if longest > max_side:
        scale = max_side / longest
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def image_size(rgb: np.ndarray) -> Tuple[int, int]:
    return int(rgb.shape[1]), int(rgb.shape[0])

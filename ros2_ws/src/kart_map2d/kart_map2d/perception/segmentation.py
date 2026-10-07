"""Drivable-area and obstacle segmentation for the kart cameras.

The mapper must work from camera + IMU alone, in any environment (campus
plazas, car parks, lawns), not only on marked tracks. Classical colour or
texture heuristics tuned to one surface break the moment the surface changes,
so we use a semantic segmentation network trained on Cityscapes: its 19-class
street taxonomy (road, sidewalk, people, vehicles, vegetation, ...) transfers
well to any outdoor ground-vehicle scene and directly answers the two
questions the mapper asks of every pixel: "can the kart drive here?" and
"is this an obstacle?".

SegFormer-B0 is chosen because it is the smallest model of the family that
still tops the Cityscapes leaderboard for its size, and it comfortably fits
the 6 GB laptop GPU in fp16 at the checkpoint's native 1024x1024 input.

Pixel categories (Cityscapes train ids, stored as uint8 in the class map):
- drivable: road, sidewalk. NOT terrain/grass - the kart can physically cross
  a lawn but must not plan onto it, so grass is kept out of the drivable set.
- obstacle: buildings, walls, fences, poles, traffic lights/signs, people,
  riders, all vehicle classes, and vegetation (trees and bushes at kart
  height are things to avoid, unlike flat terrain).
- neutral: terrain and sky - neither drivable nor a mappable obstacle.
- ``KART_ID`` (255): the kart's own body. The wheels intrude into the lower
  image corners; those pixels belong to no scene class and are excluded from
  every mask so they never pollute the map.

Results are cached per frame under ``outputs/seg/<run>/<camera>/`` as
compressed npz holding only the uint8 class map; all masks derive from it.

Importing this module never imports torch or transformers: the class-id
tables, masks, :class:`SegmentationResult`, the caching helpers and
:func:`get_segmenter` (with the zero-network classical backend) all work on
a torch-free install. torch is imported only when the neural backend is
constructed (:class:`Segmenter`), so a classical-only deployment - the kart
bundle built by ``scripts/make_deploy_bundle.py --seg classical`` - never
needs it on disk.
"""

from __future__ import annotations

import hashlib
import importlib.util
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from kart_map2d.core.measurements import CameraFrame

MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

#: Where a converted (safetensors) copy of the checkpoint is kept, relative to
#: the working directory; a deploy bundle ships the weights at the same place.
MODEL_CACHE_DIR = Path("outputs/cache/models")

#: Accepted ``seg_mode`` values everywhere (pipeline, CLI, live node).
SEG_MODES: tuple[str, ...] = ("auto", "neural", "classical")

#: Cityscapes evaluation classes in train-id order; the checkpoint's output
#: channel i is CITYSCAPES_CLASSES[i]. Verified against the model config at
#: load time so a checkpoint swap cannot silently change the cached format.
CITYSCAPES_CLASSES: tuple[str, ...] = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

DRIVABLE_CLASSES: tuple[str, ...] = ("road", "sidewalk")
NEUTRAL_CLASSES: tuple[str, ...] = ("terrain", "sky")
OBSTACLE_CLASSES: tuple[str, ...] = (
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

#: Class-map value for the kart's own body; outside the Cityscapes id range.
KART_ID = 255


def class_ids(names: Sequence[str]) -> frozenset[int]:
    """Map Cityscapes class names to train ids (raises on unknown names)."""
    return frozenset(CITYSCAPES_CLASSES.index(name) for name in names)


DRIVABLE_IDS = class_ids(DRIVABLE_CLASSES)
OBSTACLE_IDS = class_ids(OBSTACLE_CLASSES)
NEUTRAL_IDS = class_ids(NEUTRAL_CLASSES)

# Lookup table: class-map value -> category (0 = none/kart, 1..3 below).
_DRIVABLE, _OBSTACLE, _NEUTRAL = 1, 2, 3
_CATEGORY_LUT = np.zeros(256, dtype=np.uint8)
_CATEGORY_LUT[list(DRIVABLE_IDS)] = _DRIVABLE
_CATEGORY_LUT[list(OBSTACLE_IDS)] = _OBSTACLE
_CATEGORY_LUT[list(NEUTRAL_IDS)] = _NEUTRAL


@lru_cache(maxsize=8)
def default_kart_mask(
    height: int,
    width: int,
    bottom_fraction: float = 0.15,
    corner_width_fraction: float = 0.30,
    corner_height_fraction: float = 0.38,
) -> NDArray[np.bool_]:
    """Static mask of the kart's own body in the image (True = kart pixel).

    The ZED X GS units sit ~0.56 m above ground looking forward-down, so the
    kart's nose fills the bottom strip and the front wheels rise into the
    lower corners. A fixed geometric mask (bottom ``bottom_fraction`` of the
    rows plus two lower-corner triangles) is used instead of a per-run
    variance estimate: the mount is rigid, so the mask is identical across
    runs, and a closed form keeps segmentation deterministic and testable.
    """
    rows = np.arange(height, dtype=np.float64).reshape(-1, 1)
    cols = np.arange(width, dtype=np.float64).reshape(1, -1)
    mask = np.broadcast_to(rows >= (1.0 - bottom_fraction) * height, (height, width)).copy()
    # Triangles anchored at the bottom corners: full corner width at the
    # bottom row, tapering to nothing corner_height_fraction up the image.
    rise = (height - 1 - rows) / max(corner_height_fraction * height, 1.0)
    reach = corner_width_fraction * width
    mask |= cols < reach * (1.0 - rise)  # lower-left
    mask |= (width - 1 - cols) < reach * (1.0 - rise)  # lower-right
    mask.flags.writeable = False
    return mask


@dataclass(frozen=True)
class SegmentationResult:
    """Per-frame segmentation; all masks derive from the stored class map.

    ``class_map`` is (H, W) uint8 of Cityscapes train ids at full image
    resolution, with ``KART_ID`` wherever the kart's own body was masked out.
    """

    class_map: NDArray[np.uint8]

    @property
    def drivable_mask(self) -> NDArray[np.bool_]:
        """True where the kart may drive (road + sidewalk, never grass)."""
        return _CATEGORY_LUT[self.class_map] == _DRIVABLE

    @property
    def obstacle_mask(self) -> NDArray[np.bool_]:
        """True on structures, people, vehicles and vegetation to avoid."""
        return _CATEGORY_LUT[self.class_map] == _OBSTACLE

    @property
    def neutral_mask(self) -> NDArray[np.bool_]:
        """True on terrain and sky: neither drivable nor an obstacle."""
        return _CATEGORY_LUT[self.class_map] == _NEUTRAL

    @property
    def kart_mask(self) -> NDArray[np.bool_]:
        """True on the kart's own body (excluded from every scene class)."""
        return self.class_map == KART_ID

    @property
    def drivable_fraction(self) -> float:
        """Drivable share of the scene pixels (kart body excluded)."""
        scene = self.class_map != KART_ID
        n = int(scene.sum())
        return float(np.count_nonzero(self.drivable_mask)) / n if n else 0.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, class_map=self.class_map)

    @classmethod
    def load(cls, path: Path) -> "SegmentationResult":
        with np.load(path) as data:
            return cls(np.ascontiguousarray(data["class_map"], dtype=np.uint8))


def _import_torch():
    """torch, imported on first use by the neural backend only.

    Kept out of module scope so that everything else in this module (and the
    whole classical path) runs on a torch-free install; the error names the
    fix instead of a bare ModuleNotFoundError deep in a call stack.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "the neural segmentation backend needs torch + transformers "
            "(requirements/neural-seg.txt; on a Jetson install NVIDIA's JetPack torch "
            "wheel, not the PyPI build) - or use seg_mode='classical', which needs neither"
        ) from exc
    return torch


def _installed(module: str) -> bool:
    """Whether ``module`` is importable, WITHOUT importing it (find_spec only)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def local_model_dir(model_name: str = MODEL_NAME, cache_dir: Path = MODEL_CACHE_DIR) -> Path:
    """Directory holding the converted checkpoint (``model.safetensors`` + configs)."""
    return Path(cache_dir) / model_name.rsplit("/", 1)[-1]


def _load_torch_weights(model_name: str, cache_dir: Path = MODEL_CACHE_DIR):
    """Convert a torch-format checkpoint to safetensors, then load normally.

    The conversion runs once: the safetensors copy is kept under
    ``outputs/cache/models/<name>/`` and later loads go straight to
    ``from_pretrained``, which also applies the transformers-5 weight
    renamings (the checkpoint predates the SegFormer module refactor).
    ``weights_only=True`` avoids the pickle code-execution path on this
    trusted upstream file.
    """
    import shutil

    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file
    from transformers import AutoModelForSemanticSegmentation

    torch = _import_torch()
    local = local_model_dir(model_name, cache_dir)
    if not (local / "model.safetensors").exists():
        local.mkdir(parents=True, exist_ok=True)
        config = hf_hub_download(model_name, "config.json")
        weights = hf_hub_download(model_name, "pytorch_model.bin")
        state = torch.load(weights, map_location="cpu", weights_only=True)
        state = {k: v.contiguous() for k, v in state.items()}
        shutil.copyfile(config, local / "config.json")
        save_file(state, local / "model.safetensors", metadata={"format": "pt"})
    return AutoModelForSemanticSegmentation.from_pretrained(local)


class Segmenter:
    """SegFormer-B0 (Cityscapes) wrapper producing :class:`SegmentationResult`.

    Runs in fp16 on CUDA when available and falls back to fp32 on CPU. The
    model download / load is deferred to the first ``segment`` call so that
    constructing a Segmenter (e.g. in tests, with an injected fake model)
    costs nothing. ``model`` and ``processor`` can be injected for testing;
    both must follow the HuggingFace calling convention.

    Construction is the point where torch is imported (ImportError with the
    install hint when it is missing); transformers follows on the first
    ``segment`` call.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        fp16: bool = True,
        apply_kart_mask: bool = True,
        kart_mask: NDArray[np.bool_] | None = None,
        model=None,
        processor=None,
    ) -> None:
        torch = _import_torch()
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if (fp16 and self.device.startswith("cuda")) else torch.float32
        self.apply_kart_mask = apply_kart_mask
        self._custom_kart_mask = kart_mask
        self._model = model
        self._processor = processor
        if model is not None:
            model.to(self.device)
            if self.dtype is torch.float16:
                model.half()
        # Wall-clock accounting so validation can report ms/frame without
        # instrumenting the call sites.
        self.frames_processed = 0
        self.inference_seconds = 0.0

    # ------------------------------------------------------------- model
    def load(self) -> None:
        """Load the model now instead of on the first ``segment`` call.

        The live node calls this before spinning so the multi-second load
        never stalls a mapping-timer tick (and with it the IMU callbacks).
        """
        self._load()

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        torch = _import_torch()
        # A converted local copy (written by _load_torch_weights, or shipped
        # in a deploy bundle) is used first so an offline kart never touches
        # the hub; it holds exactly the weights the hub path ends up loading.
        local = local_model_dir(self.model_name)
        if self._processor is None:
            source = local if (local / "preprocessor_config.json").exists() else self.model_name
            self._processor = AutoImageProcessor.from_pretrained(source)
        if (local / "model.safetensors").exists():
            model = AutoModelForSemanticSegmentation.from_pretrained(local)
        else:
            try:
                model = AutoModelForSemanticSegmentation.from_pretrained(self.model_name)
            except ValueError:
                # The NVIDIA checkpoint ships torch-format weights only, which
                # transformers 5 refuses to torch.load on torch < 2.6. The file
                # is a trusted upstream release and weights_only=True avoids
                # the pickle code-execution path, so load it ourselves.
                model = _load_torch_weights(self.model_name)
        id2label = model.config.id2label
        names = tuple(str(id2label[i]).lower() for i in sorted(id2label))
        if names != CITYSCAPES_CLASSES:
            raise ValueError(
                f"{self.model_name} labels {names!r} do not match the Cityscapes "
                "train-id order this module (and its caches) are built around"
            )
        model.eval().to(self.device)
        if self.dtype is torch.float16:
            model.half()
        self._model = model

    # --------------------------------------------------------- inference
    def segment(self, image_bgr: NDArray[np.uint8]) -> SegmentationResult:
        """Segment one BGR uint8 image (any resolution)."""
        return self.segment_batch([image_bgr])[0]

    def segment_batch(self, images_bgr: Sequence[NDArray[np.uint8]]) -> list[SegmentationResult]:
        """Segment several BGR uint8 images in one forward pass.

        Images may have different sizes: the processor rescales each to the
        network input, and the logits are upsampled back to every image's own
        resolution before the argmax, so the class map is pixel-aligned with
        the input frame.
        """
        if not images_bgr:
            return []
        self._load()
        torch = _import_torch()
        start = time.perf_counter()
        rgb = [np.ascontiguousarray(img[:, :, ::-1]) for img in images_bgr]
        inputs = self._processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, self.dtype)
        with torch.inference_mode():
            logits = self._model(pixel_values=pixel_values).logits
            results = []
            for i, img in enumerate(images_bgr):
                h, w = img.shape[:2]
                up = torch.nn.functional.interpolate(
                    logits[i : i + 1].float(), size=(h, w), mode="bilinear", align_corners=False
                )
                class_map = up.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
                results.append(SegmentationResult(self._masked(class_map)))
        self.inference_seconds += time.perf_counter() - start
        self.frames_processed += len(images_bgr)
        return results

    def _masked(self, class_map: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if not self.apply_kart_mask:
            return class_map
        mask = self._custom_kart_mask
        if mask is None:
            mask = default_kart_mask(*class_map.shape)
        class_map[mask] = KART_ID
        return class_map

    @property
    def ms_per_frame(self) -> float:
        """Mean wall-clock milliseconds per segmented frame so far."""
        return 1e3 * self.inference_seconds / self.frames_processed if self.frames_processed else 0.0


# ------------------------------------------------------------------ factory
def neural_weights_available(model_name: str = MODEL_NAME) -> bool:
    """True when the SegFormer checkpoint is already on disk (no download)."""
    if (local_model_dir(model_name) / "model.safetensors").exists():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(model_name, "pytorch_model.bin"), str)
    except Exception:
        return False


def neural_backend_available() -> bool:
    """True when the neural backend can run here without installing or downloading.

    Requires torch and transformers to be installed and the default
    checkpoint to be on disk. The packages are probed with
    ``importlib.util.find_spec``, which does NOT import them, so asking costs
    milliseconds and a torch-free process stays torch-free.
    """
    return _installed("torch") and _installed("transformers") and neural_weights_available()


def resolve_seg_mode(mode: str) -> str:
    """``auto`` -> ``neural`` when :func:`neural_backend_available`, else ``classical``.

    ``neural`` / ``classical`` pass through unchanged; anything else raises
    ValueError. Never imports torch.
    """
    if mode == "auto":
        return "neural" if neural_backend_available() else "classical"
    if mode in ("neural", "classical"):
        return mode
    raise ValueError(f"unknown segmenter mode {mode!r} (use auto | neural | classical)")


def get_segmenter(mode: str = "auto", **kwargs):
    """Return a drivable/obstacle segmenter backend.

    ``"neural"`` returns the SegFormer :class:`Segmenter` (this is where
    torch gets imported), ``"classical"`` the zero-network
    :class:`~kart_map2d.perception.classical_ground.ClassicalGroundSegmenter`
    (torch-free), and ``"auto"`` picks neural when torch, transformers and
    the checkpoint are all present (:func:`resolve_seg_mode`), falling back
    to classical otherwise, so a machine that never installed torch or never
    downloaded the model still segments. ``kwargs`` are forwarded to the
    chosen backend's constructor; both backends accept
    ``segment(image_bgr)`` and expose ``ms_per_frame``.
    """
    mode = resolve_seg_mode(mode)
    if mode == "neural":
        return Segmenter(**kwargs)
    from kart_map2d.perception.classical_ground import ClassicalGroundSegmenter

    return ClassicalGroundSegmenter(**kwargs)


# ------------------------------------------------------------------ caching
def cache_path(cache_root: Path, run: str, camera: str, frame: CameraFrame) -> Path:
    """Cache file for one frame: ``<root>/<run>/<camera>/<image stem>.npz``."""
    return Path(cache_root) / run / camera / (Path(frame.image_path).stem + ".npz")


def load_or_compute(
    frames: Sequence[CameraFrame],
    run: str,
    camera: str | None = None,
    segmenter: Segmenter | None = None,
    cache_root: Path = Path("outputs/seg"),
    batch_size: int = 4,
    kart_mask: NDArray[np.bool_] | None = None,
) -> list[SegmentationResult]:
    """Return segmentation for ``frames``, computing only the uncached ones.

    ``run`` is the recording's directory name (e.g. ``sept3_20260905_130304``)
    and ``camera`` defaults to the frames' own camera name. A ``segmenter`` is
    only constructed (and the model only loaded) when at least one frame is
    missing from the cache, so replaying a fully cached run needs no GPU.

    ``kart_mask`` (optional, full-resolution bool, True = kart body) replaces
    the default geometric mask - e.g. the per-camera mask learned by
    :func:`kart_map2d.perception.classical_ground.load_or_learn_kart_mask`.
    Cached class maps have their mask baked in, so a custom mask gets its own
    cache namespace (``<camera>__m<digest>``); the legacy per-camera cache is
    left untouched for callers without a mask. Because masking happens AFTER
    the network's argmax, a cached legacy entry can be upgraded exactly by
    stamping ``KART_ID`` over the extra pixels whenever ``kart_mask`` is a
    superset of the default geometric mask - no GPU needed; only genuinely
    novel frames reach the model.
    """
    if camera is None:
        camera = frames[0].camera if frames else ""
    cache_cam = camera
    if kart_mask is not None:
        kart_mask = np.ascontiguousarray(kart_mask, dtype=bool)
        digest = hashlib.sha256(
            np.packbits(kart_mask).tobytes() + repr(kart_mask.shape).encode("ascii")
        ).hexdigest()[:10]
        cache_cam = f"{camera}__m{digest}"
    paths = [cache_path(cache_root, run, cache_cam, f) for f in frames]
    results: list[SegmentationResult | None] = [
        SegmentationResult.load(p) if p.exists() else None for p in paths
    ]
    missing = [i for i, r in enumerate(results) if r is None]
    if missing and kart_mask is not None:
        h, w = kart_mask.shape
        if bool(((kart_mask | default_kart_mask(h, w)) == kart_mask).all()):
            still_missing = []
            for i in missing:
                legacy = cache_path(cache_root, run, camera, frames[i])
                if legacy.exists():
                    class_map = SegmentationResult.load(legacy).class_map.copy()
                    if class_map.shape == kart_mask.shape:
                        class_map[kart_mask] = KART_ID
                        upgraded = SegmentationResult(class_map)
                        upgraded.save(paths[i])
                        results[i] = upgraded
                        continue
                still_missing.append(i)
            missing = still_missing
    if missing:
        if segmenter is None:
            segmenter = Segmenter() if kart_mask is None else Segmenter(kart_mask=kart_mask)
        for start in range(0, len(missing), batch_size):
            block = missing[start : start + batch_size]
            images = []
            for i in block:
                image = cv2.imread(frames[i].image_path)
                if image is None:
                    raise FileNotFoundError(f"cannot read camera frame {frames[i].image_path}")
                images.append(image)
            for i, result in zip(block, segmenter.segment_batch(images)):
                result.save(paths[i])
                results[i] = result
    return [r for r in results if r is not None]

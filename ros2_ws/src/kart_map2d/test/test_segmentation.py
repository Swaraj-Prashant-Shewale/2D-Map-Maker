"""Tests for kart_map2d.perception.segmentation (no network / GPU needed)."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

try:
    import torch
except ImportError:  # torch-free install (e.g. the ROS 2 Humble image): the
    torch = None  # class-id, mask and cache-format tests below still run

from kart_map2d.core.measurements import CameraFrame
from kart_map2d.perception import segmentation as seg
from kart_map2d.perception.segmentation import (
    CITYSCAPES_CLASSES,
    DRIVABLE_IDS,
    KART_ID,
    NEUTRAL_IDS,
    OBSTACLE_IDS,
    SegmentationResult,
    Segmenter,
    class_ids,
    default_kart_mask,
    load_or_compute,
)

# --------------------------------------------------------------- class ids


def test_cityscapes_train_id_order() -> None:
    # The cached npz format depends on these exact train ids.
    assert len(CITYSCAPES_CLASSES) == 19
    assert CITYSCAPES_CLASSES.index("road") == 0
    assert CITYSCAPES_CLASSES.index("sidewalk") == 1
    assert CITYSCAPES_CLASSES.index("vegetation") == 8
    assert CITYSCAPES_CLASSES.index("terrain") == 9
    assert CITYSCAPES_CLASSES.index("sky") == 10
    assert CITYSCAPES_CLASSES.index("person") == 11
    assert CITYSCAPES_CLASSES.index("bicycle") == 18


def test_category_sets_partition_all_classes() -> None:
    assert DRIVABLE_IDS == {0, 1}  # road + sidewalk only
    assert NEUTRAL_IDS == class_ids(("terrain", "sky"))
    assert DRIVABLE_IDS.isdisjoint(OBSTACLE_IDS)
    assert DRIVABLE_IDS.isdisjoint(NEUTRAL_IDS)
    assert OBSTACLE_IDS.isdisjoint(NEUTRAL_IDS)
    assert DRIVABLE_IDS | OBSTACLE_IDS | NEUTRAL_IDS == set(range(19))


def test_grass_is_not_drivable_but_vegetation_is_obstacle() -> None:
    assert class_ids(("terrain",)) <= NEUTRAL_IDS
    assert class_ids(("terrain",)).isdisjoint(DRIVABLE_IDS)
    assert class_ids(("vegetation", "person", "building", "car")) <= OBSTACLE_IDS


def test_kart_id_outside_class_range() -> None:
    assert KART_ID == 255
    assert KART_ID not in DRIVABLE_IDS | OBSTACLE_IDS | NEUTRAL_IDS


def test_class_ids_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        class_ids(("hovercraft",))


# --------------------------------------------------------------- kart mask


def test_default_kart_mask_geometry() -> None:
    h, w = 600, 960
    mask = default_kart_mask(h, w)
    assert mask.shape == (h, w) and mask.dtype == np.bool_
    assert mask[h - 1].all()  # bottom row fully kart
    assert not mask[0].any()  # top row fully scene
    assert mask[h - 60, 5] and mask[h - 60, w - 6]  # corner triangles
    assert not mask[h - 200, w // 2]  # mid image above the bottom strip
    np.testing.assert_array_equal(mask, mask[:, ::-1])  # left/right symmetric


# --------------------------------------------------------- mask derivation


def _synthetic_class_map() -> np.ndarray:
    class_map = np.full((10, 10), CITYSCAPES_CLASSES.index("sky"), dtype=np.uint8)
    class_map[6:, :] = 0  # road
    class_map[:6, :4] = CITYSCAPES_CLASSES.index("building")
    class_map[5, 5:] = CITYSCAPES_CLASSES.index("terrain")
    class_map[9, 9] = KART_ID
    return class_map


def test_masks_derive_from_class_map() -> None:
    result = SegmentationResult(_synthetic_class_map())
    assert result.drivable_mask[7, 3] and not result.drivable_mask[2, 2]
    assert result.obstacle_mask[2, 2] and not result.obstacle_mask[7, 3]
    assert result.neutral_mask[5, 7] and result.neutral_mask[0, 8]  # terrain + sky
    assert result.kart_mask[9, 9]
    # Kart pixel belongs to no category.
    assert not (result.drivable_mask | result.obstacle_mask | result.neutral_mask)[9, 9]
    # Categories partition the scene pixels.
    union = result.drivable_mask | result.obstacle_mask | result.neutral_mask | result.kart_mask
    assert union.all()
    # 39 road pixels of 99 scene pixels (one kart pixel excluded).
    assert result.drivable_fraction == pytest.approx(39 / 99)


def test_result_save_load_roundtrip(tmp_path) -> None:
    result = SegmentationResult(_synthetic_class_map())
    path = tmp_path / "sub" / "000001.npz"
    result.save(path)
    loaded = SegmentationResult.load(path)
    assert loaded.class_map.dtype == np.uint8
    np.testing.assert_array_equal(loaded.class_map, result.class_map)


# ------------------------------------------------------------ mocked model
needs_torch = pytest.mark.skipif(torch is None, reason="the neural backend needs torch")


class _FakeProcessor:
    """Stands in for the HF image processor: fixed-size pixel_values."""

    def __call__(self, images, return_tensors):
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros(len(images), 3, 16, 16)}


if torch is not None:

    class _FakeModel(torch.nn.Module):
        """Left half road, right half building, independent of the input."""

        def forward(self, pixel_values):
            n = pixel_values.shape[0]
            logits = torch.zeros(n, 19, 4, 4)
            logits[:, 0, :, :2] = 10.0  # road
            logits[:, 2, :, 2:] = 10.0  # building
            return SimpleNamespace(logits=logits)


def _fake_segmenter(**kwargs) -> Segmenter:
    return Segmenter(device="cpu", fp16=False, model=_FakeModel(), processor=_FakeProcessor(), **kwargs)


@needs_torch
def test_segment_with_mocked_model() -> None:
    segmenter = _fake_segmenter()
    image = np.zeros((60, 40, 3), dtype=np.uint8)
    result = segmenter.segment(image)
    assert result.class_map.shape == (60, 40)
    assert result.class_map.dtype == np.uint8
    assert result.class_map[5, 2] == 0  # road on the left
    assert result.class_map[5, 37] == 2  # building on the right
    assert result.drivable_mask[5, 2] and result.obstacle_mask[5, 37]
    assert result.class_map[59, 20] == KART_ID  # bottom strip is kart body
    assert segmenter.frames_processed == 1
    assert segmenter.ms_per_frame > 0.0


@needs_torch
def test_segment_batch_matches_single() -> None:
    segmenter = _fake_segmenter()
    images = [np.zeros((30, 20, 3), dtype=np.uint8), np.zeros((60, 40, 3), dtype=np.uint8)]
    results = segmenter.segment_batch(images)
    assert [r.class_map.shape for r in results] == [(30, 20), (60, 40)]
    single = _fake_segmenter().segment(images[1])
    np.testing.assert_array_equal(results[1].class_map, single.class_map)
    assert segmenter.segment_batch([]) == []


@needs_torch
def test_kart_mask_can_be_disabled_or_overridden() -> None:
    image = np.zeros((60, 40, 3), dtype=np.uint8)
    plain = _fake_segmenter(apply_kart_mask=False).segment(image)
    assert not plain.kart_mask.any()
    custom = np.zeros((60, 40), dtype=bool)
    custom[0, 0] = True
    overridden = _fake_segmenter(kart_mask=custom).segment(image)
    assert overridden.class_map[0, 0] == KART_ID
    assert not overridden.kart_mask[59, 20]  # default bottom strip not applied


def _frames_on_disk(tmp_path, n=3, shape=(30, 20)) -> list:
    frames = []
    for i in range(n):
        path = tmp_path / f"{i:06d}.jpg"
        cv2.imwrite(str(path), np.zeros((*shape, 3), dtype=np.uint8))
        frames.append(CameraFrame(float(i), str(path), "zed_right"))
    return frames


def test_load_or_compute_kart_mask_seeds_from_legacy_cache(tmp_path) -> None:
    """A custom kart mask that is a superset of the geometric default is
    served by STAMPING the legacy cached class maps - no model, no torch.
    (Masking happens after the network argmax, so the stamp is exact.)"""
    frames = _frames_on_disk(tmp_path)
    root = tmp_path / "seg"
    legacy_map = _synthetic_class_map_hw(30, 20)
    legacy_map[default_kart_mask(30, 20)] = KART_ID  # what a legacy compute wrote
    for f in frames:
        SegmentationResult(legacy_map).save(seg.cache_path(root, "runB", "zed_right", f))

    learned = default_kart_mask(30, 20).copy()
    learned[10:14, 5:9] = True  # superset: geometric + a learned protrusion
    results = load_or_compute(frames, "runB", cache_root=root, kart_mask=learned)
    assert len(results) == 3
    for r in results:
        assert (r.class_map[learned] == KART_ID).all()
        # outside the mask the legacy classes are untouched
        assert (r.class_map[~learned] == legacy_map[~learned]).all()
    # the upgraded results live in a mask-namespaced cache, legacy untouched
    dirs = sorted(p.name for p in (root / "runB").iterdir())
    assert "zed_right" in dirs and any(d.startswith("zed_right__m") for d in dirs)
    np.testing.assert_array_equal(
        SegmentationResult.load(seg.cache_path(root, "runB", "zed_right", frames[0])).class_map,
        legacy_map,
    )
    # second call is served from the namespaced cache (still no model)
    again = load_or_compute(frames, "runB", cache_root=root, kart_mask=learned)
    for a, b in zip(results, again):
        np.testing.assert_array_equal(a.class_map, b.class_map)


def _synthetic_class_map_hw(h: int, w: int) -> np.ndarray:
    class_map = np.full((h, w), CITYSCAPES_CLASSES.index("sky"), dtype=np.uint8)
    class_map[h // 2 :, :] = 0  # road
    return class_map


@needs_torch
def test_load_or_compute_non_superset_mask_recomputes(tmp_path) -> None:
    """A mask that does NOT contain the geometric default cannot be stamped
    over legacy caches (their baked mask cannot be undone): the model runs."""
    frames = _frames_on_disk(tmp_path)
    root = tmp_path / "seg"
    legacy_map = _synthetic_class_map_hw(30, 20)
    legacy_map[default_kart_mask(30, 20)] = KART_ID
    for f in frames:
        SegmentationResult(legacy_map).save(seg.cache_path(root, "runC", "zed_right", f))
    small = np.zeros((30, 20), dtype=bool)
    small[0, 0] = True  # not a superset of the default mask
    segmenter = _fake_segmenter(kart_mask=small)
    results = load_or_compute(frames, "runC", segmenter=segmenter, cache_root=root, kart_mask=small)
    assert segmenter.frames_processed == 3  # model had to run
    for r in results:
        assert r.class_map[0, 0] == KART_ID
        assert r.class_map[29, 10] != KART_ID  # default strip NOT applied


@needs_torch
def test_load_or_compute_uses_cache(tmp_path) -> None:
    frames = []
    for i in range(3):
        path = tmp_path / f"{i:06d}.jpg"
        cv2.imwrite(str(path), np.zeros((30, 20, 3), dtype=np.uint8))
        frames.append(CameraFrame(float(i), str(path), "zed_right"))

    first = _fake_segmenter()
    results = load_or_compute(frames, "runA", segmenter=first, cache_root=tmp_path / "seg", batch_size=2)
    assert len(results) == 3 and first.frames_processed == 3
    for frame in frames:
        assert seg.cache_path(tmp_path / "seg", "runA", "zed_right", frame).exists()

    # Second pass must be served from the cache: the model is never called.
    second = _fake_segmenter()
    cached = load_or_compute(frames, "runA", segmenter=second, cache_root=tmp_path / "seg")
    assert second.frames_processed == 0
    for a, b in zip(results, cached):
        np.testing.assert_array_equal(a.class_map, b.class_map)

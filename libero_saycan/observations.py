"""Observation utilities shared by smoke tests and later policy adapters."""

import numpy as np


def camera_frame(observation, key="agentview_image", flip_vertical=True):
    """Return a contiguous uint8 RGB image from a LIBERO camera observation."""
    if key not in observation:
        available = sorted(observation.keys())
        raise KeyError(
            "Camera key {!r} is absent. Available observation keys: {}".format(
                key, available
            )
        )

    frame = np.asarray(observation[key])
    if frame.ndim != 3:
        raise ValueError(
            "Expected a 3D camera image for {!r}, got shape {}".format(
                key, frame.shape
            )
        )

    if frame.shape[-1] not in (3, 4) and frame.shape[0] in (3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.shape[-1] not in (3, 4):
        raise ValueError(
            "Expected HWC RGB/RGBA or CHW RGB/RGBA image, got {}".format(
                frame.shape
            )
        )

    frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if flip_vertical:
        frame = np.flip(frame, axis=0)
    return np.ascontiguousarray(frame)


def observation_shapes(observation):
    """Return JSON-safe shape/dtype metadata without serializing image pixels."""
    summary = {}
    for key, value in observation.items():
        array = np.asarray(value)
        summary[key] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    return summary

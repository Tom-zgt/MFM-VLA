"""Import this before anything from `libero` to pin down headless GL and loader quirks.

Two things bite on a modern stack:
  - MuJoCo picks a GL backend at import time, so the env vars must be set first.
  - LIBERO stores task init states as pickled numpy via `torch.save`, which
    torch >= 2.6 refuses to load under its default `weights_only=True`.

If `../nvgl` exists (NVIDIA userspace EGL matching the kernel driver), prefer
that over Mesa llvmpipe so every wrist look is cheap enough for a real loop.
"""

import os
from pathlib import Path

_NVGL = Path(__file__).resolve().parents[1] / "nvgl"
if _NVGL.is_dir() and (_NVGL / "libEGL_nvidia.so.0").exists():
    os.environ["LD_LIBRARY_PATH"] = (
        str(_NVGL) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    )
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(
        _NVGL / "egl_vendor.d" / "10_nvidia.json"
    )
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch

if not getattr(torch.load, "_libero_patched", False):
    _orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    _load._libero_patched = True
    torch.load = _load

# robosuite decodes segmentation IDs as uint8 * 2**8, which NumPy >= 2 rejects
# instead of silently wrapping. Decode in int32 with identical semantics.
import mujoco
import numpy as np
from robosuite.utils import binding_utils as _binding_utils

if not getattr(_binding_utils.MjRenderContext.read_pixels, "_libero_patched", False):

    def _read_pixels(self, width, height, depth=False, segmentation=False):
        viewport = mujoco.MjrRect(0, 0, width, height)
        rgb_img = np.empty((height, width, 3), dtype=np.uint8)
        depth_img = np.empty((height, width), dtype=np.float32) if depth else None
        mujoco.mjr_readPixels(rgb=rgb_img, depth=depth_img, viewport=viewport, con=self.con)
        ret_img = rgb_img
        if segmentation:
            channels = rgb_img.astype(np.int32)
            seg_img = channels[:, :, 0] + channels[:, :, 1] * (2**8) + channels[:, :, 2] * (2**16)
            seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
            seg_ids = np.full((self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32)
            for i in range(self.scn.ngeom):
                geom = self.scn.geoms[i]
                if geom.segid != -1:
                    seg_ids[geom.segid + 1, 0] = geom.objtype
                    seg_ids[geom.segid + 1, 1] = geom.objid
            ret_img = seg_ids[seg_img]
        if depth:
            return (ret_img, depth_img)
        return ret_img

    _read_pixels._libero_patched = True
    _binding_utils.MjRenderContext.read_pixels = _read_pixels

"""Conservative collision envelopes from the official Tiangong Pro USD.

The companion archive contains the original hull boundary vertices and the
revolute joint origins from tiangong_pro_v2.usd. Some lower-body collision
meshes are disabled in the simulator. They still bound the visible legs.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


HOME_ANGLES = {
    f"{joint}_{side}_joint": angle
    for side in ("l", "r")
    for joint, angle in (
        ("hip_roll", 0.0), ("hip_pitch", -0.5), ("hip_yaw", 0.0),
        ("knee_pitch", 1.0), ("ankle_pitch", -0.5), ("ankle_roll", 0.0),
    )
}


@lru_cache(maxsize=1)
def _model():
    with np.load(Path(__file__).with_name("lower_body.npz")) as data:
        joints = tuple(zip(data["joint_names"], data["parents"], data["children"],
                           data["offsets"], data["axes"]))
        shapes = {}
        for link in data["link_names"]:
            vertices = data[f"vertices_{link}"]
            planes = ConvexHull(vertices).equations
            # A subset of the exact planes cheaply rejects most nearby points.
            directions = np.rint(planes[:, :3] * 3).astype(np.int8)
            _, first = np.unique(directions, axis=0, return_index=True)
            shapes[link] = (vertices.min(0), vertices.max(0), planes[first], planes)
    return joints, shapes


class LowerBody:
    """Pose the pelvis and both legs at one lower-body encoder snapshot."""

    def __init__(self, angles):
        missing = HOME_ANGLES.keys() - angles.keys()
        if missing:
            raise ValueError(f"missing lower-body encoders: {', '.join(sorted(missing))}")
        if not all(np.isfinite(angles[name]) for name in HOME_ANGLES):
            raise ValueError("lower-body encoders must be finite")

        joints, shapes = _model()
        frames = {"pelvis": (np.eye(3), np.zeros(3))}
        for joint, parent, child, offset, axis in joints:
            parent_rotation, parent_position = frames[parent]
            rotation = parent_rotation @ Rotation.from_rotvec(axis * angles[joint]).as_matrix()
            frames[child] = rotation, parent_position + parent_rotation @ offset

        self._hulls = []
        for link, (low, high, coarse, planes) in shapes.items():
            rotation, position = frames[link]
            center = position + rotation @ ((high + low) * .5)
            radius = np.abs(rotation) @ ((high - low) * .5)
            self._hulls.append((center - radius, center + radius,
                                rotation, position, low, high, coarse, planes))

    def collides(self, vertices, yaw, clearance=.008):
        """Test waist-frame points against the hulls using absolute body yaw in radians."""
        points = np.asarray(vertices, dtype=float)
        rotation = Rotation.from_rotvec([0., 0., yaw]).as_matrix()
        pelvis = points @ rotation.T
        for outer_low, outer_high, rotation, position, low, high, coarse, planes in self._hulls:
            nearby = pelvis[np.all((pelvis >= outer_low - clearance)
                                   & (pelvis <= outer_high + clearance), axis=1)]
            if not len(nearby):
                continue
            local = (nearby - position) @ rotation
            local = local[np.all((local >= low - clearance) & (local <= high + clearance), axis=1)]
            if not len(local):
                continue
            inside = np.all(local @ coarse[:, :3].T + coarse[:, 3] <= clearance, axis=1)
            candidates = local[inside]
            if len(candidates) and np.any(np.all(candidates @ planes[:, :3].T
                                                + planes[:, 3] <= clearance, axis=1)):
                return True
        return False

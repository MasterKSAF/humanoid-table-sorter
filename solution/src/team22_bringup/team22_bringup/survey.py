"""Register RGB-D views in waist_yaw_link and inventory the visible work surface."""
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .grasp import origin, point
from .vision import HUES, MOUSE, find_tray


HEAD = ["head_yaw_joint", "head_pitch_joint", "head_roll_joint"]
ANGLES = {"left": .48, "centre": 0., "right": -.48}
GRID = .0125
XY0 = np.array([.12, -1.50])
SHAPE = (126, 242)


def cloud(depth, k, transform):
    v, u = np.mgrid[:depth.shape[0]:3, :depth.shape[1]:3]
    d = depth[::3, ::3].astype(float)/1000
    valid = (d >= .15) & (d <= 2.5)
    rays = np.stack(((u-k[2])*d/k[0], (v-k[5])*d/k[4], d), axis=-1)
    return rays[valid] @ transform[:3, :3].T + transform[:3, 3]


def plane_from_torso(p):
    data = p[(p[:, 0] > .22) & (p[:, 0] < 1.) & (abs(p[:, 1]) < .75)
             & (p[:, 2] > -.28) & (p[:, 2] < -.06)][::3]
    if len(data) < 100:
        raise ValueError("no table surface")
    xy = np.column_stack((data[:, :2], np.ones(len(data))))
    best = np.zeros(len(data), dtype=bool)
    rng = np.random.default_rng(22)
    for idx in (rng.choice(len(data), 3, replace=False) for _ in range(80)):
        if abs(np.linalg.det(xy[idx])) < .002:
            continue
        coeff = np.linalg.solve(xy[idx], data[idx, 2])
        if np.linalg.norm(coeff[:2]) > .25:
            continue
        inside = abs(data[:, 2]-xy @ coeff) < .004
        if inside.sum() > best.sum():
            best = inside
    if best.sum() < max(90, len(data)*.36) or min(np.ptp(data[best, :2], axis=0)) < .18:
        raise ValueError("ambiguous table plane")
    return np.linalg.lstsq(xy[best], data[best, 2], rcond=None)[0]


def cells(xy):
    return np.floor((xy-XY0)/GRID).astype(int)


def identify_items(items, task):
    """Pool 2 appearance hypotheses. Keep ambiguous matches anonymous."""
    guesses = {}
    for item in items:
        if item["names"]:
            continue
        a = item["appearance"]
        x, y, z = item["extent"]
        prefix = None
        if z < .065 and max(x, y) > .10 and a["dark"] > .15:
            prefix = "Ubisoft_"
        elif z > .13 and max(x, y) > .09 and a["pale"] > .65:
            prefix = "Deskstar_"
        elif max(x, y) < .085:
            if a["blue"] > .18 and z < .13:
                prefix = "Beta_Glucan"
            elif a["yellow"] > .14 and .08 < z < .145:
                prefix = "Wilton_"
            elif a["yellow"] > .14 and z > .16:
                prefix = "Marc_Anthony_"
        matches = [name for name in task["objects"] if prefix and name.startswith(prefix)]
        if len(matches) == 1:
            guesses.setdefault(matches[0], []).append(item)
    for name, group in guesses.items():
        if len(group) == 1:
            group[0]["hypothesis"] = name


class Survey:
    def __init__(self):
        model = ET.parse(Path(__file__).with_name("upper_body.urdf"))
        self.neck = [model.find(f"./joint[@name='{name}']") for name in HEAD]
        self.eye = origin(model.find("./joint[@name='camera_head_joint']"))
        self.body = origin(model.find("./joint[@name='camera_body_front_joint']"))
        self.views = {}
        self.colors = {}
        self.tray_shapes = {}

    def head_pose(self, angles):
        pose = np.eye(4)
        for j, value in zip(self.neck, angles):
            pose = pose @ origin(j)
            axis = np.fromstring(j.find("axis").get("xyz"), sep=" ")
            pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(value*axis).as_matrix()
        return pose @ self.eye

    def add(self, camera, bgr, depth, k, result, task, head=None):
        sector, pose = "torso", self.body
        if camera == "head":
            if head is None or abs(head[1]-.42) > .075:
                return
            sector = min(ANGLES, key=lambda key: abs(ANGLES[key]-head[0]))
            if abs(ANGLES[sector]-head[0]) > .07:
                return
            pose = self.head_pose(head)
        objects = {}
        if result.get("pick"):
            objects[MOUSE] = point(pose, result["pick"]["xyz_m"])
        for tray, color in task.get("trays", {}).items():
            location = find_tray(bgr, depth, color, k) if color in HUES else None
            if location:
                objects[tray] = point(pose, location["xyz_m"])
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, (HUES[color][0], 5, 135), (HUES[color][1], 145, 255))
                _, blobs = cv2.connectedComponents(mask)
                u, v = location["pixel"]
                if blobs[v, u]:
                    patch = cloud(np.where(blobs == blobs[v, u], depth, 0), k, pose)
                    patch = patch[(np.linalg.norm(patch-objects[tray], axis=1) < .31)
                                  & (abs(patch[:, 2]-objects[tray][2]) < .010)]
                    if len(patch) > 40:
                        rect = cv2.minAreaRect(patch[:, :2].astype(np.float32))
                        if .15 < min(rect[1]) and max(rect[1]) < .45:
                            hull = cv2.boxPoints((rect[0], tuple(x+.024 for x in rect[1]), rect[2]))
                            self.tray_shapes[sector, tray] = hull
        self.views[sector] = cloud(depth, k, pose), objects
        d = depth[::3, ::3]
        self.colors[sector] = cv2.cvtColor(bgr[::3, ::3], cv2.COLOR_BGR2HSV)[(d >= 150) & (d <= 2500)]

    def inventory(self, task):
        plane = plane_from_torso(self.views["torso"][0])
        parts, colors, objects = [], [], {}
        # Prefer near-camera estimates for the mouse when available.
        for key in [*ANGLES, "torso"]:
            if key in self.views:
                xyz, marks = self.views[key]
                parts.append(xyz)
                colors.append(self.colors[key])
                objects.update(marks)
        p = np.concatenate(parts)
        ij = cells(p[:, :2])
        h = p[:, 2]-p[:, :2] @ plane[:2]-plane[2]
        valid = ((ij >= 0).all(1) & (ij < SHAPE).all(1) & (h >= -.009) & (h < .32))
        p, ij, h = p[valid], ij[valid], h[valid]
        color = np.concatenate(colors)[valid]
        seen = np.zeros(SHAPE, np.uint8)
        seen[tuple(ij.T)] = 1
        surface = p[abs(h) < .007]
        edge = float(np.percentile(surface[:, 0], 2)-.028)
        high = (h > .013) & (p[:, 0] > edge)
        occupancy = np.zeros(SHAPE, np.uint8)
        tray_grid = np.zeros_like(occupancy)
        trays = sorted(task.get("trays", {}))
        for idx, name in enumerate(trays, 1):
            for (_, tray), hull in self.tray_shapes.items():
                if tray == name:
                    cv2.fillConvexPoly(tray_grid, cells(hull)[:, ::-1].astype(np.int32), idx)
        tray_id = tray_grid[tuple(ij.T)]
        item = high & ((tray_id == 0) | (h > .085))
        occupancy[tuple(ij[item].T)] = 1
        # Fill outlined rims before labeling so each tray is counted once.
        closed = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), anchor=(0, 0))
        closed |= occupancy
        outlines, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(closed, outlines, -1, 1, cv2.FILLED)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
        components, tags = [], {}
        for name, xyz in objects.items():
            if name in trays:
                continue
            i, j = cells(xyz[:2])
            if 0 <= i < SHAPE[0] and 0 <= j < SHAPE[1]:
                label = labels[i, j]
                if label:
                    tags.setdefault(label, []).append(name)
        for label in range(1, n):
            if stats[label, cv2.CC_STAT_AREA] < 8:
                continue
            select = item & (labels[tuple(ij.T)] == label)
            group = p[select]
            if len(group):
                hue, sat, val = color[select].T
                components.append(dict(names=tags.get(label, []), center=np.median(group, axis=0).tolist(),
                                       extent=np.ptp(group, axis=0).tolist(), label=label,
                                       appearance=dict(dark=float(np.mean(val < 85)),
                                                       pale=float(np.mean((sat < 45) & (val > 155))),
                                                       blue=float(np.mean((hue > 90) & (hue < 135) & (sat > 45))),
                                                       yellow=float(np.mean((hue > 12) & (hue < 40) & (sat > 50))))))
        items = list(components)
        identify_items(items, task)
        for idx, name in enumerate(trays, 1):
            group = p[(tray_id == idx) & (abs(h) < .085)]
            if name in objects and len(group):
                components.append(dict(names=[name], center=objects[name].tolist(), extent=np.ptp(group, axis=0).tolist(),
                                       label=n+idx))
        trays = set(trays)
        found = {name for c in components for name in c["names"]}
        # Coverage of the work area enclosing the trays and all located objects.
        coverage = 0.
        if components:
            locs = np.array([cells(c["center"][:2]) for c in components])
            lo, hi = locs.min(0), locs.max(0)+1
            coverage = float(np.mean(seen[lo[0]:hi[0], lo[1]:hi[1]]))
        complete = (set(ANGLES).issubset(self.views) and trays and trays.issubset(found)
                    and MOUSE in found and len(items) == len(task["objects"]) and coverage >= .90
                    and not any(len(c["names"]) > 1 for c in components))
        return dict(views=sorted(self.views), components=components, coverage=coverage, complete=bool(complete),
                    plane=plane, edge=edge, seen=seen, labels=labels, tray_grid=tray_grid,
                    mouse_point=objects.get(MOUSE), high=p[high],
                    high_labels=np.where(item[high], labels[tuple(ij[high].T)], -1))


def obstacles(scene, exclude_mouse=False):
    points = scene["high"]
    if exclude_mouse:
        label = next(c["label"] for c in scene["components"] if MOUSE in c["names"])
        # Restrict permitted contact to the target component near the observed body.
        # A cable connected to that component can extend across the rest of the table.
        near_body = np.linalg.norm(points[:, :2]-scene["mouse_point"][:2], axis=1) <= .075
        points = points[(scene["high_labels"] != label) | ~near_body]
    return scene["plane"], scene["edge"], cKDTree(points)

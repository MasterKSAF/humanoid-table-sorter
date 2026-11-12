"""One-object baseline for sort2. Input is aligned BGR + optical depth in mm."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


MOUSE = "Razer_Taipan_Black_Ambidextrous_Gaming_Mouse"
HUES = {"red": (0, 13), "blue": (90, 115), "green": (50, 82)}


@lru_cache(maxsize=2)
def mouse_views(filename):
    ref = cv2.imread(str(Path(__file__).with_name(filename)), cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise RuntimeError(f"Missing {filename}")
    # Rotate around the body. The cable is deliberately outside the reference.
    h, w = ref.shape
    views = []
    for scale in (0.55, 0.8, 1.0, 1.3):
        for angle in range(0, 360, 30):
            rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale / 2)
            width, height = np.ceil(np.abs(rot[:, :2]) @ [w, h]).astype(int)
            rot[:, 2] += [(width-w)/2, (height-h)/2]
            small = cv2.warpAffine(ref, rot, (width, height), borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=float(np.median(ref[0])))
            views.append(small)
    return views


def surface_point(depth, x, y, k):
    patch = depth[max(0, y-2):y+3, max(0, x-2):x+3]
    valid = patch[(patch >= 150) & (patch <= 1500)]
    if valid.size < patch.size * 0.8 or np.ptp(valid) > 25:
        return None
    mm = int(np.median(valid))
    point = {"pixel": [x, y], "depth_mm": mm}
    if k is not None:
        z = mm / 1000
        point["xyz_m"] = [round((x-k[2])*z/k[0], 4), round((y-k[5])*z/k[4], 4), z]
    return point


def find_mouse(bgr, depth, k):
    gray = cv2.cvtColor(cv2.resize(bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
    best = None
    best_score = 0.79
    for template in mouse_views("mouse_reference.png") + mouse_views("mouse_top.png"):
        h, w = template.shape
        if h > gray.shape[0] or w > gray.shape[1]:
            continue
        scores = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (u, v) = cv2.minMaxLoc(scores)
        if score < best_score:
            continue
        point = surface_point(depth, 2*u+w, 2*v+h, k)
        if point:
            best = dict(point, match=round(score, 3))
            best_score = score
    return best


def find_tray(bgr, depth, color, k):
    # The simulated trays have a pale floor and a more saturated rim of the same hue.
    low, high = HUES[color]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    floor = cv2.inRange(hsv, (low, 6, 225), (high, 32, 255))
    rim = cv2.inRange(hsv, (low, 33, 145), (high, 130, 255))
    floor[(depth < 150) | (depth > 1500)] = 0
    floor[[0, -1], :] = 0
    floor[:, [0, -1]] = 0
    contours, _ = cv2.findContours(floor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < 25 or cv2.contourArea(contour) < 0.45*w*h:
            continue
        roi = np.zeros((h+2, w+2), np.uint8)
        cv2.drawContours(roi, [contour - [x-1, y-1]], -1, 255, cv2.FILLED)
        roi[1:-1,1:-1] &= floor[y:y+h,x:x+w]
        distance = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
        _, clearance, _, (dx, dy) = cv2.minMaxLoc(distance)
        u, v = x+dx-1, y+dy-1
        # Trace four bands out from the clear patch. Two rim crossings are needed,
        # even when part of the tray falls outside the camera's field of view.
        band = max(3, int(clearance/3))
        margin = max(8, min(w, h)//3)
        strips = [rim[max(0,y-margin):v,max(0,u-band):u+band+1],
                  rim[v:y+h+margin,max(0,u-band):u+band+1],
                  rim[max(0,v-band):v+band+1,max(0,x-margin):u].T,
                  rim[max(0,v-band):v+band+1,u:x+w+margin].T]
        if sum(s.size > 0 and np.count_nonzero(np.mean(s != 0, axis=1) > 0.5) >= 2 for s in strips) < 2:
            continue
        point = surface_point(depth, u, v, k)
        if point and clearance >= 12 and (k is None or clearance*point["depth_mm"]/max(k[0], k[4]) >= 25):
            candidates.append((clearance, point))
    return max(candidates, key=lambda c: c[0])[1] if candidates else None


def observe(bgr, depth, task, k=None):
    if depth.dtype != np.uint16 or depth.shape != bgr.shape[:2]:
        raise ValueError("Expected aligned 16UC1 millimetres")
    result = {"object": MOUSE, "tray": None, "color": None, "pick": None, "place": None}
    if task.get("task") != "sort2" or task.get("objects_from") != "pool 2":
        return result
    category = task.get("objects", {}).get(MOUSE)
    tray = task.get("categories", {}).get(category)
    color = task.get("trays", {}).get(tray)
    if not tray or color not in HUES:
        return result
    result.update(tray=tray, color=color, pick=find_mouse(bgr, depth, k),
                  place=find_tray(bgr, depth, color, k))
    return result


def annotate(bgr, result):
    marked = bgr.copy()
    for key, number, color in (("pick", "1", (30, 235, 100)), ("place", "2", (0, 190, 255))):
        point = result[key]
        if point:
            x, y = point["pixel"]
            cv2.circle(marked, (x, y), 10, (22, 22, 22), 4, cv2.LINE_AA)
            cv2.circle(marked, (x, y), 10, color, 2, cv2.LINE_AA)
            cv2.drawMarker(marked, (x, y), color, cv2.MARKER_CROSS, 9, 1)
            label = f"{number} | {point['depth_mm']} mm"
            origin = (min(max(0, x-10), max(0, bgr.shape[1]-125)), y-15 if y>32 else y+26)
            for ink, thickness in (((22, 22, 22), 4), (color, 1)):
                cv2.putText(marked, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, ink, thickness, cv2.LINE_AA)
    return marked

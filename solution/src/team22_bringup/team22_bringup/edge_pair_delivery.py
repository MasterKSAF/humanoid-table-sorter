"""Carry an object held between two hands from below the table edge."""

from itertools import product

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation, Slerp


SIDES = ('left', 'right')
VERTICAL = np.array([0., 0., 1.])


def obstacle_geometry(obj):
    low, high = np.asarray(obj['bounds'])
    corners = np.asarray((obj.get('geometry') or {}).get('corners_xy_m', ()))
    if corners.ndim == 2 and corners.shape[1] == 2 and len(corners) >= 3:
        try:
            hull = ConvexHull(corners)
            return low, high, corners[hull.vertices], hull.equations
        except QhullError:
            pass
    corners = np.array([[low[0], low[1]], [low[0], high[1]], [high[0], high[1]], [high[0], low[1]]])
    planes = np.array([[-1., 0., low[0]], [1., 0., -high[0]], [0., -1., low[1]], [0., 1., -high[1]]])
    return low, high, corners, planes


def edge_pair_delivery(routes, item, held, yaw, points, center, solve, clear, *,
                       start_ready=None, edge=None, withdraw=.045, obstacles=None,
                       lift_profile=None, placements=None, route_resolution=(.035, .18)):
    """Return a rigid paired lift, body turn and release, or None if blocked.

    All points are in the frozen survey frame. ``solve`` takes the side, hand
    frame origin and rotation, waist yaw and previous joint values. ``clear``
    tests both arms, including their current fingers, throughout a motion.
    ``obstacles`` can replace frozen objects with their post-transfer positions.
    A nonzero withdrawal assumes the object is supported at the front edge.
    ``start_ready`` can specify a staged return after the opened retreat.
    ``lift_profile`` specifies displacement and rotation pairs from the extracted
    pose. ``placements`` specifies waist yaw, release center, object rotation,
    approach height and an optional release IK seed for checked tray placements.
    """
    points = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or center.shape != (3,):
        return None
    if not np.isfinite(points).all() or not np.isfinite(center).all():
        return None
    edge = routes.edge + .015 if edge is None else edge
    normal_length = np.linalg.norm(routes.normal)
    normal = routes.normal / normal_length
    floor = routes.floor / normal_length
    tray_low, tray_high = np.asarray(item['tray']['bounds'])
    relative_cloud = points - center
    objects = tuple(routes.objects if obstacles is None else obstacles)
    solids = [obstacle_geometry(obj) for obj in objects if obj['kind'] != item['kind']
              and not obj['kind'].startswith('tray_')]
    occupied = [(polygon, planes) for lo, hi, polygon, planes in solids
                if np.all(lo[:2] <= tray_high[:2]) and np.all(hi[:2] >= tray_low[:2])]
    rims = []
    for obj in objects:
        if not obj['kind'].startswith('tray_'):
            continue
        lo, hi = np.asarray(obj['bounds'])
        for axis in (0, 1):
            near, far = lo.copy(), hi.copy()
            near[axis], far[axis] = hi[axis] - .012, lo[axis] + .012
            rims.extend(((lo, far), (near, hi)))
    held = {side: np.asarray(held[side], dtype=float) for side in SIDES}
    ready_path = []
    if start_ready is not None:
        if not isinstance(start_ready, (tuple, list)):
            start_ready = [start_ready]
        ready_path = [{side: np.asarray(q[side], dtype=float) for side in SIDES} for q in start_ready]

    def hand_frame(side, pair, angle):
        arm = routes.arms[side]
        return routes.world((arm.wrist(pair[side]) @ arm.hand)[:3], angle)

    frames = {side: hand_frame(side, held, yaw) for side in SIDES}
    wrist_offsets = np.stack([routes.world(routes.arms[side].wrist(held[side])[:3, 3], yaw) - center
                              for side in SIDES])
    shoulders = np.stack([routes.arms[side].origins[0][:3, 3] for side in SIDES])
    reaches = np.array([sum(np.linalg.norm(p[:3, 3]) for p in routes.arms[side].origins[1:])
                        for side in SIDES])

    def pose(at, turn, angle, seed):
        pair = {}
        for side, frame in frames.items():
            q = solve(side, at + turn @ (frame[:, 3] - center), turn @ frame[:, :3], angle, seed[side])
            if q is None:
                return None
            pair[side] = np.asarray(q, dtype=float)
        return pair

    def cloud_clear(cloud, underneath):
        over = cloud[:, 0] >= edge
        height = cloud[over] @ normal - floor
        if underneath:
            # Rounded leading edge fitted from RGB-D, radius 21 mm.
            radius = .021
            side = np.maximum(0., edge + radius - cloud[over, 0])
            half = np.sqrt(np.maximum(0., radius**2 - side**2))
            if np.any((height > -radius - half + .003) & (height < -radius + half - .003)):
                return False
        elif np.any(height < .005):
            return False
        if any(np.any(np.all((cloud >= lo - .004) & (cloud <= hi + .004), axis=1)) for lo, hi in rims):
            return False
        for lo, hi, _, planes in solids:
            nearby = cloud[np.all((cloud >= lo - .004) & (cloud <= hi + .004), axis=1)]
            if len(nearby) and np.any(np.max(nearby[:, :2] @ planes[:, :2].T + planes[:, 2], axis=1) <= .004):
                return False
        return True

    def payload_poses(pair, angle):
        offsets, turns = [], []
        for side, initial in frames.items():
            current = hand_frame(side, pair, angle)
            turn = current[:, :3] @ initial[:, :3].T
            offsets.append(current[:, 3] + turn @ (center - initial[:, 3]))
            turns.append(turn)
        return offsets, turns

    def carries(pair, angle, underneath):
        offsets, turns = payload_poses(pair, angle)
        if (np.linalg.norm(offsets[0] - offsets[1]) > .007
                or Rotation.from_matrix(turns[0] @ turns[1].T).magnitude() > .06):
            return False
        return all(cloud_clear(relative_cloud @ turn.T + at, underneath) for turn, at in zip(turns, offsets))

    def segment(a, b, a_yaw, b_yaw, *, underneath=False, opened=False):
        delta = {side: b[side] - a[side] for side in SIDES}
        step = max(abs(b_yaw - a_yaw) / .055, max(np.max(np.abs(q)) / .045 for q in delta.values()))
        for fraction in np.linspace(0., 1., 1 + max(1, int(np.ceil(step)))):
            pair = {side: a[side] + fraction * delta[side] for side in SIDES}
            angle = a_yaw + fraction * (b_yaw - a_yaw)
            if not clear(pair, angle, opened=opened, underneath=underneath):
                return False
            if not opened and not carries(pair, angle, underneath):
                return False
        return True

    def route(start, a, b, a_rot, b_rot, angle, *, underneath=False):
        delta = Rotation.from_matrix(b_rot @ a_rot.T).magnitude()
        linear_step, angular_step = route_resolution
        count = max(1, int(np.ceil(max(np.linalg.norm(b - a) / linear_step, delta / angular_step))))
        turns = Slerp([0., 1.], Rotation.from_matrix([a_rot, b_rot]))
        waypoints = []
        for fraction in np.linspace(0., 1., count + 1)[1:]:
            q = pose(a + fraction * (b - a), turns(fraction).as_matrix(), angle, start)
            if q is None or not segment(start, q, angle, angle, underneath=underneath):
                return None
            waypoints.append(q)
            start = q
        return waypoints

    def placement_targets():
        site = np.asarray(item['release'])
        heading = routes.yaw + np.arctan2(site[1], site[0])
        targets = []
        directions = (0., -.20, .20, -.40, .40, -.62, .62)
        offsets = (0., -.24, .24, -.48, .48, -.72, .72, -1.08, 1.08)
        tilts = ((0., 0.), (.32, 0.), (-.32, 0.), (0., .32), (0., -.32),
                 (.72, 0.), (-.72, 0.), *product((-.95, -.72, .72, .95), (-.48, -.24, .24, .48)))
        for direction, offset in product(directions, offsets):
            angle = heading + direction
            if not -2.94 <= angle <= 3.11:
                continue
            for tilt in tilts:
                turn = Rotation.from_euler('zyx', [angle - yaw + offset, *tilt]).as_matrix()
                footprint = relative_cloud @ turn.T
                low = tray_low[:2] + .007 - footprint[:, :2].min(0)
                high = tray_high[:2] - .007 - footprint[:, :2].max(0)
                if np.any(low > high):
                    continue
                xy = np.stack([np.clip(site[:2], low, high), low, high, (low + high) / 2,
                               [low[0], high[1]], [high[0], low[1]]])
                free = np.ones(len(xy), dtype=bool)
                if occupied:
                    try:
                        footprint_axes = ConvexHull(footprint[:, :2]).equations[:, :2]
                    except QhullError:
                        continue
                    for polygon, planes in occupied:
                        axes = np.vstack((footprint_axes, planes[:, :2]))
                        held_projection = footprint[:, :2] @ axes.T
                        obstacle_projection = polygon @ axes.T
                        delta = xy @ axes.T
                        gap = np.maximum(obstacle_projection.min(0) - held_projection.max(0) - delta,
                                         held_projection.min(0) + delta - obstacle_projection.max(0))
                        free &= np.max(gap, axis=1) > .007
                for point, gap in product(xy[free], (.012, .040, .075)):
                    at = np.r_[point, tray_high[2] - footprint[:, 2].min() + gap]
                    wrists = at + wrist_offsets @ turn.T
                    reach = np.linalg.norm(np.stack([routes.local(w, angle) for w in wrists]) - shoulders, axis=1)
                    spare = np.min(reaches - reach)
                    if spare < .002:
                        continue
                    cost = gap + .40 * np.linalg.norm(point - site[:2]) + .030 * abs(offset) + .02 * sum(map(abs, tilt))
                    cost += 2. * max(0., .040 - spare)
                    targets.append((cost, angle, at, turn))
        shortlist, by_yaw = [], {}
        for candidate in sorted(targets, key=lambda target: target[0]):
            angle = candidate[1]
            if by_yaw.get(angle, 0) >= 4:
                continue
            by_yaw[angle] = by_yaw.get(angle, 0) + 1
            shortlist.append(candidate)
        return shortlist

    if placements is None:
        targets = [(angle, low, turn, .045, None) for _, angle, low, turn in placement_targets()]
    else:
        targets = []
        for angle, low, turn, rise, seed in placements:
            footprint = relative_cloud @ turn.T + low
            if (not -2.94 <= angle <= 3.11 or rise <= 0.
                    or np.any(footprint[:, :2].min(0) < tray_low[:2] + .007)
                    or np.any(footprint[:, :2].max(0) > tray_high[:2] - .007)
                    or footprint[:, 2].min() < tray_high[2] + .012):
                continue
            # Keep the drop footprint clear even when it starts above the solid.
            if occupied:
                try:
                    axes = ConvexHull(footprint[:, :2]).equations[:, :2]
                except QhullError:
                    continue
                for polygon, planes in occupied:
                    test_axes = np.vstack((axes, planes[:, :2]))
                    held_xy = footprint[:, :2] @ test_axes.T
                    solid_xy = polygon @ test_axes.T
                    gap = np.maximum(solid_xy.min(0) - held_xy.max(0), held_xy.min(0) - solid_xy.max(0))
                    if np.max(gap) <= .007:
                        break
                else:
                    targets.append((angle, low, turn, rise, seed))
            else:
                targets.append((angle, low, turn, rise, seed))
    if not targets:
        return None

    identity = np.eye(3)
    outside = center - [withdraw, 0., 0.]
    extraction = route(held, center, outside, identity, identity, yaw, underneath=True) if withdraw else [held]
    if extraction is None:
        return None
    peaks = []
    at = outside
    lift = extraction.copy() if withdraw else []
    last = extraction[-1]
    previous_turn = identity
    if lift_profile is None:
        lift_profile = [(height * VERTICAL, identity)
                        for height in (.020, .040, .065, .095, .130, .170, .210, .250)]
    for displacement, tilt in lift_profile:
        up = outside + displacement
        step = route(last, at, up, previous_turn, tilt, yaw, underneath=True)
        if step is None:
            break
        lift.extend(step)
        last, at = lift[-1], up
        previous_turn = tilt
        if displacement[2] >= .065 and clear(last, yaw, opened=False, underneath=False) and carries(last, yaw, False):
            peaks.append((len(lift), up, last, tilt))
    if not peaks:
        return None

    turns = {}
    for angle, low, orientation, rise, seed in targets:
        if angle not in turns:
            valid = (peak for peak in reversed(peaks) if segment(peak[2], peak[2], yaw, angle))
            turns[angle] = list(valid) if placements is not None else [next(valid, None)]
        rigid_turn = Rotation.from_rotvec((angle - yaw) * VERTICAL).as_matrix()
        above = low + rise * VERTICAL
        for peak in turns[angle]:
            if peak is None:
                continue
            count, up, pair, tilt = peak
            # An unseeded explicit placement follows the continuous IK branch.
            if seed is not None or placements is None:
                low_pose = pose(low, orientation, angle, pair if seed is None else seed)
                if low_pose is None:
                    continue
            after_turn = rigid_turn @ up
            turn_orientation = rigid_turn @ tilt
            if seed is None:
                transfer = route(pair, after_turn, above, turn_orientation, orientation, angle)
                if transfer is None:
                    continue
                release = route(transfer[-1], above, low, orientation, orientation, angle)
                if release is None:
                    continue
            else:
                raised = route(low_pose, low, above, orientation, orientation, angle)
                if raised is None:
                    continue
                backward = route(raised[-1], above, after_turn, orientation, turn_orientation, angle)
                if backward is None or not segment(pair, backward[-1], angle, angle):
                    continue
                transfer = [*reversed(backward), raised[-1]]
                release = [*reversed(raised[:-1]), low_pose]
            retreat = [*reversed(release[:-1]), transfer[-1]]
            if ready_path:
                retreat += [*reversed(transfer[:-1]), pair, *ready_path]
            previous = release[-1]
            opened = []
            for q in retreat:
                if not segment(previous, q, angle, angle, opened=True):
                    if placements is None:
                        break
                    for side in SIDES:
                        middle = dict(previous, **{side: q[side]})
                        if (segment(previous, middle, angle, angle, opened=True)
                                and segment(middle, q, angle, angle, opened=True)):
                            opened.append(middle)
                            break
                    else:
                        break
                opened.append(q)
                previous = q
            else:
                offsets, rotations = payload_poses(release[-1], angle)
                footprint = np.concatenate([relative_cloud @ turn.T + at
                                            for turn, at in zip(rotations, offsets)])[:, :2]
                footprint = np.stack((footprint.min(0), footprint.max(0)))
                return dict(lift=lift[:count], tray_yaw=angle, transfer=transfer,
                            release=release, retreat=opened, release_footprint=footprint.tolist(),
                            item=dict(item, release=low.tolist()))
    return None

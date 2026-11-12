#!/usr/bin/env python3
"""Sort the current ROS 2 table from one RGB-D survey. Run inside nssr-sim.

--recorded reads a planning-input NPZ and exercises the same queue and IK
without ROS. The live run emits a small JSONL log to stdout.
"""
import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from functools import cache
import json
import os
from pathlib import Path
import sys
from tempfile import gettempdir
from threading import Event, Lock
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'solution/src/team22_bringup'), str(ROOT / 'web-panel')]
from team22_bringup.grasp import Arm
from team22_bringup.batch_transit import empty_transition
from team22_bringup.cable_bimanual import (
    _section_hits_table, cable_bimanual_plan, cable_edge_receive_plan, check_cable_hook_transport,
)
from team22_bringup.cable_withdrawal import withdrawal_waypoints as cable_withdrawal_waypoints
from team22_bringup.hand_geometry import HandGeometry
from team22_bringup.lower_body import LowerBody
from team22_bringup.mouse_scoop import (
    DISTAL, body_geometry, low_object_slide_plan, mouse_edge_receive_plan, mouse_scoop_plan,
    supports_low_object_slide,
)
from team22_bringup.posture import ready_angles as base_ready_angles, working_hand_command
from team22_bringup.robot_surface import RobotSurfaceMask
from team22_bringup.scene_geometry import planning_edge, table_boundary
from team22_bringup.sort_order import measured_dimensions, prioritize_by_size
from team22_bringup.survey import HEAD, ANGLES
from team22_bringup.table_contact import fit_contact, split as split_contact_vertices, table_pickups
from team22_bringup.table_vision import CATALOG, TableVision
from body_turn import rotation
from grasp_gaze import gaze_angles
from tray_placement import release_sites

SIDES = ('left', 'right')
WAIST = 'body_yaw_joint'
CYLINDERS = ('lotion', 'gold_jar', 'blue_jar')
TWO_ARM_MOUSE = ('mouse', 'white_mouse')
TABLE_PINCH = (*TWO_ARM_MOUSE, 'cable')
TOP = (*TABLE_PINCH, 'gelatin_box', 'green_speaker', 'alarm_clock')
OPEN_LEFT = [1.] * 6
OPEN = {
    'cylinder': [.9512, .9501, .9455, 1., .5842, 0.],
    'top': [1., 1., 1., 1., 1., .45],
    'other': [.52, .63, 1., 1., .84, .38],
}
CLOSE = {
    'cylinder': [.25, .25, .25, .25, .10, .45],
    'top': [.24, .24, .24, .24, .10, .45],
    'other': [.52, .60, .31, .21, .18, .46],
}
TRAY_OFFSETS = (.05, 0., -.04)
TABLE_GAP = .006
TOUCH_FLOOR = -.002
CENTER_TOLERANCE = .007
READY_RISE = .05
PALM_FACE = np.array([.0275, 0., .09])
BOX_PRELOAD = .014
BOX_LOAD_LIMIT = 12.
BOX_PRESSURE_TARGET = 8.
BOX_GRIP_PRESSURE = 2.
WAIST_SPEED = .25
READY_WAIST_SPEED = .42
HAND_SPEED = .8
MOUSE_MOTION_TOLERANCE = np.deg2rad(1.)
FINGERS = list('123456')
LOG = None
LOG_LOCK = Lock()
SIM_TIME = None
SIM_START = None


def event(stage, **data):
    row = json.dumps(dict(stage=stage, sim_time=SIM_TIME, **data), ensure_ascii=False)
    with LOG_LOCK:
        print(row, flush=True)
        if LOG is not None:
            LOG.write(row + '\n')


@contextmanager
def measure(stage, **data):
    start = time.monotonic()
    sim_start = SIM_TIME
    try:
        yield
    finally:
        first = SIM_START if sim_start is None else sim_start
        elapsed = None if first is None or SIM_TIME is None else round(SIM_TIME - first, 3)
        wall = round(time.monotonic() - start, 3)
        event(stage, seconds=wall, wall_seconds=wall, sim_start=first,
              sim_end=SIM_TIME, sim_seconds=elapsed, **data)


def cartesian_rates(arm, frames, point, error, angle, speed_limit):
    axes = np.array([frame[:3, :3] @ axis for frame, axis in zip(frames, arm.axes)])
    offsets = point - np.array([frame[:3, 3] for frame in frames])
    jacobian = np.vstack((np.cross(axes, offsets).T, .13 * axes.T))
    twist = np.r_[5.5 * error, .13 * 2.5 * angle]
    rates = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6) * .015**2, twist)
    return rates * min(1., speed_limit / max(.001, float(np.max(abs(rates)))))


def queue_for(scene, task, points, *, skip_box=False, selected=None):
    observed = {obj['kind']: obj for obj in scene['objects'] if obj['status'] == 'observed'}
    expected = [(kind, name) for kind, (_, name) in CATALOG.items() if name in task['objects']]

    def family(kind):
        if kind in CYLINDERS:
            return 'cylinder'
        if kind in TOP:
            return 'top'
        return 'two_hands' if kind == 'drive' else 'other'

    # Reserve placement sites by height independently of the pickup sequence.
    # Tall items retain their tray positions even when smaller ones go first.
    expected.sort(key=lambda pair: -(measured_dimensions(observed.get(pair[0], {})) or (0., 0., 0.))[2])
    queue, omitted, tray_counts = [], {}, {}
    for kind, name in expected:
        if skip_box and kind == 'drive':
            omitted[kind] = 'requested_skip'
            continue
        target = observed.get(kind)
        category = task['objects'][name]
        tray_kind = 'tray_' + task['trays'][task['categories'][category]]
        slot = tray_counts.get(tray_kind, 0)
        tray_counts[tray_kind] = slot + 1
        if selected is not None and kind not in selected:
            continue
        tray = observed.get(tray_kind)
        if target is None or tray is None:
            omitted[kind] = 'object_missing' if target is None else 'tray_missing'
            continue
        bounds = np.array(tray['bounds'])
        if np.all((bounds[0, :2] <= target['xyz'][:2]) & (target['xyz'][:2] <= bounds[1, :2])):
            omitted[kind] = 'already_in_tray'
            continue
        fixed_slot = task['task'] == 'sort2' and slot < len(TRAY_OFFSETS)
        if fixed_slot:
            axis = 1 if kind == 'lotion' and slot == 0 else 0
            site = np.array(tray['xyz'])
            site[axis] += -.070 if axis == 1 else TRAY_OFFSETS[slot]
            if kind == 'gold_jar':
                site[1] += .040
            if kind == 'drive':
                site[0] = tray['xyz'][0] + .005
                site[1] = bounds[1, 1] - .055
            left, right = np.asarray(target['bounds'])[:, axis] - target['xyz'][axis]
            margin = TABLE_GAP + CENTER_TOLERANCE if kind == 'lotion' else TABLE_GAP
            low = bounds[0, axis] - left + margin
            high = bounds[1, axis] - right - margin
            if low <= high:
                site[axis] = np.clip(site[axis], low, high)
            sites = [site.tolist()]
        else:
            sites = release_sites(target, tray, points) or [tray['xyz']]
            sites.sort(key=lambda p: np.linalg.norm(p[:2]))
        queue.append(dict(kind=kind, family=family(kind), target=target, tray=tray,
                          release=sites[0], alternatives=sites[1:], tray_slot=slot if fixed_slot else None,
                          prefer_far_placement=kind == 'drive' and task['task'] == 'sort2',
                          approach_mode='edge_slide' if kind in TABLE_PINCH else 'lift'))
    omitted.update({name: 'unsupported' for name in task['objects'] if name not in dict(expected).values()})
    edge_x = scene.get('table_geometry', {}).get('edge_x_m')
    return prioritize_by_size(queue, edge_x=edge_x), omitted


def retain_contact_surfaces(scene, points):
    normal = np.r_[-np.array(scene['plane'][:2]), 1.]
    raised = points[points @ normal - scene['plane'][2] > .007]
    for obj in scene['objects']:
        if obj['status'] != 'observed' or obj['kind'].startswith('tray_') or 'contact_points' in obj:
            continue
        if obj['kind'] not in TABLE_PINCH and not supports_low_object_slide(obj):
            continue
        low, high = np.asarray(obj['bounds'])
        surface = raised[np.all((raised >= low - .008) & (raised <= high + .008), axis=1)]
        _, keep = np.unique(np.round(surface / .006).astype(int), axis=0, return_index=True)
        if len(keep) > 250:
            keep = keep[np.linspace(0, len(keep) - 1, 250).astype(int)]
        obj['contact_points'] = surface[keep].round(5).tolist()


def measure_tray_contents(scene, points):
    contents = {}
    for tray in scene['objects']:
        if not tray['kind'].startswith('tray_') or tray['status'] != 'observed':
            continue
        low, high = np.asarray(tray['bounds'])
        inside = np.all((points[:, :2] > low[:2] + .012) & (points[:, :2] < high[:2] - .012), axis=1)
        contents_cloud = points[inside & (points[:, 2] > high[2] + .020)]
        if len(contents_cloud) >= 24:
            bottom, top = np.quantile(contents_cloud, [.02, .98], axis=0)
            bottom[:2] = np.maximum(low[:2], bottom[:2] - .012)
            top[:2] = np.minimum(high[:2], top[:2] + .012)
            bottom[2] = low[2]
            contents[tray['kind']] = [bottom.tolist(), top.tolist()]
    return contents


def show_queue(scene, yaw, queue, omitted, *, task=None):
    survey = {} if task is None else dict(task=task, scene={
        'plane': scene['plane'], 'table_geometry': scene.get('table_geometry'),
        'tray_contents': scene.get('tray_contents', {}),
        'objects': [{key: obj[key] for key in ('kind', 'status', 'xyz', 'bounds', 'geometry', 'contact_points') if key in obj}
                    for obj in scene['objects'] if obj['status'] == 'observed'],
    })
    event('memory', waist_rad=yaw, plane=scene['plane'],
          centers_m={obj['kind']: obj['xyz'] for obj in scene['objects'] if obj['status'] == 'observed'},
          queue=[dict(kind=p['kind'], grasp=p['family'], tray=p['tray']['kind'], release_m=p['release'],
                      order_priority=p.get('order_priority')) for p in queue],
          omitted=omitted, **survey)


def next_route(routes, item, reserved, *, box_removed=False, resume_pinch=False, cable_probe=False,
               flat_mouse_receiver=False, lower_body=None, start_yaw=None, edge_batch=False,
               upright_box=False, leave_cable_hand=False):
    occupied = reserved.setdefault(item['tray']['kind'], [])
    sites = [item['release'], *item['alternatives']]
    radius = np.max(np.ptp(item['target']['bounds'], axis=0)[:2]) / 2

    def spacing(point):
        gap = min((np.linalg.norm(np.subtract(point[:2], xy)) - r - radius - .004
                   for xy, r in occupied), default=0.)
        return min(gap, 0.)

    if item.get('tray_slot') is None and item['family'] == 'cylinder' and spacing(item['release']) < 0.:
        lo, hi = np.array(item['tray']['bounds'])[:, 1]
        for delta in (.02, -.02, .035, -.035, .05, -.05):
            point = (np.array(item['release']) + [0., delta, 0.]).tolist()
            if lo + radius + .012 < point[1] < hi - radius - .012 and spacing(point) == 0.:
                sites.insert(0, point)
                break
    sites.sort(key=spacing, reverse=True)
    item = dict(item, release=sites[0], alternatives=sites[1:])
    ready = {side: ready_angles(side) for side in SIDES}
    if item.get('low_slide_only'):
        route = None
    elif item['kind'] in TWO_ARM_MOUSE:
        if cable_probe:
            route = mouse_scoop_plan(routes, item, ready, box_removed=box_removed)
        elif edge_batch and not flat_mouse_receiver:
            route = mouse_edge_receive_plan(routes, item, ready, box_removed=box_removed, probe_only=False,
                                            under_table=True, lower_body=lower_body, start_yaw=start_yaw,
                                            try_other_hand=item.get('adaptive_batch', False),
                                            planning_seconds=item.get('planning_seconds'))
        else:
            route = mouse_edge_receive_plan(routes, item, ready, box_removed=box_removed,
                                            probe_only=False, flat_receiver=flat_mouse_receiver,
                                            outside_receiver=flat_mouse_receiver, lower_body=lower_body,
                                            start_yaw=start_yaw)
    elif item['kind'] == 'cable' and not resume_pinch:
        if cable_probe:
            route = cable_bimanual_plan(routes, item, ready, box_removed=box_removed, probe_only=True)
        elif edge_batch:
            adaptive = item.get('adaptive_batch', False)
            preferred = item.get('preferred_hand', 'left' if item['target']['xyz'][1] >= 0. else 'right')
            sides = (preferred, 'right' if preferred == 'left' else 'left') if adaptive else (None,)
            started = time.monotonic()
            duration = item.get('planning_seconds')
            route = None
            for side_index, receiver in enumerate(sides):
                deadline = (started + duration * (.60 if side_index == 0 else 1.)
                            if duration is not None else None)
                route = cable_edge_receive_plan(
                    routes, item, ready, box_removed=box_removed, under_table=True, lower_body=lower_body,
                    start_yaw=start_yaw, hook=True, leave_hand_after_release=True,
                    receiver_side=receiver, deadline=deadline)
                if route is None:
                    continue
                route['hook_leave_hand_after_release'] = leave_cable_hand
                route['hook_operator_withdrawal'] = True
                route['hook_roll_dump'] = not leave_cable_hand and not route.get('hook_two_finger')
                if (route.get('hook_whole_hand') or route.get('hook_open_fingers')) and not item.get('manual_edge_stage'):
                    side = route['grip_side']
                    start = route['grasp'][-1][side][-1]
                    curl_limit = routes.arms[side].limits[0 if side == 'left' else 1, -1]
                    target = (start + curl_limit) * .5
                    delta = target - start

                    def curl(goals, fraction=1.):
                        result = {key: np.asarray(value).copy() for key, value in goals.items()}
                        if side in result:
                            result[side][-1] = np.clip(result[side][-1] + fraction * delta,
                                                       *routes.arms[side].limits[:, -1])
                        return result

                    closing = [step for step in route['close'] if step['phase'] == 'close']
                    for index, step in enumerate(closing, 1):
                        step['q'] = curl(step['q'], index / len(closing))
                        step['phase'] = 'close_curl'
                    for key in ('grasp', 'hook_departure', 'pusher_retreat', 'edge_probe',
                                'lift', 'transfer', 'release'):
                        if route.get(key):
                            route[key] = [curl(goals) for goals in route[key]]
                    route['hook_half_curl_rad'] = float(target)
                    base_yaw = route['tray_yaw']
                    yaw_offsets = (0., .58) if (adaptive and side == 'right'
                                               and item.get('occupied_footprints')) else (0.,)
                    for offset in yaw_offsets:
                        try:
                            adjusted = low_cable_transport(routes, dict(route, tray_yaw=base_yaw + offset))
                            if offset and not check_cable_hook_transport(routes, adjusted, lower_body):
                                raise RuntimeError('cable: obstructed turn or placement')
                        except RuntimeError as error:
                            if not adaptive:
                                raise
                            event('route_rejected', kind='cable', side=side, reason=str(error),
                                  tray_yaw=base_yaw + offset)
                            continue
                        route = adjusted
                        break
                    else:
                        route = None
                        continue
                if adaptive and route.get('probe_only'):
                    route = None
                    continue
                break
        else:
            route = cable_edge_receive_plan(routes, item, ready, box_removed=box_removed)
    elif item['family'] == 'two_hands':
        route = two_hand_plan(routes, item, ready, roll_box=edge_batch and not upright_box)
    else:
        route = None
        candidate = routes.candidate(item, ready)
        if candidate is not None:
            side, pick_yaw, tray_yaw, pick, put, item = candidate
            route = dict(item=item, pick_yaw=pick_yaw, tray_yaw=tray_yaw,
                         preshape={side: pick['preshape']}, grip={side: pick['grip']},
                         approach=[{side: q} for q in pick['approach']], grasp=[],
                         lift=[{side: pick['lift']}], transfer=[{side: put[0]}],
                         release=[{side: put[1]}], retreat=[{side: q} for q in (put[0], *pick.get('return', ()))])
            route.update(pocket=routes.arms[side].pocket.copy(), shape=pick['shape'],
                         pickup_center=pick['point'], placement_center=put[2])
            if 'touch' in pick:
                route['touch'] = pick['touch']
    if item.get('allow_low_slide') and (route is None or route.get('probe_only') or route.get('box_table_tip')):
        if supports_low_object_slide(item['target']):
            event('side_slide_fallback', kind=item['kind'])
            route = low_object_slide_plan(routes, item, ready, box_removed=box_removed,
                                          lower_body=lower_body, start_yaw=start_yaw)
    if route and route.get('edge_receive') and 'receive_point' not in route:
        return None
    if (route and not route.get('probe_only') and not route.get('box_table_tip')
            and not route.get('mouse_staging_only')):
        occupied.append((route['item']['release'][:2], radius))
    return route


def low_cable_transport(routes, plan):
    """Plan supported transport, a roll over the tray and finger extension once."""
    side = plan['grip_side']
    arm = routes.arms[side]
    departure = plan.get('hook_departure')
    if not departure and plan['item'].get('adaptive_batch'):
        departure = plan.get('edge_probe')
    if not departure:
        raise RuntimeError('cable: low departure is unavailable')
    start_pair = departure[-1]
    start = start_pair[side]
    initial = arm.tool(start)
    receive = np.asarray(plan['receive_point'])
    coil_start = initial[:3, 3] + initial[:3, :3] @ (receive - arm.pocket)
    yaw = plan['tray_yaw']
    tray = plan['item']['tray']
    low, high = np.asarray(tray['bounds'])
    center = np.asarray(tray['xyz'])[:2]
    outward = center / np.linalg.norm(center)
    preferred = high[:2] - .080
    near = center - .035 * outward
    sites = ([near + fraction * (preferred - near) for fraction in (1., .75, .50, .25, 0.)]
             if plan.get('hook_open_fingers') else
             [low[:2] + [.080, .080]])
    occupied = plan['item'].get('occupied_footprints', {})
    if plan.get('hook_open_fingers') and occupied:
        candidates = [*sites, *[np.array([x, y])
            for y in (low[1] + .080, low[1] + .065, center[1] - .065)
            for x in (center[0], low[0] + .065, high[0] - .065)]]
        def free(xy):
            return all(np.any(xy + .073 < np.asarray(bounds)[0]) or
                       np.any(xy - .073 > np.asarray(bounds)[1])
                       for bounds in occupied.values())
        available = [xy for xy in candidates if free(xy)]
        if available:
            sites = sorted(available, key=lambda xy: np.linalg.norm(xy - preferred))
    sites = [np.clip(xy, low[:2] + .055, high[:2] - .055) for xy in sites]
    lower, upper = arm.limits + [[.018], [-.018]]
    inverted = np.column_stack(([0., 0., -1.], [0., 1., 0.], [1., 0., 0.]))
    if plan.get('hook_open_fingers'):
        # Let the fingers hang toward the tray instead of leaving the palm
        # horizontal with the coil wrapped around the back of the hand.
        inverted = Rotation.from_rotvec([0., np.deg2rad(65.), 0.]).as_matrix() @ inverted
    blend = Slerp([0., 1.], Rotation.from_matrix([initial[:3, :3], inverted]))
    release_hand = (np.ones(6) if plan.get('hook_open_fingers') else
                    np.array([.55, .55, .55, .55, .60, 0.]))
    mesh = np.concatenate(list(routes.hands[side].vertices(
        routes.hands[side].nominal_angles(release_hand)).values()))
    def solve(coil, orient, seed, position_weights=(3., 3., 1.)):
        point = coil + orient @ (arm.pocket - receive)
        roll_weight = .30 if plan.get('hook_open_fingers') and np.linalg.norm(orient - inverted) < .1 else .15
        def residual(q):
            pose = arm.tool(q)
            turn = Rotation.from_matrix(orient.T @ pose[:3, :3]).as_rotvec()
            return np.r_[position_weights * (pose[:3, 3] - point), roll_weight * turn, .00003 * (q - seed)]
        fit = least_squares(residual, np.clip(seed, lower + 1e-7, upper - 1e-7),
                            bounds=(lower, upper), max_nfev=70)
        # A requested bottom pose may be beyond joint travel. Use its bounded
        # closest pose rather than waiting for an unreachable exact alignment.
        return fit.x

    def pair(q):
        return dict(start_pair, **{side: q})

    # A hanging coil must clear the rim before crossing it. Carry the support
    # point 24 cm above the rim, then release inside the tray, beyond its center.
    heights = [high[2] + .14] if plan.get('hook_open_fingers') else [low[2] + .028]
    for height in heights:
        for xy in sites:
            destination = routes.local(np.r_[xy, height], yaw)
            stage = destination.copy()
            seed = start
            takeoff = []
            transport_start = coil_start.copy()
            if plan.get('hook_open_fingers'):
                raised_coil = coil_start.copy()
                raised_coil[2] = max(coil_start[2], high[2] + .24)
                seed = solve(raised_coil, initial[:3, :3], seed)
                takeoff.append(pair(seed))
                raised_pose = arm.tool(seed)
                transport_start = raised_pose[:3, 3] + raised_pose[:3, :3] @ (receive - arm.pocket)
                stage[2] = max(height + .025, transport_start[2])
            else:
                stage[2] = max(height + .025, min(coil_start[2], high[2] + .065))
            carry_height = float(stage[2])
            transfer = []
            transport_samples = np.linspace(0., 1., 8)[1:]
            for fraction in transport_samples:
                coil = (1. - fraction) * transport_start + fraction * stage
                orient = initial[:3, :3] if plan.get('hook_open_fingers') else blend(fraction).as_matrix()
                seed = solve(coil, orient, seed)
                if seed is None:
                    break
                transfer.append(pair(seed))
            else:
                if plan.get('hook_open_fingers'):
                    # Keep the coil supported during transport. Turn the wrist
                    # only once the receiving hand is over the tray opening.
                    at_tray = arm.tool(seed)
                    wrist_origin = at_tray[:3, 3] - at_tray[:3, :3] @ arm.pocket
                    for fraction in np.linspace(0., 1., 9)[1:]:
                        orient = blend(fraction).as_matrix()
                        hanging = stage.copy()
                        hanging[2] = max(low[2] + .035,
                                         wrist_origin[2] + .060 * fraction + (orient @ receive)[2])
                        seed = solve(hanging, orient, seed)
                        transfer.append(pair(seed))
                    # The wrist has risen 6 cm while the fingers now point down
                    # into the opening. Release there, without another lift.
                    destination[2] = stage[2] = height = hanging[2]
                descent = []
                for z in np.linspace(stage[2], height, max(2, int((stage[2] - height) / .03) + 2))[1:]:
                    seed = solve(np.r_[destination[:2], z], inverted, seed)
                    if seed is None:
                        break
                    descent.append(pair(seed))
                else:
                    release_pose = arm.tool(seed)
                    release_point = release_pose[:3, 3]
                    release_orient = release_pose[:3, :3]
                    mesh_offset = (mesh - arm.pocket) @ release_orient.T
                    release_support = release_point + release_orient @ (receive - arm.pocket)
                    support_world = routes.world(release_support, yaw)
                    if plan.get('hook_open_fingers') and (
                            np.linalg.norm(support_world[:2] - xy) > .030 or
                            release_orient[2, 2] > -.48 or
                            np.any(support_world[:2] < low[:2] + .077) or
                            np.any(support_world[:2] > high[:2] - .077)):
                        continue
                    obstacle_top = routes.highest
                    minimum_rise = .18
                    if plan['item'].get('timed_batch'):
                        # Detach vertically, clear this tray, then withdraw at constant height.
                        # Do not use the height of distant objects as the tray clearance.
                        obstacle_top = max(high[2], low[2] + .015)
                        minimum_rise = .075
                    rise = max(minimum_rise, obstacle_top + .035 -
                               (release_point[2] + mesh_offset[:, 2].min()))
                    retreat = []
                    exits = np.linspace(0., rise, max(3, int(rise / .06) + 2))[1:]
                    for lift in exits:
                        # Lift from the attained release pose, with the open
                        # fingers still hanging down and no sideways withdrawal.
                        seed = solve(release_support + [0., 0., lift], release_orient, seed,
                                     position_weights=(3., 3., 6.))
                        if seed is None:
                            break
                        retreat.append(pair(seed))
                    else:
                        raised_pose = arm.tool(seed)
                        raised_support = raised_pose[:3, 3] + raised_pose[:3, :3] @ (receive - arm.pocket)
                        withdraw_high = []
                        for back in (.060, .120, .180, .240):
                            seed = solve(raised_support + [-back, 0., 0.], raised_pose[:3, :3], seed,
                                         position_weights=(3., 3., 6.))
                            withdraw_high.append(pair(seed))
                        plan.update(hook_low_transport=True, hook_roll_dump=False,
                                    hook_operator_withdrawal=False, lift=takeoff, transfer=transfer,
                                    release=descent, retreat=retreat, cable_withdraw_high=withdraw_high,
                                    release_hand={side: release_hand},
                                    cable_lower_center_m=np.r_[xy, height],
                                    release_content_top_m=float(low[2] + .015 +
                                        np.ptp(plan['item']['target']['bounds'], axis=0)[2]),
                                    cable_exit_rise_m=rise)
                        plan['item'] = dict(plan['item'], release=np.r_[xy, height].tolist())
                        placed_xy = support_world[:2]
                        plan['release_footprint'] = [(placed_xy - .065).tolist(),
                                                     (placed_xy + .065).tolist()]
                        event('cable_low_transport_plan', release_center_m=np.r_[xy, height].tolist(),
                              bottom_gap_mm=round(1000 * (height - low[2]), 1),
                              exit_rise_mm=round(1000 * rise, 1), release_hand=release_hand.tolist(),
                              no_squeeze=bool(plan.get('hook_open_fingers')),
                              carry_height_m=carry_height,
                              wrist_rise_during_roll_mm=60 if plan.get('hook_open_fingers') else 0,
                              finger_down_deg=round(float(np.degrees(np.arcsin(np.clip(
                                  -arm.tool(descent[-1][side])[2, 2], -1., 1.)))), 1),
                              planned_support_m=support_world.tolist())
                        return plan
    raise RuntimeError('cable: no low roll, placement and upward exit route')


class Routes:
    """Small Cartesian IK problem in the frozen survey frame, with a 6 mm floor."""

    def __init__(self, scene, yaw):
        self.yaw = yaw
        self.objects = scene['objects']
        self.normal = np.r_[-np.asarray(scene['plane'])[:2], 1.]
        self.floor = scene['plane'][2]
        self.edge = planning_edge(scene)
        self.highest = max(obj['bounds'][1][2] for obj in scene['objects'])
        self.mouse_bounds = []
        for obj in self.objects:
            if obj['kind'] not in TWO_ARM_MOUSE:
                continue
            low, high = np.array(obj['bounds'])
            footprint = np.asarray(obj.get('geometry', {}).get('corners_xy_m', ()))
            if footprint.size:
                low[:2] = np.minimum(low[:2], footprint.min(0))
                high[:2] = np.maximum(high[:2], footprint.max(0))
            self.mouse_bounds.append((low - .010, high + .010))
        self.arms = {side: Arm(side) for side in SIDES}
        self.hands = {side: HandGeometry(side) for side in SIDES}

    def local(self, xyz, yaw):
        return rotation(self.yaw - yaw) @ xyz

    def world(self, xyz, yaw):
        return rotation(yaw - self.yaw) @ xyz

    def heading(self, point, side):
        return self.yaw + np.arctan2(point[1], point[0]) - (.27 if side == 'left' else -.27)

    def shape(self, side, family, grip, preshape=None):
        hand = self.hands[side]
        preshape = OPEN[family] if preshape is None else preshape
        joints = hand.nominal_angles(preshape)
        frames = hand.frames(joints)
        roots = {j['name']: j['child'] for j in hand.joints}
        self.arms[side].pocket = np.mean([
            frames[roots[f'{side}_{finger}_1_joint']][:3, 3]
            for finger in ('index', 'middle', 'ring', 'little')
        ], axis=0) + [.05, 0., 0.]
        states = [hand.nominal_angles(preshape + fraction * np.subtract(grip, preshape))
                  for fraction in (0., .5, 1.)]
        return np.concatenate([v for q in states for v in hand.vertices(q).values()])

    def floor_height(self, side, points, orient, yaw, goal):
        offsets = (points - self.arms[side].pocket) @ orient.T
        normal = self.local(self.normal, yaw)
        return self.floor - goal[:2] @ normal[:2] - (offsets @ normal).min() + TABLE_GAP + .003

    def clears_mouse(self, points, radii):
        for low, high in self.mouse_bounds:
            gap = np.maximum(0., np.maximum(low - points, points - high))
            if np.any(np.einsum('ij,ij->i', gap, gap) <= radii ** 2):
                return False
        return True

    def clear(self, side, q, points, yaw, *, touch=None, gap=TABLE_GAP):
        arm = self.arms[side]
        pose = arm.wrist(q) @ arm.hand
        forearm, radii = arm.samples(q)
        fingers = points @ pose[:3, :3].T + pose[:3, 3]
        if touch is not None:
            tips = touch @ pose[:3, :3].T + pose[:3, 3]
            fingers = np.concatenate((fingers, tips))
        local = np.concatenate((forearm[:112], fingers))
        radii = np.r_[radii[:112], np.zeros(len(fingers))]
        # The first two arm segments are inside their shoulder mounts.
        x, y, z = local[32:].T
        radius = radii[32:]
        if np.any((x > -.19 - radius) & (x < .10 + radius) & (abs(y) < .17 + radius)
                  & (z > -.22 - radius) & (z < .43 + radius)):
            return False
        world = local @ rotation(yaw - self.yaw).T
        over = world[:, 0] >= self.edge - radii
        margins = np.full(len(local), gap)
        if touch is not None:
            margins[-len(touch):] = TOUCH_FLOOR
        return (np.all(world[over] @ self.normal - self.floor - radii[over] >= margins[over])
                and self.clears_mouse(world, radii))

    def link(self, side, start, goal, points, yaw):
        count = 2 + int(np.max(abs(goal - start)) / .10)
        return all(self.clear(side, q, points, yaw) for q in np.linspace(start, goal, count))

    def fit(self, side, goal, orient, seed, *, model=None):
        arm = self.arms[side] if model is None else model
        return fit_pose(arm, goal, orient, seed)

    def raised_ready(self, side, start, shape, yaw):
        pose = self.arms[side].tool(start)
        q = self.fit(side, pose[:3, 3] + [0., 0., .13], pose[:3, :3], start)
        return q if q is not None and self.link(side, start, q, shape, yaw) else None

    def raised_elbow(self, side, waypoint, hand_rise):
        arm = self.arms[side]
        pose = arm.tool(waypoint)
        target = pose[:3, 3] + [0., 0., hand_rise]
        elbow_z = arm.frames(waypoint)[3][2, 3] + .051

        def residual(q):
            actual = arm.tool(q)
            turn = Rotation.from_matrix(pose[:3, :3].T @ actual[:3, :3]).as_rotvec()
            return np.r_[actual[:3, 3] - target, .12 * turn,
                         arm.frames(q)[3][2, 3] - elbow_z, .002 * (q - waypoint)]

        bounds = arm.limits + [[.018], [-.018]]
        result = least_squares(residual, np.clip(waypoint, *bounds), bounds=bounds, max_nfev=36, ftol=2e-4)
        if np.linalg.norm(result.fun[:3]) <= .003 and abs(result.fun[6]) <= .002:
            return result.x
        return None

    def straight_pickup(self, side, start, grasp, orient, shape, yaw):
        pose = self.arms[side].tool(start)
        rotate = Slerp([0., 1.], Rotation.from_matrix([pose[:3, :3], orient]))
        path = [start]
        for fraction in (.25, .50, .75, 1.):
            target = pose[:3, 3] + fraction * (grasp - pose[:3, 3])
            q = self.fit(side, target, rotate(fraction).as_matrix(), path[-1])
            if q is None or not self.link(side, path[-1], q, shape, yaw):
                return None
            path.append(q)
        return path[1:]

    def pickup(self, item, side, yaw, start):
        family = item['family']
        target = self.local(item['target']['xyz'], yaw)
        bottom, top = np.array(item['target']['bounds'])[:, 2]
        if family != 'cylinder':
            target[2] = top - min(.018, (top - bottom) / 4)
        if item['kind'] == 'gold_jar':
            target[2] += .014
        grip = [.24, .24, .24, .24, .19, 0.] if item['kind'] == 'lotion' else CLOSE[family]
        preshape = OPEN_LEFT if side == 'left' and item['kind'] in ('mouse', 'white_mouse', 'cable') else OPEN[family]
        shape = self.shape(side, family, grip, preshape)
        mirror = 1. if side == 'right' else -1.
        options = ((.02, -.10, -.20, 0.), (-.32, -.08, -.16, 0.),
                   (.32, -.20, -.12, 0.), (-.60, -.25, -.12, 0.))
        if item['kind'] == 'lotion':
            options = ((-.32, 0., 0., 0.), *options)
        if family != 'cylinder':
            options = ((0., 0., 0., 0.), (-.65, 0., .24, .70),
                       (.65, 0., .24, .70), (-1.2, 0., -.30, .52), (1.2, 0., -.30, .52))
        raised = None
        for heading, pitch, roll, tilt in options:
            heading *= mirror
            fingers = np.array([np.cos(heading) * np.cos(pitch), np.sin(heading) * np.cos(pitch), np.sin(pitch)])
            palm = (mirror * np.array([-np.sin(heading), np.cos(heading), 0.])
                    if family == 'cylinder' else np.array([0., 0., -1.]))
            orient = np.column_stack((palm, np.cross(fingers, palm), fingers))
            orient = orient @ Rotation.from_euler('yz', [tilt, mirror * roll]).as_matrix()
            grasp = target.copy()
            grasp[2] = max(grasp[2], self.floor_height(side, shape, orient, yaw, grasp))
            if grasp[2] > top + (.014 if family == 'cylinder' else .025):
                continue
            lift = grasp.copy()
            lift[2] = max(grasp[2] + .10, self.highest + (grasp[2] - bottom) + .022)
            straight = self.straight_pickup(side, start, grasp, orient, shape, yaw) if item['kind'] == 'gold_jar' else None
            if item['kind'] == 'gold_jar' and straight is None:
                continue
            q_grasp = straight[-1] if straight else self.fit(side, grasp, orient, start)
            if q_grasp is None:
                continue
            q_lift = self.fit(side, lift, orient, q_grasp)
            if q_lift is None or not self.link(side, q_grasp, q_lift, shape, yaw):
                continue
            approach = straight or [q_grasp]
            if not straight and (item['kind'] == 'blue_jar' or not self.link(side, start, q_grasp, shape, yaw)):
                via = self.fit(side, grasp + [0., 0., .055], orient, q_lift)
                if item['kind'] == 'blue_jar':
                    if via is None:
                        continue
                    early = self.raised_elbow(side, start + .25 * (via - start), .080)
                    raised_via = self.raised_elbow(side, via, .100)
                    if early is None or raised_via is None or not all(
                            self.link(side, first, last, shape, yaw)
                            for first, last in zip((start, early, raised_via, via),
                                                   (early, raised_via, via, q_grasp))):
                        continue
                    approach = [early, raised_via, via, q_grasp]
                elif via is not None and self.link(side, via, q_grasp, shape, yaw) and self.link(side, start, via, shape, yaw):
                    approach.insert(0, via)
                else:
                    if raised is None:
                        raised = self.raised_ready(side, start, shape, yaw)
                    if raised is None or not self.link(side, raised, q_lift, shape, yaw):
                        continue
                    approach = [raised, q_lift, q_grasp]
            return dict(approach=approach, lift=q_lift, orient=orient, shape=shape, grip=grip, preshape=preshape,
                        bottom_offset=grasp[2] - bottom, height=lift[2], point=self.world(grasp, yaw))
        return None

    def placement(self, item, side, yaw, pick):
        shape = pick['shape']
        ready = ready_angles(side)
        high = self.local(item['release'], yaw)
        high[2] = pick['height']
        aim = np.arctan2(high[1], high[0]) - np.arctan2(pick['orient'][1, 2], pick['orient'][0, 2])
        for delta in (0., aim, aim - .28, aim + .28):
            orient = rotation(delta) @ pick['orient']
            low = high.copy()
            opening_gap = {'blue_jar': .065, 'gold_jar': .070, 'lotion': .050}.get(item['kind'], .045)
            low[2] = max(item['tray']['bounds'][1][2] + pick['bottom_offset'] + opening_gap,
                         self.floor_height(side, shape, orient, yaw, low))
            q_high = self.fit(side, high, orient, pick['lift'])
            q_low = None if q_high is None else self.fit(side, low, orient, q_high)
            if q_low is None:
                continue
            links = ((pick['lift'], q_high), (q_high, q_low))
            if not all(self.link(side, a, b, shape, yaw) for a, b in links):
                continue
            if self.link(side, q_high, ready, shape, yaw):
                back = ()
            else:
                raised = self.raised_ready(side, ready, shape, yaw)
                if raised is None or not self.link(side, q_high, raised, shape, yaw):
                    continue
                back = (raised,)
            elevated = []
            pose = self.arms[side].tool(q_high)
            next_pose = back[0] if back else ready
            for rise in (.025, .050):
                start = elevated[-1] if elevated else q_high
                q = self.fit(side, pose[:3, 3] + [0., 0., rise], pose[:3, :3], start)
                if q is None or not self.link(side, start, q, shape, yaw):
                    break
                elevated.append(q)
            while elevated and not self.link(side, elevated[-1], next_pose, shape, yaw):
                elevated.pop()
            pick['return'] = (*elevated, *back)
            return q_high, q_low, self.world(low, yaw)
        return None

    def pickups(self, item, side, start):
        if side == 'left' and item['kind'] in TABLE_PINCH:
            yield from table_pickups(self, item, side, start)
            return
        source = self.heading(item['target']['xyz'], side)
        preferred = .15 if item['kind'] == 'gold_jar' else 0.
        offsets = (.27, .30, .25, 0., -.27) if item['kind'] == 'lotion' else (preferred, 0., -.27, .27)
        for offset in dict.fromkeys(offsets):
            pick_yaw = source + offset
            pick = self.pickup(item, side, pick_yaw, start)
            if pick is not None:
                yield pick_yaw, pick

    def candidate(self, item, positions):
        tray = item['tray']
        preferred_side = {'tray_red': 'left', 'tray_pink': 'left', 'tray_blue': 'right'}.get(
            tray['kind'], 'right' if tray['xyz'][1] < 0 else 'left')
        adaptive = item.get('adaptive_batch', False)
        if adaptive:
            preferred_side = item.get('preferred_hand', preferred_side)
        sides = (preferred_side,)
        if adaptive and item['kind'] in CYLINDERS:
            sides += ('right' if preferred_side == 'left' else 'left',)
        lotion = item['kind'] == 'lotion'

        def pickup_trials(side):
            lotion_picks = []
            for pick_yaw, pick in self.pickups(item, side, positions[side]):
                if adaptive and lotion:
                    lotion_picks.append((pick_yaw, pick))
                yield pick_yaw, pick, False
            for pick_yaw, pick in lotion_picks:
                yield pick_yaw, pick, True

        for side in sides:
            for pick_yaw, pick, allow_lotion_turn in pickup_trials(side):
                for site in [item['release'], *item['alternatives'][:2]]:
                    selected = dict(item, release=site)
                    if lotion and not allow_lotion_turn:
                        tray_headings = (pick_yaw,)
                    else:
                        drop = self.heading(site, side)
                        facing = (drop, drop - .27, drop + .27)
                        wider = (drop - .50, drop + .50, drop - .80, drop + .80)
                        if lotion:
                            tray_headings = tuple(yaw for yaw in (*facing, *wider) if yaw < pick_yaw)
                        elif item['kind'] == 'blue_jar':
                            tray_headings = (*facing, pick_yaw, *wider)
                        else:
                            tray_headings = (pick_yaw, *facing, *wider)
                    for tray_yaw in tray_headings:
                        place = self.placement(selected, side, tray_yaw, pick)
                        if place is None:
                            continue
                        steps = 2 + int(abs(tray_yaw - pick_yaw) / .2)
                        if all(self.clear(side, pick['lift'], pick['shape'], yaw)
                               for yaw in np.linspace(pick_yaw, tray_yaw, steps)[1:-1]):
                            return side, pick_yaw, tray_yaw, pick, place, selected
        return None


def fit_pose(arm, goal, orient, seed, tolerance=.012):
    lower, upper = arm.limits + [[.018], [-.018]]

    def residual(q):
        pose = arm.tool(q)
        angle = Rotation.from_matrix(orient.T @ pose[:3, :3]).as_rotvec()
        return np.r_[pose[:3, 3] - goal, .12 * angle, .002 * (q - seed)]

    fit = least_squares(residual, np.clip(seed, lower + 1e-6, upper - 1e-6),
                        bounds=(lower, upper), max_nfev=36, ftol=2e-4)
    if np.linalg.norm(fit.fun[:3]) < tolerance and np.linalg.norm(fit.fun[3:6]) < .014:
        return fit.x
    return None


@cache
def ready_angles(side):
    arm = Arm(side)
    seed = base_ready_angles(side)
    pose = arm.tool(seed)
    ready = fit_pose(arm, pose[:3, 3] + [0., 0., READY_RISE], pose[:3, :3], seed, tolerance=.001)
    if ready is None:
        raise RuntimeError(f'{side}: raised ready pose is unreachable')
    return ready


def two_hand_plan(routes, item, positions, *, roll_box=False):
    """Plan an upright carry or tip the box around its lower edge on the table."""
    from itertools import product
    from scipy.spatial import ConvexHull, cKDTree
    from bimanual import interarm_gap
    from team22_bringup import grasp

    arms = {side: Arm(side) for side in SIDES}
    with np.load(Path(grasp.__file__).with_name('robot_surfaces.npz')) as surfaces:
        torso = surfaces['waist_yaw_link']
        # The two links at each shoulder overlap their intended mounting surfaces.
        links = {side: [surfaces[name.replace('_joint', '_link')][::4] for name in arm.names[2:]]
                 for side, arm in arms.items()}
    body_min, body_max = torso.min(0) - .003, torso.max(0) + .003
    body_planes = ConvexHull(torso).equations
    body_tree = cKDTree(torso)
    ready = {side: ready_angles(side) for side in SIDES}
    signs = dict(left=1., right=-1.)
    target = item['target']
    object_center = np.asarray(target['xyz'])
    bounds = np.asarray(target['bounds'])
    bottom, top = bounds[:, 2]
    geometry = target.get('geometry', {})
    major = np.argmax(np.ptp(bounds, axis=0)[:2])
    axis = np.array(geometry.get('long_axis_xy', np.eye(2)[major]), dtype=float)
    axis /= np.linalg.norm(axis)
    corners = np.asarray(geometry.get('corners_xy_m', [[x, y] for x in bounds[:, 0] for y in bounds[:, 1]]))
    box = np.array([[*xy, z] for xy in corners for z in (bottom, top)])
    tray = np.asarray(item['tray']['bounds'])
    # Keep the returning arms clear even if the placed box rests on the rim.
    tray_contents_high = tray[1] + np.array([0., 0., top - bottom + .015])
    active = 'left' if object_center[1] > 0. else 'right'
    support = 'right' if active == 'left' else 'left'
    preshape = {active: [1., 1., 1., 1., 0., 1.], support: [.08, .08, .08, .08, 0., 1.]}
    grip = {active: [.16, .16, .16, .16, .18, .45]}
    if roll_box:
        preshape = {side: [1., 1., 1., 1., 0., 1.] for side in SIDES}
        grip = {}
    parts = {side: hand.vertices(hand.nominal_angles(preshape[side])) for side, hand in routes.hands.items()}
    shape = {side: np.concatenate(list(vertices.values())) for side, vertices in parts.items()}
    for side, command in grip.items():
        hand = routes.hands[side]
        shape[side] = np.concatenate([shape[side], *hand.vertices(hand.nominal_angles(command)).values()])
    arms[active].pocket = PALM_FACE

    def clear(side, q, yaw, *, tray_contents=False):
        arm = arms[side]
        frames = arm.frames(q)
        hand = frames[-1] @ arm.hand
        fingers = shape[side] @ hand[:3, :3].T + hand[:3, 3]
        samples, radii = arm.samples(q)
        samples = np.concatenate((samples[:112], fingers))
        radii = np.r_[radii[:112], np.zeros(len(fingers))]
        world = samples @ rotation(yaw - routes.yaw).T
        if tray_contents:
            distance = np.maximum(0., np.maximum(tray[0] - world, world - tray_contents_high))
            if np.any(np.einsum('ij,ij->i', distance, distance) < (radii + .004)**2):
                return False
        over = world[:, 0] >= routes.edge - radii
        if np.any(world[over] @ routes.normal - routes.floor - radii[over] < TABLE_GAP):
            return False
        if not routes.clears_mouse(world, radii):
            return False
        for vertices, frame in zip(links[side], frames[2:]):
            points = vertices @ frame[:3, :3].T + frame[:3, 3]
            near = points[np.all((points > body_min) & (points < body_max), axis=1)]
            if len(near) and (np.any(body_tree.query(near)[0] < .003)
                              or np.any((near @ body_planes[:, :3].T + body_planes[:, 3]).max(1) < .002)):
                return False
        near = fingers[np.all((fingers > body_min) & (fingers < body_max), axis=1)]
        return not np.any(body_tree.query(near)[0] < .003)

    def relative(pair):
        return arms['left'].tool(pair['left'])[:3, 3] - arms['right'].tool(pair['right'])[:3, 3]

    def clear_pair(pair, yaw, *, tray_contents=False):
        return (all(clear(side, pair[side], yaw, tray_contents=tray_contents) for side in SIDES)
                and interarm_gap(arms, pair) >= .002)

    def link(start, end, yaw, separation=None, *, tray_contents=False):
        delta = {side: end[side] - start[side] for side in SIDES}
        count = 2 + int(max(np.max(abs(d)) for d in delta.values()) / .10)
        for fraction in np.linspace(0., 1., count):
            pair = {side: start[side] + fraction * delta[side] for side in SIDES}
            if not clear_pair(pair, yaw, tray_contents=tray_contents):
                return False
            if separation is not None:
                palms = relative(pair)
                gap = (abs(np.linalg.norm(palms) - np.linalg.norm(separation)) if narrow
                       else np.linalg.norm(palms - separation))
                if gap > .010:
                    return False
        return True

    def fit_pair(seed, goals, orient):
        pair = {side: fit_pose(arms[side], goals[side], orient[side], seed[side], tolerance=.008)
                for side in SIDES}
        return pair if all(q is not None for q in pair.values()) else None

    def path(seed, goals, orient, yaw, separation=None, *, tray_contents=False):
        steps = []
        for goal in goals:
            pair = fit_pair(seed, goal, orient)
            if pair is None or not link(seed, pair, yaw, separation, tray_contents=tray_contents):
                return None
            steps.append(pair)
            seed = pair
        return steps

    def connect(start, end, yaw, *, tray_contents=False):
        if link(start, end, yaw, tray_contents=tray_contents):
            return [end]
        before = {side: arms[side].tool(start[side]) for side in SIDES}
        after = {side: arms[side].tool(end[side]) for side in SIDES}
        turns = {side: Slerp([0., 1.], Rotation.from_matrix([before[side][:3, :3], after[side][:3, :3]]))
                 for side in SIDES}
        steps = []
        seed = start
        for fraction in (.25, .5, .75):
            goals = {side: (1 - fraction) * before[side][:3, 3] + fraction * after[side][:3, 3]
                     + np.sin(np.pi * fraction) * np.array([.06, signs[side] * .10, .14]) for side in SIDES}
            orient = {side: turns[side](fraction).as_matrix() for side in SIDES}
            pair = fit_pair(seed, goals, orient)
            if pair is None or not link(seed, pair, yaw, tray_contents=tray_contents):
                return None
            steps.append(pair)
            seed = pair
        return [*steps, end] if link(seed, end, yaw, tray_contents=tray_contents) else None

    thumb_clearance_roll = {side: Rotation.from_rotvec([0., 0., roll]).as_matrix()
                            for side, roll in dict(left=-.115, right=.20).items()}

    def orientations(tangent, normal, tilt, cant=0., *, roll=False):
        fingers = np.cos(tilt) * tangent + [0., 0., np.sin(tilt)]
        palms = {side: -signs[side] * normal for side in SIDES}
        return {side: rotation(signs[side] * cant) @ np.column_stack((palm, np.cross(fingers, palm), fingers))
                      @ (thumb_clearance_roll[side] if roll else np.eye(3))
                for side, palm in palms.items()}

    if roll_box:
        # Push the upper part of the near end face. The far bottom edge supports the box.
        side, passive = 'right', 'left'
        center = np.r_[corners.mean(0), bottom]
        yaw = routes.yaw + np.arctan2(center[1], center[0]) + .62
        forward = axis if axis[0] >= 0. else -axis
        normal = routes.local(-np.r_[forward, 0.], yaw)
        tangent = np.array([normal[1], -normal[0], 0.])
        origin = routes.local(center, yaw)
        faces = (corners - center[:2]) @ -forward
        edge = origin + faces.min() * normal
        height = min(.175, top - bottom - .020)
        press_start = origin + faces.max() * normal - .016 * tangent + [0., 0., height]
        arm = arms[side]
        arm.pocket = PALM_FACE
        lower, upper = arm.limits + [[.018], [-.018]]
        angle_deg = 60.

        def press_pose(fraction, gap):
            turn = Rotation.from_rotvec(fraction * np.deg2rad(angle_deg) * tangent).as_matrix()
            target = edge + turn @ (press_start - edge + gap * normal) + [0., 0., .017 * fraction]
            tilt = -.22 + .47 * fraction
            fingers = np.cos(tilt) * tangent + [0., 0., np.sin(tilt)]
            palm = -normal
            orient = np.column_stack((palm, np.cross(fingers, palm), fingers)) @ thumb_clearance_roll[side]
            return target, orient

        def press_step(seed, target, orient):
            q0 = seed[side]
            for start in (q0, q0 + [-.4, -.35, -.4, 0., 0., 0., 0.],
                          q0 + [-.6, -.45, .4, .2, -.6, .25, 0.]):
                def residual(q):
                    pose = arm.tool(q)
                    return np.r_[pose[:3, 3] - target, .10 * (pose[:3, 0] - orient[:, 0]),
                                 .008 * (pose[:3, 2] - orient[:, 2]), .004 * (q - start)]

                fit = least_squares(residual, np.clip(start, lower + 1e-6, upper - 1e-6),
                                    bounds=(lower, upper), max_nfev=48, ftol=2e-4)
                pair = {side: fit.x, passive: ready[passive]}
                if (np.linalg.norm(fit.fun[:3]) < .008 and np.linalg.norm(fit.fun[3:6]) < .030
                        and link(seed, pair, yaw)):
                    return pair
            return None

        world_turn = rotation(yaw - routes.yaw)
        payload = box @ world_turn - edge
        obstacles = [*routes.mouse_bounds, *(np.asarray(obj['bounds']) for obj in routes.objects
                                            if obj['kind'].startswith('tray_'))]
        for angle in np.linspace(0., np.pi / 2, 10):
            turn = Rotation.from_rotvec(angle * tangent).as_matrix()
            footprint = (edge + payload @ turn.T) @ world_turn.T
            low, high = footprint[:, :2].min(0), footprint[:, :2].max(0)
            if low[0] < routes.edge + TABLE_GAP or any(
                    np.all((high + TABLE_GAP >= obstacle_low[:2]) & (low - TABLE_GAP <= obstacle_high[:2]))
                    for obstacle_low, obstacle_high in obstacles):
                return None

        point, orient = press_pose(0., -.006)
        initial = positions[side]
        for seed in (initial, initial + [-.5, -.5, -.7, 0., 0., 0., 0.],
                     initial + [-1.1, -.85, .4, .8, -1.2, .4, 0.],
                     initial + [-.3, -1.2, .85, 1.2, -1.4, .6, 0.]):
            q = fit_pose(arm, point, orient, seed, tolerance=.006)
            if q is not None:
                grasp = {side: q, passive: ready[passive]}
                if clear_pair(grasp, yaw):
                    break
        else:
            return None
        point, orient = press_pose(0., .020)
        contact = press_step(grasp, point, orient)
        if contact is None:
            return None
        for elevation, gap in ((.075, .016), (.095, .020), (.065, .040), (.085, .045), (.055, .062)):
            high = press_step(contact, point + gap * normal + [0., 0., elevation], orient)
            approach = connect(positions, high, yaw) if high is not None else None
            if approach is not None:
                break
        else:
            return None

        tip = []
        seed = grasp
        for fraction in np.linspace(.125, 1., 8):
            point, orient = press_pose(fraction, -.006 - .024 * fraction)
            seed = press_step(seed, point, orient)
            if seed is None:
                return None
            tip.append({side: seed[side]})
        away = press_step(seed, point + .040 * tangent + [0., 0., .060], orient)
        home = connect(away, ready, yaw) if away is not None else None
        if home is None:
            return None
        return dict(pick_yaw=yaw, approach=[*approach, contact], grasp=[{side: grasp[side]}],
                    box_table_tip=tip, box_tip_retreat=[away, *home], box_tip_angle=angle_deg,
                    box_tip_push_side=side, box_tip_push_axis=(-normal).tolist(),
                    box_grasp_mode='open_palm_table_tip', box_grip_height_m=height,
                    grip=grip, preshape=preshape, item=item)

    _, tips = split_contact_vertices(parts[support], cable=True)
    fingertips = np.concatenate([vertices for finger, vertices in tips.items() if finger != 'thumb'])
    face_center = corners.mean(0)
    face_heading = np.arctan2(face_center[1], face_center[0])
    axis_heading = np.arctan2(axis[1], axis[0])
    axis_bearing = (axis_heading - face_heading + np.pi / 2) % np.pi - np.pi / 2
    grip_options = [(True, True, (top - bottom) / 2, axis_bearing - angle) for angle in (.72, .97, -.72)]
    grip_options.append((False, True, .145, -.22))
    grip_options.extend((False, False, height, delta) for height, delta in product((.125, .145), (-.22, 0., .22)))
    for narrow, exact, grip_height, pick_delta in grip_options:
        roll = exact and not narrow
        inward = thumb_clearance_roll[support][0] if roll else np.array([1., 0., 0.])
        depth = fingertips @ inward
        leading = fingertips[depth >= depth.max() - .004].mean(0)
        arms[support].pocket = leading + (depth.max() - leading @ inward) * inward
        across = np.array([-axis[1], axis[0]])
        forward_axis = axis if axis[0] >= 0. else -axis
        if narrow:
            center = np.r_[face_center, bottom + grip_height]
            directions, tilts, setback = (axis,), (-.40, 0.), 0.
        else:
            center = np.r_[object_center[:2], min(bottom + grip_height, top - .020)]
            directions = (across,) if exact else (across, axis)
            tilts = (0.,) if exact else (0., -.35, -.60)
            setback = min(.040, .34 * np.ptp(corners @ forward_axis)) if exact else .019
        high_z = max(center[2] + .105, routes.highest + center[2] - bottom + .022)
        pick_yaw = routes.yaw + np.arctan2(center[1], center[0]) + pick_delta
        preload = .006 if exact else BOX_PRELOAD
        local_center = routes.local(center - np.r_[setback * forward_axis, 0.], pick_yaw)
        prism = box @ rotation(routes.yaw - pick_yaw).T - local_center
        for world_direction in directions:
            normal = routes.local(np.r_[world_direction, 0.], pick_yaw)
            if normal[1] < 0:
                normal = -normal
                world_direction = -world_direction
            if normal[1] < .45:
                continue
            tangent = np.array([normal[1], -normal[0], 0.])
            faces = (corners - center[:2]) @ world_direction
            offsets = {'left': faces.max() * normal, 'right': faces.min() * normal}

            def goals(point, spacing=0., swing=0.):
                turn = rotation(swing)
                return {side: point + turn @ (offsets[side] + signs[side] * spacing * normal) for side in SIDES}

            for tilt in tilts:
                cant = 0. if narrow or tilt == 0. else .15
                orient = orientations(tangent, normal, tilt, cant, roll=roll)
                contact = goals(local_center, -preload)
                squeeze = {}
                for side, sign in signs.items():
                    seed = positions[side]
                    for elbow in (seed, seed + [-.5, sign * .5, sign * .7, 0., 0., 0., 0.],
                                  seed + [-1.1, sign * .85, -sign * .4, .8, sign * 1.2, .4, 0.]):
                        q = fit_pose(arms[side], contact[side], orient[side], elbow, tolerance=.005)
                        if q is not None and clear(side, q, pick_yaw):
                            squeeze[side] = q
                            break
                    else:
                        break
                if len(squeeze) < len(SIDES) or interarm_gap(arms, squeeze) < .002:
                    continue
                elevated = np.r_[local_center[:2], high_z]
                lift = []
                for fraction in (.15, .3, .5, .75, 1.):
                    level = (1 - fraction) * local_center + fraction * elevated
                    remaining_cant = max(0., 1. - 2. * fraction)
                    pivot = goals(level, -preload)
                    seed = lift[-1] if lift else squeeze
                    step = path(seed, [pivot], orientations(tangent, normal, tilt, cant * remaining_cant, roll=roll),
                                pick_yaw, relative(seed))
                    if step is None:
                        break
                    lift.extend(step)
                if len(lift) != 5:
                    continue
                # Solve backwards from the squeeze so the elbows keep their IK branch.
                above = local_center.copy()
                above[2] = max(local_center[2] + .065, top + .055)
                middle = (local_center + above) / 2.
                open_goals = [goals(point, .029) for point in (local_center, middle, above)]
                entry = path(squeeze, [goals(local_center, .008), *open_goals], orient, pick_yaw)
                if entry is None:
                    continue
                precontact, *entry = entry
                if cant:
                    straight = orientations(tangent, normal, tilt)
                    swivel_back = path(entry[-1], [goals(above, .029)], straight, pick_yaw)
                    if swivel_back is None:
                        continue
                    high_open = swivel_back[-1]
                else:
                    high_open = entry.pop()
                approach = connect(positions, high_open, pick_yaw)
                if approach is None:
                    continue
                approach.extend(reversed(entry))
                separation = relative(lift[-1])
                bearing = np.arctan2(item['release'][1], item['release'][0])
                tray_deltas = (0.,) if exact else (-.10, .05, -.15, .20)
                if exact and not narrow:
                    tray_deltas = (.06, 0.)
                for tray_delta in tray_deltas:
                    tray_yaw = routes.yaw + bearing + tray_delta
                    steps = 2 + int(abs(tray_yaw - pick_yaw) / .15)
                    if not all(link(lift[-1], lift[-1], yaw) for yaw in np.linspace(pick_yaw, tray_yaw, steps)):
                        continue
                    world_turn = rotation(tray_yaw - routes.yaw)
                    swing = -np.arctan2(tangent[1], tangent[0]) if exact else 0.
                    carry_tilt = 0. if narrow else tilt

                    def carried(fraction):
                        return (rotation(fraction * swing)
                                @ Rotation.from_rotvec(fraction * carry_tilt * normal).as_matrix())

                    end_turn = carried(1.)
                    # Contain the box even if friction lets it turn less than the palms.
                    envelope = np.concatenate([prism @ carried(fraction).T
                                               for fraction in np.linspace(0., 1., 6)])
                    projected = envelope @ world_turn.T
                    low_xy = tray[0, :2] - projected[:, :2].min(0) + TABLE_GAP
                    high_xy = tray[1, :2] - projected[:, :2].max(0) - TABLE_GAP
                    if np.any(low_xy > high_xy):
                        continue
                    center_offset = (world_turn @ end_turn @ prism.mean(0))[:2]
                    requested = np.asarray(item['release'][:2]) - center_offset
                    if item.get('prefer_far_placement'):
                        requested = np.clip(requested, low_xy, high_xy)
                        requested[1] = high_xy[1]
                    elif exact and not narrow and np.any((requested < low_xy) | (requested > high_xy)):
                        continue
                    requested = np.clip(requested, low_xy, high_xy)
                    near = np.clip([0., 0.], low_xy, high_xy)
                    if roll:
                        near[1] = requested[1]
                    heights = (.34, .36, .38) if exact else (.29, .25)
                    if item.get('prefer_far_placement'):
                        # Keep every grasp variant toward the far wall; do not
                        # silently fall back to the near corner of the tray.
                        placements = (((1., 0., .16), (1., 0., .18), (.95, 0., .16), (.85, 0., .16))
                                      if exact and not narrow else
                                      tuple((blend, 0., .18) for blend in
                                            (1., .95, .85, .70, .55, .40, .25, .15, .075)))
                    elif exact and not narrow:
                        placements = ((.65, .010, .16), (0., 0., .18), (1., 0., .18))
                    else:
                        blends = (1., .70, .35, 0.) if narrow else (1., .15, 0.)
                        placements = tuple((blend, 0., .18) for blend in blends)
                    for (blend, depth, wrist_pitch), height in product(placements, heights):
                        pivot = near + blend * (requested - near)
                        pivot[1] = min(high_xy[1], pivot[1] + depth)
                        site = np.r_[pivot + center_offset, item['release'][2]]
                        destination = routes.local(np.r_[pivot, max(high_z, height)], tray_yaw)
                        transfer = []
                        for fraction in (.2, .4, .6, .8, 1.):
                            hand_tilt = tilt - fraction * carry_tilt
                            if roll:
                                hand_tilt -= wrist_pitch * fraction
                            carry_turn = carried(fraction)
                            point = (1. - fraction) * elevated + fraction * destination
                            point[2] = max(point[2], routes.highest - (prism @ carry_turn.T)[:, 2].min() + .022)
                            hand_turn = rotation(fraction * swing)
                            orient = {side: hand_turn @ matrix
                                      for side, matrix in orientations(tangent, normal, hand_tilt, roll=roll).items()}
                            seed = transfer[-1] if transfer else lift[-1]
                            # Compare each rotating segment to the previous palm span.
                            grip_span = relative(seed) if exact else separation
                            phase = path(seed, [goals(point, -preload, fraction * swing)], orient, tray_yaw,
                                         grip_span)
                            if phase is None:
                                break
                            transfer.extend(phase)
                        else:
                            destination = point
                            margins = ((.006, .015, .035, .055, .080, .105, .130, .145,
                                        .16, .18, .20, .22, .24, .26, .28, .30, .32, np.inf) if exact
                                       else (.015, .05, .085, .12, .16, .20, .24, np.inf))
                            grip_span = relative(transfer[-1]) if exact else separation
                            for margin in margins:
                                down = destination.copy()
                                down[2] = min(destination[2], tray[1, 2] - envelope[:, 2].min() + margin)
                                release = path(transfer[-1],
                                               [goals((1. - t) * destination + t * down, -preload, swing)
                                                for t in (.33, .67, 1.)],
                                               orient, tray_yaw, grip_span)
                                if release is None:
                                    continue
                                # Lift both wrists at their release XY before moving them out of the tray.
                                release_pair = release[-1]
                                release_points = {side: arms[side].tool(release_pair[side])[:3, 3] for side in SIDES}
                                for rise in (.085, .065, .050, .035):
                                    up_step = np.array([0., 0., rise])
                                    raised = path(release_pair, [{side: point + up_step
                                                                 for side, point in release_points.items()}],
                                                  orient, tray_yaw, tray_contents=True)
                                    if raised is None:
                                        continue
                                    for distance in (.060, .095, .120):
                                        outward_shift = routes.local(np.array([-distance, 0., 0.]), tray_yaw)
                                        outward_points = {side: point + up_step + outward_shift
                                                          for side, point in release_points.items()}
                                        outward = path(raised[-1], [outward_points], orient, tray_yaw,
                                                        tray_contents=True)
                                        if outward is None:
                                            continue
                                        home = connect(outward[-1], ready, tray_yaw, tray_contents=True)
                                        if home is not None:
                                            retreat = [*raised, *outward]
                                            break
                                    else:
                                        continue
                                    break
                                else:
                                    continue
                                pickup_retreat = connect(lift[-1], ready, pick_yaw)
                                if pickup_retreat is None:
                                    continue
                                placed = (prism @ end_turn.T + down) @ world_turn.T
                                footprint = np.array([placed[:, :2].min(0), placed[:, :2].max(0)])
                                return dict(pick_yaw=pick_yaw, tray_yaw=tray_yaw, approach=approach,
                                            precontact=precontact, grasp=[squeeze],
                                            lift=lift, transfer=transfer, release=release, retreat=[*retreat, *home[:-1]],
                                            pickup_retreat=pickup_retreat,
                                            grip=grip, preshape=preshape, box_width=float(np.ptp(faces)),
                                            box_clear=clear_pair, box_preload=preload,
                                            box_grasp_mode='narrow_fork' if narrow else 'single_and_press',
                                            box_grip_side=active, box_press_side=support,
                                            box_contact_points={side: arm.pocket.tolist() for side, arm in arms.items()},
                                            box_grip_height_m=float(center[2] - bottom),
                                            box_release_height_m=float(down[2]),
                                            box_release_clearance_m=float(down[2] + (prism @ end_turn.T)[:, 2].min()
                                                                          - tray[1, 2]),
                                            release_footprint=footprint,
                                            item=dict(item, release=site.tolist()))
    return None


def snapshot(frames, angles, stamp, task, frame_angles, *, skip_box=False, selected=None):
    with measure('recognition_and_queue'):
        vision = TableVision()
        clouds = {}
        for camera, frame in frames.items():
            pose = frame_angles[camera]
            cloud = vision.points(camera, frame, pose)
            delta = pose[WAIST] - angles[WAIST]
            if delta:
                transform = rotation(delta)
                cloud['points'] = cloud['points'] @ transform.T
                cloud['pose'] = cloud['pose'].copy()
                cloud['pose'][:3, :] = transform @ cloud['pose'][:3, :]
            clouds[camera] = cloud
        scene = vision.observe_clouds(clouds, angles, stamp, task, audit_views=False)
        scene['table_geometry'] = table_boundary({'torso': frames['torso']}, angles, vision.rig, scene['plane'])
        points = np.concatenate([cloud['points'] for cloud in clouds.values()])
        retain_contact_surfaces(scene, points)
        scene['tray_contents'] = measure_tray_contents(scene, points)
        queue, omitted = queue_for(scene, task, points, skip_box=skip_box, selected=selected)
        show_queue(scene, angles[WAIST], queue, omitted, task=task)
        return scene, angles[WAIST], queue


def live(*, skip_box=False, resume=None, selected=None, force_box=False, flat_mouse_receiver=False,
         edge_batch=False, upright_box=False, fast_batch=False, cup_mouse=False, cup_trial_release=False,
         leave_cable_hand=False, ignore_force_checks=False, timed_batch=False, sim_budget=260.,
         adaptive_batch=False):
    import rclpy
    cancel_lookahead = Event()
    from bodyctrl_msgs.msg import CmdSetMotorPosition, CmdSetMotorSpeed, SetMotorPosition, SetMotorSpeed
    from bodyctrl_msgs.srv import SetForce, SetSpeed
    from cv_bridge import CvBridge
    from geometry_msgs.msg import WrenchStamped
    from message_filters import Subscriber, TimeSynchronizer
    from rclpy.duration import Duration
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import CameraInfo, Image, JointState
    from std_msgs.msg import Empty, String
    from rosgraph_msgs.msg import Clock

    def seconds(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    class WristContact(RuntimeError):
        def __init__(self, side, load, label):
            super().__init__(f'{side}: wrist load {load:.2f} N in {label}')
            self.side, self.load = side, load

    class WaypointAlignment(RuntimeError):
        pass

    class Robot:
        def __init__(self, node, executor):
            self.executor = executor
            self.angles = {}
            self.history = deque(maxlen=80)
            self.task = None
            self.done_pub = node.create_publisher(Empty, '/sim/task/done', 1)
            node.create_subscription(Clock, '/clock', self.clock, qos_profile_sensor_data)
            self.box_removed = bool(resume and resume.get('box_removed'))
            self.receive_guard = None
            self.images = None
            self.wrench, self.hand = {}, {}
            self.arm = {side: Arm(side) for side in SIDES}
            self.ids = {name: base + i for side, base in (('left', 11), ('right', 21))
                        for i, name in enumerate(self.arm[side].names)}
            self.motion = {group: node.create_publisher(CmdSetMotorPosition, f'/{group}/cmd_pos', 2)
                           for group in ('arm', 'head', 'waist')}
            self.brakes = {group: node.create_publisher(CmdSetMotorSpeed, f'/{group}/cmd_vel', 2)
                           for group in self.motion}
            self.hand_pub = {side: node.create_publisher(JointState, f'/inspire_hand/ctrl/{side}_hand', 2)
                             for side in SIDES}
            self.clients = {(side, key): node.create_client(typ, f'/inspire_hand/set_{key}/{side}_hand')
                            for side in SIDES for key, typ in (('force', SetForce), ('speed', SetSpeed))}
            node.create_subscription(JointState, '/joint_states', self.joints, qos_profile_sensor_data)
            node.create_subscription(String, '/sim/task', self.task_message,
                                     QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            for side in SIDES:
                node.create_subscription(WrenchStamped, f'/arm_6dof_{side}',
                                         lambda m, s=side: self.force(s, m), qos_profile_sensor_data)
                node.create_subscription(JointState, f'/inspire_hand/state/{side}_hand',
                                         lambda m, s=side: self.fingers(s, m), qos_profile_sensor_data)
            self.node = node
            self.cameras = []
            self.sync = None
            self.bridge = CvBridge()
            self.vision = TableVision()
            self.force_reference = {}
            self.load_since = {}
            self.withdraw_limits = {}
            self.box_hold = None
            self.box_direct = force_box or bool(resume and resume.get('force_box'))
            force = 1. if resume and resume.get('raised_box') else .8
            self.hand_force = {side: (force,) * 6 for side in SIDES}
            self.own_arm = deque(maxlen=20)
            self.own_arm_rates = deque(maxlen=24)
            self.external_arm = False
            self.started = False
            node.create_subscription(CmdSetMotorPosition, '/arm/cmd_pos', self.observe_arm_command, 10)
            node.create_subscription(CmdSetMotorSpeed, '/arm/cmd_vel', self.observe_arm_velocity, 10)

        def clock(self, msg):
            global SIM_TIME, SIM_START
            stamp = msg.clock.sec + msg.clock.nanosec * 1e-9
            if SIM_TIME is not None and stamp < SIM_TIME - .2:
                raise RuntimeError('simulation clock reset during the run')
            SIM_TIME = stamp
            if SIM_START is None:
                SIM_START = stamp
                event('sim_timer_started', budget_seconds=sim_budget)
            # Leave one simulator tick for the stop command at the deadline.
            if adaptive_batch and stamp - SIM_START >= sim_budget - .01:
                raise RuntimeError(f'simulation budget reached ({sim_budget:g} s)')

        def joints(self, msg):
            stamp = seconds(msg)
            if self.started and self.history and stamp < self.history[-1][0] - .2:
                raise RuntimeError('simulation time reset during sort')
            self.angles = dict(zip(msg.name, msg.position))
            self.history.append((stamp, time.monotonic(), self.angles))

        def task_message(self, msg):
            task = json.loads(msg.data)
            if self.started and self.task != task:
                raise RuntimeError('task changed during sort')
            self.task = task

        def force(self, side, msg):
            f = msg.wrench.force
            self.wrench[side] = time.monotonic(), np.array([f.x, f.y, f.z])

        def fingers(self, side, msg):
            self.hand[side] = (time.monotonic(), seconds(msg), dict(zip(msg.name, msg.position)), msg.effort)

        @staticmethod
        def signature(msg):
            return tuple((cmd.name, float(np.float32(cmd.pos)), float(np.float32(cmd.spd))) for cmd in msg.cmds)

        def observe_arm_command(self, msg):
            if self.started and self.signature(msg) not in self.own_arm:
                self.external_arm = True

        @staticmethod
        def rate_signature(msg):
            return tuple((cmd.name, float(np.float32(cmd.spd))) for cmd in msg.cmds)

        def observe_arm_velocity(self, msg):
            signature = self.rate_signature(msg)
            if signature in self.own_arm_rates:
                self.own_arm_rates.remove(signature)
            elif self.started:
                self.external_arm = True

        def poll(self, predicate, timeout, label, *, load_limit=6., timeout_error=RuntimeError):
            end = time.monotonic() + timeout
            while True:
                self.executor.spin_once(timeout_sec=.018)
                if self.external_arm:
                    raise RuntimeError('another controller commanded the arm')
                if self.started and self.force_reference and not ignore_force_checks:
                    now = time.monotonic()
                    if now - self.history[-1][1] > 1.25:
                        raise RuntimeError('lost joint feedback')
                    for side, baseline in self.force_reference.items():
                        stamp, force = self.wrench[side]
                        if now - stamp > 1.25:
                            raise RuntimeError(f'{side}: lost wrench feedback')
                        load = float(np.linalg.norm(force - baseline))
                        limit = self.withdraw_limits.get(side, load_limit)
                        hard_limit = max(9.5, load_limit, self.withdraw_limits.get(side, 0.))
                        if load <= limit:
                            self.load_since.pop(side, None)
                        elif load > hard_limit or stamp - self.load_since.setdefault(side, stamp) >= .075:
                            raise WristContact(side, load, label)
                    if self.box_hold is not None:
                        self.maintain_box_pressure()
                if self.receive_guard is not None:
                    self.receive_guard()
                if predicate():
                    return
                if time.monotonic() >= end:
                    raise timeout_error(f'timeout: {label}')

        def position(self, group, rows):
            msg = CmdSetMotorPosition(cmds=[SetMotorPosition(name=i, pos=float(q), spd=float(v)) for i, q, v in rows])
            if group == 'arm':
                self.own_arm.append(self.signature(msg))
            self.motion[group].publish(msg)

        def set_arms(self, goals, speed=.55, minimum_duration=.20):
            deltas = {side: abs(goal - self.q(side)) for side, goal in goals.items()}
            duration = max(minimum_duration, max(float(delta.max()) for delta in deltas.values()) / speed)
            end = self.history[-1][0] + duration
            if self.box_hold is not None:
                self.box_hold.update(goals=goals, end=end, speed=speed, shift=np.zeros(3))
                self.send_box_targets(retime=True)
                end = self.box_hold['end']
            else:
                self.arm_targets(goals, duration, speed)
            return end

        def arm_targets(self, goals, duration, speed):
            rows = []
            for side, goal in goals.items():
                names = self.arm[side].names
                delta = abs(goal - self.q(side))
                rows.extend((self.ids[k], q, np.clip(d / duration, .02, speed)) for k, q, d in zip(names, goal, delta))
            self.position('arm', rows)

        def q(self, side):
            return np.array([self.angles[key] for key in self.arm[side].names])

        def advance_to(self, stamp, label, *, load_limit=6.):
            duration = max(0., stamp - self.history[-1][0])
            self.poll(lambda: self.history[-1][0] >= stamp, duration / .30 + 2., label, load_limit=load_limit)

        def move(self, waypoints, speed=.55, *, load_limit=6., joint_tolerance=None, position_control=False,
                 palm_tolerance=None, hold_hand=None, continuous=False):
            for index, goals in enumerate(waypoints):
                through = continuous and timed_batch and index < len(waypoints) - 1
                if hold_hand:
                    for side, command in hold_hand.items():
                        self.command_hand(side, command)
                if fast_batch and not position_control and self.box_hold is None:
                    if palm_tolerance is None:
                        self.follow_waypoint(goals, .025 if through else .009, load_limit,
                                             speed=speed, stop_on_reach=not through)
                    else:
                        target = self.box_palms(goals)
                        self.follow_waypoint(goals, .025 if through else MOUSE_MOTION_TOLERANCE,
                                             load_limit, speed=speed, stop_on_reach=not through,
                                             reached=lambda: np.all(np.linalg.norm(
                                                 self.box_palms() - target, axis=1) <=
                                                 (max(palm_tolerance, .012) if through else palm_tolerance)))
                else:
                    self.advance_to(self.set_arms(goals, speed), 'arm motion', load_limit=load_limit)
                    if joint_tolerance is not None:
                        self.align_waypoint(goals, joint_tolerance, load_limit)
                if hold_hand and not through:
                    # Leave a position target on the loaded wrists between waypoints and during the waist turn.
                    self.set_arms(goals, .16)

        def align_waypoint(self, goals, tolerance, load_limit, contact=None):
            self.follow_waypoint(goals, tolerance, load_limit, contact=contact)

        def follow_waypoint(self, goals, tolerance, load_limit, *, contact=None, speed=None, reached=None,
                            stop_on_reach=True, timed_motion=False):
            ids = {side: [self.ids[name] for name in self.arm[side].names] for side in goals}
            initial_error = max(np.max(abs(goal - self.q(side))) for side, goal in goals.items())
            timeout = initial_error / speed / .30 + 3. if speed is not None else 4.
            finish = self.history[-1][0] + max(.20, initial_error / speed + .10) if timed_motion else None
            last = -np.inf

            def steer():
                nonlocal last
                if contact is not None and contact():
                    return True
                errors = {side: goal - self.q(side) for side, goal in goals.items()}
                worst_error = max(np.max(abs(error)) for error in errors.values())
                stamp = self.history[-1][0]
                if finish is not None:
                    if stamp >= finish:
                        return True
                elif worst_error <= tolerance and (reached is None or reached()):
                    return True
                if stamp - last >= .028:
                    rates = ({side: min(5., speed / max(worst_error, 1e-6)) * error for side, error in errors.items()}
                             if speed is not None else
                             {side: np.clip(3. * error, -.22, .22) for side, error in errors.items()})
                    self.velocity('arm', ((motor, rate) for side in goals for motor, rate in
                                          zip(ids[side], rates[side])))
                    last = stamp
                return False

            converged = False
            try:
                self.poll(steer, timeout, 'arm motion', load_limit=load_limit,
                          timeout_error=WaypointAlignment)
                converged = True
            finally:
                if np.isfinite(last) and not self.external_arm and (stop_on_reach or not converged):
                    if contact is not None and contact():
                        self.set_arms(goals, .16)
                    else:
                        self.velocity('arm', ((motor, 0.) for names in ids.values() for motor in names))

        def align_center(self, plan, routes, phase, object_offset=None):
            kind = plan['item']['kind']
            side = next(iter(plan['preshape']))
            arm = self.arm[side]
            arm.pocket = plan['pocket']
            if phase == 'pickup':
                waypoint, reference_yaw = plan['approach'][-1], plan['pick_yaw']
            else:
                waypoint, reference_yaw = plan['release'][-1], plan['tray_yaw']
            nominal = arm.tool(waypoint[side])[:3, :3]
            target = np.array(plan[f'{phase}_center'])
            if object_offset is not None:
                target[2] += nominal[2] @ object_offset
            return self.correct_center(side, routes, kind, phase, target,
                                       nominal, reference_yaw, plan['shape'],
                                       duration=3.5 if phase == 'pickup' else 4., object_offset=object_offset)

        def release_near_site(self, plan, routes, object_offset):
            item = plan['item']
            side = next(iter(plan['preshape']))
            arm = self.arm[side]
            arm.pocket = plan['pocket']
            q = self.q(side)
            yaw = self.angles[WAIST]
            pose = arm.tool(q)
            target = plan['placement_center'].copy()
            nominal = arm.tool(plan['release'][-1][side])
            target[2] += nominal[2, :3] @ object_offset
            center = routes.world(pose[:3, 3] + pose[:3, :3] @ object_offset, yaw)
            delta = target - center
            error = float(np.linalg.norm(delta))
            xy_error = float(np.linalg.norm(delta[:2]))
            low, high = np.asarray(item['tray']['bounds'])[:, :2]
            radius = np.max(np.ptp(item['target']['bounds'], axis=0)[:2]) / 2 + .008
            inside = np.all((low + radius <= center[:2]) & (center[:2] <= high - radius))
            if error > .012 or not inside or not routes.clear(side, q, plan['shape'], yaw):
                return False
            event('release_within_tray', kind=item['kind'], error_mm=round(1000 * error, 1),
                  xy_error_mm=round(1000 * xy_error, 1))
            return True

        def correct_center(self, side, routes, kind, phase, target, nominal, reference_yaw, shape,
                           *, tolerance=CENTER_TOLERANCE, duration=2., load_limit=6., object_offset=None,
                           touch=None, angular_tolerance=None, speed_limit=.30):
            arm = self.arm[side]
            point = arm.pocket if object_offset is None else arm.pocket + object_offset
            deadline = self.history[-1][0] + duration
            before = after = None
            last_command = -np.inf
            ids = [self.ids[name] for name in arm.names]
            moving = aligned = False

            def correct():
                nonlocal before, after, last_command, moving, aligned
                stamp = self.history[-1][0]
                yaw = self.angles[WAIST]
                q = self.q(side)
                frames = arm.frames(q)
                pose = frames[-1] @ arm.hand
                palm = pose[:3, 3] + pose[:3, :3] @ point
                error = routes.local(target, yaw) - palm
                after = float(np.linalg.norm(error))
                if before is None:
                    before = after
                orientation = rotation(reference_yaw - yaw) @ nominal
                angle = Rotation.from_matrix(orientation @ pose[:3, :3].T).as_rotvec()
                angle_ok = angular_tolerance is None or np.linalg.norm(angle) <= angular_tolerance
                gap = TABLE_GAP if touch is None else .002
                aligned = after <= tolerance and angle_ok and routes.clear(side, q, shape, yaw, touch=touch, gap=gap)
                if aligned or stamp >= deadline:
                    return True
                if stamp - last_command < .028:
                    return False
                rates = cartesian_rates(arm, frames, palm, error, angle, speed_limit)
                for scale in (1., .5, .25):
                    if routes.clear(side, q + .10 * scale * rates, shape, yaw, touch=touch, gap=gap):
                        break
                else:
                    if moving:
                        self.velocity('arm', ((i, 0.) for i in ids))
                        moving = False
                    last_command = stamp
                    return False
                self.velocity('arm', zip(ids, scale * rates))
                last_command = stamp
                moving = True
                return False

            try:
                with measure('encoder_alignment', kind=kind, phase=phase):
                    self.poll(correct, duration / .30 + 1., f'encoder alignment at {phase}', load_limit=load_limit)
            finally:
                if moving and not self.external_arm:
                    self.velocity('arm', ((i, 0.) for i in ids))
                if before is not None:
                    event('center_error', kind=kind, phase=phase, before_mm=round(1000 * before, 1),
                          after_mm=round(1000 * after, 1), tolerance_mm=1000 * tolerance)
            return bool(aligned)

        def measured_hand(self, side, routes, command):
            hand = routes.hands[side]
            angles = hand.nominal_angles(command)
            angles.update({key: self.angles[key] for key in angles if key in self.angles})
            return hand.vertices(angles)

        def contact_vertices(self, side, routes, command):
            return split_contact_vertices(self.measured_hand(side, routes, command))

        def guard_receiver(self, plan, routes, clearance=.003):
            side = plan['grip_side']
            arm = self.arm[side]
            meshes = body_geometry()[3][side]
            previous = None

            def guard():
                nonlocal previous
                if self.angles is previous:
                    return
                previous = self.angles
                frames = arm.frames(self.q(side))
                palm = frames[-1] @ arm.hand
                hand = np.concatenate(list(self.measured_hand(side, routes, plan['preshape'][side]).values()))
                vertices = [hand @ palm[:3, :3].T + palm[:3, 3]]
                vertices += [mesh @ frame[:3, :3].T + frame[:3, 3]
                             for mesh, frame in zip(meshes, frames[2:])]
                if LowerBody(self.angles).collides(np.concatenate(vertices), self.angles[WAIST], clearance=clearance):
                    raise RuntimeError(f'{side}: receiver within {1000 * clearance:g} mm of the leg')

            return guard

        def receiver_geometry(self, side, routes, command):
            yaw = self.angles[WAIST]
            arm = self.arm[side]
            pose = arm.wrist(self.q(side)) @ arm.hand
            normal = routes.local(routes.normal, yaw)
            hand = self.measured_hand(side, routes, command)
            vertices = np.concatenate(list(hand.values())) @ pose[:3, :3].T + pose[:3, 3]
            furthest = routes.world(vertices.T, yaw)[0].max()
            tips = np.concatenate([hand[name] for name in DISTAL])
            tip_top = (tips @ pose[:3, :3].T @ normal + pose[:3, 3] @ normal - routes.floor).max()
            cosine = (pose[:3, 0] @ normal) / np.linalg.norm(normal)
            tilt = np.rad2deg(np.arccos(np.clip(cosine, -1., 1.)))
            return float(furthest), float(tip_top), float(tilt)

        def position_right_receiver(self, routes, setup):
            settings = self.connect(('right',))
            self.poll(lambda: all(future.done() for _, _, future in settings), 4., 'hand settings')
            for key, side, future in settings:
                if not getattr(future.result(), f'{key}_accepted'):
                    raise RuntimeError(f'{side}: {key} setting rejected')
            side = 'right'
            start, yaw = np.asarray(setup['start']), setup['yaw']
            start_tolerance = .015 if 'right_clear' in setup else .035
            if abs(self.angles[WAIST] - yaw) > .01 or np.max(abs(self.q(side) - start)) > start_tolerance:
                raise RuntimeError('right receiver start pose changed')
            command = setup['preshape']
            plan = dict(grip_side=side, preshape={side: command})
            path = [{side: np.asarray(q)} for q in setup['path']]
            under_table = setup.get('under_table', False)
            from_under_table = setup.get('from_under_table', False)
            if from_under_table and np.max(abs(self.q('left') - setup['left_start'])) > .025:
                raise RuntimeError('left mouse pose changed before receiver adjustment')

            def alignment(q, yaw):
                pose = routes.arms[side].tool(q)
                center = routes.world(pose[:3, 3], yaw)
                fingers = routes.world(pose[:3, 2], yaw)
                return float(center[1]), float(np.arctan2(fingers[1], fingers[0]))

            start_y, _ = alignment(self.q(side), self.angles[WAIST])
            self.receive_guard = self.guard_receiver(plan, routes)
            if under_table or from_under_table:
                self.receive_guard = self.guard_mouse_receiver_under_table(routes, plan)
            if 'right_clear' in setup:
                self.receive_guard = self.guard_receiver(plan, routes, clearance=0.)
                self.move([{side: np.asarray(setup['right_clear'])}], speed=.20, joint_tolerance=.018)
                self.receive_guard = self.guard_receiver(plan, routes)
                self.receive_guard()
            self.command_hand(side, command)
            if setup.get('turn'):
                self.turn_for_right_receiver(routes, setup)
                yaw = setup['turn'][-1]['yaw']
            target_y, target_heading = alignment(path[-1][side], yaw)
            normal = routes.local(routes.normal, yaw)
            palm = routes.arms[side].tool(path[-1][side])[:3, 0]
            target_tilt = np.rad2deg(np.arccos(np.clip(palm @ normal / np.linalg.norm(normal), -1., 1.)))
            tilt_limit = max(8., target_tilt + 2.5)
            with measure('right_receiver_approach'):
                self.move(path, speed=.32, joint_tolerance=.025)
                self.align_waypoint(path[-1], .012, 6.)
                self.set_arms(path[-1], .12)
                edge = routes.edge + .015
                settled_since = None

                def receiver_settled():
                    nonlocal settled_since
                    furthest, tip_top, tilt = self.receiver_geometry(side, routes, command)
                    palm_y, heading = alignment(self.q(side), self.angles[WAIST])
                    heading_error = np.arctan2(np.sin(heading - target_heading), np.cos(heading - target_heading))
                    edge_ok = (abs(furthest - edge - .030) <= .008 and tip_top <= -.048 if under_table
                               else furthest <= edge and tip_top <= .004)
                    if (not edge_ok or tilt > tilt_limit
                            or abs(palm_y - target_y) >= .015 or abs(heading_error) >= np.deg2rad(6.)):
                        settled_since = None
                        return False
                    stamp = self.history[-1][0]
                    if settled_since is None:
                        settled_since = stamp
                    return stamp - settled_since >= .25

                self.poll(receiver_settled, 3., 'right receiving geometry')
            furthest, tip_top, tilt = self.receiver_geometry(side, routes, command)
            palm_y, heading = alignment(self.q(side), self.angles[WAIST])
            event('mouse_receiver_ready', side=side, tilt_deg=round(tilt, 1),
                  edge_clearance_mm=round(1000 * (edge - furthest), 1),
                  fingertip_top_mm=round(1000 * tip_top, 1),
                  shift_right_mm=round(1000 * (start_y - palm_y), 1),
                  finger_heading_deg=round(float(np.rad2deg(heading)), 1))
            self.receive_guard = None

        def guard_mouse_receiver_under_table(self, routes, plan):
            right_guard = self.guard_receiver(plan, routes, clearance=.006)
            edge = routes.edge + .012
            previous = None

            def guard():
                nonlocal previous
                if self.angles is previous:
                    return
                previous = self.angles
                right_guard()
                yaw = self.angles[WAIST]
                fore = routes.local(np.array([1., 0., 0.]), yaw)
                normal = routes.local(routes.normal, yaw)
                parts = {}
                for side in SIDES:
                    arm = self.arm[side]
                    palm = arm.wrist(self.q(side)) @ arm.hand
                    command = plan['preshape'].get(side, OPEN_LEFT)
                    parts[side] = [mesh @ palm[:3, :3].T + palm[:3, 3]
                                   for mesh in self.measured_hand(side, routes, command).values()]
                if any(_section_hits_table(mesh, fore, normal, routes.floor, edge, True)
                       for mesh in parts['right']):
                    raise RuntimeError('right: receiving hand reached the table slab')
                left = cKDTree(np.concatenate(parts['left']))
                if np.any(left.query(np.concatenate(parts['right']), distance_upper_bound=.006)[0] < .006):
                    raise RuntimeError('mouse: clearance between hands fell below 6 mm')

            return guard

        def turn_for_right_receiver(self, routes, setup):
            if np.max(abs(self.q('left') - setup['left_start'])) > .035:
                raise RuntimeError('left arm pose changed before torso turn')
            self.force_reference['left'] = self.wrench['left'][1].copy()
            right_guard = self.receive_guard
            left_guard = self.guard_receiver(dict(grip_side='left', preshape={'left': OPEN_LEFT}), routes)

            def guard_both():
                right_guard()
                left_guard()

            self.receive_guard = guard_both
            with measure('right_receiver_turn'):
                if 'left_clear' in setup:
                    self.move([{'left': np.asarray(setup['left_clear'])}], speed=.28,
                              load_limit=8.5, joint_tolerance=.025)
                    self.force_reference['left'] = self.wrench['left'][1].copy()
                    self.load_since.pop('left', None)
                self.move([{'right': np.asarray(q)} for q in setup.get('before_turn', ())],
                          speed=.34, load_limit=8.5, joint_tolerance=.025)
                self.move_torso_with_arms(setup['turn'])
            self.receive_guard = right_guard
            self.force_reference.pop('left')
            self.load_since.pop('left', None)

        def move_torso_with_arms(self, steps, *, arm_speed=.26):
            for step in steps:
                yaw = step['yaw']
                goals = {side: np.asarray(step[side]) for side in SIDES if side in step}
                yaw_delta = abs(self.angles[WAIST] - yaw)
                arm_delta = max((float(abs(self.q(side) - q).max()) for side, q in goals.items()), default=0.)
                duration = max(.22, yaw_delta / .16, arm_delta / arm_speed)
                end = self.history[-1][0] + duration
                if goals:
                    self.arm_targets(goals, duration, arm_speed)
                if yaw_delta > .001:
                    self.position('waist', [(31, yaw, max(.02, yaw_delta / duration))])
                self.advance_to(end, 'torso and arm motion', load_limit=8.5)
                self.poll(lambda: abs(self.angles[WAIST] - yaw) < .012 and all(
                    np.max(abs(self.q(side) - q)) < .035 for side, q in goals.items()),
                    2., 'torso and arm alignment', load_limit=8.5)

        def coordinated_move(self, steps, *, speed=.55, yaw_speed=.42, hold_hand=None):
            """Track a prevalidated path with one common rate for arms and waist."""
            if not steps:
                return
            try:
                for index, step in enumerate(steps):
                    goals, yaw = step['q'], step['yaw']
                    final = index == len(steps) - 1
                    tolerance = .009 if final else .022
                    target_palms = self.box_palms(goals)
                    delta = max(max(float(np.max(abs(q - self.q(side)))) / speed
                                    for side, q in goals.items()),
                                abs(yaw - self.angles[WAIST]) / yaw_speed)
                    last = -np.inf
                    if hold_hand:
                        for side, command in hold_hand.items():
                            self.command_hand(side, command)

                    def steer():
                        nonlocal last
                        errors = {side: q - self.q(side) for side, q in goals.items()}
                        waist_error = yaw - self.angles[WAIST]
                        if (max(float(np.max(abs(e))) for e in errors.values()) <= tolerance
                                and abs(waist_error) <= (.008 if final else .018)
                                and (not final or np.all(np.linalg.norm(
                                    self.box_palms() - target_palms, axis=1) <= .010))):
                            return True
                        stamp = self.history[-1][0]
                        if stamp - last >= .028:
                            duration = max(.001, abs(waist_error) / yaw_speed,
                                           max(float(np.max(abs(e))) / speed for e in errors.values()))
                            gain = min(5., 1. / duration)
                            self.velocity('arm', ((self.ids[name], gain * error)
                                for side, errors_at in errors.items()
                                for name, error in zip(self.arm[side].names, errors_at)))
                            self.velocity('waist', [(31, gain * waist_error)])
                            last = stamp
                        return False

                    self.poll(steer, delta / .30 + 4., 'coordinated transit', load_limit=9.5,
                              timeout_error=WaypointAlignment)
            finally:
                self.velocity('arm', ((self.ids[name], 0.) for side in SIDES for name in self.arm[side].names))
                self.velocity('waist', [(31, 0.)])
            self.set_arms(steps[-1]['q'], .20)
            self.position('waist', [(31, steps[-1]['yaw'], yaw_speed)])

        def prepare_mouse_press(self, routes, setup):
            if np.max(abs(self.q('right') - setup['right_start'])) > .035:
                raise RuntimeError('right receiving pose changed before mouse approach')
            command = setup['right_preshape']
            left_guard = self.receive_guard
            plan = dict(grip_side='right', preshape={'right': command})
            right_guard = self.guard_receiver(plan, routes)

            def guard_both():
                left_guard()
                right_guard()

            self.receive_guard = guard_both
            with measure('mouse_press_torso_turn'):
                self.move([{'left': np.asarray(setup['left_clear'])}], speed=.35,
                          load_limit=8.5, joint_tolerance=.025)
                self.force_reference['left'] = self.wrench['left'][1].copy()
                self.load_since.pop('left', None)
                if 'right_pre' in setup:
                    self.move([{'right': np.asarray(setup['right_pre'])}], speed=.22,
                              load_limit=8.5, joint_tolerance=.025)
                self.move_torso_with_arms(setup['turn'])

            def receiving():
                yaw = self.angles[WAIST]
                pose = self.arm['right'].tool(self.q('right'))
                fingers = routes.world(pose[:3, 2], yaw)
                heading = np.rad2deg(np.arctan2(fingers[1], fingers[0]))
                furthest, tip_top, tilt = self.receiver_geometry('right', routes, command)
                edge_gap = routes.edge + .015 - furthest
                return float(heading), edge_gap, tip_top, tilt

            def ready():
                heading, gap, tip_top, _ = receiving()
                return abs(heading) <= 4. and gap >= -.002 and tip_top <= .004

            self.poll(ready, 2., 'right palm position after torso turn', load_limit=8.5)
            heading, gap, tip_top, tilt = receiving()
            event('mouse_receiver_retained', waist_deg=round(float(np.rad2deg(self.angles[WAIST])), 2),
                  heading_deg=round(heading, 2), tilt_deg=round(tilt, 1),
                  edge_clearance_mm=round(gap * 1000, 1), fingertip_top_mm=round(tip_top * 1000, 1))

        def press_mouse_left(self, routes, setup):
            side = 'left'
            settings = self.connect(SIDES if setup.get('turn') else (side,))
            self.poll(lambda: all(future.done() for _, _, future in settings), 4., 'left hand settings',
                      load_limit=8.5)
            for key, _, future in settings:
                if not getattr(future.result(), f'{key}_accepted'):
                    raise RuntimeError(f'left: {key} setting rejected')
            yaw = setup['yaw']
            start_yaw = setup.get('start_yaw', yaw)
            if abs(self.angles[WAIST] - start_yaw) > .01 or np.max(abs(self.q(side) - setup['start'])) > .035:
                raise RuntimeError('left mouse press start pose changed')
            command = setup['preshape']
            plan = dict(grip_side=side, preshape={side: command})
            self.receive_guard = self.guard_receiver(plan, routes)
            if 'right_lock' in setup:
                reference = setup['right_lock']
                if np.max(abs(self.q('right') - reference['q'])) > .025:
                    raise RuntimeError('right receiving pose changed before mouse press')
                preshape = {'left': command, 'right': setup['right_preshape']}
                guard_left = self.receive_guard
                guard_right = self.guard_mouse_receiver_under_table(
                    routes, dict(grip_side='right', preshape=preshape))
                hold_right = self.mouse_receiver_lock(routes, preshape, reference)

                def guard_and_hold():
                    guard_left()
                    guard_right()
                    hold_right()

                self.receive_guard = guard_and_hold
            if setup.get('turn'):
                self.prepare_mouse_press(routes, setup)
            open_after = setup.get('open_after', -1)
            path = [{side: np.asarray(q)} for q in setup['path']]
            if open_after == -1:
                self.grip({side: command}, load_limit=8.5)
            with measure('mouse_press_approach'):
                for index, goal in enumerate(path):
                    self.move([goal], speed=.47, load_limit=8.5, joint_tolerance=.035)
                    if index == open_after:
                        self.grip({side: command}, load_limit=8.5)
                self.align_waypoint(path[-1], .016, 8.5)
                self.set_arms(path[-1], .14)

            arm = self.arm[side]
            normal = routes.local(routes.normal, yaw)
            normal /= np.linalg.norm(normal)
            center = routes.local(np.asarray(setup['mouse_center']), yaw)
            pad = np.asarray(setup['palm_pad'])
            hover_stamp = self.wrench[side][0]
            hover_forces = []

            def hover_sample():
                nonlocal hover_stamp
                stamp, force = self.wrench[side]
                if stamp > hover_stamp:
                    hover_stamp = stamp
                    hover_forces.append(force)
                return len(hover_forces) >= 5

            self.poll(hover_sample, 1., 'left hover force', load_limit=8.5)
            reference = np.median(hover_forces, axis=0)
            self.force_reference[side] = reference
            self.load_since.pop(side, None)

            def contact():
                q = self.q(side)
                wrist = arm.wrist(q)
                palm = wrist @ arm.hand
                delta = palm[:3, 3] + palm[:3, :3] @ pad - center
                pressure = normal @ wrist[:3, :3] @ (self.wrench[side][1] - reference)
                error = np.linalg.norm(delta - (delta @ normal) * normal)
                return float(pressure), float(error), float((delta + center) @ normal)

            pressure_since = None

            def pressed():
                nonlocal pressure_since
                pressure, error, height = contact()
                if pressure < 1.2 or error > .022 or abs(height - setup['mouse_top']) > .016:
                    pressure_since = None
                    return False
                stamp = self.wrench[side][0]
                if pressure_since is None:
                    pressure_since = stamp
                return stamp - pressure_since >= .10

            with measure('mouse_press_descent'):
                for q in setup['press']:
                    self.advance_to(self.set_arms({side: np.asarray(q)}, .12), 'left mouse press')
                    try:
                        self.poll(pressed, .32, 'left palm contact', timeout_error=WaypointAlignment)
                    except WaypointAlignment:
                        continue
                    pressure, error, _ = contact()
                    event('mouse_pressed', side=side, pressure_n=round(pressure, 2),
                          palm_error_mm=round(error * 1000, 1))
                    return
            pressure, error, height = contact()
            event('mouse_press_unconfirmed', pressure_n=round(pressure, 2), palm_error_mm=round(error * 1000, 1),
                  above_mouse_mm=round(1000 * (height - setup['mouse_top']), 1))

        def slide_mouse_left(self, routes, setup, item):
            self.connect(())
            if abs(self.angles[WAIST] - setup['yaw']) > .015 or any(
                    np.max(abs(self.q(side) - setup['start'][side])) > .035 for side in SIDES):
                raise RuntimeError('mouse slide start pose changed')
            self.force_reference = {side: self.wrench[side][1].copy() for side in SIDES}
            guards = [self.guard_receiver(dict(grip_side=side, preshape=setup['preshape']), routes)
                      for side in SIDES]

            def guard_both():
                for guard in guards:
                    guard()

            self.receive_guard = guard_both
            left = self.arm['left']
            pad = np.asarray(setup['palm_pad'])
            normal = routes.normal / np.linalg.norm(routes.normal)

            def palm_state(side, point):
                arm = self.arm[side]
                wrist = arm.wrist(self.q(side))
                pose = wrist @ arm.hand
                yaw = self.angles[WAIST]
                position = routes.world(pose[:3, 3] + pose[:3, :3] @ point, yaw)
                force = routes.world(wrist[:3, :3] @ self.wrench[side][1], yaw)
                return position, float(normal @ force)

            start, press_start = palm_state('left', pad)
            _, right_baseline = palm_state('right', np.zeros(3))
            right_baseline = setup.get('right_unloaded_force_n', right_baseline)
            if 'release_left' in setup:
                self.grip({'left': np.asarray(setup['release_left'])}, load_limit=8.5)
                self.poll(lambda: all(self.hand['left'][2].get(name, 0.) >= .94 for name in FINGERS[:4]),
                          2., 'left fingers clear for mouse slide', load_limit=8.5)
            if setup.get('cover_left'):
                self.grip({'left': np.asarray(setup['preshape']['left'])}, load_limit=8.5)
                settled_after = self.hand['left'][1] + .40
                self.poll(lambda: self.hand['left'][1] >= settled_after, 2., 'left fingers around mouse',
                          load_limit=8.5)
            steps = setup['steps']
            while steps and steps[0].get('phase') in ('relief', 'turn'):
                phase = steps[0]['phase']
                self.move_torso_with_arms(steps[:1], arm_speed=.20)
                _, pressure = palm_state('left', pad)
                event('mouse_wrist_relief' if phase == 'relief' else 'mouse_body_turn',
                      normal_force_n=round(pressure, 2))
                steps = steps[1:]
            lock = setup.get('right_lock')
            if lock is not None:
                if setup.get('right_restore'):
                    self.move_torso_with_arms(setup['right_restore'], arm_speed=.20)
                hold_right = self.mouse_receiver_lock(routes, setup['preshape'], lock)
                under_table = lock.get('under_table', False)
                guard_right = (self.guard_mouse_receiver_under_table(
                    routes, dict(grip_side='right', preshape=setup['preshape'])) if under_table else None)
                pressure_since = None

                def guard_and_hold():
                    nonlocal pressure_since
                    guard_both()
                    if guard_right is not None:
                        guard_right()
                    hold_right()
                    if setup.get('max_pressure_n') is not None:
                        _, pressure = palm_state('left', pad)
                        stamp = self.history[-1][0]
                        if pressure <= setup['max_pressure_n']:
                            pressure_since = None
                        elif pressure_since is None:
                            pressure_since = stamp
                        elif stamp - pressure_since >= .08:
                            raise WristContact('left', pressure, 'mouse slide pressure')

                self.receive_guard = guard_and_hold
                settle_until = self.history[-1][0] + 2.2

                def receiver_settled():
                    error, angle = hold_right(feedback_only=True)
                    return ((error <= .004 and angle <= np.deg2rad(1.5))
                            or (not under_table and self.history[-1][0] >= settle_until))

                self.poll(receiver_settled, 7., 'right palm encoder lock', load_limit=8.5)

            def push_step(step):
                if not setup.get('pressure_compliance'):
                    self.move_torso_with_arms([step], arm_speed=.20)
                    return
                nominal = np.asarray(step['left'])
                frames = left.frames(nominal)
                palm = frames[-1] @ left.hand
                point = palm[:3, 3] + palm[:3, :3] @ pad
                up = routes.local(normal, self.angles[WAIST])
                rise = cartesian_rates(left, frames, point, .006 * up, np.zeros(3), .3) / 5.5
                stamp = self.history[-1][0]
                finish = stamp + max(.22, float(np.max(abs(self.q('left') - nominal))) / .22)
                last_sent, previous = -np.inf, -1.

                def reached():
                    nonlocal last_sent, previous
                    now = self.history[-1][0]
                    _, pressure = palm_state('left', pad)
                    fraction = float(np.clip((pressure - 4.5) / 2.5, 0., 1.))
                    target = np.clip(nominal + fraction * rise, *left.limits)
                    if now - last_sent >= .08 and (abs(fraction - previous) > .15 or now >= finish):
                        self.arm_targets({'left': target}, max(.16, finish - now), .22)
                        last_sent, previous = now, fraction
                    if now < finish:
                        return False
                    measured = left.wrist(self.q('left')) @ left.hand
                    error = measured[:3, 3] + measured[:3, :3] @ pad - point
                    vertical = float(error @ up)
                    horizontal = np.linalg.norm(error - vertical * up)
                    angle = Rotation.from_matrix(palm[:3, :3].T @ measured[:3, :3]).magnitude()
                    return (horizontal <= setup.get('step_tolerance_m', .006)
                            and abs(vertical) <= .008 and angle <= np.deg2rad(5.))

                reached()
                self.poll(reached, (finish - stamp) / .30 + 4., 'compliant mouse slide', load_limit=8.5)

            with measure('mouse_slide'):
                for index, step in enumerate(steps, start=1):
                    if lock is not None:
                        step = {key: value for key, value in step.items() if key != 'right'}
                    push_step(step)
                    _, pressure = palm_state('left', pad)
                    feedback = {}
                    if lock is not None:
                        error, angle = hold_right(feedback_only=True)
                        feedback = dict(right_error_mm=round(error * 1000, 1),
                                        right_error_deg=round(np.rad2deg(angle), 2))
                    event('mouse_slide_step', step=index, normal_force_n=round(pressure, 2), **feedback)
            end, press_end = palm_state('left', pad)
            event('mouse_slide_ended', palm_travel_mm=round(1000 * float(np.linalg.norm(end[:2] - start[:2])), 1),
                  normal_force_start_n=round(press_start, 2), normal_force_end_n=round(press_end, 2))
            plan = dict(item=item, grip_side='right', preshape=setup['preshape'],
                        receive_point=setup['receive_point'], receive_camera=setup.get('receive_camera', 'head'),
                        catch_mouse='catch' in setup)
            aligned = bool(setup.get('catch_now', False))
            if 'catch' in setup and not aligned:
                if setup['steps']:
                    aligned = self.edge_center_aligned(plan, routes)
                elif 'measured_catch_error_m' in setup:
                    error = setup['measured_catch_error_m']
                    aligned = 0. <= error <= .015
                    event('edge_center', kind='mouse', receiver='right', error_mm=round(1000 * error, 1),
                          tolerance_mm=15., source='synchronized_capture')
            if lock is not None:
                self.receive_guard = guard_both
                error, angle = hold_right(feedback_only=True)
                self.set_arms({'right': self.q('right')}, .12)
                event('mouse_receiver_lock', error_mm=round(error * 1000, 1), angle_deg=round(np.rad2deg(angle), 2))
            if aligned and 'catch' in setup:
                command = np.asarray(setup['catch'])
                self.grip({'right': command}, load_limit=8.5)
                settled_after = self.hand['right'][1] + .40
                self.poll(lambda: self.hand['right'][1] >= settled_after, 2., 'right mouse grip feedback',
                          load_limit=8.5)
                opposed, _, positions, loads = self.grip_state('right', command)
                contact = (positions[:4] - command[:4] > .035) & (loads[:4] > .15)
                _, right_pressure = palm_state('right', np.zeros(3))
                support = abs(right_pressure - right_baseline)
                engaged = opposed or np.count_nonzero(contact) >= 2 or bool(contact.any() and support >= .7)
                event('mouse_caught', engaged=engaged, measured=positions.round(3).tolist(),
                      efforts=loads.round(3).tolist(), support_change_n=round(support, 2))
                if engaged and 'carry' in setup:
                    self.carry_mouse_shell(setup['carry'])

        def carry_mouse_shell(self, carry):
            with measure('mouse_pair_lift'):
                self.move_torso_with_arms(carry['lift'], arm_speed=.24)
            opposed, _, positions, loads = self.grip_state('right', np.asarray(carry['right_grip']))
            contact = (positions[:4] - carry['right_grip'][:4] > .035) & (loads[:4] > .15)
            if not opposed and np.count_nonzero(contact) < 2:
                event('mouse_pair_hold_lost')
                return
            if carry.get('transfer'):
                with measure('mouse_pair_transfer'):
                    self.move_torso_with_arms(carry['transfer'], arm_speed=.32)
            if carry.get('release'):
                with measure('mouse_pair_release'):
                    self.grip({side: np.asarray(q) for side, q in carry['release'].items()}, load_limit=8.5)
                    self.move_torso_with_arms(carry['retreat'], arm_speed=.32)

        def retry_mouse_shell(self, routes, setup):
            self.connect(())
            yaw = self.angles[WAIST]
            if abs(yaw - setup['yaw']) > .015 or any(
                    np.max(abs(self.q(side) - setup['start'][side])) > .035 for side in SIDES):
                raise RuntimeError('mouse retry start pose changed')
            self.force_reference = {side: self.wrench[side][1].copy() for side in SIDES}
            normal = routes.local(routes.normal, yaw)
            normal /= np.linalg.norm(normal)
            pad = {side: np.asarray(setup['pads'][side]) for side in SIDES}
            preshape = setup['preshape']
            guards = [self.guard_receiver(dict(grip_side=side, preshape=preshape), routes) for side in SIDES]

            def guard_pair():
                for guard in guards:
                    guard()

            def pose_of(side, frames):
                pose = frames[-1] @ self.arm[side].hand
                return pose[:3, 3] + pose[:3, :3] @ pad[side], pose[:3, :3]

            def left_load():
                wrist = self.arm['left'].wrist(self.q('left'))
                return float(normal @ wrist[:3, :3] @ self.wrench['left'][1])

            unloaded = left_load()
            self.receive_guard = guard_pair
            for side, command in setup.get('close', {}).items():
                self.command_hand(side, command)
            regrasp_index = -1
            for index, step in enumerate(setup.get('regrasp', ())):
                pressing = step.get('phase') == 'press'
                if pressing and left_load() - unloaded >= 2.:
                    break
                goals = {'left': np.asarray(step['left'])}
                self.advance_to(self.set_arms(goals, .26), 'mouse upper approach', load_limit=8.5)
                self.align_waypoint(goals, .025, 8.5,
                                    contact=(lambda: left_load() - unloaded >= 2.) if pressing else None)
                self.set_arms({'left': self.q('left')}, .16)
                regrasp_index = index
            event('mouse_upper_regrasp', step=regrasp_index, force_n=round(left_load() - unloaded, 2))
            if 'upper_grip' in setup:
                upper = np.asarray(setup['upper_grip'])
                self.grip({'left': upper}, load_limit=8.5)
                opposed, _, positions, efforts = self.grip_state('left', upper)
                event('mouse_upper_grip', engaged=opposed, force_n=round(left_load() - unloaded, 2),
                      measured=positions.round(3).tolist(), efforts=efforts.round(3).tolist())
                if not opposed:
                    return
            if 'receiver_under' in setup:
                receiver = setup['receiver_under']
                if np.max(abs(self.q('left') - receiver['start']['left'])) > .018:
                    event('mouse_receiver_blocked', reason='upper hold outside validated pose')
                    return
                right_open = np.asarray(receiver['grip'])
                for step in receiver['steps']:
                    if step['phase'] == 'open_outside':
                        self.grip({'right': right_open}, load_limit=8.5)
                    else:
                        self.move_torso_with_arms([step], arm_speed=.26)
                right = self.arm['right']
                palm = right.wrist(self.q('right')) @ right.hand
                hand = self.measured_hand('right', routes, right_open)
                tips = np.concatenate([hand[name] for name in DISTAL]) @ palm[:3, :3].T + palm[:3, 3]
                depth = routes.world(tips.T, yaw)[0].max() - routes.edge - .015
                tip_top = (tips @ routes.local(routes.normal, yaw) - routes.floor).max()
                event('mouse_receiver_under', depth_mm=round(1000 * depth, 1),
                      tip_height_mm=round(1000 * tip_top, 1))
                if 'receiver_extract' not in setup:
                    return
                self.move_torso_with_arms(setup['receiver_extract'], arm_speed=.26)
            if 'rise_m' not in setup:
                return

            start = {side: pose_of(side, self.arm[side].frames(self.q(side))) for side in SIDES}
            left_offset = start['left'][0] - start['right'][0]
            goal = start['right'][0] + setup['rise_m'] * normal
            physical_edge = routes.edge + .015
            last = self.history[-1][0] - .08
            inset = 0.
            measured = {}

            def carry():
                nonlocal last, inset
                now = self.history[-1][0]
                dt = now - last
                if dt < .06:
                    return False
                last = now
                guard_pair()
                q = {side: self.q(side) for side in SIDES}
                frames = {side: self.arm[side].frames(q[side]) for side in SIDES}
                pose = {side: pose_of(side, frames[side]) for side in SIDES}
                pressure = left_load() - unloaded
                if pressure > 6.:
                    raise WristContact('left', pressure, 'mouse shell squeeze')
                inset = float(np.clip(inset + .003 * min(dt, .12) * (2.4 - pressure), -.004, .012))
                targets = {'left': pose['right'][0] + left_offset - inset * normal,
                           'right': goal if pressure >= 1. else pose['right'][0]}
                rates = {}
                for side in SIDES:
                    point, orientation = pose[side]
                    angle = Rotation.from_matrix(start[side][1] @ orientation.T).as_rotvec()
                    rates[side] = cartesian_rates(self.arm[side], frames[side], point,
                                                  targets[side] - point, angle, .32)
                meshes = {side: np.concatenate(list(self.measured_hand(side, routes, preshape[side]).values()))
                          for side in SIDES}
                for scale in (1., .5, .25, 0.):
                    vertices = {}
                    for side in SIDES:
                        arm = self.arm[side]
                        trial = np.clip(q[side] + .10 * scale * rates[side], *arm.limits)
                        palm = arm.wrist(trial) @ arm.hand
                        vertices[side] = meshes[side] @ palm[:3, :3].T + palm[:3, 3]
                    points = np.concatenate(list(vertices.values()))
                    over = routes.world(points.T, yaw)[0] >= physical_edge
                    height = points @ routes.local(routes.normal, yaw) - routes.floor
                    if (not np.any(over & (height < -.002))
                            and cKDTree(vertices['left']).query(vertices['right'])[0].min() >= .005):
                        break
                measured.update(pressure_n=round(pressure, 2),
                                left_mm=round(1000 * float(normal @ (pose['left'][0] - start['left'][0])), 1),
                                right_mm=round(1000 * float(normal @ (pose['right'][0] - start['right'][0])), 1))
                reached = (np.linalg.norm(goal - pose['right'][0]) < .005
                           and np.linalg.norm(targets['left'] - pose['left'][0]) < .007 and pressure >= 1.)
                self.velocity('arm', [(self.ids[name], 0. if reached else scale * rate)
                                      for side in SIDES for name, rate in zip(self.arm[side].names, rates[side])])
                return reached

            try:
                self.poll(carry, 10., 'coupled mouse lift', load_limit=8.5)
            finally:
                event('mouse_coupled_lift', **measured)
            self.set_arms({side: self.q(side) for side in SIDES}, .12)

        def mouse_receiver_lock(self, routes, preshape, reference):
            side = 'right'
            arm = self.arm[side]
            left = self.arm['left']
            nominal = arm.wrist(reference['q']) @ arm.hand
            target = routes.world(nominal[:3, 3], reference['yaw']) + reference.get('shift', np.zeros(3))
            right_parts = tuple(self.measured_hand(side, routes, preshape[side]).values())
            right_shape = np.concatenate(right_parts)
            left_shape = np.concatenate(list(self.measured_hand('left', routes, preshape['left']).values()))
            right_links = body_geometry()[3][side]
            edge = routes.edge + .015
            under_table = reference.get('under_table', False)
            ids = [self.ids[name] for name in arm.names]
            last = -np.inf

            def hold(feedback_only=False):
                nonlocal last
                stamp = self.history[-1][0]
                if not feedback_only and stamp - last < .07:
                    return
                yaw = self.angles[WAIST]
                q = self.q(side)
                frames = arm.frames(q)
                pose = frames[-1] @ arm.hand
                point = pose[:3, 3]
                error = routes.local(target, yaw) - point
                orient = rotation(reference['yaw'] - yaw) @ nominal[:3, :3]
                angle = Rotation.from_matrix(orient @ pose[:3, :3].T).as_rotvec()
                distance, rotation_error = np.linalg.norm(error), np.linalg.norm(angle)
                if feedback_only:
                    return float(distance), float(rotation_error)
                last = stamp
                rates = np.zeros(len(q))
                if distance > .003 or rotation_error > np.deg2rad(1.):
                    velocity = cartesian_rates(arm, frames, point, error, angle, .17)
                    left_pose = left.wrist(self.q('left')) @ left.hand
                    other = cKDTree(left_shape @ left_pose[:3, :3].T + left_pose[:3, 3])
                    body = LowerBody(self.angles)
                    for scale in (1., .5, .25):
                        trial = q + .14 * scale * velocity
                        if np.any(trial < arm.limits[0]) or np.any(trial > arm.limits[1]):
                            continue
                        chain = arm.frames(trial)
                        palm = chain[-1] @ arm.hand
                        hand = right_shape @ palm[:3, :3].T + palm[:3, 3]
                        gap = .006 if under_table else .005
                        if other.query(hand, distance_upper_bound=gap)[0].min() < gap:
                            continue
                        if under_table:
                            fore = routes.local(np.array([1., 0., 0.]), yaw)
                            normal = routes.local(routes.normal, yaw)
                            if any(_section_hits_table(mesh @ palm[:3, :3].T + palm[:3, 3],
                                                       fore, normal, routes.floor, edge - .003, True)
                                   for mesh in right_parts):
                                continue
                        elif routes.world(hand.T, yaw)[0].max() > edge - .004:
                            continue
                        vertices = [hand, *(v @ f[:3, :3].T + f[:3, 3] for v, f in zip(right_links, chain[2:]))]
                        if not body.collides(np.concatenate(vertices), yaw, clearance=.006):
                            rates = scale * velocity
                            break
                self.velocity('arm', zip(ids, rates))

            return hold

        def press_mouse_fingers(self, setup):
            side = 'left'
            self.connect(())
            if abs(self.angles[WAIST] - setup['yaw']) > .015 or np.max(abs(self.q(side) - setup['start'])) > .035:
                raise RuntimeError('left palm moved before finger press')
            command = np.asarray(setup['command'])
            self.force_reference = {side: self.wrench[side][1].copy()}
            if 'target' in setup:
                self.move([{side: np.asarray(setup['target'])}], speed=.14, load_limit=8.5, joint_tolerance=.012)
            _, _, _, before = self.hand[side]
            self.grip({side: command}, load_limit=8.5)
            settled_after = self.hand[side][1] + .40
            self.poll(lambda: self.hand[side][1] >= settled_after, 2., 'left finger press feedback', load_limit=8.5)
            _, _, positions, efforts = self.hand[side]
            stroke = np.array([positions.get(name, np.nan) for name in FINGERS])
            loads = abs(np.asarray(efforts))
            loaded = loads[:4] > .25 if len(loads) == len(FINGERS) else np.zeros(4, dtype=bool)
            blocked = (stroke[:4] > command[:4] + .02) & loaded
            event('mouse_fingers_pressed', engaged=np.flatnonzero(blocked).tolist(),
                  command=command.tolist(), measured=stroke.round(3).tolist(), efforts=loads.round(3).tolist(),
                  effort_before=np.abs(np.asarray(before)).round(3).tolist())

        def fingertip_gaps(self, side, routes, preshape):
            _, tips = self.contact_vertices(side, routes, preshape)
            pose = self.arm[side].wrist(self.q(side)) @ self.arm[side].hand
            normal = routes.local(routes.normal, self.angles[WAIST])
            return {
                finger: float((vertices @ pose[:3, :3].T @ normal + pose[:3, 3] @ normal - routes.floor).min())
                for finger, vertices in tips.items()
            }

        def touch_table(self, plan, routes, side):
            kind = plan['item']['kind']
            touch = plan['touch']
            arm = self.arm[side]
            arm.pocket = plan['pocket']
            tips = np.concatenate(list(touch['tips'].values()))
            reference_yaw = plan['pick_yaw']
            if touch.get('scoop'):
                pose = arm.tool(touch['waypoints'][0])
                target = routes.world(pose[:3, 3], reference_yaw)
                return self.correct_center(side, routes, kind, 'scoop_entry', target, pose[:3, :3],
                                           reference_yaw, touch['solid'], tolerance=.0015, duration=5.,
                                           touch=tips, angular_tolerance=.020, speed_limit=.19)
            force_before = self.wrench[side][1]
            for index, q in enumerate(touch['waypoints']):
                thumb_first = len(touch['waypoints']) == 2 and index == 0
                phase = 'touch_thumb' if thumb_first else 'touch_thumb_and_index'
                pose = arm.tool(q)
                target = routes.world(pose[:3, 3], reference_yaw)
                offset = None
                if index:
                    normal = routes.local(routes.normal, reference_yaw)
                    thumb = touch['tips']['thumb']
                    offset = thumb[np.argmin(thumb @ pose[:3, :3].T @ normal)] - arm.pocket
                    target += routes.world(pose[:3, :3] @ offset, reference_yaw)
                    measured = arm.tool(self.q(side))
                    anchor = measured[:3, 3] + measured[:3, :3] @ offset
                    target[:2] = routes.world(anchor, self.angles[WAIST])[:2]
                aligned = self.correct_center(side, routes, kind, phase, target, pose[:3, :3],
                                              reference_yaw, touch['solid'], tolerance=.0015 if index else .002,
                                              duration=5., touch=tips, angular_tolerance=.009 if index else .024,
                                              speed_limit=.19, object_offset=offset)
                gaps = self.fingertip_gaps(side, routes, plan['preshape'][side])
                thumb_down = -.004 <= gaps['thumb'] <= .003
                index_down = -.004 <= gaps['index'] <= .003
                contacted = thumb_down and (gaps['index'] > .003 if thumb_first else index_down)
                load = float(np.linalg.norm(self.wrench[side][1] - force_before))
                event('table_contact', kind=kind, phase=phase, aligned=aligned, in_contact_band=contacted,
                      gaps_mm={key: round(1000 * gap, 1) for key, gap in gaps.items()}, force_change_n=round(load, 2))
                if not aligned or not contacted:
                    return False
            return True

        def close_at_table(self, plan, routes, side, completed=0):
            kind = plan['item']['kind']
            arm = self.arm[side]
            arm.pocket = plan['pocket']
            steps = plan['touch']['close']
            yaw = plan['pick_yaw']
            if plan['touch'].get('scoop'):
                return self.scoop_cable(plan, routes, side, completed)
            height = max((arm.tool(step['q'])[2, 3] for step in steps[:completed]), default=-np.inf)
            values = steps[completed - 1]['command'] if completed else plan['preshape'][side]
            for index, step in enumerate(steps[completed:], completed + 1):
                solid, tips = self.contact_vertices(side, routes, values)
                pose = arm.tool(step['q'])
                target = routes.world(pose[:3, 3], yaw)
                # An object can block a finger before its nominal stroke is complete.
                height = max(height, target[2])
                target[2] = height
                if not self.correct_center(side, routes, kind, 'close_clearance', target, pose[:3, :3],
                                           yaw, solid, tolerance=.0025, duration=3.,
                                           touch=np.concatenate(list(tips.values())),
                                           angular_tolerance=.03, speed_limit=.18):
                    return False
                values = step['command']
                self.grip({side: values})
                event('pinch_step', kind=kind, step=index)
            return True

        def scoop_cable(self, plan, routes, side, completed):
            touch = plan['touch']
            steps = touch['close']
            drop = touch['scoop_drop_m']
            yaw = plan['pick_yaw']
            arm = self.arm[side]
            values = steps[completed - 1]['command'] if completed else plan['preshape'][side]

            def height():
                return routes.world(arm.tool(self.q(side))[:3, 3], self.angles[WAIST])[2]

            entry = arm.tool(touch['waypoints'][0])[:3, 3]
            start_height = routes.world(entry, yaw)[2] if completed else height()
            close_sent_at = 0.
            for index, step in enumerate(steps[completed:], completed):
                stroke = float(np.max(abs(np.subtract(step['command'], values))))
                duration = max(.20, .8 * stroke / HAND_SPEED)
                close_sent_at = time.monotonic()
                self.command_hand(side, step['command'])
                goal = arm.tool(step['q'])[:3, 3]
                if routes.world(goal, yaw)[2] <= height():
                    end = self.set_arms({side: step['q']}, .16, duration)
                else:
                    end = self.history[-1][0] + duration
                self.advance_to(end, 'cable scoop', load_limit=8.)
                values = step['command']
                event('pinch_step', kind='cable', step=index + 1)

            while_closing = start_height - height()
            contact_since = None
            strong_effort = .25
            efforts = np.zeros(5)
            blocked = np.zeros(5, dtype=bool)

            def holding():
                nonlocal contact_since, efforts, blocked
                received, stamp, positions, effort = self.hand[side]
                # The sixth actuator rotates the thumb. Its flexion is channel five.
                if len(effort) != 6 or received < close_sent_at or time.monotonic() - received > 1.25:
                    contact_since = None
                    return False
                efforts = np.abs(np.asarray(effort[:5]))
                measured = np.array([positions.get(channel, np.nan) for channel in FINGERS[:5]])
                blocked = (measured - steps[-1]['command'][:5] > .025) & (measured < .95)
                if (not np.all(efforts >= strong_effort)
                        or not np.all(blocked)
                        or start_height - height() < drop - .003):
                    contact_since = None
                    return False
                if contact_since is None:
                    contact_since = stamp
                return stamp - contact_since >= .12

            try:
                self.poll(holding, 3., 'cable finger contact', load_limit=8., timeout_error=WaypointAlignment)
                engaged = True
            except WaypointAlignment:
                engaged = False
            event('cable_scoop', commanded_mm=1000 * drop, while_closing_mm=round(1000 * while_closing, 1),
                  measured_mm=round(1000 * (start_height - height()), 1))
            event('cable_finger_contact', engaged=engaged, min_effort=strong_effort,
                  efforts=efforts.round(3).tolist(), blocked=blocked.tolist())
            if not engaged:
                self.grip({side: plan['preshape'][side]}, load_limit=8.)
            return engaged

        def lift_from_table(self, plan, routes, side, *, load_limit=6.):
            kind = plan['item']['kind']
            if kind == 'cable':
                arm = self.arm[side]
                start = self.q(side)
                pose = arm.tool(start)
                yaw = self.angles[WAIST]
                goal = fit_contact(arm, pose[:3, 3] + [0., 0., .040], pose[:3, :3], start)
                hand = routes.hands[side]
                angles = hand.nominal_angles(plan['grip'][side])
                angles.update({key: self.angles[key] for key in angles if key in self.angles})
                vertices = np.concatenate(list(hand.vertices(angles).values()))
                normal = routes.local(routes.normal, yaw)

                def gap(q):
                    frame = arm.wrist(q) @ arm.hand
                    return (vertices @ (frame[:3, :3].T @ normal)).min() + frame[:3, 3] @ normal - routes.floor

                # The measured fingers can already touch the table. Validate their
                # upward separation from that starting pose and a clear endpoint.
                if goal is None or not routes.clear(side, goal, vertices, yaw, gap=.002):
                    raise RuntimeError('cable: no clear tabletop exit')
                path = np.linspace(start, goal, 2 + max(8, int(np.max(abs(goal - start)) / .025)))
                if (np.any(np.diff([gap(q) for q in path]) < -.0002)
                        or not all(routes.clear(side, q, np.empty((0, 3)), yaw, gap=.002) for q in path)):
                    raise RuntimeError('cable: tabletop exit does not separate upward')
                self.move([{side: goal}], speed=.17, load_limit=load_limit)

                def raised():
                    q = self.q(side)
                    return (arm.tool(q)[2, 3] - pose[2, 3] >= .036
                            and routes.clear(side, q, vertices, self.angles[WAIST], gap=.002))

                self.poll(raised, 3., 'cable tabletop exit', load_limit=load_limit)
                event('cable_table_exit', raised_mm=round(1000 * (arm.tool(self.q(side))[2, 3] - pose[2, 3]), 1))
                return
            solid, tips = self.contact_vertices(side, routes, plan['grip'][side])
            pose = self.arm[side].tool(self.q(side))
            yaw = self.angles[WAIST]
            target = routes.world(pose[:3, 3], yaw) + [0., 0., .040]
            if not self.correct_center(side, routes, kind, 'leave_table', target, pose[:3, :3],
                                       yaw, solid, tolerance=.004, duration=3.5,
                                       touch=np.concatenate(list(tips.values())),
                                       angular_tolerance=.030, speed_limit=.17, load_limit=load_limit):
                raise RuntimeError(f'{kind}: wrist did not clear the tabletop')

        def held_geometry(self, side, routes, target):
            pose = self.arm[side].tool(self.q(side))
            yaw = self.angles[WAIST]
            center = routes.world(pose[:3, 3], yaw)
            orient = rotation(yaw - routes.yaw) @ pose[:3, :3]
            corners = np.stack(np.meshgrid(*np.asarray(target['bounds']).T), axis=-1).reshape(-1, 3)
            return (corners - center) @ orient, (np.asarray(target['xyz']) - center) @ orient

        def raise_above(self, side, routes, kind, phase, shape, offsets, *, load_limit=6.):
            arm = self.arm[side]
            yaw = self.angles[WAIST]
            pose = arm.tool(self.q(side))
            orient = pose[:3, :3]
            target = routes.world(pose[:3, 3], yaw)
            clearance = pose[2, 3] + (offsets @ orient[2]).min() - routes.highest
            if fast_batch and phase == 'lift_loaded' and clearance >= .015:
                event('clearance_before_transfer', kind=kind, mm=round(1000 * clearance, 1))
                return
            target[2] = max(target[2], routes.highest + .030 - (offsets @ orient[2]).min())
            if not self.correct_center(side, routes, kind, phase, target, orient, yaw, shape,
                                       tolerance=.010, duration=4., load_limit=load_limit):
                raise RuntimeError(f'{kind}: {phase} did not reach the required height')
            pose = arm.tool(self.q(side))
            clearance = pose[2, 3] + (offsets @ pose[2, :3]).min() - routes.highest
            if clearance < TABLE_GAP:
                raise RuntimeError(f'{kind}: {phase} has insufficient clearance')
            event('clearance_before_transfer' if phase == 'lift_loaded' else 'clearance_before_return',
                  kind=kind, mm=round(1000 * clearance, 1))

        def raise_and_ready(self, side, routes, kind, shape, pocket):
            self.arm[side].pocket = pocket
            self.raise_above(side, routes, kind, 'lift_clear', shape, shape - pocket, load_limit=9.5)
            yaw = self.angles[WAIST]
            ready = ready_angles(side)
            raised = routes.raised_ready(side, ready, shape, yaw)
            if raised is None or not routes.link(side, self.q(side), raised, shape, yaw):
                raise RuntimeError(f'{kind}: no elevated return to ready')
            self.move([{side: raised}, {side: ready}], load_limit=9.5)

        def approach(self, plan, routes):
            kind = plan['item']['kind']
            waypoints = plan['approach']
            sides = waypoints[0]
            visited = [{side: self.q(side) for side in sides}]
            mouse_motion = fast_batch and (kind in TWO_ARM_MOUSE or plan.get('low_object_slide')) and plan.get(
                'edge_receive')
            center_approach = not fast_batch or kind == 'blue_jar'
            try:
                with measure('approach', kind=kind):
                    touch = plan.get('touch')
                    # Steer the final descent near the table from encoder feedback.
                    count = len(touch['waypoints']) if touch else int(
                        center_approach and 'pickup_center' in plan and len(waypoints) > 1 and kind != 'gold_jar')
                    prefix = waypoints[:len(waypoints) - count]
                    if len(sides) == 2:
                        resume_limit = .35 if resume else .14
                        for index in reversed(range(len(prefix))):
                            if all(np.max(abs(self.q(side) - prefix[index][side])) < resume_limit for side in sides):
                                visited = [{side: ready_angles(side) for side in sides}, *prefix[:index]]
                                prefix = prefix[index:]
                                break
                    if touch:
                        side = next(iter(sides))
                        arm = self.arm[side]
                        current = arm.wrist(self.q(side))
                        first_touch = arm.wrist(touch['waypoints'][0])
                        tilt = Rotation.from_matrix(current[:3, :3].T @ first_touch[:3, :3]).magnitude()
                        if np.linalg.norm(current[:3, 3] - first_touch[:3, 3]) < .012 and tilt < .09:
                            # Resume at a reached contact without moving sideways from the tabletop.
                            visited = [{side: ready_angles(side)}, *prefix]
                            prefix = ()
                    for goals in prefix:
                        careful = kind == 'cable' or len(sides) == 2
                        tolerance = np.deg2rad(2.) if kind == 'cable' else .028
                        if not center_approach and not touch and 'pickup_center' in plan and goals is prefix[-1]:
                            side = next(iter(goals))
                            arm = self.arm[side]
                            arm.pocket = plan['pocket']
                            target = np.asarray(plan['pickup_center'])

                            def reached_grasp():
                                q, yaw = self.q(side), self.angles[WAIST]
                                center = routes.world(arm.tool(q)[:3, 3], yaw)
                                return (np.linalg.norm(center - target) <= CENTER_TOLERANCE
                                        and routes.clear(side, q, plan['shape'], yaw))

                            self.follow_waypoint(goals, .022 if kind == 'gold_jar' else .009, 6., speed=.55,
                                                 reached=reached_grasp)
                        else:
                            if mouse_motion:
                                target = self.box_palms(goals)
                                self.follow_waypoint(goals, np.deg2rad(2.), 6., speed=.42,
                                                     reached=lambda: np.all(np.linalg.norm(
                                                         self.box_palms() - target, axis=1) <= .015))
                                visited.append(goals)
                                continue
                            position_control = (kind in ('drive', 'gold_jar') or
                                                (fast_batch and kind == 'blue_jar' and goals is prefix[-1]))
                            speed = .42 if fast_batch and careful and goals is prefix[0] and len(prefix) > 1 else .34
                            align = careful and (not fast_batch or kind == 'drive')
                            if fast_batch and self.box_direct and kind == 'drive' and goals is prefix[-1]:
                                align = False
                            self.move([goals], speed=speed if careful else .55,
                                      joint_tolerance=tolerance if align else None,
                                      load_limit=8. if kind == 'cable' else 6., position_control=position_control,
                                      palm_tolerance=.010 if mouse_motion else None)
                        visited.append(goals)
                    if kind == 'mouse' and plan.get('edge_receive'):
                        pusher = plan['pusher_side']
                        measured, target = (self.box_palms(q)[SIDES.index(pusher)]
                                            for q in (None, waypoints[-1]))
                        event('mouse_palm_on_top' if plan.get('top_slide') else 'mouse_palm_behind', side=pusher,
                              palm_error_mm=round(1000 * float(np.linalg.norm(measured - target)), 1),
                              palm_m=routes.world(measured, self.angles[WAIST]).round(4).tolist())
                        if flat_mouse_receiver:
                            receiver = plan['grip_side']
                            self.align_waypoint({receiver: goals[receiver]}, .012, 6.)
                            self.set_arms({receiver: goals[receiver]}, .12)
                            furthest, tip_top, tilt = self.receiver_geometry(receiver, routes, plan['preshape'][receiver])
                            edge = plan['receiver_edge_x_m'] + .003
                            event('mouse_receiver_ready', side=receiver, tilt_deg=round(tilt, 1),
                                  edge_clearance_mm=round(1000 * (edge - furthest), 1),
                                  fingertip_top_mm=round(1000 * tip_top, 1))
                            if furthest > edge or tip_top > .004 or tilt > 4.5:
                                raise WaypointAlignment('mouse: receiving fingers are outside the edge target')
                    if kind == 'cable' and prefix:
                        side = next(iter(sides))
                        gaps = self.fingertip_gaps(side, routes, plan['preshape'][side])
                        if min(gaps.values()) >= .012:
                            self.force_reference[side] = self.wrench[side][1]
                            self.load_since.pop(side, None)
                if touch:
                    aligned = self.touch_table(plan, routes, next(iter(sides)))
                else:
                    aligned = not center_approach or 'pickup_center' not in plan or self.align_center(plan, routes, 'pickup')
                if not aligned:
                    with measure('withdraw_after_alignment', kind=kind):
                        self.move(reversed(visited if count else visited[:-1]), speed=.42)
                    event('skipped', kind=kind, reason='encoder_alignment_outside_tolerance')
                    return False
            except WaypointAlignment:
                event('approach_alignment_failed', kind=kind, waypoint=len(visited),
                      errors_deg={side: round(float(np.rad2deg(np.max(abs(self.q(side) - goals[side])))), 1)
                                  for side in sides},
                      joint_errors_deg={side: dict(zip(self.arm[side].names, np.rad2deg(
                          self.q(side) - goals[side]).round(1).tolist())) for side in sides})
                with measure('withdraw_after_alignment', kind=kind):
                    self.move(reversed(visited), speed=.42)
                for side in sides:
                    self.ready_hand(side)
                event('skipped', kind=kind, reason='encoder_waypoint_outside_tolerance')
                return False
            except WristContact as contact:
                if contact.side not in sides:
                    raise
                event('approach_contact', kind=kind, side=contact.side, waypoint=len(visited),
                      load_n=round(contact.load, 2))
                self.brake('arm', [self.ids[name] for side in sides for name in self.arm[side].names])
                self.load_since.clear()
                self.withdraw_limits[contact.side] = max(6., contact.load + 2.)
                try:
                    with measure('withdraw_after_contact', kind=kind):
                        self.move(reversed(visited), speed=.42)
                finally:
                    self.withdraw_limits.clear()
                for side in sides:
                    self.ready_hand(side)
                event('skipped', kind=kind, reason=str(contact))
                return False
            return True

        def command_hand(self, side, values):
            self.hand_pub[side].publish(JointState(name=FINGERS, position=list(values)))

        def ready_hand(self, side):
            self.command_hand(side, OPEN_LEFT if side == 'left' else working_hand_command(side))

        def grip(self, commands, *, load_limit=6.):
            start = self.history[-1][0]
            duration = 0.
            for side, values in commands.items():
                positions = self.hand[side][2]
                stroke = np.max(abs(np.clip([positions.get(name, 1.) for name in FINGERS], 0., 1.) - values))
                duration = max(duration, float(stroke) * .8 / HAND_SPEED)
                self.command_hand(side, values)
            self.advance_to(start + duration, 'finger motion', load_limit=load_limit)

        def set_grip_force(self, kind, hands):
            pending = {}
            for side in SIDES:
                ratios = (.8,) * 6
                if kind in ('drive', 'cable', *TWO_ARM_MOUSE) and side in hands:
                    ratios = (1.,) * 6
                if ratios == self.hand_force[side]:
                    continue
                request = SetForce.Request()
                for i, ratio in enumerate(ratios):
                    setattr(request, f'force{i}_ratio', ratio)
                pending[side] = (ratios, self.clients[side, 'force'].call_async(request))
            if pending:
                self.poll(lambda: all(future.done() for _, future in pending.values()), 2.,
                          'finger effort setting', load_limit=9.5)
                for side, (ratios, future) in pending.items():
                    if not future.result().force_accepted:
                        raise RuntimeError(f'{side}: force setting rejected')
                    self.hand_force[side] = ratios

        @contextmanager
        def allow_table_contact(self, kind, sides=('left',), limit=np.inf, *, edge_slide=False):
            if kind not in TABLE_PINCH and not edge_slide:
                yield
                return
            previous = {side: self.withdraw_limits.get(side) for side in sides}
            self.withdraw_limits.update(dict.fromkeys(sides, limit))
            try:
                yield
            finally:
                for side, limit in previous.items():
                    if limit is None:
                        self.withdraw_limits.pop(side, None)
                    else:
                        self.withdraw_limits[side] = limit
                    self.load_since.pop(side, None)

        def scoop_feedback(self, commands, since, receiver=None, *, paired=False, mouse_pusher=None):
            feedback, stamps = {}, []
            for side in SIDES:
                received, stamp, positions, effort = self.hand[side]
                measured = np.array([positions.get(channel, np.nan) for channel in FINGERS[:4]])
                effort = np.abs(np.asarray(effort[:4]))
                fresh = received >= since and time.monotonic() - received <= 1.25 and len(effort) == 4
                if fresh and (side == receiver or paired):
                    deflection = measured - commands[side][:4]
                    contact = (deflection > .045) & (effort >= .35)
                    if side == mouse_pusher:
                        contact |= (deflection > .030) & (effort >= .80)
                    loaded = np.flatnonzero((measured <= 1.12) & contact).tolist()
                elif fresh and receiver is None:
                    loaded = np.flatnonzero((measured < .95) & (measured - commands[side][:4] > .025)
                                            & (effort >= .25)).tolist()
                else:
                    loaded = []
                feedback[side] = dict(loaded=loaded, position=measured.round(3).tolist(),
                                      effort=effort.round(3).tolist())
                if fresh and loaded:
                    stamps.append(stamp)
            return feedback, stamps

        def move_scoop_steps(self, steps, commands):
            for step in steps:
                fingers = step['command']
                stroke = max(np.max(abs(np.subtract(fingers[side], commands[side]))) for side in SIDES)
                sent_at = time.monotonic()
                for side in SIDES:
                    self.command_hand(side, fingers[side])
                duration = max(.20, .8 * stroke / HAND_SPEED)
                phase = step.get('phase')
                if phase == 'close_curl':
                    end = self.set_arms(step['q'], .55, duration)
                elif phase in ('open', 'close'):
                    end = self.history[-1][0] + duration
                elif fast_batch:
                    self.follow_waypoint(step['q'], .007, 9.5, speed=.20)
                    commands = fingers
                    continue
                else:
                    end = self.set_arms(step['q'], .20, duration)
                self.advance_to(end, phase or 'scoop motion', load_limit=9.5)
                commands = fingers
            return sent_at

        def push_scoop_steps(self, steps, plan):
            receiver = plan['grip_side']
            pusher = plan['pusher_side']
            mouse_motion = plan['item']['kind'] in TWO_ARM_MOUSE or plan.get('low_object_slide')
            cup = 'cup_gap_above_rim_m' in plan or plan.get('mouse_staging_only')
            timed_push = timed_batch and not plan.get('mouse_lateral_first')
            palm_tolerance = np.array([.008 if side == receiver else .010 for side in SIDES])
            joint_tolerance = MOUSE_MOTION_TOLERANCE if mouse_motion else .075
            receiver_tolerance = joint_tolerance if cup else .035
            segment = None
            segment_started = time.monotonic()
            segment_sim_start = SIM_TIME
            position_bias = np.zeros(7)

            def finish_segment():
                if segment is not None:
                    event('mouse_slide_segment', kind=plan['item']['kind'], phase=segment,
                          seconds=round(time.monotonic() - segment_started, 3),
                          sim_start=segment_sim_start, sim_end=SIM_TIME,
                          sim_seconds=round(SIM_TIME - segment_sim_start, 3))

            for index, step in enumerate(steps):
                next_segment = step.get('segment', 'contact')
                if next_segment != segment:
                    finish_segment()
                    segment, segment_started = next_segment, time.monotonic()
                    segment_sim_start = SIM_TIME
                goals = step['q']
                for side in SIDES:
                    self.command_hand(side, step['command'][side])
                free_move = segment in ('around_back', 'receiver_approach', 'around_back_and_receiver')
                if cup and timed_batch:
                    if segment != 'right_shift':
                        # Keep the contact stroke in velocity mode, where it yields at the table edge.
                        speed = .40 if free_move and plan['item'].get('adaptive_batch') else .26
                        next_edge = (free_move and index + 1 < len(steps)
                                     and steps[index + 1].get('segment') == 'edge_shift')
                        if next_edge and plan['item'].get('adaptive_batch'):
                            points = {receiver: plan['receive_point'], pusher: PALM_FACE}
                            expected = self.box_palms(goals, points)[SIDES.index(receiver)]

                            def receiver_in_place():
                                actual = self.box_palms(contact_points=points)[SIDES.index(receiver)]
                                return np.linalg.norm(actual - expected) <= .006

                            try:
                                self.follow_waypoint(goals, np.deg2rad(2.), 9.5, speed=speed,
                                                     reached=receiver_in_place)
                            except WaypointAlignment:
                                finish_segment()
                                event('receiver_not_in_place', kind=plan['item']['kind'], side=receiver)
                                return index + 1
                            continue
                        self.follow_waypoint(goals, None, 9.5, speed=speed, timed_motion=True,
                                             stop_on_reach=index == len(steps) - 1)
                        continue
                    speed = .30
                    start = self.history[-1][0]
                    end = self.set_arms(goals, speed)

                    # Correct steady contact load on a fixed schedule.
                    # The next waypoint starts on time regardless of encoder error.
                    finish = end + .55
                    last_update = max(start, end - .10)

                    def correct_push():
                        nonlocal last_update, position_bias
                        stamp = self.history[-1][0]
                        if stamp >= finish:
                            return True
                        dt = stamp - last_update
                        if dt >= .04:
                            error_at = goals[pusher] - self.q(pusher)
                            correction = np.clip(error_at, -.10, .10)
                            position_bias = np.clip(position_bias + 4. * min(dt, .08) * correction,
                                                    -.20, .20)
                            adjusted = np.clip(goals[pusher] + position_bias + .5 * error_at,
                                               self.arm[pusher].limits[0] + .005,
                                               self.arm[pusher].limits[1] - .005)
                            self.set_arms({pusher: adjusted}, speed, minimum_duration=.08)
                            last_update = stamp
                        return False

                    self.poll(correct_push, (finish - start) / .30 + 2., 'mouse slide', load_limit=9.5)
                    continue

                contact_points = {receiver: plan['receive_point'],
                                  pusher: step.get('push_point', plan.get('push_point', PALM_FACE))}
                target = self.box_palms(goals, contact_points)
                tolerance = palm_tolerance.copy()
                last = index == len(steps) - 1
                if mouse_motion and not last:
                    tolerance[SIDES.index(pusher)] = .014 if cup else .013

                def error():
                    return np.linalg.norm(self.box_palms(contact_points=contact_points) - target, axis=1)

                def reached():
                    distances = error()
                    return (np.max(abs(self.q(receiver) - goals[receiver])) <= receiver_tolerance
                            and distances[SIDES.index(receiver)] <= tolerance[SIDES.index(receiver)]
                            and (cup or distances[SIDES.index(pusher)] <= tolerance[SIDES.index(pusher)]))

                # Keep correcting the receiving wrist from encoders even after its planned joint goals stop changing.
                step_tolerance = np.deg2rad(1.7) if cup else joint_tolerance
                try:
                    self.follow_waypoint(goals, step_tolerance, 9.5,
                                         speed=.42 if timed_push and free_move else .26, reached=reached,
                                         stop_on_reach=not cup or last)
                except WaypointAlignment:
                    finish_segment()
                    event('edge_push_missed', kind=plan['item']['kind'], step=index + 1,
                          segment=segment,
                          error_mm=dict(zip(SIDES, (1000 * error()).round(1).tolist())),
                          goal_rad={side: goals[side].round(5).tolist() for side in SIDES},
                          actual_rad={side: self.q(side).round(5).tolist() for side in SIDES},
                          joint_error_deg={side: round(float(np.degrees(np.max(abs(goals[side] - self.q(side))))), 2)
                                           for side in SIDES})
                    return index + 1
            finish_segment()
            return None

        def close_scoop_pair(self, plan, routes):
            steps = plan['close']
            kind = plan['item']['kind']
            receiver = plan['grip_side'] if plan.get('edge_receive') else None
            paired = plan.get('paired_carry', False)
            hook = plan.get('hook_carry', False)
            lift_after_close = fast_batch and paired and kind in TWO_ARM_MOUSE
            receiver_before = self.edge_support_force(receiver, routes) if plan.get('under_table') and not fast_batch else None
            before = self.box_palms()
            scoop_steps = sum(step.get('phase') == 'scoop_open' for step in steps)
            if scoop_steps:
                if fast_batch and receiver and (paired or hook):
                    missed = self.push_scoop_steps(steps[:scoop_steps], plan)
                    if missed is not None:
                        return 'open_scoop_did_not_reach', missed
                    if not plan.get('mouse_table_sweep'):
                        self.set_arms(steps[scoop_steps - 1]['q'], .16)
                else:
                    self.move_scoop_steps(steps[:scoop_steps], plan['preshape'])
                goals = steps[scoop_steps - 1]['q']
                if not fast_batch:
                    try:
                        self.align_waypoint(goals, np.deg2rad(2.), 9.5)
                    except WaypointAlignment:
                        error = dict(zip(SIDES, (1000 * np.linalg.norm(
                            self.box_palms() - self.box_palms(goals), axis=1)).round(1).tolist()))
                        if not receiver or error[receiver] > 8. or error[plan['pusher_side']] > 16.:
                            event('scoop_open_missed', kind=kind, error_mm=error)
                            return 'open_scoop_did_not_reach', scoop_steps
                        event('edge_push_within_tolerance', kind=kind, error_mm=error)
                        self.set_arms({receiver: goals[receiver]}, .14)
                    else:
                        self.set_arms(goals, .14)
                after = self.box_palms()
                if receiver:
                    push_index = SIDES.index(plan['pusher_side'])
                    event('edge_push', kind=kind, receiver=receiver, pusher=plan['pusher_side'],
                          travel_mm=round(1000 * np.linalg.norm(after[push_index, :2] - before[push_index, :2]), 1))
                else:
                    event('scoop_open', kind=kind, convergence_mm=round(1000 * (
                        np.linalg.norm(before[0, :2] - before[1, :2]) - np.linalg.norm(after[0, :2] - after[1, :2])), 1))
            if resume and resume.get('hold_before_edge_grasp') and kind == 'cable':
                return 'manual_edge_stage', scoop_steps
            close_steps = steps[scoop_steps:]
            if hook and not close_steps:
                return None, len(steps)
            if receiver and 'receive_point' in plan and not fast_batch and not hook:
                first_close = next(i for i, step in enumerate(close_steps) if step['phase'] == 'close')
                if first_close:
                    self.move_scoop_steps(close_steps[:first_close], plan['preshape'])
                aligned = (self.edge_transfer_aligned(plan, routes, receiver_before, close_steps[first_close]['q'])
                           if receiver_before is not None else self.edge_center_aligned(plan, routes))
                if not aligned:
                    return 'object_center_not_on_receiving_palm', scoop_steps + first_close
                close_steps = close_steps[first_close:]
            close_sent_at = self.move_scoop_steps(close_steps, plan['preshape'])
            if plan.get('hook_two_finger'):
                side = plan['grip_side']
                _, _, positions, efforts = self.hand[side]
                event('cable_two_finger_closed', side=side, force_ratio=1.,
                      command=plan['grip'][side],
                      position={name: positions.get(name) for name in (FINGERS[3], FINGERS[4])},
                      effort={name: float(efforts[i]) for i, name in ((3, 'index'), (4, 'thumb'))})
            commands = steps[-1]['command']
            contact_since = None
            feedback = {}

            def holding():
                nonlocal contact_since, feedback
                feedback, stamps = self.scoop_feedback(commands, close_sent_at, receiver, paired=paired)
                if len(stamps) != len(SIDES):
                    contact_since = None
                    return False
                stamp = min(stamps)
                if contact_since is None:
                    contact_since = stamp
                return stamp - contact_since >= .12

            if lift_after_close or (receiver and not paired):
                mouse_pusher = plan['pusher_side'] if lift_after_close and plan['item'].get('adaptive_batch') else None
                feedback, stamps = self.scoop_feedback(commands, close_sent_at, receiver, paired=paired,
                                                      mouse_pusher=mouse_pusher)
                engaged = bool(stamps)
            else:
                try:
                    self.poll(holding, 2., 'grasp contact', load_limit=9.5, timeout_error=WaypointAlignment)
                    engaged = True
                except WaypointAlignment:
                    engaged = False
            event('edge_contact' if receiver else 'scoop_contact', kind=kind, engaged=engaged, hands=feedback,
                  descent_mm=dict(zip(SIDES, (1000 * (before[:, 2] - self.box_palms()[:, 2])).round(1).tolist())),
                  table_gap_mm={side: {finger: round(1000 * gap, 1) for finger, gap in
                                       self.fingertip_gaps(side, routes, commands[side]).items()} for side in SIDES},
                  joint_error_deg={side: round(float(np.rad2deg(np.max(abs(self.q(side) - steps[-1]['q'][side])))), 1)
                                   for side in SIDES})
            if (lift_after_close and plan['item'].get('adaptive_batch') and not engaged
                    and not any(hand['loaded'] for hand in feedback.values())):
                return 'no_contact_on_both_hands', len(steps)
            return (None if lift_after_close or engaged or (receiver and not paired) else 'no_contact_on_both_hands',
                    len(steps))

        def edge_support_force(self, side, routes):
            stamp, force = self.wrench[side]
            wrist = self.arm[side].wrist(self.q(side))
            normal = routes.local(routes.normal, self.angles[WAIST])
            return stamp, float(normal @ wrist[:3, :3] @ force / np.linalg.norm(normal))

        def edge_transfer_aligned(self, plan, routes, receiver_before, goals):
            receiver = plan['grip_side']
            pusher = plan['pusher_side']
            yaw = self.angles[WAIST]

            def marker(side, q, local_point, angle):
                arm = self.arm[side]
                hand = arm.wrist(q) @ arm.hand
                return routes.world(hand[:3, 3] + hand[:3, :3] @ local_point, angle)

            point = np.asarray(plan['receive_point'])
            pad = np.asarray(plan.get('push_point', PALM_FACE))
            expected = marker(receiver, goals[receiver], point, plan['pick_yaw'])
            receiving = marker(receiver, self.q(receiver), point, yaw)
            intended_push = marker(pusher, goals[pusher], pad, plan['pick_yaw'])
            actual_push = marker(pusher, self.q(pusher), pad, yaw)
            carried = np.asarray(plan.get('edge_object_center', expected)) + actual_push - intended_push
            gap = float(np.linalg.norm((carried - receiving)[:2]))
            drift = float(np.linalg.norm(receiving - expected))
            settled = time.monotonic()
            try:
                self.poll(lambda: self.wrench[receiver][0] >= settled, .35, 'edge support feedback',
                          load_limit=9.5, timeout_error=WaypointAlignment)
            except WaypointAlignment:
                event('edge_encoder_center', kind=plan['item']['kind'], receiver=receiver, fresh=False)
                return False
            stamp, force = self.edge_support_force(receiver, routes)
            support = abs(force - receiver_before[1])
            kind = plan['item']['kind']
            tolerance = .015 if kind in TWO_ARM_MOUSE else .025
            fresh = stamp > receiver_before[0] and time.monotonic() - stamp <= 1.25
            event('edge_encoder_center', kind=kind, receiver=receiver, error_mm=round(1000 * gap, 1),
                  receiver_drift_mm=round(1000 * drift, 1), load_change_n=round(support, 2),
                  fresh=fresh, tolerance_mm=1000 * tolerance)
            return fresh and gap <= tolerance and drift <= .010 and support >= .30

        def edge_center_aligned(self, plan, routes):
            kind = plan['item']['kind']
            receiver = plan['grip_side']
            camera = plan.get('receive_camera', 'head')
            frames = deque(maxlen=1)
            topics = ((Image, 'color/image_raw'), (Image, 'depth/image_raw'), (CameraInfo, 'color/camera_info'))
            inputs = [Subscriber(self.node, typ, f'/cam_{camera}/{topic}', qos_profile=qos_profile_sensor_data)
                      for typ, topic in topics]
            sync = TimeSynchronizer(inputs, 4)
            sync.registerCallback(lambda *msgs: frames.append(msgs))
            try:
                self.poll(lambda: bool(frames), 2., 'receiving palm depth',
                          load_limit=9.5, timeout_error=WaypointAlignment)
            except WaypointAlignment:
                event('edge_center', kind=kind, receiver=receiver, reason='no_fresh_depth')
                return False
            finally:
                for stream in inputs:
                    self.node.destroy_subscription(stream.sub)
            rgb, depth, intrinsics = frames[-1]
            stamp = seconds(rgb)
            t, _, q = min(self.history, key=lambda row: abs(row[0] - stamp))
            if abs(stamp - t) > .08:
                event('edge_center', kind=kind, receiver=receiver, reason='no_matching_encoder_pose')
                return False
            frame = (self.bridge.imgmsg_to_cv2(rgb, 'bgr8'), self.bridge.imgmsg_to_cv2(depth, 'passthrough'),
                     np.asarray(intrinsics.k))
            catch_mouse = plan.get('catch_mouse', False)
            cloud = self.vision.points(camera, frame, q, mask_arms=not catch_mouse)
            center = None
            hand_vertices = []
            for side, arm in self.arm.items():
                pose = arm.wrist([q[name] for name in arm.names]) @ arm.hand
                hand = routes.hands[side]
                angles = hand.nominal_angles(plan['preshape'][side])
                angles.update({key: q[key] for key in angles if key in q})
                vertices = np.concatenate(list(hand.vertices(angles).values()))
                hand_vertices.append(vertices @ pose[:3, :3].T + pose[:3, 3])
                if side == receiver:
                    center = pose[:3, 3] + pose[:3, :3] @ plan['receive_point']
            points = cloud['points']
            normal = routes.local(routes.normal, q[WAIST])
            height = points @ normal - routes.floor
            radius = np.linalg.norm(np.ptp(plan['item']['target']['bounds'], axis=0)[:2]) / 2 + .042
            distance_xy = np.linalg.norm(points[:, :2] - center[:2], axis=1)
            color = cloud['colors'].mean(axis=1)
            if catch_mouse:
                world = routes.world(points.T, q[WAIST]).T
                support = routes.world(center, q[WAIST])
                lip = routes.edge + .015
                nearby = ((abs(world[:, 0] - support[0]) < .063) & (abs(world[:, 1] - support[1]) < .048)
                          & (world[:, 0] < lip - .004) & ((world[:, 0] < lip - .026) | (height > .010))
                          & (height > -.035) & (height < .047) & (color < 201.))
                points, _ = RobotSurfaceMask().filter(points[nearby], q, tolerance=.004)
                if len(points) >= 18 and (np.linalg.norm(np.ptp(points[:, :2], axis=0)) < .025
                                          or np.ptp(points @ normal) < .008):
                    event('edge_center', kind=kind, receiver=receiver, reason='object_profile_too_small')
                    return False
            else:
                nearby = (distance_xy <= radius) & (height > -.060) & (height < .065) & (color < 172.)
                points = points[nearby]
                distance, _ = cKDTree(np.concatenate(hand_vertices)).query(points)
                points = points[distance > .012]
            if len(points) < 18:
                event('edge_center', kind=kind, receiver=receiver, reason='object_occluded', points=len(points))
                return False
            observed = np.percentile(points, [8., 92.], axis=0).mean(axis=0)
            error = float(np.linalg.norm(observed[:2] - center[:2]))
            hold_mouse = flat_mouse_receiver or (resume and resume.get('hold_after_edge_grasp'))
            tolerance = .015 if kind == 'mouse' and (hold_mouse or catch_mouse) else .025
            event('edge_center', kind=kind, receiver=receiver, error_mm=round(1000 * error, 1), points=len(points),
                  tolerance_mm=1000 * tolerance, palm_m=center.round(4).tolist(), object_m=observed.round(4).tolist())
            return error <= tolerance

        def withdraw_scoop_pair(self, plan, steps_run, *, partial_push=False):
            if plan.get('hold_on_miss'):
                event('edge_pose_retained', kind=plan['item']['kind'])
                return
            high_exit = plan.get('abort_after_push')
            push_steps = sum(step.get('phase') == 'scoop_open' for step in plan['close'])
            if partial_push:
                passed = [step['q'] for step in plan['close'][:steps_run - 1]]
                self.move([*reversed(passed), plan['approach'][-1], *plan['pickup_retreat']],
                          speed=.30, load_limit=9.5, position_control=True)
            elif high_exit and steps_run >= push_steps:
                self.grip(plan['preshape'], load_limit=9.5)
                self.move(high_exit, speed=.42, load_limit=9.5,
                          palm_tolerance=.010 if plan['item']['kind'] in TWO_ARM_MOUSE else None)
            else:
                start = plan['close'][steps_run - 1]['command']
                self.move_scoop_steps(plan['abort_close'][-steps_run:], start)
                self.move(plan['pickup_retreat'], speed=.42, load_limit=9.5)
            for side in SIDES:
                self.ready_hand(side)

        def grip_state(self, side, command):
            _, _, positions, effort = self.hand[side]
            position = np.array([positions.get(name, np.nan) for name in FINGERS])
            blocked = (position - command > .045) & (np.asarray(command) < .95)
            available = len(effort) == 6 and np.isfinite(position).all()
            load = abs(np.asarray(effort)) if available else np.zeros(6)
            thumb = np.any(blocked[4:]) and np.max(load[4:]) > .12
            opposed = bool(thumb and np.any(blocked[:4] & (load[:4] > .10)))
            empty = bool(available and not np.any(blocked[:4]) and not np.any(blocked[4:]))
            return opposed, empty, position, load

        def grip_feedback(self, side, kind, command):
            opposed, empty, position, load = self.grip_state(side, command)
            event('grip_feedback', kind=kind, opposed=opposed, empty=empty,
                  position=position.round(3).tolist(), effort=load.round(3).tolist())
            return empty

        def aim(self, yaw, pitch):
            self.position('head', [(1, 0., .72), (2, pitch, .72), (3, yaw, .72)])

        def turn(self, yaw, target, routes, *, load_limit=6., speed=WAIST_SPEED):
            target = routes.local(target, yaw)
            heading, pitch = gaze_angles(target, [self.angles[k] for k in HEAD[:2]], self.vision.rig)
            self.aim(heading, pitch)
            distance = abs(self.angles[WAIST] - yaw)
            self.position('waist', [(31, yaw, speed)])
            if fast_batch:
                self.poll(lambda: abs(self.angles[WAIST] - yaw) <= .008,
                          distance / speed / .30 + 2., 'torso turn', load_limit=load_limit)
            else:
                self.advance_to(self.history[-1][0] + distance / speed, 'torso turn', load_limit=load_limit)

        def configure(self, sides):
            pending = []
            for (side, key), client in self.clients.items():
                if side not in sides:
                    continue
                request = (SetForce if key == 'force' else SetSpeed).Request()
                for i in range(6):
                    setattr(request, f'{key}{i}_ratio', self.hand_force[side][i] if key == 'force' else HAND_SPEED)
                pending.append((key, side, client.call_async(request)))
            return pending

        def connect(self, hand_sides=SIDES):
            self.poll(lambda: self.task and len(self.wrench) == len(self.hand) == 2
                      and all(k in self.angles for k in (*self.ids, *HEAD, WAIST))
                      and all(client.service_is_ready() for client in self.clients.values()), 12., 'ROS feeds')
            if resume and self.task != resume['task']:
                raise RuntimeError('saved coordinates belong to a different task')
            self.started = True
            self.force_reference = {side: self.wrench[side][1] for side in hand_sides}
            return self.configure(hand_sides)

        def survey(self, worker):
            started = time.monotonic()
            self.cameras = [Subscriber(self.node, typ, f'/cam_{camera}/{topic}',
                                       qos_profile=qos_profile_sensor_data)
                            for camera in ('head', 'torso')
                            for typ, topic in ((Image, 'color/image_raw'), (Image, 'depth/image_raw'),
                                               (CameraInfo, 'color/camera_info'))]
            self.sync = TimeSynchronizer(self.cameras, 8)
            self.sync.registerCallback(lambda *msgs: setattr(self, 'images', msgs))
            goals = {side: ready_angles(side) for side in SIDES}
            ready_at = None
            pending = self.connect()
            for side in SIDES:
                self.ready_hand(side)
            begin = time.monotonic()
            last_stamp = self.history[-1][0]
            arms_ready_at = self.set_arms(goals)
            self.position('waist', [(31, 0., READY_WAIST_SPEED)])
            frames, poses = {}, {}

            def arm_progress():
                nonlocal ready_at
                if ready_at is None and self.history[-1][0] >= arms_ready_at:
                    ready_at = time.monotonic()
                    event('ready_both', seconds=round(ready_at - begin, 3), rise_mm=1000 * READY_RISE)

            sectors = sorted(ANGLES.items(), key=lambda sector: sector[1], reverse=self.angles[HEAD[0]] > 0.)
            for name, yaw in sectors:
                self.aim(yaw, .42)
                captured_pose = None

                def aligned_frame():
                    nonlocal captured_pose
                    arm_progress()
                    if self.images is None:
                        return False
                    stamp = seconds(self.images[0])
                    if stamp <= last_stamp:
                        return False
                    t, _, q = min(self.history, key=lambda row: abs(row[0] - stamp))
                    if (abs(stamp - t) < .06 and abs(q[WAIST]) < .025
                            and abs(q[HEAD[0]] - yaw) < .045 and abs(q[HEAD[1]] - .42) < .042):
                        captured_pose = q
                        return True
                    return False

                self.poll(aligned_frame, 15., f'survey {name}')
                last_stamp = seconds(self.images[0])
                views = [(f'head_{name}', 0)]
                if name == 'centre':
                    views.append(('torso', 3))
                for camera, index in views:
                    rgb, depth, intrinsics = self.images[index:index + 3]
                    frames[camera] = (self.bridge.imgmsg_to_cv2(rgb, 'bgr8'),
                                      self.bridge.imgmsg_to_cv2(depth, 'passthrough'), np.array(intrinsics.k))
                    poses[camera] = captured_pose
            for camera in self.cameras:
                self.node.destroy_subscription(camera.sub)
            self.cameras.clear()
            self.sync = None
            self.images = None
            event('survey_once', seconds=round(time.monotonic() - begin, 3), views=list(frames))
            future = worker.submit(snapshot, frames, poses['torso'], last_stamp, self.task, poses,
                                   skip_box=skip_box, selected=selected)

            def complete():
                arm_progress()
                return ready_at is not None and future.done() and all(f.done() for _, _, f in pending)

            self.poll(complete, 10., 'ready and survey processing')
            for key, side, setting in pending:
                if not getattr(setting.result(), f'{key}_accepted'):
                    raise RuntimeError(f'{side}: {key} setting rejected')
            event('startup', seconds=round(time.monotonic() - started, 3))
            return future.result()

        def observe_tray_contents(self, tray, survey_yaw):
            frames = deque(maxlen=1)
            topics = ((Image, 'color/image_raw'), (Image, 'depth/image_raw'), (CameraInfo, 'color/camera_info'))
            inputs = [Subscriber(self.node, typ, f'/cam_head/{topic}', qos_profile=qos_profile_sensor_data)
                      for typ, topic in topics]
            sync = TimeSynchronizer(inputs, 4)
            sync.registerCallback(lambda *msgs: frames.append(msgs))
            try:
                self.poll(lambda: bool(frames), 2., 'tray depth', timeout_error=WaypointAlignment)
            except WaypointAlignment:
                return None
            finally:
                for stream in inputs:
                    self.node.destroy_subscription(stream.sub)
            rgb, depth, intrinsics = frames[-1]
            stamp = seconds(rgb)
            recorded, _, pose = min(self.history, key=lambda row: abs(row[0] - stamp))
            if abs(recorded - stamp) > .08:
                return None
            frame = (self.bridge.imgmsg_to_cv2(rgb, 'bgr8'), self.bridge.imgmsg_to_cv2(depth, 'passthrough'),
                     np.asarray(intrinsics.k))
            points = self.vision.points('head', frame, pose)['points']
            points = points @ rotation(pose[WAIST] - survey_yaw).T
            return measure_tray_contents({'objects': [tray]}, points).get(tray['kind'])

        def velocity(self, group, rows):
            msg = CmdSetMotorSpeed(cmds=[SetMotorSpeed(name=i, spd=float(rate)) for i, rate in rows])
            if group == 'arm':
                self.own_arm_rates.append(self.rate_signature(msg))
            self.brakes[group].publish(msg)

        def brake(self, group, ids):
            self.velocity(group, ((i, 0.) for i in ids))
            self.brakes[group].wait_for_all_acked(Duration(seconds=.15))

        def stop(self):
            for group, ids in (('arm', self.ids.values()), ('head', (1, 2, 3)), ('waist', (31,))):
                self.brake(group, ids)

        def box_palms(self, goals=None, contact_points=None):
            if contact_points is None and self.box_hold is not None:
                contact_points = self.box_hold['contact_points']
            palms = []
            for side in SIDES:
                arm = self.arm[side]
                pose = arm.wrist(self.q(side) if goals is None else goals[side]) @ arm.hand
                point = PALM_FACE if contact_points is None else contact_points[side]
                palms.append(pose[:3, 3] + pose[:3, :3] @ point)
            return np.array(palms)

        def box_axis(self, contact_points=None):
            left, right = self.box_palms(contact_points=contact_points)
            axis = left - right
            return axis / max(.001, np.linalg.norm(axis))

        def box_pressure(self, reference, contact_points=None):
            axis = self.box_axis(contact_points)
            return {side: float(sign * axis @ self.arm[side].wrist(self.q(side))[:3, :3]
                                @ (self.wrench[side][1] - reference[side]))
                    for side, sign in zip(SIDES, (1., -1.))}

        def send_box_targets(self, *, retime=False):
            hold = self.box_hold
            targets = {}
            nominal = hold['goals']
            left, right = self.box_palms(nominal)
            axis = left - right
            axis /= max(.001, np.linalg.norm(axis))
            for side, sign in zip(SIDES, (-1., 1.)):
                arm = self.arm[side]
                q = nominal[side]
                frames = arm.frames(q)
                pose = frames[-1] @ arm.hand
                palm = pose[:3, 3] + pose[:3, :3] @ hold['contact_points'][side]
                axes = np.array([frame[:3, :3] @ direction for frame, direction in zip(frames, arm.axes)])
                offsets = palm - np.array([frame[:3, 3] for frame in frames])
                jacobian = np.vstack((np.cross(axes, offsets).T, .13 * axes.T))
                correction = np.r_[hold['shift'] + sign * hold['inset'] * axis, 0., 0., 0.]
                damping = np.eye(6) * .015**2
                delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping, correction)
                lower = np.maximum(arm.limits[0] - q, -.18)
                upper = np.minimum(arm.limits[1] - q, .18)
                bounded = np.clip(delta, lower, upper)
                free = bounded == delta
                if not np.all(free):
                    # Keep the wrist pose by sharing the clipped motion with the joints that can still move.
                    active = jacobian[:, free]
                    bounded[free] += active.T @ np.linalg.solve(active @ active.T + damping,
                                                               correction - jacobian @ bounded)
                targets[side] = q + np.clip(bounded, lower, upper)
            yaw = self.angles[WAIST]
            for fraction in (1., .5, .25):
                candidate = {side: nominal[side] + fraction * (targets[side] - nominal[side]) for side in SIDES}
                if hold['clear'](candidate, yaw):
                    break
            else:
                candidate = nominal
                fraction = 0.
            hold['applied_inset'] = fraction * hold['inset']
            if retime:
                corrected_duration = max(float(np.max(abs(goal - self.q(side))))
                                         for side, goal in candidate.items()) / hold['speed']
                hold['end'] = max(hold['end'], self.history[-1][0] + corrected_duration)
            duration = max(.14, hold['end'] - self.history[-1][0])
            self.arm_targets(candidate, duration, hold['speed'])

        def maintain_box_pressure(self):
            hold = self.box_hold
            stamp = self.history[-1][0]
            elapsed = stamp - hold['stamp']
            if elapsed < .08:
                return
            hold['stamp'] = stamp
            pressure = np.array(list(self.box_pressure(hold['forces']).values()))
            hold['pressure'] = .65 * hold['pressure'] + .35 * pressure
            low, high = np.min(hold['pressure']), np.max(hold['pressure'])
            error = min(BOX_PRESSURE_TARGET - low, BOX_PRESSURE_TARGET + 1. - high)
            inset = np.clip(hold['inset'] + .0022 * min(elapsed, .12) * error,
                            hold.get('min_inset', -.008), hold['max_inset'])
            shift = hold['shift']
            if hold.get('align_palms'):
                midpoint_error = (self.box_palms(hold['goals']) - self.box_palms()).mean(0)
                shift = hold['shift'].copy()
                shift[:2] = np.clip(shift[:2] + .16 * midpoint_error[:2], -.035, .035)
            if abs(inset - hold['inset']) > .00025 or np.linalg.norm(shift - hold['shift']) > .0005:
                hold['inset'] = inset
                hold['shift'] = shift
                self.send_box_targets()

        def squeeze_box(self, plan):
            palm_only = 'box_table_tip' in plan
            forces = {side: self.wrench[side][1] for side in SIDES}
            goals = plan['grasp'][-1]
            contact_points = plan.get('box_contact_points', dict.fromkeys(SIDES, PALM_FACE))
            loaded_since = None

            def span():
                return float(np.linalg.norm(np.diff(self.box_palms(contact_points=contact_points), axis=0)))

            def engaged():
                nonlocal loaded_since
                stamp = self.history[-1][0]
                if (span() > plan['box_width'] + .010
                        or min(self.box_pressure(forces, contact_points).values()) < BOX_GRIP_PRESSURE):
                    loaded_since = None
                    return False
                if loaded_since is None:
                    loaded_since = stamp
                return stamp - loaded_since >= .10

            self.move([plan['precontact']], speed=.34, load_limit=BOX_LOAD_LIMIT, position_control=True)
            if palm_only and not fast_batch:
                try:
                    self.align_waypoint(plan['precontact'], .020, BOX_LOAD_LIMIT, contact=engaged)
                except WaypointAlignment:
                    pass
            self.advance_to(self.set_arms(goals, .30), 'box squeeze', load_limit=BOX_LOAD_LIMIT)
            try:
                self.align_waypoint(goals, .010, BOX_LOAD_LIMIT, contact=engaged)
            except WaypointAlignment:
                pass
            max_inset = np.clip((plan['box_width'] - 2 * plan['box_preload'] - .038) / 2, 0., .022)
            inset = min(.012, max_inset) if palm_only or self.box_direct else 0.
            self.box_hold = dict(forces=forces, clear=plan['box_clear'], contact_points=contact_points, shift=np.zeros(3),
                                 inset=inset, max_inset=max_inset, pressure=np.zeros(2), stamp=self.history[-1][0],
                                 align_palms=palm_only)
            end = self.set_arms(goals, .20)
            self.advance_to(end, 'box squeeze', load_limit=BOX_LOAD_LIMIT)
            if palm_only or not self.box_direct:
                pressure_time = 2.3 if palm_only and self.box_direct else 6.
                try:
                    self.poll(engaged, pressure_time, 'box palm pressure', load_limit=BOX_LOAD_LIMIT,
                              timeout_error=WaypointAlignment)
                except WaypointAlignment:
                    pass
            if plan['grip']:
                self.grip(plan['grip'], load_limit=BOX_LOAD_LIMIT)
            held = engaged()
            event('bimanual_contact', kind=plan['item']['kind'], engaged=bool(held),
                  press_side=plan.get('box_press_side'),
                  palm_distance_mm=round(1000 * span(), 1),
                  applied_inset_mm=round(1000 * self.box_hold['applied_inset'], 1),
                  palm_error_mm=dict(zip(SIDES, (1000 * np.linalg.norm(self.box_palms() - self.box_palms(goals),
                                                                      axis=1)).round(1).tolist())),
                  force_change_n={side: round(load, 2) for side, load in self.box_pressure(forces).items()})
            if palm_only:
                event('box_palm_alignment', actual_mm=(1000 * self.box_palms()).round(1).tolist(),
                      target_mm=(1000 * self.box_palms(goals)).round(1).tolist())
            if not held and not self.box_direct:
                self.box_hold = None
                return None
            return self.box_palms()[:, 2], forces

        def align_box_placement(self, plan, routes):
            hold = self.box_hold
            nominal = self.box_palms(plan['release'][-1]).mean(0)
            target = routes.world(nominal, plan['tray_yaw'])
            last = -np.inf

            def error():
                actual = routes.world(self.box_palms().mean(0), self.angles[WAIST])
                return target[:2] - actual[:2]

            def align():
                nonlocal last
                diff = error()
                if np.linalg.norm(diff) <= CENTER_TOLERANCE:
                    return True
                stamp = self.history[-1][0]
                if stamp - last >= .12:
                    offset = routes.local(np.r_[diff, 0.], self.angles[WAIST])
                    hold['shift'] = np.clip(hold['shift'] + .45 * offset, -.026, .026)
                    self.send_box_targets()
                    last = stamp
                return False

            before = float(np.linalg.norm(error()))
            try:
                self.poll(align, 4., 'box placement center', load_limit=BOX_LOAD_LIMIT, timeout_error=WaypointAlignment)
            except WaypointAlignment:
                pass
            after = float(np.linalg.norm(error()))
            event('center_error', kind='drive', phase='placement', before_mm=round(1000 * before, 1),
                  after_mm=round(1000 * after, 1), tolerance_mm=1000 * CENTER_TOLERANCE)
            return after <= CENTER_TOLERANCE

        def box_still_held(self):
            pressure = self.box_pressure(self.box_hold['forces'])
            engaged = {side: load >= 1. for side, load in pressure.items()}
            event('box_hold_before_release', engaged=engaged,
                  force_change_n={side: round(load, 2) for side, load in pressure.items()})
            return all(engaged.values())

        def continue_raised_box(self, plan, routes):
            goals = plan['lift'][-1]
            yaw = self.angles[WAIST]
            if (abs(yaw - plan['pick_yaw']) > .04
                    or any(np.max(abs(self.q(side) - goals[side])) > .32 for side in SIDES)):
                raise RuntimeError('arms are outside the saved lifted box pose')
            max_inset = np.clip((plan['box_width'] - 2 * plan['box_preload'] - .038) / 2, 0., .022)
            self.box_hold = dict(forces={side: np.zeros(3) for side in SIDES},
                                 clear=plan['box_clear'], shift=np.zeros(3), inset=max_inset,
                                 contact_points=plan.get('box_contact_points', dict.fromkeys(SIDES, PALM_FACE)),
                                 max_inset=max_inset, pressure=np.zeros(2), stamp=self.history[-1][0])
            end = self.set_arms(goals, .24)
            self.box_hold['shift'][2] = .032
            self.send_box_targets()
            self.advance_to(end, 'resume box squeeze', load_limit=BOX_LOAD_LIMIT)
            with measure('turn_to_tray', kind='drive'):
                self.turn(plan['tray_yaw'], plan['item']['release'], routes, load_limit=BOX_LOAD_LIMIT)
            with measure('transfer', kind='drive'):
                self.move(plan['transfer'], load_limit=BOX_LOAD_LIMIT)

        def carry_to_tray(self, plan, routes, completed=0):
            item = plan['item']
            kind = item['kind']
            paired_scoop = 'close' in plan
            paired_carry = plan.get('paired_carry', False)
            hook_carry = plan.get('hook_carry', False)
            hook_grip = {plan['grip_side']: plan['grip'][plan['grip_side']]} if hook_carry else None
            low_slide = plan.get('low_object_slide', False)
            mouse_tolerance = .010 if (kind in TWO_ARM_MOUSE or low_slide) and plan.get('edge_receive') else None
            mouse_hold = fast_batch and paired_carry and mouse_tolerance is not None
            carry_grip = plan['grip'] if mouse_hold else hook_grip
            contact_limit = 6. if plan.get('top_slide') else np.inf
            if completed:
                side = next(iter(plan['preshape']))
                arm = self.arm[side]
                steps = plan.get('touch', {}).get('close', ())
                if (not 0 < completed <= len(steps) or abs(self.angles[WAIST] - plan['pick_yaw']) > .04
                        or np.linalg.norm(arm.wrist(self.q(side))[:3, 3]
                                          - arm.wrist(steps[completed - 1]['q'])[:3, 3]) > .028):
                    raise RuntimeError(f'{kind}: wrist has moved since the recorded pinch step')
            else:
                if not plan.get('prepositioned') and (len(plan['preshape']) == 2 or kind == 'cable'):
                    ready = {side: ready_angles(side) for side in SIDES}
                    tolerance = np.deg2rad(2.)
                    if any(np.max(abs(self.q(side) - goal)) > tolerance for side, goal in ready.items()):
                        self.move([ready], speed=.34, load_limit=8.,
                                  joint_tolerance=None if fast_batch else tolerance, palm_tolerance=mouse_tolerance)
                for side, values in plan['preshape'].items():
                    self.command_hand(side, values)
                with measure('orient', kind=kind):
                    self.turn(plan['pick_yaw'], item['target']['xyz'], routes, load_limit=9.5,
                              speed=READY_WAIST_SPEED if fast_batch else WAIST_SPEED)
                if kind == 'cable' or paired_scoop:
                    for side in plan['preshape']:
                        self.force_reference[side] = self.wrench[side][1]
                        self.load_since.pop(side, None)
                with self.allow_table_contact(kind, plan['preshape'], contact_limit, edge_slide=low_slide):
                    if not self.approach(plan, routes):
                        return False, None
            box_hold = None
            with measure('grasp', kind=kind):
                if paired_scoop:
                    if hook_carry:
                        _, support_before = self.edge_support_force(plan['grip_side'], routes)
                    with self.allow_table_contact(kind, SIDES, contact_limit, edge_slide=low_slide):
                        reason, steps_run = self.close_scoop_pair(plan, routes)
                        if reason:
                            if reason == 'manual_edge_stage':
                                if resume.get('raise_cable_wrist'):
                                    side = plan['grip_side']
                                    arm = self.arm[side]
                                    start = self.q(side)
                                    candidates = []
                                    for limit in arm.limits[:, -1]:
                                        goal = start.copy()
                                        goal[-1] = limit
                                        hand = arm.wrist(goal) @ arm.hand
                                        candidates.append((hand[2, 2], goal))
                                    goal = max(candidates, key=lambda candidate: candidate[0])[1]
                                    self.move([{side: goal}], speed=.55, load_limit=9.5,
                                              position_control=True, hold_hand={side: plan['preshape'][side]})
                                    event('cable_wrist_raised_for_operator', side=side,
                                          joint=arm.names[-1], target_rad=float(goal[-1]),
                                          actual_rad=float(self.q(side)[-1]))
                                event('cable_ready_for_manual_grasp', receiver=plan['grip_side'])
                                return False, None
                            self.withdraw_scoop_pair(plan, steps_run, partial_push=fast_batch and (paired_carry or hook_carry)
                                                     and reason == 'open_scoop_did_not_reach')
                            event('skipped', kind=kind, reason=reason)
                            return False, None
                elif 'touch' in plan:
                    side = next(iter(plan['preshape']))
                    with self.allow_table_contact(kind):
                        closed = self.close_at_table(plan, routes, side, completed)
                    if not closed:
                        self.command_hand(side, plan['preshape'][side])
                        self.raise_and_ready(side, routes, kind, plan['shape'], plan['pocket'])
                        self.ready_hand(side)
                        event('skipped', kind=kind, reason='finger_clearance_outside_tolerance')
                        return False, None
                elif item['family'] == 'two_hands':
                    palm_push = plan.get('box_tip_push_side') and 'precontact' not in plan
                    if palm_push:
                        self.move(plan['grasp'], speed=.45, load_limit=BOX_LOAD_LIMIT)
                    else:
                        box_hold = self.squeeze_box(plan)
                    if not palm_push and box_hold is None:
                        self.grip(plan['preshape'], load_limit=9.5)
                        self.move([*reversed(plan['approach']), {side: ready_angles(side) for side in SIDES}],
                                  speed=.42, load_limit=9.5)
                        for side in SIDES:
                            self.ready_hand(side)
                        event('skipped', kind=kind, reason='no_pressure_from_both_palms')
                        return False, None
                else:
                    self.move(plan['grasp'])
                    self.grip(plan['grip'])
                    if kind == 'gold_jar':
                        side, command = next(iter(plan['grip'].items()))
                        try:
                            self.poll(lambda: self.grip_state(side, command)[0], .55,
                                      'gold jar finger contact', timeout_error=WaypointAlignment)
                        except WaypointAlignment:
                            self.grip_feedback(side, kind, command)
                            self.command_hand(side, plan['preshape'][side])
                            self.raise_and_ready(side, routes, kind, plan['shape'], plan['pocket'])
                            self.ready_hand(side)
                            event('skipped', kind=kind, reason='no_finger_contact_before_lift')
                            return False, None
            probe_only = plan.get('probe_only')
            probe = plan['lift'] if probe_only else plan.get('edge_probe')
            receiver = plan.get('grip_side') if plan.get('edge_receive') else None
            hold_edge = not paired_carry and resume and resume.get('hold_after_edge_grasp')
            if receiver and hold_edge and not probe:
                event('edge_grasp_closed', kind=kind, receiver=receiver)
                return False, None
            if hook_carry:
                departure = plan.get('hook_departure')
                if departure:
                    with measure('hook_departure', kind=kind):
                        self.move(departure, speed=.45 if plan.get('hook_low_transport') else .14 if kind == 'cable' else .26,
                                  load_limit=9.5, joint_tolerance=None if plan.get('hook_low_transport') else .025,
                                  position_control=True, hold_hand=hook_grip)
                else:
                    with measure('hook_pusher_clear', kind=kind):
                        retreat = plan['pusher_retreat']
                        if timed_batch and item.get('adaptive_batch'):
                            self.move(retreat[:1], speed=.28, load_limit=9.5, joint_tolerance=.025,
                                      position_control=True, hold_hand=hook_grip)
                            retreat = retreat[1:]
                        self.move(retreat, speed=.42 if timed_batch and item.get('adaptive_batch') else .28,
                                  load_limit=9.5, joint_tolerance=.025,
                                  position_control=True, hold_hand=hook_grip)
                    with measure('hook_lift', kind=kind):
                        self.move(probe, speed=.24, load_limit=9.5, joint_tolerance=.025, position_control=True,
                                  hold_hand=hook_grip)
                if departure or probe:
                    _, support_after = self.edge_support_force(receiver, routes)
                    event('hook_support', kind=kind, receiver=receiver,
                          load_change_n=round(support_after - support_before, 2))
                if hold_edge:
                    return False, None
            elif paired_scoop and probe:
                with measure('scoop_probe', kind=kind), self.allow_table_contact(kind, SIDES):
                    heights = self.box_palms()[:, 2]
                    self.move(probe, speed=.22, load_limit=9.5,
                              joint_tolerance=None if fast_batch else .028)
                    feedback, _ = self.scoop_feedback(plan['grip'], since=0., receiver=receiver,
                                                     paired=paired_carry)
                    event('probe_feedback', kind=kind, hands=feedback,
                          wrist_lift_mm=dict(zip(SIDES, (1000 * (self.box_palms()[:, 2] - heights)).round(1).tolist())))
                    holding = SIDES if paired_carry else (receiver,) if receiver else ()
                    lost = any(not feedback[side]['loaded'] for side in holding)
                    if receiver and not lost and hold_edge:
                        event('edge_held', kind=kind, receiver=receiver)
                        return False, None
                    if probe_only or lost:
                        self.move(plan['probe_return'], speed=.22, load_limit=9.5, joint_tolerance=.028)
                        self.withdraw_scoop_pair(plan, len(plan['close']))
                        if lost:
                            event('skipped', kind=kind, reason='empty_receiver_after_probe')
                        return False, None
            if probe_only:
                with measure('mouse_probe', kind=kind):
                    self.move(plan['lift'], speed=.20, joint_tolerance=.028)
                    empty = {side: self.grip_feedback(side, kind, plan['grip'][side]) for side in SIDES}
                    event('probe_feedback', kind=kind, commanded_lift_mm=30, empty=empty)
                    self.move(plan['pickup_retreat'], speed=.20, joint_tolerance=.028)
                    self.grip(plan['preshape'])
                    self.move(plan['retreat'], speed=.34)
                    for side in SIDES:
                        self.ready_hand(side)
                return False, None
            held_offsets = object_offset = None
            if 'pocket' in plan:
                side = next(iter(plan['preshape']))
                corners, object_offset = self.held_geometry(side, routes, item['target'])
                held_offsets = np.concatenate((corners, plan['shape'] - plan['pocket']))
                event('held_offset', kind=kind, offset_m=object_offset.tolist())
            load_limit = BOX_LOAD_LIMIT if box_hold is not None or kind in ('drive', 'cable') else 9.5 if paired_scoop else 6.
            if 'box_table_tip' in plan:
                push_side = plan.get('box_tip_push_side')
                if push_side:
                    self.box_hold = None
                    self.move(plan.get('box_tip_clear_support', ()), speed=.48, load_limit=load_limit)
                    with measure('box_table_tip', kind=kind):
                        tip = plan['box_table_tip']
                        self.move(tip, speed=.68, load_limit=load_limit)
                        try:
                            self.align_waypoint({push_side: tip[-1][push_side]}, .030, load_limit)
                        except WaypointAlignment:
                            pass
                    arm = self.arm[push_side]
                    actual = arm.wrist(self.q(push_side)) @ arm.hand
                    target = arm.wrist(tip[-1][push_side]) @ arm.hand
                    palm_error = (actual[:3, :3] - target[:3, :3]) @ PALM_FACE + actual[:3, 3] - target[:3, 3]
                    event('box_table_push_feedback', kind=kind, side=push_side,
                          commanded_deg=plan['box_tip_angle'],
                          palm_error_mm=round(float(1000 * np.linalg.norm(palm_error)), 1))
                else:
                    initial_axis = self.box_axis()
                    self.box_hold['align_palms'] = False
                    self.box_hold['min_inset'] = min(.004, self.box_hold['max_inset'])
                    with measure('box_table_tip', kind=kind):
                        self.move(plan['box_table_tip'], speed=.65, load_limit=load_limit)
                    final_axis = self.box_axis()
                    angle = np.rad2deg(np.arccos(np.clip(initial_axis @ final_axis, -1., 1.)))
                    event('box_table_tip_feedback', kind=kind, commanded_deg=plan['box_tip_angle'],
                          angle_deg=round(float(angle), 1), final_axis_z=round(float(final_axis[2]), 3),
                          force_change_n={side: round(force, 2)
                                          for side, force in self.box_pressure(box_hold[1]).items()})
                with measure('box_tip_withdraw', kind=kind):
                    self.box_hold = None
                    for side, command in plan['preshape'].items():
                        self.command_hand(side, command)
                    self.move(plan['box_tip_retreat'], speed=.40, load_limit=load_limit)
                    for side in SIDES:
                        self.ready_hand(side)
                event('box_table_tip_done', kind=kind)
                return False, None
            with measure('lift', kind=kind):
                if 'touch' in plan:
                    with self.allow_table_contact(kind):
                        self.lift_from_table(plan, routes, side, load_limit=load_limit)
                with self.allow_table_contact(kind, plan['preshape'], edge_slide=low_slide):
                    self.move(plan['lift'], speed=(.48 if timed_batch else .34) if mouse_hold else .55,
                              load_limit=load_limit, position_control=hook_carry or mouse_hold,
                              joint_tolerance=.025 if hook_carry and not plan.get('hook_low_transport') else None, palm_tolerance=mouse_tolerance,
                              hold_hand=carry_grip, continuous=timed_batch)
                if paired_scoop:
                    feedback, _ = self.scoop_feedback(plan['grip'], since=0., receiver=receiver,
                                                     paired=paired_carry)
                    event('scoop_after_lift', kind=kind, hands=feedback)
                    if (paired_carry and not (fast_batch and kind in TWO_ARM_MOUSE)
                            and any(not feedback[side]['loaded'] for side in SIDES)):
                        raise RuntimeError(f'{kind}: lost two-hand contact during lift')
                if held_offsets is not None:
                    self.raise_above(side, routes, kind, 'lift_loaded', plan['shape'], held_offsets)
                if box_hold is not None:
                    box_height, forces = box_hold
                    bottom = item['target']['bounds'][0][2]
                    obstacle_top = max((obj['bounds'][1][2] for obj in routes.objects
                                        if obj['kind'] != kind), default=routes.floor)

                    def clearance():
                        return bottom + np.min(self.box_palms()[:, 2] - box_height) - obstacle_top

                    if clearance() < TABLE_GAP:
                        self.box_hold['shift'][2] = np.clip(TABLE_GAP - clearance() + .006, 0., .050)
                        self.send_box_targets()
                        try:
                            self.poll(lambda: clearance() >= TABLE_GAP, 3., 'box lift height',
                                      load_limit=load_limit, timeout_error=WaypointAlignment)
                        except WaypointAlignment:
                            pass
                    pressure = self.box_pressure(forces)
                    event('box_pressure_after_lift', force_change_n={side: round(load, 2) for side, load in pressure.items()})
                    if min(pressure.values()) < .65 and not self.box_direct:
                        self.box_hold = None
                        self.grip(plan['preshape'], load_limit=load_limit)
                        self.move(plan['pickup_retreat'], speed=.42, load_limit=load_limit)
                        for side in SIDES:
                            self.ready_hand(side)
                        event('skipped', kind=kind, reason='box_slipped_during_lift')
                        return False, None
                    remaining = clearance()
                    required = 0. if self.box_direct else TABLE_GAP
                    if remaining < required:
                        raise RuntimeError(f'{kind}: lift clearance {1000 * remaining:.1f} mm')
                    event('clearance_before_transfer', kind=kind, mm=round(1000 * remaining, 1))
            if held_offsets is not None and self.grip_feedback(side, kind, plan['grip'][side]):
                self.command_hand(side, plan['preshape'][side])
                self.raise_and_ready(side, routes, kind, plan['shape'], plan['pocket'])
                self.ready_hand(side)
                event('skipped', kind=kind, reason='empty_grip')
                return False, None
            transfer_limit = load_limit if box_hold is not None or 'close' in plan else 6.
            coordinated = timed_batch and plan.get('coordinated_transfer')
            if plan['tray_yaw'] != plan['pick_yaw'] and not coordinated:
                with measure('turn_to_tray', kind=kind):
                    if carry_grip:
                        for side, command in carry_grip.items():
                            self.command_hand(side, command)
                    self.turn(plan['tray_yaw'], item['release'], routes, load_limit=transfer_limit,
                              speed=.42 if timed_batch else WAIST_SPEED)
            elif kind == 'lotion':
                event('waist_held', kind=kind, yaw_rad=round(self.angles[WAIST], 4))
            if box_hold is not None:
                event('box_pressure_after_turn', force_change_n={
                    side: round(load, 2) for side, load in self.box_pressure(box_hold[1]).items()})
            with measure('transfer', kind=kind):
                if coordinated:
                    self.coordinated_move(coordinated, speed=.48, hold_hand=carry_grip)
                    event('turn_overlapped_transfer', kind=kind)
                elif box_hold is not None and fast_batch:
                    for step, goals in enumerate(plan['transfer'], 1):
                        self.move([goals], speed=.42 if timed_batch else .30, load_limit=transfer_limit)
                        palms = self.box_palms()
                        event('box_transfer', step=step, span_mm=round(1000 * np.linalg.norm(palms[0] - palms[1]), 1),
                              force_change_n={side: round(load, 2) for side, load in
                                              self.box_pressure(box_hold[1]).items()})
                else:
                    self.move(plan['transfer'], speed=(.48 if timed_batch else .34) if mouse_hold else .55,
                              load_limit=transfer_limit, position_control=hook_carry or mouse_hold,
                              joint_tolerance=.025 if hook_carry and not plan.get('hook_low_transport') else None, palm_tolerance=mouse_tolerance,
                              hold_hand=carry_grip, continuous=timed_batch)
            if mouse_hold:
                feedback, _ = self.scoop_feedback(plan['grip'], 0., receiver=receiver, paired=True)
                event('mouse_hold_after_transfer', hands=feedback)
            if fast_batch and box_hold is not None:
                event('box_pressure_after_transfer', force_change_n={
                    side: round(load, 2) for side, load in self.box_pressure(box_hold[1]).items()})
            return True, object_offset

        def lower_hook_at_tray(self, plan, routes):
            side = plan['grip_side']
            arm = self.arm[side]
            start = self.q(side).copy()
            initial = arm.tool(start)
            hand = arm.wrist(start) @ arm.hand
            low, high = np.asarray(plan['item']['tray']['bounds'])
            # Reach the near part of the tray using the encoder-driven placement shown in this scene.
            support_world = np.minimum(low + np.array([.082, .080, .028]), high - .025)
            support_local = routes.local(support_world, self.angles[WAIST])
            receive = np.asarray(plan['receive_point'])
            point = support_local + hand[:3, :3] @ (arm.pocket - receive)
            lower, upper = arm.limits + [[.018], [-.018]]

            def residual(values):
                pose = arm.tool(values)
                turn = Rotation.from_matrix(initial[:3, :3].T @ pose[:3, :3]).as_rotvec()
                return np.r_[pose[:3, 3] - point, .20 * turn, .00003 * (values - start)]

            goal = least_squares(residual, np.clip(start, lower + 1e-7, upper - 1e-7),
                                 bounds=(lower, upper), max_nfev=60, ftol=1e-6, gtol=1e-8).x
            self.move([{side: goal}], speed=.40, load_limit=9.5,
                      position_control=True, hold_hand={side: plan['grip'][side]})
            event('hook_lowered_for_operator', target_support_m=support_world.tolist())

        def withdraw_cable_demonstration(self, plan):
            side = plan['grip_side']
            waypoints = cable_withdrawal_waypoints(self.arm[side], self.q(side))
            self.move([{side: q} for q in waypoints], speed=.55, load_limit=9.5,
                      position_control=True, hold_hand={side: OPEN_LEFT})
            event('cable_operator_withdrawal', side=side, waypoints=len(waypoints))

        def withdraw_cable_pinch(self, plan):
            side = plan['grip_side']
            arm = self.arm[side]
            start = self.q(side)
            pose = arm.tool(start)
            goal = fit_contact(arm, pose[:3, 3] + [0., 0., .14], pose[:3, :3], start)
            if goal is None:
                raise RuntimeError('cable: no vertical withdrawal after two-finger release')
            self.move([{side: goal}, {side: ready_angles(side)}], speed=.50, load_limit=9.5,
                      position_control=True, hold_hand={side: OPEN_LEFT})
            event('cable_pinch_withdrawal', side=side, rise_mm=140)

        def dump_cable_in_tray(self, routes, tray, *, carry_from_current=False):
            side = 'left'
            arm = self.arm[side]
            start = self.q(side).copy()
            low, high = np.asarray(tray['bounds'])
            center = np.asarray(tray['xyz'])[:2]
            outward = center / np.linalg.norm(center)
            site = np.clip(center + .025 * outward, low[:2] + .065, high[:2] - .065)
            yaw = routes.heading(tray['xyz'], side) if carry_from_current else self.angles[WAIST]
            lower, upper = arm.limits + [[.018], [-.018]]
            receive = np.array([.0543, .0286, .1694])

            def solve(point, orient, seed):
                def residual(values):
                    pose = arm.tool(values)
                    turn = Rotation.from_matrix(orient.T @ pose[:3, :3]).as_rotvec()
                    return np.r_[pose[:3, 3] - point, .20 * turn, .00003 * (values - seed)]
                fit = least_squares(residual, np.clip(seed, lower + 1e-7, upper - 1e-7),
                                    bounds=(lower, upper), max_nfev=100)
                if np.linalg.norm(fit.fun[:3]) > .012 or np.linalg.norm(fit.fun[3:6]) > .012:
                    raise RuntimeError('cable: inverted release or upward retreat is outside arm reach')
                return fit.x

            raised = start
            if carry_from_current:
                pose = arm.tool(start)
                raised = solve(pose[:3, 3] + [0., 0., .12], pose[:3, :3], start)
            fingers = np.array([1., 0., 0.])
            normal = np.array([0., 0., 1.])
            upright = np.column_stack((normal, np.cross(fingers, normal), fingers))
            coil = routes.local(np.r_[site, .20], yaw)
            point = coil + upright @ (arm.pocket - receive)
            above = solve(point, upright, raised)
            roll = []
            seed = above
            for angle in np.linspace(0., np.pi, 13)[1:]:
                orient = Rotation.from_rotvec(angle * fingers).as_matrix() @ upright
                point = coil + orient @ (arm.pocket - receive)
                seed = solve(point, orient, seed)
                roll.append(seed)
            # Raise vertically before any return across the occupied tray.
            retreat = []
            for rise in (.045, .09, .15):
                seed = solve(point + [0., 0., rise], orient, seed)
                retreat.append(seed)
            event('cable_roll_dump_plan', roll_steps=len(roll), retreat_rise_mm=150,
                  yaw_rad=float(yaw), release_center_m=np.r_[site, .20].tolist())
            closed = {side: np.zeros(6)}
            if carry_from_current:
                self.grip(closed, load_limit=9.5)
                self.move([{side: raised}], speed=.55, load_limit=9.5,
                          position_control=True, hold_hand=closed)
                self.turn(yaw, tray['xyz'], routes, load_limit=9.5, speed=.50)
            self.move([{side: above}, *[{side: q} for q in roll]], speed=.60,
                      load_limit=9.5, position_control=True, hold_hand=closed)
            self.grip({side: OPEN_LEFT}, load_limit=9.5)
            self.move([{side: q} for q in retreat], speed=.60, load_limit=9.5,
                      position_control=True, hold_hand={side: OPEN_LEFT})
            self.move([{side: ready_angles(side)}], speed=.60, load_limit=9.5,
                      position_control=True, hold_hand={side: OPEN_LEFT})
            event('cable_roll_dump_executed', side=side, palm_down=True, retreat_rise_mm=150)

        def resume_held_offset(self, plan, routes, held):
            side = next(iter(plan['preshape']))
            arm = self.arm[side]
            arm.pocket = plan['pocket']
            kind = plan['item']['kind']
            if self.grip_feedback(side, kind, plan['grip'][side]):
                raise RuntimeError(f'{kind}: cannot resume with an empty hand')
            if 'offset_m' in held:
                return np.array(held['offset_m'])
            pose = arm.tool(plan['approach'][-1][side])
            orient = rotation(plan['pick_yaw'] - routes.yaw) @ pose[:3, :3]
            offset = np.subtract(plan['item']['target']['xyz'], plan['pickup_center']) @ orient
            event('resume_held_offset_estimate', kind=kind, offset_m=offset.tolist())
            return offset

        def run(self, worker):
            if resume and resume.get('dump_cable'):
                settings = self.connect()
                self.poll(lambda: all(f.done() for _, _, f in settings), 4., 'hand settings')
                for key, side, setting in settings:
                    if not getattr(setting.result(), f'{key}_accepted'):
                        raise RuntimeError(f'{side}: {key} setting rejected')
                self.dump_cable_in_tray(Routes(resume['scene'], resume['yaw']),
                                        resume['dump_cable'], carry_from_current=True)
                return
            if resume and 'retry_mouse_shell' in resume:
                self.retry_mouse_shell(Routes(resume['scene'], resume['yaw']), resume['retry_mouse_shell'])
                return
            if resume and 'press_mouse_fingers' in resume:
                self.press_mouse_fingers(resume['press_mouse_fingers'])
                return
            if resume and 'receive_right' in resume:
                self.position_right_receiver(Routes(resume['scene'], resume['yaw']), resume['receive_right'])
                return
            if resume and 'press_mouse_left' in resume:
                self.press_mouse_left(Routes(resume['scene'], resume['yaw']), resume['press_mouse_left'])
                return
            if resume and 'slide_mouse_left' in resume:
                self.slide_mouse_left(Routes(resume['scene'], resume['yaw']),
                                      resume['slide_mouse_left'], resume['queue'][0])
                return
            if resume:
                scene, yaw, queue = (resume[key] for key in ('scene', 'yaw', 'queue'))
            else:
                scene, yaw, queue = self.survey(worker)
            if cup_mouse and not adaptive_batch and not resume and not skip_box and selected is None:
                pending_kinds = {item['kind'] for item in queue}
                missing = [kind for kind, (_, name) in CATALOG.items()
                           if name in self.task['objects'] and kind not in pending_kinds]
                if missing:
                    raise RuntimeError('batch: objects absent from the fresh queue: ' + ', '.join(missing))
            if not queue:
                return
            held = resume.get('held') if resume else None
            closing = resume.get('closing') if resume else None
            if held and held['kind'] != queue[0]['kind']:
                raise RuntimeError('held object must be first in the saved queue')
            if closing and (held or closing['kind'] != queue[0]['kind']):
                raise RuntimeError('object being pinched must be first in the saved queue')
            routes = Routes(scene, yaw)
            reserved = {}

            def needs_current_pose(kind):
                return (edge_batch and kind in TABLE_PINCH) or (flat_mouse_receiver and kind in TWO_ARM_MOUSE)

            def prepare(item, angles, box_removed, start_yaw, geometry=routes, placements=reserved):
                with measure('ik', kind=item['kind']):
                    if resume and item['kind'] in resume.get('prepared_plans', {}):
                        if abs(start_yaw - resume['prepared_start_yaw']) > np.deg2rad(4.):
                            raise RuntimeError('prepared route: torso has moved since planning')
                        if any(np.max(abs(np.array([angles[name] for name in self.arm[side].names])
                                          - ready_angles(side))) > np.deg2rad(3.) for side in SIDES):
                            raise RuntimeError('prepared route: arms are outside the planned start')
                        def restore(value):
                            if isinstance(value, dict):
                                return {key: np.asarray(row) if key in SIDES and isinstance(row, list) and len(row) == 7
                                        else restore(row) for key, row in value.items()}
                            return [restore(row) for row in value] if isinstance(value, list) else value
                        return restore(resume['prepared_plans'][item['kind']])
                    guarded = needs_current_pose(item['kind']) or (
                        item.get('allow_low_slide') and supports_low_object_slide(item['target']))
                    return next_route(geometry, item, placements, box_removed=box_removed,
                                      resume_pinch=bool((closing or held) and (closing or held)['kind'] == item['kind']),
                                      cable_probe=bool(resume and resume.get('cable_probe')),
                                      flat_mouse_receiver=flat_mouse_receiver and item['kind'] in TWO_ARM_MOUSE,
                                      lower_body=LowerBody(angles) if guarded else None,
                                      start_yaw=start_yaw if guarded else None, edge_batch=edge_batch,
                                      upright_box=upright_box, leave_cable_hand=leave_cable_hand)

            def submit(item):
                return worker.submit(prepare, item, self.angles, self.box_removed, self.angles[WAIST])

            # Only the independent cylinder prefix may execute before the full
            # batch is ready. All planning uses the same frozen survey/encoders.
            prefix_count = (3 if fast_batch and timed_batch and not resume and len(queue) > 3
                            and all(item['family'] == 'cylinder' for item in queue[:3]) else 0)
            prefix_ready = Future()

            def publish_prefix(plans, angles):
                if any(plan is None for plan in plans):
                    raise RuntimeError('batch: incomplete cylinder prefix')
                cleared, placed = set(), []
                legs = LowerBody(angles)
                # Finish these transitions before publication. The worker will
                # never mutate a prefix plan while the motion thread uses it.
                for plan, following in zip(plans, plans[1:]):
                    item = plan['item']
                    cleared.add(item['kind'])
                    size = np.ptp(item['target']['bounds'], axis=0)
                    center = np.asarray(item['release'], dtype=float)
                    placed.append(dict(kind='placed_' + item['kind'],
                                       bounds=[center - size / 2 - .015, center + size / 2 + .015]))
                    objects = [obj for obj in scene['objects'] if obj['kind'] not in cleared]
                    geometry = Routes(dict(scene, objects=objects + placed), yaw)
                    prefix = plan['retreat'][:1]
                    start = {side: ready_angles(side).copy() for side in SIDES}
                    for step in [*plan['release'][-1:], *prefix]:
                        start.update(step)
                    end = {side: ready_angles(side).copy() for side in SIDES}
                    end.update(following['approach'][0])
                    commands = {side: [working_hand_command(side),
                                plan.get('release_hand', plan['preshape']).get(side, working_hand_command(side)),
                                following['preshape'].get(side, working_hand_command(side))]
                                for side in SIDES}
                    with measure('transition_plan', kind=item['kind'], next_kind=following['item']['kind']):
                        bridge = empty_transition(geometry, start, plan['tray_yaw'], end,
                                                  following['pick_yaw'], commands, legs)
                    if bridge:
                        plan['exit_transition'] = dict(prefix=prefix, steps=bridge)
                    event('transition_route', kind=item['kind'], next_kind=following['item']['kind'],
                          direct=bool(bridge), prepared_with_prefix=True)
                prefix_ready.set_result(list(plans))
                event('batch_prefix_ready', kinds=[plan['item']['kind'] for plan in plans])

            def prepare_batch(angles):
                plans = []
                start_yaw = angles[WAIST]
                cleared = set()
                footprints = {}
                content_heights = {}
                for index, item in enumerate(queue):
                    planning_scene = dict(scene, objects=[obj for obj in scene['objects']
                                                          if obj['kind'] not in cleared])
                    tray_kind = item['tray']['kind']
                    item = dict(item, timed_batch=timed_batch, occupied_footprints=footprints.get(
                        tray_kind, item.get('occupied_footprints', {})),
                        occupied_heights=content_heights.get(tray_kind, {}))
                    if item['kind'] in TWO_ARM_MOUSE:
                        # Keep the verified slide and the separate turn with the mouse held.
                        item = dict(item, occupied_footprints={}, occupied_heights={}, cup_mouse=cup_mouse,
                                    cup_trial_release=cup_trial_release, mouse_top_push=False,
                                    mouse_table_sweep=False, timed_batch=False,
                                    mouse_last_in_batch=index == len(queue) - 1)
                    geometry = Routes(planning_scene, yaw)
                    plan = prepare(item, angles, self.box_removed or 'drive' in cleared, start_yaw, geometry)
                    plans.append(plan)
                    if plan is not None:
                        if not plan.get('probe_only') and not plan.get('box_table_tip'):
                            cleared.add(item['kind'])
                        if 'release_footprint' in plan:
                            footprints.setdefault(tray_kind, {})[item['kind']] = plan['release_footprint']
                            if 'release_content_top_m' in plan:
                                content_heights.setdefault(tray_kind, {})[item['kind']] = plan['release_content_top_m']
                        start_yaw = plan.get('tray_yaw', plan['pick_yaw'])
                    if prefix_count and len(plans) == prefix_count:
                        publish_prefix(plans, angles)
                if timed_batch:
                    cleared, placed = set(), []
                    legs = LowerBody(angles)
                    for index, (plan, following) in enumerate(zip(plans, plans[1:])):
                        if plan is None or following is None:
                            continue
                        item = plan['item']
                        cleared.add(item['kind'])
                        objects = [obj for obj in scene['objects'] if obj['kind'] not in cleared]
                        # Include already deposited contents when checking the empty hands.
                        target = item['target']
                        size = np.ptp(target['bounds'], axis=0)
                        center = np.asarray(item['release'], dtype=float)
                        if 'release_footprint' in plan:
                            lo, hi = np.asarray(plan['release_footprint'], dtype=float)
                            bottom = item['tray']['bounds'][0][2]
                            bounds = [np.r_[lo, bottom], np.r_[hi, plan.get(
                                'release_content_top_m', bottom + size[2])]]
                        else:
                            bounds = [center - size / 2 - .015, center + size / 2 + .015]
                        placed.append(dict(kind='placed_' + item['kind'], bounds=np.asarray(bounds).tolist()))
                        if prefix_count and index < prefix_count:
                            # Already published, including its completed exit.
                            # The final prefix item retains the normal READY exit.
                            continue
                        geometry = Routes(dict(scene, objects=objects + placed), yaw)
                        if plan.get('hook_low_transport'):
                            opened = {side: [plan['release_hand'].get(side, working_hand_command(side))]
                                      for side in SIDES}
                            ready = {side: ready_angles(side).copy() for side in SIDES}
                            path = plan['cable_withdraw_high']
                            bridge = None
                            with measure('transition_plan', kind=item['kind'], next_kind=following['item']['kind']):
                                start = dict(ready, **plan['retreat'][-1])
                                target = dict(start, **path[0])
                                target_yaw = plan['tray_yaw'] + .20 * (
                                    following['pick_yaw'] - plan['tray_yaw'])
                                # The vertical detach is already complete. Clear the
                                # tray by 60 mm, then return directly to READY.
                                outward = empty_transition(geometry, start, plan['tray_yaw'], target,
                                                            target_yaw, opened, legs)
                                home = (empty_transition(geometry, target, target_yaw, ready,
                                                         target_yaw, opened, legs)
                                        if outward is not None else None)
                                if home is not None:
                                    # Each entire joint-linear edge was sampled above.
                                    # Keep its endpoint: tracking every audit sample
                                    # added repeated deceleration to the empty-hand exit.
                                    bridge = [outward[-1], home[-1]]
                                    plan['exit_transition'] = dict(prefix=[], steps=bridge,
                                                                   prepositioned=False,
                                                                   cable_short_exit=True)
                            event('transition_route', kind=item['kind'], next_kind=following['item']['kind'],
                                  direct=bool(bridge))
                            continue
                        prefix = plan['retreat'][:1]
                        start = {side: ready_angles(side).copy() for side in SIDES}
                        for step in [*plan['release'][-1:], *prefix]:
                            start.update(step)
                        end = {side: ready_angles(side).copy() for side in SIDES}
                        end.update(following['approach'][0])
                        commands = {side: [working_hand_command(side),
                                    plan.get('release_hand', plan['preshape']).get(side, working_hand_command(side)),
                                    following['preshape'].get(side, working_hand_command(side))]
                                    for side in SIDES}
                        with measure('transition_plan', kind=item['kind'], next_kind=following['item']['kind']):
                            bridge = empty_transition(geometry, start, plan['tray_yaw'], end,
                                                      following['pick_yaw'], commands, legs)
                        if bridge:
                            plan['exit_transition'] = dict(prefix=prefix, steps=bridge)
                        elif item['kind'] == 'drive':
                            # Once the opened hands clear the box, withdraw and
                            # turn together; retain the next object's approach.
                            ready = {side: ready_angles(side).copy() for side in SIDES}
                            for count in (1, 2, 3):
                                prefix = plan['retreat'][:count]
                                start = {side: ready_angles(side).copy() for side in SIDES}
                                for step in [*plan['release'][-1:], *prefix]:
                                    start.update(step)
                                bridge = empty_transition(geometry, start, plan['tray_yaw'], ready,
                                                          following['pick_yaw'], commands, legs)
                                if bridge:
                                    plan['exit_transition'] = dict(prefix=prefix, steps=bridge,
                                                                   prepositioned=False)
                                    break
                        event('transition_route', kind=item['kind'], next_kind=following['item']['kind'],
                              direct=bool(bridge))
                return plans

            if not fast_batch:
                pending = None if needs_current_pose(queue[0]['kind']) else submit(queue[0])
            if resume:
                with measure('resume_prepare'):
                    settings = self.connect()
                    self.poll(lambda: all(f.done() for _, _, f in settings), 4., 'hand settings')
                    for key, side, setting in settings:
                        if not getattr(setting.result(), f'{key}_accepted'):
                            raise RuntimeError(f'{side}: {key} setting rejected')
                    if resume.get('complete_cable_release'):
                        self.grip({'left': OPEN_LEFT}, load_limit=9.5)
                        event('released', kind='cable', resumed_at_release=True)
                        self.withdraw_cable_demonstration({'grip_side': 'left'})
                        self.ready_hand('left')
                        self.move([{side: ready_angles(side) for side in SIDES}],
                                  speed=.55, load_limit=9.5, position_control=True)
                        event('cycle_executed', kind='cable', hands=['left', 'right'])
                    if resume.get('fresh_start'):
                        for side in SIDES:
                            self.ready_hand(side)
                        self.position('waist', [(31, 0., .16)])
                        self.move([{side: ready_angles(side) for side in SIDES}], speed=.55, load_limit=9.5)
                    for side, family in resume.get('recover', {}).items():
                        recovery = Routes(scene, yaw)
                        if family == 'cable':
                            with self.allow_table_contact(family):
                                self.grip({side: OPEN_LEFT}, load_limit=9.5)
                                arm = self.arm[side]
                                start = self.q(side)
                                raised = start.copy()
                                elbow = arm.names.index(f'elbow_pitch_{side[0]}_joint')
                                raised[elbow] = max(arm.limits[0, elbow] + .02, start[elbow] - .32)
                                solid, tips = self.contact_vertices(side, recovery, OPEN_LEFT)
                                vertices = np.concatenate([solid, *tips.values()])
                                ready = ready_angles(side)
                                current_yaw = self.angles[WAIST]
                                if not recovery.link(side, raised, ready, vertices, current_yaw):
                                    raise RuntimeError('resume: no elevated cable withdrawal')
                                self.move([{side: raised}], speed=.40, load_limit=9.5)
                                self.poll(lambda: recovery.link(side, self.q(side), ready, vertices, current_yaw),
                                          2., 'cable withdrawal', load_limit=9.5)
                                self.move([{side: ready}], speed=.55, load_limit=9.5)
                            self.ready_hand(side)
                            continue
                        shape = recovery.shape(side, family, CLOSE[family])
                        self.raise_and_ready(side, recovery, 'resume', shape, recovery.arms[side].pocket)
                        self.ready_hand(side)
            def validate_batch(plans):
                missing = [item['kind'] for item, plan in zip(queue, plans) if plan is None]
                event('batch_routes', reachable=[item['kind'] for item, plan in zip(queue, plans) if plan],
                      missing=missing)
                if cup_mouse:
                    incomplete = [item['kind'] for item, plan in zip(queue, plans)
                                  if plan and item['kind'] in TWO_ARM_MOUSE and not (
                                      plan.get('cup_release') or (cup_trial_release and
                                      0. <= plan.get('cup_gap_above_rim_m', np.inf) <= .220))]
                    for item, plan in zip(queue, plans):
                        if item['kind'] in incomplete and 'cup_gap_above_rim_m' in plan:
                            event('mouse_placement_too_high', kind=item['kind'],
                                  clearance_mm=round(1000 * plan['cup_gap_above_rim_m'], 1))
                    if missing or incomplete:
                        raise RuntimeError('batch: incomplete routes for ' + ', '.join(missing + incomplete))
                if any(item['kind'] in TWO_ARM_MOUSE and plan is None for item, plan in zip(queue, plans)):
                    raise RuntimeError('mouse: complete two-hand route required for the batch')

            deferred, slide_retries, mouse_retries, cleared = [], [], [], set()
            retry_hands = {}
            footprints, content_heights = {}, {}
            refresh_needed = False
            observed = {item['kind']: item for item in queue}

            def observed_contents(world_scene):
                contents, heights = {}, {}
                for tray, bounds in world_scene.get('tray_contents', {}).items():
                    bounds = np.asarray(bounds)
                    contents[tray] = {'observed': bounds[:, :2].tolist()}
                    heights[tray] = {'observed': bounds[1, 2]}
                trays = [obj for obj in world_scene['objects'] if obj['kind'].startswith('tray_')]
                for obj in world_scene['objects']:
                    if obj['kind'].startswith('tray_') or obj['status'] != 'observed':
                        continue
                    center = np.asarray(obj['xyz'])[:2]
                    for tray in trays:
                        low, high = np.asarray(tray['bounds'])[:, :2]
                        if np.all((low <= center) & (center <= high)):
                            kind, name = obj['kind'], tray['kind']
                            bounds = np.asarray(obj['bounds'])
                            contents.setdefault(name, {})[kind] = bounds[:, :2].tolist()
                            heights.setdefault(name, {})[kind] = bounds[1, 2]
                            break
                return contents, heights

            if adaptive_batch:
                footprints, content_heights = observed_contents(scene)

            primary = [item for item in queue if item['kind'] not in TWO_ARM_MOUSE]
            mice = [item for item in queue if item['kind'] in TWO_ARM_MOUSE]

            def retry_order(pending, world_scene, items):
                updated = [items.get(item['kind'], item) for item in pending]
                edge = world_scene.get('table_geometry', {}).get('edge_x_m')
                return prioritize_by_size(updated, edge_x=edge)

            def adaptive_sequence():
                yield from ((1, item) for item in primary)
                retry = retry_order(deferred, scene, observed)
                if retry:
                    event('retry_pass', kinds=[item['kind'] for item in retry])
                    yield from ((2, item) for item in retry)
                yield from ((3, item) for item in retry_order(slide_retries, scene, observed))
                yield from ((2, item) for item in mice)
                yield from ((3, item) for item in mouse_retries)

            def remember_placement(plan, removed, placements, occupied, heights):
                item = plan['item']
                kind, tray = item['kind'], item['tray']['kind']
                removed.add(kind)
                radius = np.max(np.ptp(item['target']['bounds'], axis=0)[:2]) / 2
                placements.setdefault(tray, []).append((item['release'][:2], radius))
                if 'release_footprint' in plan:
                    occupied.setdefault(tray, {})[kind] = plan['release_footprint']
                    bottom = item['tray']['bounds'][0][2]
                    height = np.ptp(item['target']['bounds'], axis=0)[2]
                    heights.setdefault(tray, {})[kind] = plan.get('release_content_top_m', bottom + height)

            def plan_adaptive(item, attempt, world_scene, survey_yaw, angles, start_yaw, removed,
                              placements, occupied, heights, preferred_hand=None):
                kind, tray = item['kind'], item['tray']['kind']
                item = dict(item, adaptive_batch=True, timed_batch=timed_batch,
                            low_slide_only=attempt == 3 and kind not in TWO_ARM_MOUSE,
                            occupied_footprints=occupied.get(tray, {}),
                            occupied_heights=heights.get(tray, {}),
                            allow_low_slide=attempt >= 2 and kind not in TWO_ARM_MOUSE)
                if preferred_hand:
                    item['preferred_hand'] = preferred_hand
                if kind == 'cable':
                    item['planning_seconds'] = 28. if attempt == 1 else 46.
                if kind in TWO_ARM_MOUSE:
                    item.update(cup_mouse=cup_mouse, cup_trial_release=cup_trial_release, timed_batch=False,
                                parallel_receiver=True, mouse_last_in_batch=len(mice) == 1,
                                planning_seconds=75. if primary else 110.)
                planning_scene = dict(world_scene, objects=[obj for obj in world_scene['objects']
                                                             if obj['kind'] not in removed])
                geometry = Routes(planning_scene, survey_yaw)
                plan = prepare(item, angles, 'drive' in removed, start_yaw, geometry,
                               {name: list(sites) for name, sites in placements.items()})
                if plan is None and not primary and attempt == 2 and cup_mouse and kind in TWO_ARM_MOUSE:
                    plan = prepare(dict(item, mouse_stage_only=True, planning_seconds=18.), angles,
                                   'drive' in removed, start_yaw, geometry, {})
                if (plan is None or plan.get('probe_only') or plan.get('box_table_tip')
                        or (cup_mouse and kind in TWO_ARM_MOUSE and not (
                            plan.get('mouse_staging_only') or plan.get('cup_release') or (cup_trial_release and
                            0. <= plan.get('cup_gap_above_rim_m', np.inf) <= .220)))):
                    plan = None
                return geometry, plan

            projected = {(attempt, item['kind']): Future()
                         for item in queue for attempt in ((2,) if item['kind'] in TWO_ARM_MOUSE else (1, 2))}
            planned_exits, exit_yaws = {}, {}
            box_contents = Future()

            def apply_box_observation(tray, bounds, occupied, heights):
                if bounds is not None:
                    bounds = np.asarray(bounds)
                    occupied.setdefault(tray, {})['drive'] = bounds[:, :2].tolist()
                    heights.setdefault(tray, {})['drive'] = bounds[1, 2]

            def plan_ahead(world_scene, survey_yaw, angles, known):
                removed, pending = set(), []
                placements = {}
                occupied, heights = observed_contents(world_scene)
                start_yaw = angles[WAIST]
                previous = None

                def schedule():
                    yield from ((1, item) for item in primary)
                    yield from ((2, item) for item in retry_order(pending, world_scene, known))
                    yield from ((2, item) for item in mice)

                for attempt, item in schedule():
                    if cancel_lookahead.is_set():
                        return
                    key = (attempt, item['kind'])
                    if item['kind'] in TWO_ARM_MOUSE and 'drive' in removed:
                        while not box_contents.done():
                            if cancel_lookahead.wait(.1):
                                return
                        apply_box_observation(item['tray']['kind'], box_contents.result(), occupied, heights)
                    geometry, plan = plan_adaptive(item, attempt, world_scene, survey_yaw, angles, start_yaw,
                                                   removed, placements, occupied, heights)
                    projected[key].set_result((frozenset(removed), start_yaw, geometry, plan))
                    if plan:
                        if previous and (previous[1].get('hook_low_transport') or
                                         previous[1]['item']['kind'] == 'drive'):
                            old_key, departing = previous
                            objects = list(geometry.objects)
                            for tray, contents in occupied.items():
                                tray_obj = next(obj for obj in world_scene['objects'] if obj['kind'] == tray)
                                for kind, (low, high) in contents.items():
                                    bottom = tray_obj['bounds'][0][2]
                                    bounds = [list(low) + [bottom], list(high) + [heights[tray][kind]]]
                                    objects.append(dict(kind='placed_' + kind, bounds=bounds))
                            exit_geometry = Routes(dict(world_scene, objects=objects), survey_yaw)
                            opened = {side: [departing.get('release_hand', departing['preshape']).get(
                                side, working_hand_command(side))]
                                      for side in SIDES}
                            ready = {side: ready_angles(side).copy() for side in SIDES}
                            legs = LowerBody(angles)
                            tray_yaw = departing['tray_yaw']
                            with measure('transition_plan', kind=departing['item']['kind'], next_kind=item['kind']):
                                if departing.get('hook_low_transport'):
                                    start = dict(ready, **departing['retreat'][-1])
                                    target = dict(start, **departing['cable_withdraw_high'][0])
                                    target_yaw = tray_yaw + .20 * (plan['pick_yaw'] - tray_yaw)
                                    outward = empty_transition(exit_geometry, start, tray_yaw, target,
                                                                target_yaw, opened, legs)
                                    home = (empty_transition(exit_geometry, target, target_yaw, ready,
                                                             target_yaw, opened, legs) if outward else None)
                                    if home:
                                        planned_exits[old_key] = (key, target_yaw, [], [outward[-1], home[-1]])
                                else:
                                    start = dict(ready, **departing['release'][-1])
                                    for count, goal in enumerate(departing['retreat'][:3], 1):
                                        start.update(goal)
                                        bridge = empty_transition(exit_geometry, start, tray_yaw, ready,
                                                                  plan['pick_yaw'], opened, legs)
                                        if bridge:
                                            planned_exits[old_key] = (key, plan['pick_yaw'],
                                                                      departing['retreat'][1:count], bridge)
                                            break
                        remember_placement(plan, removed, placements, occupied, heights)
                        start_yaw = plan['tray_yaw']
                        previous = (key, plan)
                    elif attempt == 1:
                        pending.append(item)

            def defer(item, attempt, reason, plan=None):
                retry_slide = (attempt == 2 and item['kind'] not in TWO_ARM_MOUSE and plan is not None
                               and not plan.get('low_object_slide') and supports_low_object_slide(item['target']))
                retry_mouse = (not primary and attempt == 2 and item['kind'] in TWO_ARM_MOUSE and plan is not None
                               and reason == 'grasp_or_transfer_failed' and SIM_TIME - SIM_START < 120.)
                event('deferred' if attempt == 1 or retry_slide or retry_mouse else 'unresolved', kind=item['kind'],
                      attempt=attempt, reason=reason)
                if attempt == 1:
                    deferred.append(item)
                elif retry_slide:
                    slide_retries.append(item)
                elif retry_mouse:
                    mouse_retries.append(item)

            if adaptive_batch and primary:
                lookahead = worker.submit(plan_ahead, scene, yaw, dict(self.angles), observed)
            else:
                lookahead = None
            if fast_batch and not adaptive_batch:
                def plan_all(angles):
                    with measure('batch_plan'):
                        return prepare_batch(angles)

                batch = worker.submit(plan_all, dict(self.angles))
                batch_checked = False
                with measure('batch_initial_wait'):
                    # A failed worker also wakes this wait, including failures
                    # before the independent prefix could be published.
                    self.poll(lambda: batch.done() or (prefix_count and prefix_ready.done()),
                              150., 'batch IK', load_limit=9.5)
                if batch.done():
                    plans = batch.result()
                    validate_batch(plans)
                    batch_checked = True
                else:
                    plans = prefix_ready.result()
            sequence = adaptive_sequence() if adaptive_batch else ((1, item) for item in queue)
            for index, (attempt, item) in enumerate(sequence):
                if fast_batch and not adaptive_batch and not batch_checked and (index >= prefix_count or batch.done()):
                    if not batch.done():
                        with measure('wait_for_batch_tail'):
                            self.poll(batch.done, 150., 'remaining batch IK', load_limit=9.5)
                    plans = batch.result()
                    validate_batch(plans)
                    batch_checked = True
                kind = item['kind']
                cycle_sim_start, cycle_wall_start = SIM_TIME, time.monotonic()
                last_item = (kind in TWO_ARM_MOUSE and len(mice) == 1 if adaptive_batch
                             else index == len(queue) - 1)
                last_attempt = not mice and (index == len(queue) - 1 and not deferred if attempt == 1
                                             else index == len(queue) + len(deferred) - 1)
                if kind == 'drive':
                    self.withdraw_limits.update(dict.fromkeys(SIDES, np.inf))
                else:
                    for side in SIDES:
                        self.withdraw_limits.pop(side, None)
                if adaptive_batch:
                    if refresh_needed:
                        cancel_lookahead.set()
                        scene, yaw, refreshed = self.survey(worker)
                        observed = {entry['kind']: entry for entry in refreshed}
                        for known, fresh in zip((footprints, content_heights), observed_contents(scene)):
                            for tray, entries in fresh.items():
                                known.setdefault(tray, {}).update(entries)
                        refresh_needed = False
                    current = observed.get(kind)
                    if current is None:
                        defer(item, attempt, 'no_longer_on_table')
                        continue
                    event('plan_attempt', kind=kind, attempt=attempt)
                    reused = False
                    if lookahead and attempt < 3 and not cancel_lookahead.is_set():
                        upcoming = projected[attempt, kind]
                        with measure('wait_for_ik', kind=kind, attempt=attempt):
                            self.poll(lambda: upcoming.done() or lookahead.done(), 150., 'lookahead IK',
                                      load_limit=9.5)
                        if upcoming.done():
                            expected, expected_yaw, geometry, planned = upcoming.result()
                            expected_yaw = exit_yaws.get((attempt, kind), expected_yaw)
                            if cleared == expected and abs(self.angles[WAIST] - expected_yaw) <= .04:
                                routes, plan = geometry, planned
                                reused = True
                                event('lookahead_route', kind=kind, attempt=attempt, available=bool(plan))
                        if not reused:
                            cancel_lookahead.set()
                    if not reused:
                        if kind in TWO_ARM_MOUSE and box_contents.done():
                            apply_box_observation(current['tray']['kind'], box_contents.result(),
                                                  footprints, content_heights)
                        future = worker.submit(plan_adaptive, current, attempt, scene, yaw, dict(self.angles),
                                               self.angles[WAIST], cleared, reserved, footprints, content_heights,
                                               retry_hands.get(kind))
                        with measure('wait_for_ik', kind=kind, attempt=attempt):
                            self.poll(future.done, 150., 'adaptive IK', load_limit=9.5)
                        routes, plan = future.result()
                    if plan is None:
                        defer(item, attempt, 'no_complete_route')
                        continue
                elif fast_batch:
                    plan = plans[index]
                else:
                    if pending is None:
                        pending = submit(item)
                    if not pending.done():
                        with measure('wait_for_ik', kind=kind):
                            timeout = 90. if edge_batch or (flat_mouse_receiver and kind in TWO_ARM_MOUSE) else 20.
                            self.poll(pending.done, timeout, 'direct IK', load_limit=9.5)
                    plan = pending.result()
                    following = queue[index + 1:index + 2]
                    pending = (submit(following[0]) if following and following[0]['kind'] not in TWO_ARM_MOUSE
                               and not needs_current_pose(following[0]['kind']) else None)
                if plan is None:
                    paired = item['family'] == 'two_hands' or kind in (*TWO_ARM_MOUSE, 'cable')
                    reason = 'no_two_hand_route' if paired else 'no_direct_route'
                    event('skipped', kind=kind, reason=reason)
                    continue
                if index == 0 and resume and resume.get('recover_box'):
                    if ('pickup_retreat' not in plan or abs(self.angles[WAIST] - plan['pick_yaw']) > .04
                            or any(np.max(abs(self.q(side) - plan['lift'][-1][side])) > .14 for side in SIDES)):
                        raise RuntimeError('arms are outside the saved box withdrawal pose')
                    with measure('withdraw_box', kind=kind):
                        self.move(plan['pickup_retreat'], speed=.42, load_limit=9.5)
                    for side in SIDES:
                        self.ready_hand(side)
                    self.force_reference = {side: self.wrench[side][1] for side in SIDES}
                    self.load_since.clear()
                    continue
                item = plan['item']
                hands = list(plan['preshape'])
                hook_carry = plan.get('hook_carry', False)
                cup_release = 'cup_gap_above_rim_m' in plan
                hook_grip = {plan['grip_side']: plan['grip'][plan['grip_side']]} if hook_carry else None
                mouse_tolerance = .010 if (kind in TWO_ARM_MOUSE or plan.get('low_object_slide')) and plan.get(
                    'edge_receive') else None
                release_grip = plan['grip'] if fast_batch and mouse_tolerance and plan.get('paired_carry') else hook_grip
                event('start_item', kind=kind, hands=hands, release_m=item['release'],
                      mouse_lateral_first=bool(plan.get('mouse_lateral_first')),
                      preshape=plan['preshape'], grasp_mode=('top_slide' if plan.get('top_slide') else
                                                             'hook_carry' if hook_carry else
                                                            'cupped_hands' if cup_release else
                                                            'edge_receive' if plan.get('edge_receive') else
                                                            plan.get('box_grasp_mode')),
                      grip_side=plan.get('grip_side', plan.get('box_grip_side')),
                      press_side=plan.get('pusher_side', plan.get('box_press_side')))
                if hook_carry:
                    event('hook_route', concurrent_departure=bool(plan.get('hook_departure')),
                          roll_release=bool(plan.get('hook_roll')),
                          flip_after_open=bool(plan.get('hook_flip_after_open')))
                self.set_grip_force(kind, hands)
                if plan.get('mouse_staging_only'):
                    with measure('mouse_side_reposition', kind=kind):
                        for side, values in plan['preshape'].items():
                            self.command_hand(side, values)
                        self.turn(plan['pick_yaw'], item['target']['xyz'], routes,
                                  load_limit=9.5, speed=READY_WAIST_SPEED)
                        with self.allow_table_contact(kind, SIDES):
                            if not self.approach(plan, routes):
                                defer(item, attempt, 'side_reposition_approach_failed')
                                continue
                            self.push_scoop_steps(plan['close'], plan)
                            self.move(plan['retreat'], speed=.38, load_limit=9.5, position_control=True)
                        for side in SIDES:
                            self.ready_hand(side)
                    mouse_retries.append(item)
                    refresh_needed = True
                    continue
                if plan.get('top_slide'):
                    self.receive_guard = self.guard_receiver(plan, routes)
                already_at_tray = bool(held and index == 0)
                if already_at_tray:
                    object_offset = self.resume_held_offset(plan, routes, held)
                elif index == 0 and kind == 'drive' and resume and resume.get('raised_box'):
                    self.continue_raised_box(plan, routes)
                    object_offset = None
                else:
                    carried, object_offset = self.carry_to_tray(plan, routes, closing['steps'] if closing and index == 0 else 0)
                    if not carried:
                        self.receive_guard = None
                        if adaptive_batch:
                            side = plan.get('grip_side') or (hands[0] if len(hands) == 1 else None)
                            if side in SIDES and kind not in TWO_ARM_MOUSE:
                                retry_hands[kind] = 'right' if side == 'left' else 'left'
                            defer(item, attempt, 'grasp_or_transfer_failed', plan)
                            refresh_needed = True
                        elif cup_mouse:
                            raise RuntimeError(f'{kind}: transfer failed during the batch')
                        continue
                released = True
                watch_box = adaptive_batch and mice and kind == 'drive'
                box_at_release = None
                if watch_box:
                    center = routes.local(np.asarray(item['tray']['xyz']), self.angles[WAIST])
                    self.aim(float(np.clip(np.arctan2(center[1], center[0]), -.48, .48)), .42)
                with measure('release', kind=kind):
                    if not already_at_tray:
                        limit = BOX_LOAD_LIMIT if item['family'] == 'two_hands' else 9.5 if 'close' in plan else 6.
                        if fast_batch and item['family'] == 'two_hands' and self.box_hold is not None:
                            if 'box_release_clearance_m' in plan:
                                event('box_lower_target', clearance_above_rim_mm=round(
                                    1000 * plan['box_release_clearance_m'], 1))
                            for step, goals in enumerate(plan['release'], 1):
                                self.move([goals], speed=.20, load_limit=limit)
                                pressure = self.box_pressure(self.box_hold['forces'])
                                span = np.linalg.norm(np.diff(self.box_palms(), axis=0))
                                event('box_lower', step=step, span_mm=round(1000 * span, 1),
                                      inset_mm=round(1000 * self.box_hold['applied_inset'], 1), force_change_n={
                                    side: round(load, 2) for side, load in pressure.items()})
                        else:
                            self.move(plan['release'], speed=.26 if cup_release else .55, load_limit=limit,
                                      position_control=hook_carry or cup_release,
                                      joint_tolerance=.025 if hook_carry and not plan.get('hook_low_transport') else None,
                                      palm_tolerance=.006 if cup_release else mouse_tolerance, hold_hand=release_grip)
                    if 'placement_center' in plan and not fast_batch:
                        aligned = already_at_tray and self.release_near_site(plan, routes, object_offset)
                        if not aligned and not self.align_center(plan, routes, 'placement', object_offset):
                            if not self.release_near_site(plan, routes, object_offset):
                                raise RuntimeError(f'{kind}: release center outside tolerance')
                    if item['family'] == 'two_hands':
                        box_held = self.box_still_held()
                        aligned = (fast_batch or not (box_held or self.box_direct)
                                   or self.align_box_placement(plan, routes))
                        if not aligned and not self.box_direct:
                            raise RuntimeError('drive: release center outside tolerance')
                        self.box_hold = None
                        self.grip(plan['preshape'], load_limit=9.5)
                        height_before = self.box_palms()[:, 2]
                        self.move(plan['retreat'][:1], speed=.55, load_limit=9.5)
                        event('box_open_retreat', wrist_rise_mm=dict(zip(
                            SIDES, (1000 * (self.box_palms()[:, 2] - height_before)).round(1).tolist())))
                        if watch_box and not box_contents.done():
                            box_at_release = self.observe_tray_contents(item['tray'], yaw)
                        self.box_removed = box_held
                        released = box_held
                    elif hook_carry:
                        if plan.get('hook_roll_dump'):
                            self.dump_cable_in_tray(routes, item['tray'])
                        elif plan.get('hook_operator_withdrawal'):
                            self.lower_hook_at_tray(plan, routes)
                        side = plan['grip_side']
                        feedback, _ = self.scoop_feedback(plan['grip'], 0., receiver=side)
                        angles = self.q(side)
                        wrist_error = np.degrees(angles[-2:] - plan['release'][-1][side][-2:])
                        arm = self.arm[side]
                        palm = arm.wrist(angles) @ arm.hand
                        tilt_from_down = np.degrees(np.arccos(np.clip(-palm[2, 0], -1., 1.)))
                        _, before_open = self.edge_support_force(side, routes)
                        event('hook_before_release', kind=kind, hand=feedback[side],
                              wrist_error_deg=wrist_error.round(2).tolist(),
                              palm_tilt_from_down_deg=round(float(tilt_from_down), 1))
                        self.grip(plan.get('release_hand', {side: OPEN_LEFT}), load_limit=9.5)
                        _, after_open = self.edge_support_force(side, routes)
                        event('hook_opened', kind=kind, load_change_n=round(after_open - before_open, 2))
                        if plan.get('hook_low_transport'):
                            with measure('cable_detach_lift', kind=kind):
                                before = arm.tool(self.q(side))[:3, 3].copy()
                                self.move(plan['retreat'], speed=.55, load_limit=9.5,
                                          position_control=True, hold_hand=plan['release_hand'])
                                delta = arm.tool(self.q(side))[:3, 3] - before
                                event('cable_vertical_exit',
                                      rise_mm=round(1000 * float(delta[2]), 1),
                                      sideways_mm=round(1000 * float(np.linalg.norm(delta[:2])), 1),
                                      planned_rise_mm=round(1000 * plan['cable_exit_rise_m'], 1))
                    else:
                        if cup_release:
                            feedback, _ = self.scoop_feedback(plan['grip'], 0., receiver=plan['grip_side'], paired=True)
                            event('mouse_hold_before_release', hands=feedback)
                        release_sequence = plan.get('release_sequence')
                        if release_sequence:
                            for step in release_sequence:
                                with measure(step['phase'], kind=kind):
                                    self.move([step['q']], speed=.45, load_limit=9.5,
                                              position_control=True, hold_hand=step['command'])
                        else:
                            self.grip(plan.get('release_hand', plan['preshape']), load_limit=9.5)
                if released:
                    event('released', kind=kind, tray=item['tray']['kind'], target_m=item['release'])
                elif self.box_direct and kind == 'drive':
                    event('release_commanded', kind=kind, tray=item['tray']['kind'], target_m=item['release'])
                else:
                    event('skipped', kind=kind, reason='box_slipped_before_release')
                with measure('up_and_ready', kind=kind):
                    early_exit = planned_exits.get((attempt, kind)) if adaptive_batch and reused and released else None
                    if timed_batch and plan.get('exit_transition'):
                        transition = plan['exit_transition']
                        self.move(transition['prefix'], speed=.60, load_limit=9.5,
                                  position_control=hook_carry, continuous=True,
                                  hold_hand=plan.get('release_hand'))
                        self.coordinated_move(transition['steps'], speed=.65,
                                              hold_hand=plan.get('release_hand') if hook_carry else None)
                        if transition.get('cable_short_exit'):
                            event('cable_short_exit', kind=kind, back_mm=60,
                                  targets=len(transition['steps']),
                                  waist_yaw_rad=round(self.angles[WAIST], 4))
                        plans[index + 1]['prepositioned'] = transition.get('prepositioned', True)
                        event('next_approach_overlapped', kind=kind, next_kind=plans[index + 1]['item']['kind'])
                    elif early_exit:
                        following_key, final_yaw, prefix, steps = early_exit
                        if prefix:
                            self.move(prefix, speed=.60, load_limit=9.5, continuous=True)
                        self.coordinated_move(steps, speed=.65, hold_hand=plan.get('release_hand'))
                        exit_yaws[following_key] = final_yaw
                        event('cable_short_exit' if hook_carry else 'turn_overlapped_withdrawal',
                              kind=kind, next_kind=following_key[1])
                    elif plan.get('release_sequence') and last_item:
                        pass
                    elif timed_batch and last_item and kind in TWO_ARM_MOUSE:
                        # Both hands are already opened above the tray. Spread and
                        # lift clear; a complete return to READY is outside the task.
                        self.move(plan['retreat'][:2], speed=.55, load_limit=9.5,
                                  palm_tolerance=mouse_tolerance, continuous=True)
                    elif 'pocket' in plan and fast_batch:
                        side = hands[0]
                        self.move([*plan['retreat'], {side: ready_angles(side)}],
                                  speed=.64 if timed_batch else .55, load_limit=9.5, continuous=timed_batch)
                    elif 'pocket' in plan:
                        self.raise_and_ready(hands[0], routes, kind, plan['shape'], plan['pocket'])
                    elif hook_carry and plan.get('hook_low_transport') and not plan.get('hook_leave_hand_after_release'):
                        side = plan['grip_side']
                        self.move(plan['cable_withdraw_high'], speed=.55, load_limit=9.5,
                                  position_control=True, hold_hand=plan['release_hand'])
                        event('cable_withdraw_high', kind=kind, back_mm=240)
                        if not (adaptive_batch and last_attempt):
                            self.move([{side: ready_angles(side)}], speed=.55, load_limit=9.5,
                                      position_control=True, hold_hand=plan['release_hand'])
                    elif hook_carry and plan.get('hook_leave_hand_after_release'):
                        event('hook_hand_left_for_operator', kind=kind, side=plan['grip_side'])
                    elif hook_carry and plan.get('hook_roll_dump'):
                        pass
                    elif hook_carry and plan.get('hook_two_finger'):
                        self.withdraw_cable_pinch(plan)
                    elif hook_carry and plan.get('hook_operator_withdrawal'):
                        self.withdraw_cable_demonstration(plan)
                    elif hook_carry and fast_batch:
                        arm = self.arm[plan['grip_side']]
                        before = arm.wrist(self.q(plan['grip_side']))[2, 3]
                        for retreat_index, goals in enumerate(plan['retreat']):
                            self.follow_waypoint(goals, .025, 9.5, speed=.58)
                            if retreat_index == 0:
                                rise = arm.wrist(self.q(plan['grip_side']))[2, 3] - before
                                event('hook_open_retreat', wrist_rise_mm=round(1000 * float(rise), 1))
                    elif 'close' in plan:
                        self.move(plan['retreat'], load_limit=9.5, position_control=hook_carry,
                                  joint_tolerance=.025 if hook_carry else None, palm_tolerance=mouse_tolerance)
                    else:
                        self.move([*plan['retreat'][1:], {side: ready_angles(side) for side in hands}],
                                  speed=.64 if fast_batch else .55, load_limit=9.5)
                    if not (plan.get('hook_leave_hand_after_release') or
                            (plan.get('release_sequence') and last_item)):
                        for side in hands:
                            self.ready_hand(side)
                self.receive_guard = None
                if watch_box and not box_contents.done():
                    settled_box = self.observe_tray_contents(item['tray'], yaw) if released else None
                    bounds = settled_box if settled_box is not None else box_at_release
                    box_contents.set_result(bounds)
                    event('box_in_tray_camera', bounds_m=bounds, settled=settled_box is not None)
                if adaptive_batch:
                    if released:
                        remember_placement(plan, cleared, reserved, footprints, content_heights)
                    else:
                        defer(item, attempt, 'release_unconfirmed', plan)
                        refresh_needed = True
                event('cycle_executed', kind=kind, hands=hands, sim_start=cycle_sim_start, sim_end=SIM_TIME,
                      sim_seconds=round(SIM_TIME - cycle_sim_start, 3),
                      wall_seconds=round(time.monotonic() - cycle_wall_start, 3))
            self.set_grip_force(None, ())
            if timed_batch:
                # Allow released objects to settle without another sensor survey.
                self.advance_to(self.history[-1][0] + 1., 'final settling', load_limit=9.5)
                self.done_pub.publish(Empty())
                self.executor.spin_once(timeout_sec=.05)
                elapsed = SIM_TIME - SIM_START
                event('batch_done', sim_seconds=round(elapsed, 3), budget_seconds=sim_budget,
                      within_budget=elapsed <= sim_budget, scene_clock_seconds=SIM_TIME)

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node('team22_fast_sort')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    robot = Robot(node, executor)
    try:
        with ThreadPoolExecutor(max_workers=2 if adaptive_batch else 1) as worker:
            try:
                robot.run(worker)
            except BaseException as error:
                if robot.started and not robot.external_arm:
                    robot.stop()
                event('stopped', reason=str(error) or type(error).__name__)
                raise
            finally:
                cancel_lookahead.set()
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def recorded(path, *, skip_box=False, edge_batch=False, upright_box=False, flat_mouse_receiver=False):
    with np.load(path) as data:
        packet = json.loads(data['metadata'].item())
        scene, task = packet['scene'], packet['task']
        yaw = packet['angles'][WAIST]
        points = data['obstacles']
        retain_contact_surfaces(scene, points)
        queue, omitted = queue_for(scene, task, points, skip_box=skip_box)
    show_queue(scene, yaw, queue, omitted)
    if not queue:
        return
    routes = Routes(scene, yaw)
    reserved = {}
    for item in queue:
        with measure('ik', kind=item['kind']):
            route = next_route(routes, item, reserved, edge_batch=edge_batch, upright_box=upright_box,
                               flat_mouse_receiver=flat_mouse_receiver)
        event('recorded_plan', kind=item['kind'], reachable=route is not None,
              hands=list(route['preshape']) if route else [], release_m=route['item']['release'] if route else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--recorded', type=Path, help='check a saved NPZ without commanding the robot')
    source.add_argument('--resume', type=Path, help='continue the unchanged scene from saved coordinates and queue')
    parser.add_argument('--skip-box', action='store_true', help='leave the drive box out of the transfer queue')
    parser.add_argument('--force-box', action='store_true', help='lift and release the box without waiting for force confirmation')
    parser.add_argument('--only', nargs='+', choices=tuple(CATALOG), help='transfer only these objects from one live survey')
    parser.add_argument('--flat-mouse-receiver', action='store_true',
                        help='slide the mouse under one palm to the other hand outside the table edge')
    parser.add_argument('--edge-batch', action='store_true',
                        help='receive the mouse below the edge, hook the cable with one hand, and roll the box')
    parser.add_argument('--upright-box', action='store_true',
                        help='attempt to carry the upright box to its tray, including with --edge-batch')
    parser.add_argument('--cup-mouse', action='store_true',
                        help='receive the mouse in two cupped hands and lower it before opening')
    parser.add_argument('--cup-trial-release', action='store_true',
                        help='permit the checked mouse trial release up to 220 mm above the tray rim')
    parser.add_argument('--fast-batch', action='store_true',
                        help='plan the whole survey before picking, then run the planned poses without alignment stops')
    parser.add_argument('--leave-cable-hand', action='store_true',
                        help='stop after placing the cable and opening the receiving hand')
    parser.add_argument('--ignore-force-checks', action='store_true',
                        help='disable wrist-load guards for an explicitly requested simulator motion')
    parser.add_argument('--timed-batch', action='store_true',
                        help='stream validated transitions, overlap free motions and finish on /clock')
    parser.add_argument('--adaptive-batch', action='store_true',
                        help='run available routes and try deferred objects once after the main pass')
    parser.add_argument('--sim-budget', type=float, default=260., help='simulation seconds, including survey and planning')
    parser.add_argument('--log', type=Path, help='JSONL output (live default: team22-fast-sort.jsonl in the temp directory)')
    args = parser.parse_args()
    if args.cup_mouse and (not args.fast_batch or not args.edge_batch or args.flat_mouse_receiver
                           or args.recorded):
        parser.error('--cup-mouse requires --fast-batch --edge-batch and a live or saved survey')
    if args.cup_trial_release and not args.cup_mouse:
        parser.error('--cup-trial-release requires --cup-mouse')
    if args.adaptive_batch and (not args.fast_batch or not args.timed_batch or args.resume or args.recorded):
        parser.error('--adaptive-batch requires --fast-batch --timed-batch with a fresh survey')
    if not np.isfinite(args.sim_budget) or args.sim_budget <= 0:
        parser.error('--sim-budget must be finite and positive')
    log_path = args.log or (None if args.recorded else Path(gettempdir()) / 'team22-fast-sort.jsonl')
    with log_path.open('w', encoding='utf-8', buffering=1) if log_path else nullcontext() as LOG:
        with measure('total'):
            if args.recorded:
                recorded(args.recorded, skip_box=args.skip_box, edge_batch=args.edge_batch,
                         upright_box=args.upright_box, flat_mouse_receiver=args.flat_mouse_receiver)
            else:
                resume = json.loads(args.resume.read_text(encoding='utf-8')) if args.resume else None
                live(skip_box=args.skip_box, resume=resume, selected=args.only, force_box=args.force_box,
                     flat_mouse_receiver=args.flat_mouse_receiver, edge_batch=args.edge_batch,
                     upright_box=args.upright_box, fast_batch=args.fast_batch, cup_mouse=args.cup_mouse,
                     cup_trial_release=args.cup_trial_release, leave_cable_hand=args.leave_cable_hand,
                     ignore_force_checks=args.ignore_force_checks, timed_batch=args.timed_batch,
                     sim_budget=args.sim_budget, adaptive_batch=args.adaptive_batch)

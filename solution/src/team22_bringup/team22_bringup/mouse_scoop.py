"""Two-arm edge scoop with a short, checked lift from the frozen mouse cloud."""

from functools import lru_cache
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation, Slerp

from .edge_pair_delivery import edge_pair_delivery, obstacle_geometry
from .grasp import Arm
from .table_contact import finger_distance, fit_contact, object_frame


SIDES = ('left', 'right')
SIGNS = {'left': 1., 'right': -1.}
DISTAL = ('Link4', 'Link14', 'Link24', 'Link34')
OPEN = [1., 1., 1., 1., 1., 0.]
CLOSE = [.90, .90, .90, .90, 1., 0.]
INSERTION = .008


@lru_cache(maxsize=1)
def body_geometry():
    with np.load(Path(__file__).with_name('robot_surfaces.npz')) as data:
        torso = data['waist_yaw_link']
        links = {side: [data[name.replace('_joint', '_link')][::4] for name in Arm(side).names[2:]]
                 for side in SIDES}
    return ConvexHull(torso).equations, torso.min(0) - .003, torso.max(0) + .003, links


def mouse_scoop_plan(routes, item, positions, *, box_removed=False):
    """Return a checked pickup and 30 mm probe using the paired-plan dictionary."""
    target = item['target']
    points = np.asarray(target.get('contact_points', ()))
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 20:
        return None
    center, axis, _, up = object_frame(target, routes.normal, routes.floor)
    for angle in (-.15, -.06, -.24):
        # The axis of a partial RGB-D surface can move a few degrees between surveys.
        direction = Rotation.from_rotvec(angle * up).apply(axis)
        plan = _plan(routes, item, positions, points, center, direction, up, box_removed)
        if plan is not None:
            return plan
    return None


def _plan(routes, item, positions, points, center, axis, up, box_removed):
    yaw = routes.yaw + np.arctan2(center[1], center[0]) + .22
    normal = axis if routes.local(axis, yaw)[1] > 0 else -axis
    tangent = np.cross(normal, up)
    if routes.local(tangent, yaw)[0] < 0:
        tangent = -tangent

    arms = {side: Arm(side) for side in SIDES}
    preshape = {side: list(OPEN) for side in SIDES}
    grip = {side: list(CLOSE) for side in SIDES}
    origins, orientations, offsets, profiles = {}, {}, {}, {}
    # Oppose the open fingertips across the shell, with pitch and roll matched to each wrist.
    for side, (angle, roll, pitch) in zip(SIDES, ((.32, .30, .58), (.52, -.65, .50))):
        out = SIGNS[side] * normal
        flat = -np.cos(angle) * out + np.sin(angle) * tangent
        fingers = np.cos(pitch) * flat - np.sin(pitch) * up
        palm = np.sin(pitch) * flat + np.cos(pitch) * up
        palm = Rotation.from_rotvec(roll * fingers).apply(palm)
        orient = np.column_stack((palm, np.cross(fingers, palm), fingers))
        hand = routes.hands[side]
        commands = np.linspace(OPEN, CLOSE, 5)
        shapes = [hand.vertices(hand.nominal_angles(command)) for command in commands]
        profiles[side] = [(np.concatenate([v for k, v in mesh.items() if k not in DISTAL]),
                           np.concatenate([mesh[k] for k in DISTAL])) for mesh in shapes]
        solid, tips = profiles[side][0]
        along, height = (points - center) @ tangent, (points - center) @ up
        flank = (points - center) @ out
        low = flank[(abs(along) < .030) & (height < .030)]
        edge = np.quantile(low, .92) if len(low) > 6 else np.quantile(flank, .88)
        solid_world, tips_world = solid @ orient.T, tips @ orient.T
        raised = max(.001 - (tips_world @ up).min(), .004 - (solid_world @ up).min())
        tip_center = np.mean([shapes[0][link].mean(0) for link in DISTAL], axis=0)
        origins[side] = (center + (edge + .004 - (tips_world @ out).min()) * out
                         - (orient @ tip_center @ tangent) * tangent + raised * up)
        orientations[side] = orient
        offsets[side] = out

    contact = {side: origins[side] - INSERTION * offsets[side] for side in SIDES}
    for side in SIDES:
        hand = routes.hands[side]
        frames = hand.frames(hand.nominal_angles(CLOSE))
        local_surface = (points - contact[side]) @ orientations[side]
        if min(finger_distance(hand, frames[link], link, local_surface) for link in DISTAL) > .003:
            return None

    box = next((o for o in routes.objects if o['kind'] == 'drive' and o.get('status') == 'observed'), None)
    box_volume = None
    if box and not box_removed:
        bounds = np.asarray(box['bounds'])
        corners = box.get('geometry', {}).get('corners_xy_m')
        if corners is None:
            corners = [[x, y] for x in bounds[:, 0] for y in bounds[:, 1]]
        box_volume = ConvexHull(corners).equations, bounds[:, 2]
    planes, body_low, body_high, meshes = body_geometry()

    def in_body(points):
        near = points[np.all((points >= body_low) & (points <= body_high), axis=1)]
        return bool(len(near) and np.any((near @ planes[:, :3].T + planes[:, 3]).max(1) < .002))

    def clear(pair, closed):
        samples = {}
        for side in SIDES:
            arm = arms[side]
            frames = arm.frames(pair[side])
            hand = frames[-1] @ arm.hand
            solid, tips = profiles[side][closed]
            fingers = np.concatenate((solid, tips)) @ hand[:3, :3].T + hand[:3, 3]
            if in_body(fingers) or any(in_body(v @ f[:3, :3].T + f[:3, 3])
                                       for v, f in zip(meshes[side], frames[2:])):
                return False
            arm_points, radii = arm.samples(pair[side])
            local = np.concatenate((arm_points[:112], fingers))
            radii = np.r_[radii[:112], np.zeros(len(fingers))]
            world = routes.world(local.T, yaw).T
            over = world[:, 0] >= routes.edge - radii
            gaps = np.r_[np.full(len(local) - len(tips), .002), np.full(len(tips), -.002)]
            if np.any(world[over] @ routes.normal - routes.floor - radii[over] < gaps[over]):
                return False
            if box_volume is not None:
                faces, (low, high) = box_volume
                near = (world[:, 2] >= low - radii - .002) & (world[:, 2] <= high + radii + .002)
                side_gap = (world[near, :2] @ faces[:, :2].T + faces[:, 2]).max(1)
                if np.any(side_gap < radii[near] + .002):
                    return False
            samples[side] = ((local[32:112], .038), (fingers, 0.))
        # Separate capsules and mesh points so nearest fingertips cannot mask an arm segment.
        for first, radius_a in samples['left']:
            for second, radius_b in samples['right']:
                margin = radius_a + radius_b + .002
                if np.any(np.maximum(first.min(0) - second.max(0), second.min(0) - first.max(0)) > margin):
                    continue
                if np.any(cKDTree(second).query(first, distance_upper_bound=margin)[0] < margin):
                    return False
        return True

    def link(start, end, closed=0):
        if start is None or end is None:
            return False
        delta = {side: end[side] - start[side] for side in SIDES}
        count = 2 + int(max(np.max(abs(d)) for d in delta.values()) / .06)
        return all(clear({side: start[side] + f * delta[side] for side in SIDES}, closed)
                   for f in np.linspace(0., 1., count))

    def solve(goals, rotations, seed, precise=False):
        pair = {}
        for side in SIDES:
            arm, orient = arms[side], rotations[side]
            goal = routes.local(goals[side] + orient @ arm.pocket, yaw)
            local_orient = routes.local(orient, yaw)
            q = (fit_contact(arm, goal, local_orient, seed[side]) if precise else
                 routes.fit(side, goal, local_orient, seed[side], model=arm))
            if q is None:
                return None
            pair[side] = q
        return pair

    rough = solve(origins, orientations, positions)
    if rough is None:
        return None
    low = solve(origins, orientations, rough, precise=True)
    if low is None or not clear(low, 0):
        return None
    hover_at = {side: origins[side] + .065 * up + .020 * offsets[side] for side in SIDES}
    start = {side: routes.world((arms[side].wrist(positions[side]) @ arms[side].hand)[:3], yaw)
             for side in SIDES}
    rotation = {side: Slerp([0., 1.], Rotation.from_matrix([start[side][:, :3], orientations[side]]))
                for side in SIDES}
    path, previous = [], positions
    for t in (.25, .50, .75, 1.):
        goals = {side: ((1. - t) * start[side][:, 3] + t * hover_at[side]
                        + np.sin(np.pi * t) * routes.world(np.array([.06, SIGNS[side] * .08, .12]), yaw))
                 for side in SIDES}
        q = solve(goals, {side: rotation[side](t).as_matrix() for side in SIDES}, previous)
        if not link(previous, q):
            return None
        path.append(q)
        previous = q
    grasp = solve(contact, orientations, low, precise=True)
    closed = len(profiles['left']) - 1
    if not link(previous, low) or not link(low, grasp) or not all(clear(grasp, shape) for shape in range(1, closed + 1)):
        return None
    probe = solve({side: contact[side] + .030 * up for side in SIDES}, orientations, grasp, precise=True)
    if not link(grasp, probe, closed=closed):
        return None
    close = [dict(phase='scoop_open', q=grasp, command=preshape), dict(phase='close', q=grasp, command=grip)]
    abort_close = [dict(phase='open', q=grasp, command=preshape), dict(phase='scoop_open', q=low, command=preshape)]
    pickup_retreat = [*reversed(path), positions]
    return dict(pick_yaw=yaw, tray_yaw=yaw, approach=[*path, low], grasp=[grasp], lift=[probe],
                transfer=[], release=[], retreat=[low, *pickup_retreat], close=close, abort_close=abort_close,
                probe_return=[grasp], pickup_retreat=pickup_retreat, grip=grip, preshape=preshape, item=item,
                probe_only=True)


def _mouse_shell_plan(routes, item, positions, box_removed, lower_body, start_yaw, *,
                      reverse_receiver=False, deadline=None):
    """Push from the side, receive at the lip, then carry between the palms."""
    if item.get('mouse_table_sweep'):
        return _mouse_table_sweep_plan(routes, item, positions, box_removed, lower_body, start_yaw,
                                       reverse_receiver=reverse_receiver, deadline=deadline)
    from .cable_bimanual import _section_hits_table

    cup = item.get('cup_mouse', False)
    low_object = item.get('low_object_slide', False)
    if deadline is None:
        deadline = monotonic() + (90. if cup else 35.)
    target = item['target']
    cloud = np.asarray(target.get('contact_points', ()), dtype=float)
    if lower_body is None or routes.edge < 0. or cloud.ndim != 2 or cloud.shape[1] != 3:
        return None
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if len(cloud) < 20:
        return None
    center, _, _, up = object_frame(target, routes.normal, routes.floor)
    edge = routes.edge + .015
    floor = routes.floor / np.linalg.norm(routes.normal)
    inward = np.array([1., 0., 0.]) - up[0] * up
    inward /= np.linalg.norm(inward)
    along = np.cross(up, inward)
    receiver, pusher = SIDES if center[1] >= 0. else SIDES[::-1]
    if reverse_receiver:
        receiver, pusher = pusher, receiver
    mirror = SIGNS[pusher]
    handed = {side: np.array([1., SIGNS[side], SIGNS[side], 1., SIGNS[side], 1., SIGNS[side]])
              for side in SIDES}
    # Seed the sideways pusher and the palm-up receiver separately.
    side_seed = np.array([-.755, .056, -1.128, -.669, 1.25, -.004, .706]) * handed[pusher]
    receive_seed = np.array([-.14, .11, -.36, -.26, 1.27, .39, -1.15]) * handed[receiver]
    push_pad = np.array([.0124, .0005 * mirror, .126])
    receive_pad = np.array([.035, -.019 * mirror, .174])
    shoulder = (cloud - center) @ np.column_stack((inward, along))
    top_samples = cloud[np.linalg.norm(shoulder, axis=1) < .027]
    if low_object and len(top_samples) < 8:
        # A coil can leave a hole at its observed center.
        top_samples = cloud[np.argsort(np.linalg.norm(shoulder, axis=1))[:16]]
    if len(top_samples) < 8:
        return None
    top = center + (np.percentile((top_samples - center) @ up, 90.) - .001) * up
    open_commands = {pusher: [.90, .88, .88, .84, 0., .85], receiver: [1.] * 5 + [0.]}
    grip = {pusher: [.86, .86, .86, .76, 0., .85], receiver: [.82, .84, .93, .93, 1., .35]}
    if cup:
        open_commands[receiver] = [.94, .84, .84, .84, 1., 0.]
        grip[pusher] = [.60, .60, .60, .58, 0., .85]
    stages = (0., .25, .50, .75, 1.)
    commands = [{side: (np.asarray(open_commands[side]) * (1. - f) + np.asarray(grip[side]) * f).tolist()
                 for side in SIDES} for f in stages]
    shapes = [{side: routes.hands[side].vertices(routes.hands[side].nominal_angles(cmd[side]))
               for side in SIDES} for cmd in commands]
    release_hand = {pusher: [1., 1., 1., 1., 0., .85], receiver: [1., 1., 1., 1., 1., 0.]}
    release_shapes = ([{side: routes.hands[side].vertices(routes.hands[side].nominal_angles(
        (1. - f) * np.asarray(grip[side]) + f * np.asarray(release_hand[side]))) for side in SIDES}
                       for f in stages[1:]] if cup else list(reversed(shapes[:-1])))
    trays = [np.asarray(obj['bounds']) for obj in routes.objects if obj['kind'].startswith('tray_')]
    obstacles = []
    mouse_surface = set(map(tuple, cloud)) if cup and not low_object else set()
    for obj in routes.objects:
        if obj['kind'] == item['kind']:
            continue
        if mouse_surface and obj['kind'] == 'unknown':
            lo, hi = np.asarray(obj['bounds'])
            if (hi[2] - lo[2] <= .026 and hi[2] <= target['bounds'][1][2] - .010
                    and len(mouse_surface.intersection(map(tuple, obj.get('contact_points', ())))) >= 20):
                # A shallow fragment sharing the observed mouse surface moves with the shell.
                continue
        if cup and not obj['kind'].startswith('tray_'):
            xy = np.asarray(obj['xyz'])[:2]
            if any(np.all((xy >= tray[0, :2]) & (xy <= tray[1, :2])) for tray in trays):
                continue
        if obj['kind'] == 'drive' and box_removed:
            xy = np.asarray(obj['xyz'])[:2]
            if not any(np.all((xy >= tray[0, :2]) & (xy <= tray[1, :2])) for tray in trays):
                continue
        obstacles.append(obj)
    solids = [obstacle_geometry(obj) for obj in obstacles if not obj['kind'].startswith('tray_')]
    tray_low, tray_high = np.asarray(item['tray']['bounds'])
    contents = []
    footprints = item.get('occupied_footprints', {})
    heights = {'cable': -.037, 'drive': .112}
    heights.update(item.get('occupied_heights', {}))
    # The mouse may be placed directly on the cable in the tray.
    occupied = (((bounds, heights.get(kind, tray_high[2] + .075))
                 for kind, bounds in footprints.items()
                 if low_object or kind != 'cable')
                if isinstance(footprints, dict) else zip(footprints, heights.values()))
    for bounds, high_z in occupied:
        lo, hi = np.asarray(bounds, dtype=float)
        contents.append((np.r_[lo, tray_low[2]], np.r_[hi, high_z]))
    rims = []
    for lo, hi in trays:
        for axis in (0, 1):
            near, far = lo.copy(), hi.copy()
            near[axis], far[axis] = hi[axis] - .012, lo[axis] + .012
            rims.extend(((lo, far), (near, hi)))
    torso_planes, torso_low, torso_high, links = body_geometry()
    start_yaw = routes.yaw if start_yaw is None else start_yaw
    active_shapes = None
    top_command = dict(commands[0], **{pusher: [.72, .72, .72, .72, 1., 0.]})
    top_shapes = dict(shapes[0], **{pusher: routes.hands[pusher].vertices(
        routes.hands[pusher].nominal_angles(top_command[pusher]))})

    def in_obstacle(vertices):
        for lo, hi, _, faces in solids:
            near = vertices[np.all((vertices >= lo - .004) & (vertices <= hi + .004), axis=1)]
            if len(near) and np.any((near[:, :2] @ faces[:, :2].T + faces[:, 2]).max(1) < .004):
                return True
        return any(np.any(np.all((vertices >= lo - .004) & (vertices <= hi + .004), axis=1))
                   for lo, hi in (*rims, *contents))

    def cup_placements(placement, relative_shell, *, raised_release=False):
        yaw, release, turn, rise, seed = placement
        footprint = relative_shell @ turn.T
        shell_low, shell_high = footprint[:, :2].min(0), footprint[:, :2].max(0)
        near = tray_low[:2] + .008 - shell_low
        far = tray_high[:2] - .008 - shell_high
        if np.any(near > far):
            return []
        release_z = release[2]
        if raised_release:
            rim_gap = release_z + footprint[:, 2].min() - tray_high[2]
            release_z += np.clip(.218 - rim_gap, 0., .012)
        sites = [np.clip(release[:2], near, far)]
        for low, high in contents:
            before = low[:2] - shell_high - .012
            after = high[:2] - shell_low + .012
            sites.extend(([release[0], before[1]], [release[0], after[1]],
                          [before[0], release[1]], [after[0], release[1]],
                          [near[0], after[1]], [far[0], after[1]]))
        clear_sites = []
        for xy in sites:
            xy = np.asarray(xy)
            if np.any(xy < near) or np.any(xy > far):
                continue
            if any(not np.any(np.maximum(low[:2] - xy - shell_high,
                                         xy + shell_low - high[:2]) > .008) for low, high in contents):
                continue
            if any(np.linalg.norm(xy - prior) < .001 for prior in clear_sites):
                continue
            clear_sites.append(xy)
        return [(yaw + np.arctan2(xy[1], xy[0]) - np.arctan2(release[1], release[0]),
                 np.r_[xy, release_z], turn, rise, seed)
                for xy in sorted(clear_sites, key=lambda xy: np.linalg.norm(xy - release[:2]))]

    def solve(side, origin, orient, yaw, seed):
        if monotonic() >= deadline:
            return None
        arm = routes.arms[side]
        rotation = routes.local(orient, yaw)
        goal = routes.local(origin, yaw) + rotation @ arm.pocket
        return fit_contact(arm, goal, rotation, seed)

    def palm(side, q, yaw):
        arm = routes.arms[side]
        return routes.world((arm.wrist(q) @ arm.hand)[:3], yaw)

    def clear(pair, yaw, stage=0, *, underneath=True, hand_shapes=None):
        if monotonic() >= deadline:
            return False
        hand_shapes = ((active_shapes if active_shapes is not None and stage == 0 else shapes[stage])
                       if hand_shapes is None else hand_shapes)
        bodies = {}
        arm_clouds = {}
        hand_clouds = {}
        fore = routes.local(np.array([1., 0., 0.]), yaw)
        normal = routes.local(up, yaw)
        for side in SIDES:
            arm = routes.arms[side]
            frames = arm.frames(pair[side])
            hand = frames[-1] @ arm.hand
            arm_parts = [mesh @ frame[:3, :3].T + frame[:3, 3]
                         for mesh, frame in zip(links[side], frames[2:])]
            hand_parts = [mesh @ hand[:3, :3].T + hand[:3, 3]
                          for mesh in hand_shapes[side].values()]
            if any(_section_hits_table(part, fore, normal, floor, edge - .003, False)
                   for part in arm_parts):
                return False
            if any(_section_hits_table(part, fore, normal, floor, edge - .003,
                                       underneath and not cup and side == receiver and name != 'base_link')
                   for name, part in zip(hand_shapes[side], hand_parts)):
                return False
            local = np.concatenate([*arm_parts, *hand_parts])
            near = local[np.all((local >= torso_low) & (local <= torso_high), axis=1)]
            if len(near) and np.any((near @ torso_planes[:, :3].T + torso_planes[:, 3]).max(1) < .002):
                return False
            if lower_body.collides(local, yaw, clearance=.006) or in_obstacle(routes.world(local.T, yaw).T):
                return False
            bodies[side] = local
            arm_clouds[side] = np.concatenate(arm_parts)
            hand_clouds[side] = np.concatenate(hand_parts)
        if cup:
            # The two cups meet around the mouse; keep the full arm clearance.
            if np.any(cKDTree(bodies['left']).query(
                    arm_clouds['right'], distance_upper_bound=.006)[0] < .006):
                return False
            if np.any(cKDTree(arm_clouds['left']).query(
                    hand_clouds['right'], distance_upper_bound=.006)[0] < .006):
                return False
            return not np.any(cKDTree(hand_clouds['left']).query(
                hand_clouds['right'], distance_upper_bound=.003)[0] < .003)
        return not np.any(cKDTree(bodies['left']).query(
            bodies['right'], distance_upper_bound=.006)[0] < .006)

    def link(before, after, yaw, stage=0, **clearance):
        delta = {side: after[side] - before[side] for side in SIDES}
        steps = 2 + int(max(np.max(abs(d)) for d in delta.values()) / .040)
        return all(clear({side: before[side] + f * delta[side] for side in SIDES}, yaw, stage, **clearance)
                   for f in np.linspace(0., 1., steps))

    def cup_retreat(delivery):
        yaw = delivery['tray_yaw']
        before = delivery['release'][-1]
        frames = {side: palm(side, before[side], yaw) for side in SIDES}
        clearance = dict(underneath=False, hand_shapes=release_shapes[-1])
        for spread, tip in ((.018, .12), (.014, .065), (.010, .030)):
            pair = {}
            for side in SIDES:
                frame = frames[side]
                forward = frame[:, 2].copy()
                forward[2] = 0.
                forward /= np.linalg.norm(forward)
                origin = frame[:, 3] - spread * forward + [0., 0., .010 if side == pusher else .005]
                rotation = frame[:, :3]
                if side == receiver:
                    rotation = Rotation.from_rotvec([-mirror * tip, 0., 0.]).as_matrix() @ rotation
                pair[side] = solve(side, origin, rotation, yaw, before[side])
            if any(q is None for q in pair.values()) or not link(before, pair, yaw, **clearance):
                continue
            for index, q in enumerate(delivery['retreat'][:8]):
                if link(pair, q, yaw, **clearance):
                    return [pair, *delivery['retreat'][index:]]
        return None

    def low_cup_retreat(delivery, trial_release, ready_arc):
        """Check the return from the actual tipped and lifted wrist pose."""
        yaw = delivery['tray_yaw']
        terminal = trial_release['release_sequence'][-1]['q']
        opened = dict(underneath=False, hand_shapes=release_shapes[-1])
        for prefix in ([], trial_release['retreat']):
            start = prefix[-1] if prefix else terminal
            middle_paths = [[], [dict(start, **{pusher: positions[pusher]})],
                            [dict(start, **{receiver: positions[receiver]})]]
            if ready_arc is not None:
                middle_paths.insert(1, [ready_arc])
            for middle in middle_paths:
                suffix = [*middle, positions]
                if all(link(a, b, yaw, **opened) for a, b in zip([start, *suffix[:-1]], suffix)):
                    return [*prefix, *suffix]
        return None

    def cup_trial_release(delivery):
        """Uncover the receiver, tip its palm about the support, then leave upwards."""
        yaw = delivery['tray_yaw']
        start = delivery['release'][-1]
        frames = {side: palm(side, start[side], yaw) for side in SIDES}
        vertical = np.array([0., 0., 1.])
        upper_open = dict(grip, **{pusher: release_hand[pusher]})
        upper_shapes = dict(shapes[-1], **{pusher: release_shapes[-1][pusher]})
        opened = dict(underneath=False, hand_shapes=release_shapes[-1])
        uncovered = dict(underneath=False, hand_shapes=upper_shapes)

        def overlap_release(raise_path, tip_path):
            """Synchronize the checked joint paths, retaining their exact endpoints."""
            if not tip_path or tip_path[-1][0] < .03:
                return None
            full_push_path = [start[pusher], *[step['q'][pusher] for step in raise_path]]
            receive_path = [start[receiver], *[pair[receiver] for _, pair in tip_path]]
            receive_fractions = np.r_[0., [angle / tip_path[-1][0] for angle, _ in tip_path]]
            opening_shapes = [upper_shapes, *[dict(release_shapes[-1], **{receiver: mesh[receiver]})
                                             for mesh in release_shapes]]

            def sample(path, fractions, at):
                if at >= fractions[-1]:
                    return path[-1]
                index = max(0, int(np.searchsorted(fractions, at, side='right')) - 1)
                weight = (at - fractions[index]) / (fractions[index + 1] - fractions[index])
                return (1. - weight) * path[index] + weight * path[index + 1]

            for prefix_count, lead in ((0, 1.), (0, 2.), (1, 2.), (2, 2.), (2, 6.), (2, 18.)):
                if prefix_count >= len(raise_path):
                    continue
                push_path = full_push_path[prefix_count:]
                push_fractions = np.linspace(0., 1., len(push_path)) / lead
                fractions = sorted(set(np.round(np.r_[push_fractions, receive_fractions], 10)))
                previous = start if prefix_count == 0 else raise_path[prefix_count - 1]['q']
                sequence, receiver_open = list(raise_path[:prefix_count]), False
                for fraction in fractions[1:]:
                    pair = {pusher: sample(push_path, push_fractions, fraction),
                            receiver: sample(receive_path, receive_fractions, fraction)}
                    if not receiver_open:
                        # Opening at the initial pose can hit the upper forearm.
                        # Start both arms together and open at the first clear segment.
                        can_open = all(link(previous, pair, yaw, underneath=False, hand_shapes=mesh)
                                       for mesh in opening_shapes)
                        if can_open:
                            receiver_open = True
                        elif not link(previous, pair, yaw, **uncovered):
                            break
                    elif not link(previous, pair, yaw, **opened):
                        break
                    sequence.append(dict(phase='raise_pusher_and_tip_receiver', q=pair,
                                         command=release_hand if receiver_open else upper_open))
                    previous = pair
                else:
                    if receiver_open:
                        return sequence
            return None

        # Keep the supporting fingers closed while the upper cup moves away.
        if not all(clear(start, yaw, underneath=False,
                         hand_shapes=dict(shapes[-1], **{pusher: mesh[pusher]}))
                   for mesh in release_shapes):
            return None
        raised = None
        for rise in (.100, .080, .120, .060, .160, .040):
            previous, path = start, []
            for height in np.linspace(0., rise, 4)[1:]:
                q = solve(pusher, frames[pusher][:, 3] + height * vertical,
                          frames[pusher][:, :3], yaw, previous[pusher])
                after = None if q is None else dict(previous, **{pusher: q})
                if after is None or not link(previous, after, yaw, **uncovered):
                    break
                path.append(dict(phase='raise_pusher', q=after, command=upper_open))
                previous = after
            else:
                if all(clear(previous, yaw, underneath=False,
                             hand_shapes=dict(release_shapes[-1], **{receiver: mesh[receiver]}))
                       for mesh in release_shapes):
                    raised = previous, path
                    break
        if raised is None:
            return None
        previous, raise_path = raised
        frame = frames[receiver]
        pivot = frame[:, 3] + frame[:, :3] @ receive_pad
        normal, down = frame[:, 0], -vertical
        full_turn = float(np.arccos(np.clip(normal @ down, -1., 1.)))
        axis = np.cross(normal, down)
        if np.linalg.norm(axis) < 1e-7:
            axis = frame[:, 2].copy()
        axis /= np.linalg.norm(axis)
        strategies = [('finger_roll_negative', -frame[:, 2], np.pi),
                      ('normal_arc', axis, full_turn),
                      ('finger_roll_positive', frame[:, 2], np.pi)]
        best = None
        for mode, axis, limit in strategies:
            previous = raised[0]
            turns = np.linspace(0., limit, max(2, int(np.ceil(limit / .18)) + 1))[1:]
            tipped, reached = [], 0.
            for turn in turns:
                rotation = Rotation.from_rotvec(axis * turn).as_matrix() @ frame[:, :3]
                q = solve(receiver, pivot - rotation @ receive_pad, rotation, yaw, previous[receiver])
                after = None if q is None else dict(previous, **{receiver: q})
                if after is None or not link(previous, after, yaw, **opened):
                    # Refine the last feasible interval instead of losing a partial flip.
                    low, high = reached, float(turn)
                    for _ in range(4):
                        middle = .5 * (low + high)
                        rotation = Rotation.from_rotvec(axis * middle).as_matrix() @ frame[:, :3]
                        q = solve(receiver, pivot - rotation @ receive_pad, rotation, yaw, previous[receiver])
                        after = None if q is None else dict(previous, **{receiver: q})
                        if after is None or not link(previous, after, yaw, **opened):
                            high = middle
                        else:
                            tipped.append((middle, after))
                            previous, reached, low = after, middle, middle
                    break
                tipped.append((float(turn), after))
                previous, reached = after, float(turn)
            # Rank by the actual palm normal, not the finger axis or turn angle.
            candidates = sorted(enumerate(tipped),
                                key=lambda row: float(palm(receiver, row[1][1][receiver], yaw)[:, 0] @ down),
                                reverse=True)
            if full_turn < .01:
                candidates = [(-1, (0., raised[0]))]
            for index, (turn, pair) in candidates:
                if full_turn >= .01 and turn < .03:
                    continue
                tip_frame = palm(receiver, pair[receiver], yaw)
                palm_down = float(tip_frame[:, 0] @ down)
                if best is not None and palm_down <= best['release_palm_down']:
                    continue
                result = None
                for rise in (.050, .035, .070):
                    q = solve(receiver, tip_frame[:, 3] + rise * vertical,
                              tip_frame[:, :3], yaw, pair[receiver])
                    lifted = None if q is None else dict(pair, **{receiver: q})
                    if lifted is None or not link(pair, lifted, yaw, **opened):
                        continue
                    retreat = [lifted]
                    retreat_connected = False
                    rejoin = [] if item.get('mouse_last_in_batch') else delivery['retreat'][:8]
                    for retreat_index, retreat_pair in enumerate(rejoin):
                        if link(lifted, retreat_pair, yaw, **opened):
                            retreat = delivery['retreat'][retreat_index:]
                            retreat_connected = True
                            break
                    sequence = [dict(phase='open_pusher', q=start, command=upper_open),
                                *raise_path,
                                dict(phase='open_receiver', q=raised[0], command=release_hand),
                                *[dict(phase='tip_receiver', q=q, command=release_hand)
                                  for _, q in tipped[:index + 1]],
                                dict(phase='lift_receiver', q=lifted, command=release_hand)]
                    combined = overlap_release(raise_path, tipped[:index + 1])
                    if combined is not None:
                        sequence = [sequence[0], *combined, sequence[-1]]
                    result = dict(release_sequence=sequence, retreat=retreat,
                                  release_simultaneous=combined is not None,
                                  release_retreat_connected=retreat_connected,
                                  release_tip_angle_rad=turn,
                                  release_palm_down=palm_down,
                                  release_full_flip=bool(palm_down >= .95),
                                  release_tip_mode=mode)
                    if result is not None:
                        break
                if result is not None:
                    best = result
                    if best['release_full_flip']:
                        return best
                    break
        return best

    def hand_rotation(azimuth, pitch, side, roll=0.):
        forward = np.cos(azimuth) * inward + mirror * np.sin(azimuth) * along
        fingers = np.cos(pitch) * forward - np.sin(pitch) * up
        normal = np.sin(pitch) * forward + np.cos(pitch) * up
        if side == pusher:
            normal = -normal
        normal = Rotation.from_rotvec(mirror * roll * fingers).apply(normal)
        return np.column_stack((normal, np.cross(fingers, normal), fingers))

    def move_one(side, before, origin, rotation, yaw, seed=None):
        q = solve(side, origin, rotation, yaw, before[side] if seed is None else seed)
        if q is None:
            return None
        pair = dict(before, **{side: q})
        return pair if link(before, pair, yaw) else None

    def approach_one(side, before, origin, rotation, yaw, seed):
        approach = []
        start = palm(side, before[side], yaw)
        middle_rotation = Slerp([0., 1.], Rotation.from_matrix([start[:, :3], rotation]))(.55).as_matrix()
        if side == pusher:
            hover = origin + (.110 if cup else .065) * up
            hover_q = solve(side, hover, rotation, yaw, seed)
            if hover_q is None:
                return None
            middle_seed = .45 * before[side] + .55 * hover_q
            paths = []
            for setback, spread, rise in ((.09, .15, .08), (.16, .11, .12), (.10, .16, .14), (0., 0., 0.)):
                middle = .45 * start[:, 3] + .55 * origin - setback * inward + mirror * spread * along + rise * up
                paths.append(((middle, middle_rotation, middle_seed), (hover, rotation, hover_q)))
        else:
            paths = []
            for setback, rise in ((.085, .075), (.100, .105), (.075, .060)):
                outside = origin - setback * inward
                paths.append(((outside + rise * up, rotation, seed), (outside, rotation, seed)))
                middle = .45 * start[:, 3] + .55 * outside - .04 * inward - mirror * .055 * along + .11 * up
                paths.append(((middle, middle_rotation, before[side]), *paths[-1]))
        for path in paths:
            pair = before
            approach.clear()
            for waypoint, orient, initial in (*path, (origin, rotation, seed)):
                pair = move_one(side, pair, waypoint, orient, yaw, initial)
                if pair is None:
                    break
                approach.append(pair)
            else:
                return approach
        return None

    def slide(before, left_at, left_end, left_turn, end_turn, yaw, cloud_at, cloud_end, first_pad, last_pad,
              outward_bow=0.):
        turn = Slerp([0., 1.], Rotation.from_matrix([left_turn, end_turn]))
        count = max(2, int(np.ceil((np.linalg.norm(left_end - left_at) + 2. * outward_bow) / .025)))
        path = []
        for f in np.linspace(0., 1., count + 1)[1:]:
            # Bow away from the torso in the table plane; keep both endpoints
            # and the receiving-palm centre fixed.
            bow = 4. * f * (1. - f) * outward_bow * inward
            moved = (1. - f) * cloud_at + f * cloud_end + bow
            if in_obstacle(moved) or (low_object and lower_body.collides(
                    routes.local(moved.T, yaw).T, yaw, clearance=.006)):
                return None
            blend = f ** .55
            orient = turn(blend).as_matrix()
            pad = (1. - blend) * first_pad + blend * last_pad
            point = (1. - f) * left_at + f * left_end + bow
            before = move_one(pusher, before, point - orient @ pad, orient, yaw)
            if before is None:
                return None
            path.append(dict(phase='scoop_open', q=before, command=commands[0], push_point=pad))
        return path

    def side_contact(shift, finger_slope=.6, roll=.65):
        direction = shift / np.linalg.norm(shift)
        sideway = np.cross(direction, up)
        parts = shapes[0][pusher]
        mesh = np.concatenate(list(parts.values()))
        base = parts['base_link']
        lateral = cup and abs(direction @ along) > .8
        inner = mesh if lateral else base[base[:, 0] >= np.quantile(base[:, 0], .70)]
        digits = np.concatenate([part for name, part in parts.items() if name != 'base_link'])
        shell = cloud - center
        top_height = np.quantile(shell @ up, .97)
        back = np.quantile(shell @ direction, .025)
        span_low, span_high = np.quantile(shell @ sideway, (.02, .98))
        fingers = -mirror * sideway + (.8 if lateral else -1.2) * direction - finger_slope * up
        fingers /= np.linalg.norm(fingers)
        normal = direction - (direction @ fingers) * fingers
        normal = Rotation.from_rotvec(-mirror * roll * fingers).apply(normal / np.linalg.norm(normal))
        orient = np.column_stack((normal, np.cross(fingers, normal), fingers))
        lowest = (mesh @ orient.T @ up).min()
        inner_world = inner @ orient.T
        height = inner_world @ up - lowest + .007
        pad = (height >= .004) & (height <= top_height + .001)
        if np.count_nonzero(pad) < 7:
            return None
        front = np.quantile(inner_world[pad] @ direction, .98)
        patch = pad & (inner_world @ direction > front - .007)
        origin = (center + (.007 - lowest) * up - np.median(inner_world[patch] @ sideway) * sideway
                  + (back - .001 - front) * direction)
        digit_cloud = digits @ orient.T + origin - center
        span = digit_cloud @ sideway
        facing = (span > span_low - .001) & (span < span_high + .001) & (digit_cloud @ up <= top_height + .002)
        if np.any(digit_cloud[facing] @ direction > back + .002):
            return None
        return origin, orient, np.median(inner[patch], axis=0)

    heading = routes.yaw + np.arctan2(center[1], center[0])
    lateral_first = cup
    variants = ((.37, .060, 0.), (.32, .060, 0.), (.37, .040, 0.)) if cup else ((.27, .055, 0.),)
    force_top = cup and item.get('mouse_top_push', False)
    variants = [(yaw_offset, distance, bow, 0., force_top) for yaw_offset, distance, bow in variants]
    if low_object:
        variants.extend(((.50, .090, 0., .025, False), (.60, .140, 0., .045, False)))
        if item.get('low_cup_arc') and receiver == 'right':
            if center[0] > edge + .20:
                variants.insert(0, (.55, .135, 0., 0., False))
            else:
                variants.insert(0, (.60, .160, 0., .055, False))
    if (cup and not low_object and item.get('adaptive_batch') and receiver == 'right'
            and abs(center[1]) < .065 and center[0] > edge + .15):
        # A diagonal first stroke clears the legs before turning toward the rim.
        top_order = (True, False) if item.get('mouse_prefer_top') else (False, True)
        variants[:0] = [(.60, .160, 0., .055, top) for top in ((True,) if force_top else top_order)]
        if center[0] > edge + .20 and not force_top:
            variants.insert(0, (.55, .135, 0., 0., False))
    for yaw_offset, staging_distance, initial_bow, staging_inward, top_push in variants:
        staged = staging_inward != 0. or staging_distance >= .10
        active_shapes = None
        push_seed = side_seed
        yaw = heading - mirror * yaw_offset
        if not all(clear(positions, angle) for angle in np.linspace(start_yaw, yaw, 13)):
            continue
        middle_shift = (staging_inward * inward - mirror * staging_distance * along if lateral_first else
                        (edge + staging_distance - center[0]) / inward[0] * inward + mirror * .032 * along)
        contact = side_contact(middle_shift)
        if contact is None:
            continue
        origin, first_turn, side_pad = contact
        start, start_turn = origin, first_turn
        if top_push:
            first_turn = hand_rotation(-1.622, .147, pusher, -.139)
            mesh = np.concatenate(list(top_shapes[pusher].values()))
            rotated = mesh @ first_turn.T
            heights = rotated @ up
            patch = rotated[heights <= heights.min() + .004]
            patch_center = np.median(patch, axis=0)
            pad_world = patch[np.linalg.norm(patch - patch_center, axis=1).argmin()]
            side_pad = first_turn.T @ pad_world
            shell_heights = cloud @ up
            surface = cloud[shell_heights >= np.quantile(shell_heights, .85)]
            surface_center = np.median(surface, axis=0)
            contact_point = surface[np.linalg.norm(surface - surface_center, axis=1).argmin()]
            press_depth = min(.006, .25 * max(0., contact_point @ up - floor))
            origin = contact_point - pad_world - press_depth * up
            start, start_turn = origin, first_turn
            active_shapes = top_shapes
            push_seed = np.array([-.92917, .81622, -1.61395, -1.45657, -.39005, -.19842, .43381]) * handed[pusher]
        elif cup:
            staging = side_contact(middle_shift, .48)
            if staging is None:
                continue
            start, start_turn, _ = staging
            start = start - .055 * middle_shift / np.linalg.norm(middle_shift)
            if staging_inward:
                endpoint = solve(pusher, start, start_turn, yaw, push_seed)
                if endpoint is None or not clear(dict(positions, **{pusher: endpoint}), yaw):
                    continue
        approach = approach_one(pusher, positions, start, start_turn, yaw, push_seed)
        if approach is None:
            continue
        engaged = approach[-1]
        engage_step = []
        if cup:
            # Start behind the shell, then establish contact before the modeled slide.
            engaged = move_one(pusher, engaged, origin, first_turn, yaw, push_seed)
            if engaged is None:
                continue
            engage_step = [dict(phase='scoop_open', q=engaged,
                                command=top_command if top_push else commands[0], push_point=side_pad)]
        side_start = origin + first_turn @ side_pad
        middle_side = side_start + middle_shift
        initial_slide = slide(engaged, side_start, middle_side, first_turn, first_turn, yaw,
                              cloud, cloud + middle_shift, side_pad, side_pad, outward_bow=initial_bow)
        if initial_slide is None:
            continue
        if top_push:
            for step in initial_slide:
                step['command'] = top_command
        if item.get('mouse_stage_only'):
            before = initial_slide[-1]['q']
            raised = move_one(pusher, before, origin + middle_shift + .10 * up, first_turn, yaw)
            if raised is None or not link(raised, positions, yaw):
                continue
            for step in initial_slide:
                step['segment'] = 'right_shift'
            return dict(item=item, mouse_staging_only=True, pick_yaw=yaw, approach=approach,
                        preshape=top_command if top_push else commands[0],
                        close=[*engage_step, *initial_slide], retreat=[raised, positions],
                        edge_receive=True, receive_point=receive_pad, pusher_side=pusher,
                        grip_side=receiver, mouse_lateral_first=True)
        active_shapes = None
        if lateral_first:
            for step in initial_slide:
                step['segment'] = 'right_shift'
            # Reposition above the translated shell, then come down behind it
            # with a 55 mm gap before pushing toward the receiving palm.
            rear = side_contact(-inward, roll=.70 if staged else .65)
            if rear is None:
                continue
            rear_origin, rear_turn, rear_pad = rear
            rear_origin = rear_origin + middle_shift
            before = initial_slide[-1]['q']
            for back_index, (point, orient) in enumerate(((origin + middle_shift + .080 * up, first_turn),
                                  (rear_origin + .055 * inward + .080 * up, rear_turn),
                                  (rear_origin + .055 * inward, rear_turn), (rear_origin, rear_turn))):
                active_shapes = top_shapes if top_push and back_index == 0 else None
                before = move_one(pusher, before, point, orient, yaw)
                if before is None:
                    break
                initial_slide.append(dict(phase='scoop_open', q=before,
                                          command=top_command if top_push and back_index == 0 else commands[0],
                                          push_point=rear_pad, segment='around_back'))
            else:
                first_turn, side_pad = rear_turn, rear_pad
                middle_side = rear_origin + rear_turn @ rear_pad
            active_shapes = None
            if before is None:
                continue
        receiver_profiles = ((.306, .25, 0.), (.30, .19, 0.)) if cup else ((.306, .25, .015), (.30, .19, .008))
        if staged:
            receiver_profiles = ((.27, .25, 0.), *receiver_profiles)
        for lateral, incline, tip_gap in receiver_profiles:
            receive_turn = hand_rotation(.05, incline, receiver, .062)
            tips = np.concatenate([shapes[0][receiver][name] for name in DISTAL]) @ receive_turn.T
            all_right = np.concatenate(list(shapes[0][receiver].values())) @ receive_turn.T
            receive = np.array([edge - .008 - all_right[:, 0].max(), -mirror * lateral, 0.])
            if cup:
                index_tip = shapes[0][receiver]['Link4'] @ receive_turn.T
                receive += (floor + tip_gap - (index_tip + receive) @ up).min() * up
            else:
                receive += (floor - tip_gap - (tips + receive) @ up).min() * up
            setback = .008 if cup else .015
            receive += (edge - setback - (all_right + receive)[:, 0].max()) / inward[0] * inward
            if cup:
                receive[2] -= .010
            receiver_path = approach_one(receiver, initial_slide[-1]['q'], receive, receive_turn, yaw, receive_seed)
            if receiver_path is None:
                continue
            pre_catch = initial_slide
            concurrent_receiver = False
            final_receiver = receiver_path[-1]
            if cup and lateral_first and (item.get('timed_batch') or item.get('parallel_receiver')):
                first_back = next((i for i, step in enumerate(initial_slide)
                                   if step.get('segment') == 'around_back'), len(initial_slide))
                before_back = initial_slide[first_back - 1]['q']
                push_path = [before_back, *[step['q'] for step in initial_slide[first_back:]]]
                catch_path = [before_back, *receiver_path]
                receiver_mesh = np.concatenate(list(shapes[0][receiver].values()))

                def sample(path, fraction, side):
                    at = min(fraction * (len(path) - 1), len(path) - 1.)
                    index = min(int(at), len(path) - 2)
                    blend = at - index
                    return (1. - blend) * path[index][side] + blend * path[index + 1][side]

                fractions = sorted(set(np.linspace(0., 1., len(push_path))) |
                                   set(np.linspace(0., 1., len(catch_path))))
                merged, previous = [], before_back
                for fraction in fractions[1:]:
                    pair = {pusher: sample(push_path, fraction, pusher),
                            receiver: sample(catch_path, fraction, receiver)}
                    if not link(previous, pair, yaw):
                        break
                    separated = True
                    for blend in np.linspace(0., 1., 9):
                        q = (1. - blend) * previous[receiver] + blend * pair[receiver]
                        frame = palm(receiver, q, yaw)
                        hand_cloud = receiver_mesh @ frame[:, :3].T + frame[:, 3]
                        if np.any(cKDTree(hand_cloud).query(cloud + middle_shift,
                                  distance_upper_bound=.006)[0] < .006):
                            separated = False
                            break
                    if not separated:
                        break
                    merged.append(dict(phase='scoop_open', q=pair, command=commands[0],
                                       push_point=rear_pad, segment='around_back_and_receiver'))
                    previous = pair
                else:
                    pre_catch = [*initial_slide[:first_back], *merged]
                    concurrent_receiver = True
            if cup and not lateral_first:
                # Only the receiver's first hover may overlap the last straight push.
                previous = initial_slide[-2]['q']
                concurrent = receiver_path[0]
                receiver_mesh = np.concatenate(list(shapes[0][receiver].values()))
                separate = True
                for fraction in np.linspace(0., 1., 9):
                    q = (1. - fraction) * previous[receiver] + fraction * concurrent[receiver]
                    frame = palm(receiver, q, yaw)
                    hand_cloud = receiver_mesh @ frame[:, :3].T + frame[:, 3]
                    progress = 1. - (1. - fraction) / len(initial_slide)
                    moved = cloud + progress * middle_shift + 4. * progress * (1. - progress) * initial_bow * inward
                    if np.any(cKDTree(hand_cloud).query(moved, distance_upper_bound=.006)[0] < .006):
                        separate = False
                        break
                if separate and link(previous, concurrent, yaw):
                    pre_catch = [*initial_slide[:-1], dict(initial_slide[-1], q=concurrent)]
                    receiver_path = receiver_path[1:]
            support = receive + receive_turn @ receive_pad
            for angle, drop, setback in ((-.40, -.008, -.004),):
                pitch_mouse = Rotation.from_rotvec(mirror * angle * along).as_matrix()
                orient_mouse = Rotation.from_rotvec(mirror * .32 * up).as_matrix() @ pitch_mouse
                held_center = support + drop * up if cup else support + setback * inward + drop * up - mirror * .005 * along
                held_cloud = (cloud - center) @ orient_mouse.T + held_center
                last_top = support + pitch_mouse @ (top - center) + .012 * inward - mirror * .030 * along
                final_turn = hand_rotation(-1.622, .147, pusher, -.139)
                final_slide = slide(final_receiver, middle_side, last_top, first_turn, final_turn, yaw,
                                    cloud + middle_shift, held_cloud, side_pad, push_pad,
                                    outward_bow=.035 if cup else 0.)
                if final_slide is None:
                    continue
                if lateral_first:
                    for step in final_slide:
                        step['segment'] = 'edge_shift'
                held = final_slide[-1]['q']
                if not all(clear(held, yaw, stage) for stage in range(1, len(commands))):
                    continue
                # Require each curled hand to be close enough to meet the shell.
                for side in SIDES:
                    hand = routes.hands[side]
                    frame = palm(side, held[side], yaw)
                    shell = (held_cloud - frame[:, 3]) @ frame[:, :3]
                    links_at = hand.frames(hand.nominal_angles(grip[side]))
                    distance = min(finger_distance(hand, links_at[name], name, shell)
                                   for name in (*DISTAL, 'base_link'))
                    if not -.003 <= distance <= .006:
                        break
                else:
                    held_frames = {side: palm(side, held[side], yaw) for side in SIDES}
                    shell_in_hand = {side: (held_cloud - frame[:, 3]) @ frame[:, :3]
                                     for side, frame in held_frames.items()}

                    def delivery_clear(q, waist, *, opened=False, underneath=False):
                        if not clear(q, waist, len(commands) - 1, underneath=underneath,
                                     hand_shapes=release_shapes[-1] if opened else None):
                            return False
                        if opened or (not contents and not low_object):
                            return True
                        for side in SIDES:
                            frame = palm(side, q[side], waist)
                            moved = shell_in_hand[side] @ frame[:, :3].T + frame[:, 3]
                            if low_object and lower_body.collides(
                                    routes.local(moved.T, waist).T, waist, clearance=.006):
                                return False
                            if any(np.any(np.all((moved >= lo - .004) & (moved <= hi + .004), axis=1))
                                   for lo, hi in contents):
                                return False
                        return True

                    checked_after_release = False
                    if low_object:
                        ready_path = positions
                        cup_arc = item.get('low_cup_arc') and receiver == 'right' and staged
                        if cup_arc:
                            ready_path = [{
                                'left': np.array([-1.20093, .23544, -1.15982, -1.76650, -.44775, -.07761, -.53591]),
                                'right': np.array([.67586, -.44061, -.02828, -2.44235, -2.34920, .34657, -.29534]),
                            }, positions]
                            checked_after_release = bool(staging_inward)
                        placements = _low_object_placements(routes, item, yaw, held_cloud - held_center,
                                                            cup_arc=cup_arc,
                                                            short_lowering=checked_after_release)
                    else:
                        # Keep the shell above the shared tray while opening both hands.
                        release = np.r_[tray_low[:2] + [.070858, .051751], tray_high[2] + .247336]
                        tray_yaw = routes.yaw + np.arctan2(release[1], release[0]) - .030698
                        tray_turn = Rotation.from_euler('zyx', [1.603673, -.060048, .446866]).as_matrix()
                        ready_arc = {
                            'left': np.array([-1.20093, .23544, -1.15982, -1.76650, -.44775, -.07761, -.53591]),
                            'right': np.array([.67586, -.44061, -.02828, -2.44235, -2.34920, .34657, -.29534]),
                        }
                        ready_path = [ready_arc, positions]
                        if staged and item.get('mouse_last_in_batch') and item.get('cup_trial_release'):
                            # The release planner checks the final upward retreat for the last item.
                            ready_path = None
                        placements = [(tray_yaw, release + [0., 0., rise], tray_turn, .020, None)
                                      for rise in (0., .010)]
                        if cup:
                            cup_turn = Rotation.from_euler('zyx', [-.05219380, .06627297, -.08178732]).as_matrix() @ tray_turn
                            placements = [(tray_yaw - .07061630,
                                           release + [-.00644312, .00127530, -.015],
                                           cup_turn, .035, None)]
                            if staged:
                                placements = cup_placements(
                                    placements[0], held_cloud - held_center,
                                    raised_release=not staging_inward and item.get('cup_trial_release'))
                    lift_heights = (.020, .040, .095, .190, .285) if staged else (.020, .040, .385)
                    delivery = edge_pair_delivery(
                        routes, item, held, yaw, held_cloud, held_center, solve, delivery_clear,
                        start_ready=None if checked_after_release else ready_path,
                        edge=edge, withdraw=.015, obstacles=obstacles,
                        lift_profile=[([0., 0., h], np.eye(3)) for h in lift_heights],
                        placements=placements,
                        route_resolution=(.030, .15) if checked_after_release else (.017, .10))
                    if delivery is None:
                        continue
                    if not all(clear(delivery['release'][-1], delivery['tray_yaw'],
                                     underneath=False, hand_shapes=shape) for shape in release_shapes[:-1]):
                        continue
                    if cup:
                        if item.get('cup_trial_release'):
                            trial_release = cup_trial_release(delivery)
                            if trial_release is None:
                                continue
                            if low_object and (checked_after_release or not trial_release['release_retreat_connected']):
                                retreat = low_cup_retreat(delivery, trial_release, ready_path[0] if cup_arc else None)
                                if retreat is None:
                                    continue
                                trial_release.update(retreat=retreat, release_retreat_connected=True)
                            delivery.update(trial_release)
                        else:
                            retreat = cup_retreat(delivery)
                            if retreat is None:
                                continue
                            delivery['retreat'] = retreat
                        delivery['release_hand'] = release_hand
                        final_pair = delivery['release'][-1]
                        final_yaw = delivery['tray_yaw']
                        bottom = np.inf
                        for side in SIDES:
                            frame = palm(side, final_pair[side], final_yaw)
                            shell_z = shell_in_hand[side] @ frame[2, :3] + frame[2, 3]
                            bottom = min(bottom, shell_z.min())
                        floor_gap = float(bottom - tray_low[2])
                        delivery['cup_gap_above_rim_m'] = float(bottom - tray_high[2])
                        delivery['cup_gap_above_floor_m'] = floor_gap
                        delivery['cup_release'] = 0. <= floor_gap <= .020
                    if item.get('timed_batch'):
                        # The grasp and delivery endpoints stay unchanged. Accept a
                        # simultaneous waist/arm arc only if the full loaded meshes clear it.
                        path = delivery['transfer']
                        before_pair = delivery['lift'][-1]
                        distances = [max(float(np.max(abs(q[side] - previous[side]))) for side in SIDES)
                                     for previous, q in zip([before_pair, *path[:-1]], path)]
                        total = max(sum(distances), .001)
                        progress, before_yaw, coordinated = 0., yaw, []
                        for q, distance in zip(path, distances):
                            progress += distance
                            angle = yaw + (delivery['tray_yaw'] - yaw) * progress / total
                            count = max(3, int(max(abs(angle - before_yaw), distance) / .04) + 2)
                            if not all(delivery_clear(
                                    {side: (1. - f) * before_pair[side] + f * q[side] for side in SIDES},
                                    (1. - f) * before_yaw + f * angle)
                                    for f in np.linspace(0., 1., count)):
                                break
                            coordinated.append(dict(q=q, yaw=angle))
                            before_pair, before_yaw = q, angle
                        else:
                            if coordinated:
                                delivery['coordinated_transfer'] = coordinated
                    receive_steps = ([] if concurrent_receiver else [
                        dict(phase='scoop_open', q=q, command=commands[0], segment='receiver_approach')
                        for q in receiver_path])
                    push_steps = [*engage_step, *pre_catch, *receive_steps, *final_slide]
                    slide_path = [step['q'] for step in push_steps]
                    close_steps = [dict(phase='close', q=held, command=command) for command in commands[1:]]
                    reverse = [*reversed(slide_path[:-1]), *reversed(approach), positions]
                    return dict(delivery, pick_yaw=yaw, approach=approach, grasp=[held],
                                close=push_steps + close_steps, abort_after_push=reverse,
                                abort_close=[dict(phase='withdraw_open', q=q, command=commands[0]) for q in reverse],
                                pickup_retreat=[*reversed(approach[:-1]), positions],
                                preshape=top_command if top_push else commands[0], grip=commands[-1], paired_carry=True,
                                mouse_top_push=top_push,
                                edge_receive=True, under_table=True, probe_only=False,
                                mouse_lateral_first=lateral_first,
                                concurrent_receiver=concurrent_receiver,
                                grip_side=receiver, pusher_side=pusher, receive_point=receive_pad,
                                push_point=side_pad, receiver_edge_x_m=edge, edge_object_center=held_center)
    return None


def _mouse_table_sweep_plan(routes, item, positions, box_removed, lower_body, start_yaw, *,
                            reverse_receiver=False, deadline=None):
    """Keep a single palm frame on the table until the mouse reaches its cup."""
    from scipy.optimize import least_squares
    from .cable_bimanual import _section_hits_table

    delivery = _mouse_shell_plan(routes, dict(item, mouse_table_sweep=False, mouse_top_push=False),
                                 positions, box_removed, lower_body, start_yaw,
                                 reverse_receiver=reverse_receiver, deadline=deadline)
    if delivery is None:
        return None
    pusher, receiver = delivery['pusher_side'], delivery['grip_side']
    if pusher != 'left':
        return None
    yaw = delivery['pick_yaw']
    up = routes.normal / np.linalg.norm(routes.normal)
    floor = routes.floor / np.linalg.norm(routes.normal)
    edge = routes.edge + .012
    inward = np.array([1., 0., 0.]) - up[0] * up
    inward /= np.linalg.norm(inward)
    along = np.cross(up, inward)
    cloud = np.asarray(item['target']['contact_points'])
    center = np.array(item['target']['xyz'], dtype=float)
    center += (floor - center @ up) * up
    end_center = np.asarray(delivery['edge_object_center']).copy()
    end_center += (floor - end_center @ up) * up
    direction = end_center - center
    direction -= direction @ up * up
    direction /= np.linalg.norm(direction)
    sideway = np.cross(up, direction)
    held = delivery['grasp'][-1]
    commands = {pusher: [.90, .88, .88, .84, 0., 1.], receiver: delivery['preshape'][receiver]}
    shapes = {s: routes.hands[s].vertices(routes.hands[s].nominal_angles(commands[s])) for s in SIDES}
    torso, torso_lo, torso_hi, links = body_geometry()
    solids = []
    for obj in routes.objects:
        if obj['kind'].startswith('tray_'):
            lo, hi = np.asarray(obj['bounds'])
            for axis in (0, 1):
                a, b = hi.copy(), lo.copy()
                a[axis], b[axis] = lo[axis] + .012, hi[axis] - .012
                solids.extend(((lo, a), (b, hi)))
        elif obj['kind'] != item['kind'] and not (box_removed and obj['kind'] == 'drive'):
            solids.append(tuple(np.asarray(obj['bounds'])))

    def palm(side, q):
        arm = routes.arms[side]
        return routes.world((arm.wrist(q) @ arm.hand)[:3], yaw)

    def solve(side, origin, orient, seed, contact_pad=None):
        if deadline is not None and monotonic() >= deadline:
            return None
        arm = routes.arms[side]
        rot = routes.local(orient, yaw)
        goal = routes.local(origin, yaw) + rot @ arm.pocket
        def residual(q):
            pose = arm.tool(q)
            return np.r_[pose[:3, 3] - goal,
                         .24 * Rotation.from_matrix(rot.T @ pose[:3, :3]).as_rotvec(),
                         .00003 * (q - seed)]
        result = least_squares(residual, np.clip(seed, arm.limits[0] + .02, arm.limits[1] - .02),
                               bounds=(arm.limits[0] + .018, arm.limits[1] - .018),
                               max_nfev=80, ftol=1e-6, gtol=1e-8)
        if np.linalg.norm(result.fun[:3]) < .003 and np.linalg.norm(result.fun[3:6]) < .24 * np.deg2rad(1.):
            return result.x
        if contact_pad is not None:
            target_patch = origin + orient @ contact_pad
            target_low = (mesh @ orient.T @ up).min() + origin @ up
            def contact_residual(q):
                frame = palm(side, q)
                actual_patch = frame[:, 3] + frame[:, :3] @ contact_pad
                actual_low = (mesh @ frame[:, :3].T @ up).min() + frame[:, 3] @ up
                return np.r_[actual_patch - target_patch, 2. * (actual_low - target_low),
                             .07 * Rotation.from_matrix(orient.T @ frame[:, :3]).as_rotvec(),
                             .00003 * (q - seed)]
            result = least_squares(contact_residual, result.x,
                                   bounds=(arm.limits[0] + .018, arm.limits[1] - .018),
                                   max_nfev=80, ftol=1e-6, gtol=1e-8)
            value = contact_residual(result.x)
            if (np.linalg.norm(value[:3]) < .003 and abs(value[3]) < .004
                    and np.linalg.norm(value[4:7]) < .07 * np.deg2rad(5.)):
                return result.x
        return None

    def clear(pair, support=False, cup_contact=False):
        if deadline is not None and monotonic() >= deadline:
            return False
        fore, normal = routes.local(np.array([1., 0., 0.]), yaw), routes.local(up, yaw)
        bodies, arms, hands = {}, {}, {}
        for side in SIDES:
            model = routes.arms[side]
            frames = model.frames(pair[side]); hand = frames[-1] @ model.hand
            arm_parts = [v @ f[:3, :3].T + f[:3, 3] for v, f in zip(links[side], frames[2:])]
            hand_parts = [v @ hand[:3, :3].T + hand[:3, 3] for v in shapes[side].values()]
            if any(_section_hits_table(v, fore, normal, floor, edge, False) for v in arm_parts):
                return False
            # A tabletop stroke explicitly allows the pusher's hand to contact
            # the surface; arm links retain the ordinary 5 mm clearance.
            hand_floor = floor - .007 if support and side == pusher else floor
            if any(_section_hits_table(v, fore, normal, hand_floor, edge, False) for v in hand_parts):
                return False
            full = np.concatenate(arm_parts + hand_parts)
            near = full[np.all((full >= torso_lo) & (full <= torso_hi), axis=1)]
            if len(near) and np.any((near @ torso[:, :3].T + torso[:, 3]).max(1) < .002):
                return False
            if lower_body.collides(full, yaw, clearance=.006):
                return False
            world = routes.world(full.T, yaw).T
            if any(np.any(np.all((world >= lo - .004) & (world <= hi + .004), axis=1)) for lo, hi in solids):
                return False
            bodies[side], arms[side], hands[side] = full, np.concatenate(arm_parts), np.concatenate(hand_parts)
        # Closing the two cups intentionally brings their surfaces together.
        # Only that phase relaxes the hand gap; arm clearance stays unchanged.
        return (not np.any(cKDTree(bodies['left']).query(arms['right'], distance_upper_bound=.006)[0] < .006)
                and not np.any(cKDTree(arms['left']).query(hands['right'], distance_upper_bound=.006)[0] < .006)
                and (cup_contact or not np.any(cKDTree(hands['left']).query(
                    hands['right'], distance_upper_bound=.003)[0] < .003)))

    def link(a, b, support=False, cup_contact=False):
        count = max(3, 2 + int(max(np.max(abs(b[s] - a[s])) for s in SIDES) / .055))
        return all(clear({s: (1. - f) * a[s] + f * b[s] for s in SIDES}, support, cup_contact)
                   for f in np.linspace(0., 1., count))

    mesh = np.concatenate(list(shapes[pusher].values()))
    contact_mesh = np.concatenate([part for name, part in shapes[pusher].items()
                                   if name not in {f'Link{i}' for i in range(40, 49)}])
    body_cloud = cloud[cloud @ up > floor + .015]
    if len(body_cloud) < 12:
        return None
    seeds = [held[pusher], np.array([-.92917, .81622, -1.61395, -1.45657, -.39005, -.19842, .43381]),
             np.array([-.755, .056, -1.128, -.669, 1.25, -.004, .706])]
    for azimuth, pitch, roll in ((-1.622, .30, -.139), (-1.622, .45, -.139), (-1.45, .30, -.139)):
        forward = np.cos(azimuth) * inward + np.sin(azimuth) * along
        fingers = np.cos(pitch) * forward - np.sin(pitch) * up
        normal = Rotation.from_rotvec(roll * fingers).apply(-np.sin(pitch) * forward - np.cos(pitch) * up)
        orient = np.column_stack((normal, np.cross(fingers, normal), fingers))
        world = mesh @ orient.T
        lowest = (world @ up).min()
        contact_world = contact_mesh @ orient.T
        height = contact_world @ up - lowest + .001
        band = contact_world[(height > .004) & (height < .030)]
        if len(band) < 5:
            continue
        front = np.quantile(band @ direction, .99)
        patch = band[band @ direction > front - .004]
        back = np.quantile((body_cloud - center) @ direction, .025)
        origin = (center + (.001 - lowest) * up + (back - .002 - front) * direction
                  - np.median(patch @ sideway) * sideway)
        thumb = np.concatenate([part for name, part in shapes[pusher].items()
                                if name in {f'Link{i}' for i in range(40, 49)}]) @ orient.T + origin - center
        side_lo, side_hi = np.quantile((body_cloud - center) @ sideway, [.02, .98])
        obstructing = ((thumb @ sideway > side_lo - .001) & (thumb @ sideway < side_hi + .001)
                       & (thumb @ up < np.quantile((body_cloud - center) @ up, .98)))
        if np.any(thumb[obstructing] @ direction > back + .002):
            continue
        pad = orient.T @ np.median(patch, axis=0)
        first = next((q for seed in seeds if (q := solve(pusher, origin, orient, seed, pad)) is not None), None)
        if first is None:
            continue
        hover = origin + .080 * up
        hover_q = solve(pusher, hover, orient, first, pad)
        if hover_q is None:
            continue
        initial = palm(pusher, positions[pusher])
        middle_rotation = Slerp([0., 1.], Rotation.from_matrix([initial[:, :3], orient]))(.55).as_matrix()
        approach = None
        for setback, spread, rise in ((.09, .15, .08), (.16, .11, .12), (0., 0., 0.)):
            middle = .45 * initial[:, 3] + .55 * origin - setback * inward + spread * along + rise * up
            middle_q = solve(pusher, middle, middle_rotation, .45 * positions[pusher] + .55 * hover_q)
            if middle_q is None:
                continue
            trial = [dict(positions, **{pusher: q}) for q in (middle_q, hover_q, first)]
            if all(link(a, b, support=True) for a, b in zip([positions, *trial[:-1]], trial)):
                approach = trial
                break
        if approach is None:
            continue
        pair = approach[-1]
        steps = []
        for step in delivery['close']:
            if step.get('segment') == 'edge_shift':
                break
            q = step['q'][receiver]
            if np.max(abs(q - pair[receiver])) < .001:
                continue
            after = dict(pair, **{receiver: q})
            if not link(pair, after, support=True):
                break
            steps.append(dict(phase='scoop_open', q=after, command=commands, segment='receiver_approach', push_point=pad))
            pair = after
        else:
            pass
        if np.max(abs(pair[receiver] - held[receiver])) > .02:
            continue
        for fraction in np.linspace(0., 1., 13)[1:]:
            point = origin + fraction * (end_center - center)
            q = solve(pusher, point, orient, pair[pusher], pad)
            after = dict(pair, **{pusher: q}) if q is not None else None
            if after is None or not link(pair, after, support=True):
                break
            steps.append(dict(phase='scoop_open', q=after, command=commands, segment='table_sweep', push_point=pad))
            pair = after
        else:
            # The entire shell reaches the receiver before the upper cup moves
            # over it. Preserve the proven final two-hand enclosure and carry.
            cap_frame = palm(pusher, held[pusher])
            lift_frame = palm(pusher, pair[pusher])
            prior_shapes = shapes
            cap_path = None
            for rise in (.080, .120):
                lift = solve(pusher, lift_frame[:, 3] + rise * up, lift_frame[:, :3], pair[pusher])
                if lift is None:
                    continue
                lift_pair = dict(pair, **{pusher: lift})
                if not link(pair, lift_pair, support=True):
                    continue
                shapes = {s: routes.hands[s].vertices(routes.hands[s].nominal_angles(delivery['preshape'][s])) for s in SIDES}
                before = lift_pair
                trial = []
                for height in np.linspace(rise, .0, 6):
                    cap = solve(pusher, cap_frame[:, 3] + height * up, cap_frame[:, :3], held[pusher])
                    after = dict(before, **{pusher: cap}) if cap is not None else None
                    if after is None or not link(before, after, support=True, cup_contact=True):
                        break
                    trial.append(after)
                    before = after
                else:
                    if link(before, held, support=True, cup_contact=True):
                        cap_path = [*trial, held]
                shapes = prior_shapes
                if cap_path is not None:
                    break
            if cap_path is None:
                continue
            steps.append(dict(phase='scoop_open', q=lift_pair, command=commands, segment='cap_after_sweep', push_point=pad))
            steps.extend(dict(phase='scoop_open', q=q, command=delivery['preshape'], segment='cap_after_sweep',
                              push_point=delivery['push_point']) for q in cap_path)
            close = [step for step in delivery['close'] if step['phase'] != 'scoop_open']
            reverse = [*[step['q'] for step in reversed(steps[:-1])], *reversed(approach), positions]
            return dict(delivery, approach=approach, preshape=commands, close=steps + close,
                        pickup_retreat=[*reversed(approach[:-1]), positions],
                        abort_after_push=reverse,
                        abort_close=[dict(phase='withdraw_open', q=q, command=commands) for q in reverse],
                        mouse_table_sweep=True, mouse_top_push=False, mouse_lateral_first=False,
                        concurrent_receiver=False)
    return None


def supports_low_object_slide(target):
    """Use the slide fallback for objects low enough to approach beside the table."""
    bounds = np.asarray(target.get('bounds', ()), dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        return False
    dimensions = bounds[1] - bounds[0]
    return bool(np.all(dimensions > 0.) and dimensions[2] <= .075)


def _low_object_placements(routes, item, pickup_yaw, centered_cloud, *, cup_arc=False, short_lowering=False):
    """Build release options from the held footprint and the assigned tray."""
    low, high = np.asarray(item['tray']['bounds'])
    site = np.asarray(item['release'])[:2]
    heading = routes.yaw + np.arctan2(site[1], site[0])
    occupied = [np.asarray(bounds) for bounds in item.get('occupied_footprints', {}).values()]

    def fitting_sites(orientation, desired, corners):
        footprint = centered_cloud @ orientation.T
        payload_low, payload_high = footprint.min(0), footprint.max(0)
        near = low[:2] + .009 - payload_low[:2]
        far = high[:2] - .009 - payload_high[:2]
        if np.any(near > far):
            return
        sites = [np.clip(desired, near, far)]
        if corners:
            sites.extend(((near + far) * .5, near, far, [near[0], far[1]], [far[0], near[1]]))
        for xy in sites:
            if all(np.any(np.maximum(bounds[0] - xy - payload_high[:2],
                                     xy + payload_low[:2] - bounds[1]) > .007) for bounds in occupied):
                yield xy, payload_low[2]

    placements = []
    if cup_arc:
        reference = low[:2] + [.070858, .051751]
        yaw = routes.yaw + np.arctan2(reference[1], reference[0]) - .10131430
        orientation = (Rotation.from_euler('zyx', [-.05219380, .06627297, -.08178732]).as_matrix()
                       @ Rotation.from_euler('zyx', [1.603673, -.060048, .446866]).as_matrix())
        # Keep the checked arm arc while adapting the drop to the measured footprint.
        for xy, bottom in fitting_sites(orientation, reference + [-.00644312, .00127530], False):
            height = .217 - bottom
            z = high[2] + (height if short_lowering else min(.244336, height))
            rises = (.008, .035) if short_lowering else (.035, .007)
            placements.extend((yaw, np.r_[xy, z], orientation, rise, None) for rise in rises)
    for turn in (0., -.30, .30, -.50, .50):
        yaw = heading + turn
        if not -2.94 <= yaw <= 3.11:
            continue
        for swivel in (0., -.35, .35):
            orientation = Rotation.from_rotvec([0., 0., yaw - pickup_yaw + swivel]).as_matrix()
            for xy, bottom in fitting_sites(orientation, site, True):
                z = high[2] + .065 - bottom
                placements.append((yaw, np.r_[xy, z], orientation, .045, None))
    return placements


def low_object_slide_plan(routes, item, positions, *, box_removed=False, lower_body=None,
                          start_yaw=None, planning_seconds=None):
    """Try both side shifts and retain only complete cupped-hand routes."""
    if lower_body is None or not supports_low_object_slide(item['target']):
        return None
    x, y, _ = item['target']['xyz']
    cup_arc = x > routes.edge + .15 and abs(y) < .065 and item['tray']['xyz'][1] > 0.
    slide_item = dict(item, family='other', low_object_slide=True, low_cup_arc=cup_arc,
                      cup_mouse=True, cup_trial_release=True,
                      mouse_top_push=False, mouse_table_sweep=False, timed_batch=False)
    if planning_seconds is None:
        planning_seconds = 110. if cup_arc else 38.
    route = mouse_edge_receive_plan(
        routes, slide_item, positions, box_removed=box_removed, probe_only=False, under_table=True,
        lower_body=lower_body, start_yaw=start_yaw, try_other_hand=True, planning_seconds=planning_seconds)
    if route is not None:
        route['low_object_slide'] = True
    return route


def mouse_edge_receive_plan(routes, item, positions, *, box_removed=False, probe_only=True,
                            flat_receiver=False, outside_receiver=False, lower_body=None,
                            start_yaw=None, under_table=False, try_other_hand=False,
                            planning_seconds=None):
    """Slide the mouse across the near table edge onto one open palm."""
    def plan(deadline, reverse_receiver=False):
        if under_table:
            return _mouse_shell_plan(routes, item, positions, box_removed, lower_body, start_yaw,
                                     reverse_receiver=reverse_receiver, deadline=deadline)
        return _mouse_open_edge_plan(routes, item, positions, box_removed=box_removed, probe_only=probe_only,
                                     flat_receiver=flat_receiver, outside_receiver=outside_receiver,
                                     lower_body=lower_body, start_yaw=start_yaw,
                                     reverse_receiver=reverse_receiver, deadline=deadline)

    if planning_seconds is None and not try_other_hand:
        return plan(None)
    budget = planning_seconds if planning_seconds is not None else (90. if item.get('cup_mouse') else 35.)
    centered_mouse = (under_table and item.get('cup_mouse') and item.get('adaptive_batch')
                      and not item.get('low_object_slide') and abs(item['target']['xyz'][1]) < .065
                      and item['target']['xyz'][0] > routes.edge + .165)
    if centered_mouse and try_other_hand and not item.get('mouse_stage_only'):
        budget = max(budget, 75.)
    start = monotonic()
    deadline = start + budget
    if not try_other_hand:
        return plan(deadline)

    def usable(route):
        if route is None or (not probe_only and route.get('probe_only')):
            return False
        if probe_only or not item.get('cup_mouse') or route.get('mouse_staging_only'):
            return True
        return route.get('cup_release') or (
            item.get('cup_trial_release') and 0. <= route.get('cup_gap_above_rim_m', np.inf) <= .220)

    # Leave part of the wall-clock budget for the other approach.
    natural_receiver = 'left' if item['target']['xyz'][1] >= 0. else 'right'
    favor_right = centered_mouse or item.get('low_cup_arc')
    preferred_hand = item.get('preferred_hand', 'right' if favor_right else natural_receiver)
    first_share = .84 if favor_right else .65
    if item.get('low_cup_arc') and preferred_hand == 'left':
        first_share = .35
    preferred = plan(start + first_share * budget, reverse_receiver=preferred_hand != natural_receiver)
    if usable(preferred):
        return preferred
    alternate = plan(deadline, reverse_receiver=preferred_hand == natural_receiver)
    return alternate if usable(alternate) else None


def _mouse_open_edge_plan(routes, item, positions, *, box_removed, probe_only, flat_receiver,
                          outside_receiver, lower_body, start_yaw, reverse_receiver, deadline):
    target = item['target']
    points = np.asarray(target.get('contact_points', ()))
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 20:
        return None
    center, _, _, up = object_frame(target, routes.normal, routes.floor)
    if routes.edge < 0.:
        return None
    receiver, pusher = SIDES if center[1] >= 0. else SIDES[::-1]
    if reverse_receiver:
        receiver, pusher = pusher, receiver
    flat_receiver = flat_receiver or outside_receiver
    pair_clearance = .006 if flat_receiver else .003
    # planning_edge reserves 15 mm on the robot side of the observed physical edge.
    edge = routes.edge + .015
    forward = np.array([1., 0., 0.]) - up[0] * up
    forward /= np.linalg.norm(forward)
    lateral = np.cross(up, forward)
    hulls = []
    trays = [np.asarray(obj['bounds']) for obj in routes.objects if obj['kind'] in ('tray_blue', 'tray_red')]
    for obstacle in routes.objects:
        if obstacle['kind'] not in ('drive', 'cable', 'tray_blue', 'tray_red'):
            continue
        if obstacle['kind'] == 'drive' and box_removed:
            box_xy = np.asarray(obstacle['xyz'])[:2]
            if not any(np.all((box_xy >= tray[0, :2]) & (box_xy <= tray[1, :2])) for tray in trays):
                continue
        lo, hi = np.asarray(obstacle['bounds'])
        corners = obstacle.get('geometry', {}).get('corners_xy_m')
        if corners is None:
            corners = [[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]
        hulls.append((lo, hi, ConvexHull(corners).equations))

    def orientation(flat, pitch):
        fingers = np.cos(pitch) * flat - np.sin(pitch) * up
        palm = np.sin(pitch) * flat + np.cos(pitch) * up
        return np.column_stack((palm, np.cross(fingers, palm), fingers))

    def in_obstacle(vertices, padding):
        for lo, hi, faces in hulls:
            near = vertices[np.all((vertices >= lo - padding) & (vertices <= hi + padding), axis=1)]
            if len(near) and np.any((near[:, :2] @ faces[:, :2].T + faces[:, 2]).max(1) < padding):
                return True
        return False

    def top_slide(direction, across, angle, finger_side):
        axis = np.cos(angle) * direction + finger_side * np.sin(angle) * across
        rotation = orientation(axis, 0.) @ np.diag([-1., -1., 1.])
        rotation = rotation @ Rotation.from_rotvec([0., 0., -.12 * SIGNS[pusher]]).as_matrix()
        base = open_hands[pusher]['base_link']
        pad = base[(abs(base[:, 1]) < .026) & (base[:, 2] > .025) & (base[:, 2] < .115)]
        if len(pad) < 10:
            return None
        pad_heights = pad @ rotation.T @ up
        pad_center = np.median(pad[pad_heights <= np.quantile(pad_heights, .10)], axis=0)
        offsets = points - center
        above = offsets[np.linalg.norm(offsets - (offsets @ up)[:, None] * up, axis=1) < .030]
        if len(above) < 10:
            return None
        top = np.quantile(above @ up, .92)
        origin = center + (top - .001) * up - rotation @ pad_center
        # An extended thumb may be lower than the palm. Keep it clear of the shell.
        hand = np.concatenate([vertices for name, vertices in open_hands[pusher].items()
                               if name != 'base_link']) @ rotation.T + origin
        plane = np.column_stack((forward, lateral))
        near = hand[(hand - center) @ up < top + .004]
        neighbors = cKDTree(points @ plane).query_ball_point(near @ plane, .006)
        if any(len(indices) and point @ up < (points[indices] @ up).max() + .001
               for point, indices in zip(near, neighbors)):
            return None
        return origin, rotation

    def geometry(receiver_angle, finger_side, setback, fall, lateral_shift, receive_inset=None,
                 receive_pitch=.10):
        sign = SIGNS[receiver]
        receive_axis = np.cos(receiver_angle) * forward - sign * np.sin(receiver_angle) * lateral
        receive_rotation = orientation(receive_axis, receive_pitch)
        fingers = np.concatenate([open_hands[receiver][name] for name in DISTAL]) @ receive_rotation.T
        receive_point = routes.arms[receiver].pocket
        pocket_offset = receive_rotation @ receive_point
        receiver_origin = center + (lateral_shift - pocket_offset @ lateral) * lateral
        if outside_receiver:
            surface = open_mesh[receiver] @ receive_rotation.T
            origin_height = (receiver_origin @ routes.normal - routes.floor) / np.linalg.norm(routes.normal)
            receiver_origin += (-.001 - origin_height - (fingers @ up).max()) * up
            receive_x = edge - .004 - surface[:, 0].max()
            receiver_origin += (receive_x - receiver_origin[0]) / forward[0] * forward
        else:
            surface = open_mesh[receiver] @ receive_rotation.T if flat_receiver else fingers
            receive_x = (edge + .001 - fingers[:, 0].max() if receive_inset is None
                         else edge - receive_inset - pocket_offset[0])
            receiver_origin += (receive_x - receiver_origin[0]) / forward[0] * forward
            over = surface[:, 0] + receiver_origin[0] >= edge - .001
            if not np.any(over):
                return None
            receiver_origin += (-.001 - (surface[over] @ up).min()) * up
        shift = receiver_origin + pocket_offset - center
        shift -= (shift @ up) * up
        distance = np.linalg.norm(shift)
        if not (.008 if flat_receiver else .015) < distance < (.42 if outside_receiver else .34):
            return None
        low_cloud = points[((points - center) @ up) < .030]
        if len(low_cloud) < 15:
            return None
        for fraction in np.linspace(0., 1., 6):
            if in_obstacle(low_cloud + fraction * shift, .005):
                return None
        direction = shift / distance
        across = np.cross(up, direction)
        if outside_receiver:
            contact = top_slide(direction, across, setback, finger_side)
            if contact is None:
                return None
            push_origin, push_rotation = contact
            count = max(1, int(np.ceil(distance / .050)))
            pushes = [(push_origin + fraction * shift, push_rotation)
                      for fraction in np.linspace(0., 1., count + 1)]
            return receiver_origin, receive_rotation, pushes, shift
        fingers = finger_side * sign * across - setback * direction - fall * up
        fingers /= np.linalg.norm(fingers)
        palm = direction - (direction @ fingers) * fingers
        palm /= np.linalg.norm(palm)
        push_rotation = np.column_stack((palm, np.cross(fingers, palm), fingers))
        push_mesh = open_mesh[pusher] @ push_rotation.T
        base = open_hands[pusher]['base_link']
        inner = base[base[:, 0] >= np.quantile(base[:, 0], .65)] @ push_rotation.T
        min_height = (push_mesh @ up).min()
        inner = inner[inner @ up <= min_height + .040]
        if not len(inner):
            return None
        edge_patch = inner[(inner @ direction) >= np.quantile(inner @ direction, .85)]
        palm_front = (edge_patch @ direction).max()
        push_origin = center + (.003 - min_height) * up - np.median(edge_patch @ across) * across
        rear = np.quantile((points - center) @ direction, .02)
        push_origin += (rear - .001 - palm_front) * direction
        nonpalm = np.concatenate([mesh for name, mesh in open_hands[pusher].items() if name != 'base_link'])
        nonpalm = nonpalm @ push_rotation.T + push_origin - center
        source_width = (points - center) @ across
        near = (nonpalm @ up < ((points - center) @ up).max() + .003)
        near &= (nonpalm @ across >= source_width.min()) & (nonpalm @ across <= source_width.max())
        if np.any(nonpalm[near] @ direction > rear - .001):
            return None
        count = max(1 if flat_receiver else 2, int(np.ceil(distance / .065)))
        pushes = [(push_origin + fraction * shift, push_rotation) for fraction in np.linspace(0., 1., count + 1)]
        return receiver_origin, receive_rotation, pushes, shift

    planes, body_low, body_high, arm_meshes = body_geometry()

    def try_route(frame, yaw):
        receiver_origin, receive_rotation, pushes, shift = frame
        origins = {receiver: receiver_origin, pusher: pushes[0][0]}
        rotations = {receiver: receive_rotation, pusher: pushes[0][1]}
        grip = {side: list(command) for side, command in preshape.items()}
        grip[receiver] = [.68, .68, .68, .68, .56, preshape[receiver][5] if outside_receiver else .45]
        hand = routes.hands[receiver]
        closing_meshes = [np.concatenate(list(hand.vertices(hand.nominal_angles(command)).values()))
                          for command in np.linspace(preshape[receiver], grip[receiver], 5)[1:]]
        held = (points + shift - receiver_origin) @ receive_rotation
        held_center = (center + shift - receiver_origin) @ receive_rotation

        def solve_arm(side, goal, rotation, seed, precise, angle=yaw):
            if deadline is not None and monotonic() >= deadline:
                return None
            arm = routes.arms[side]
            orient = routes.local(rotation, angle)
            goal = routes.local(goal, angle) + orient @ arm.pocket
            if precise:
                return fit_contact(arm, goal, orient, seed)
            return routes.fit(side, goal, orient, seed)

        def solve(goals, orients, seed, *, precise=False):
            result = {}
            for side in SIDES:
                q = solve_arm(side, goals[side], orients[side], seed[side], precise)
                if q is None:
                    return None
                result[side] = q
            return result

        def clear(pair, shape, angle=yaw, carrying=False, keep_outside=True):
            if deadline is not None and monotonic() >= deadline:
                return False
            bodies = {}
            for side in SIDES:
                arm = routes.arms[side]
                frames = arm.frames(pair[side])
                palm = frames[-1] @ arm.hand
                fingers = shape if side == receiver else open_mesh[side]
                fingers = fingers @ palm[:3, :3].T + palm[:3, 3]
                links = np.concatenate([mesh @ f[:3, :3].T + f[:3, 3]
                                        for mesh, f in zip(arm_meshes[side], frames[2:])])
                local = np.concatenate((links, fingers))
                if side == receiver and lower_body is not None and lower_body.collides(local, angle):
                    return False
                near = local[np.all((local >= body_low) & (local <= body_high), axis=1)]
                if len(near) and np.any((near @ planes[:, :3].T + planes[:, 3]).max(1) < .002):
                    return False
                world = routes.world(local.T, angle).T
                if outside_receiver and side == receiver and angle == yaw and not carrying and keep_outside:
                    hand_world = world[-len(fingers):]
                    wrist_world = routes.world(frames[-1][:3, 3], angle)
                    low_hand = min((hand_world @ routes.normal).min(), wrist_world @ routes.normal)
                    if low_hand - routes.floor <= .045 and max(wrist_world[0], hand_world[:, 0].max()) > edge - .003:
                        return False
                over = world[:, 0] >= edge - .001
                if np.any(world[over] @ routes.normal - routes.floor < -.002):
                    return False
                if in_obstacle(world, .003):
                    return False
                if carrying and side == receiver:
                    cloud = routes.world((held @ palm[:3, :3].T + palm[:3, 3]).T, angle).T
                    if (cloud @ routes.normal - routes.floor).min() < .010 or in_obstacle(cloud, .007):
                        return False
                bodies[side] = local
            return not np.any(cKDTree(bodies['left']).query(
                bodies['right'], distance_upper_bound=pair_clearance)[0] < pair_clearance)

        def link(a, b, shape=open_mesh[receiver], angle=yaw, carrying=False):
            if a is None or b is None:
                return False
            delta = {side: b[side] - a[side] for side in SIDES}
            steps = 2 + int(max(np.max(abs(q)) for q in delta.values()) / .08)
            return all(clear({side: a[side] + fraction * delta[side] for side in SIDES}, shape, angle, carrying)
                       for fraction in np.linspace(0., 1., steps))

        def delivery(probe):
            closed = closing_meshes[-1]
            lifts = [probe]
            for height in (.055, .085, .115, .145):
                q = solve_arm(receiver, receiver_origin + height * up, receive_rotation,
                              lifts[-1][receiver], True)
                if q is None:
                    return None
                goal = dict(probe, **{receiver: q})
                if not link(lifts[-1], goal, closed, carrying=True):
                    return None
                lifts.append(goal)
            tray = np.asarray(item['tray']['bounds'])
            vertical = np.array([0., 0., 1.])
            for site in (item['release'], *item['alternatives']):
                heading = np.arctan2(site[1], site[0]) - SIGNS[receiver] * .32
                tray_yaw = routes.yaw + heading
                if not all(clear(lifts[-1], closed, angle, carrying=True)
                           for angle in np.linspace(yaw, tray_yaw, 20)):
                    continue
                face = heading + SIGNS[receiver] * .30
                toward = np.array([np.cos(face), np.sin(face), 0.])
                fingers = np.cos(.60) * toward - np.sin(.60) * vertical
                palm = np.sin(.60) * toward + np.cos(.60) * vertical
                rotation = np.column_stack((palm, np.cross(fingers, palm), fingers))
                footprint = (held - held_center) @ rotation.T
                x, y = site[:2]
                if (np.any([x, y] + footprint[:, :2].min(0) < tray[0, :2] + .004)
                        or np.any([x, y] + footprint[:, :2].max(0) > tray[1, :2] - .004)):
                    continue
                release = np.array([x, y, tray[1, 2] - footprint[:, 2].min() + .015])
                stages = [lifts[-1]]
                for at in (release + .060 * vertical, release):
                    q = solve_arm(receiver, at - rotation @ held_center, rotation,
                                  stages[-1][receiver], True, tray_yaw)
                    if q is None:
                        break
                    goal = dict(probe, **{receiver: q})
                    if not link(stages[-1], goal, closed, tray_yaw, carrying=True):
                        break
                    stages.append(goal)
                if len(stages) == 3:
                    _, above, low = stages
                    if link(low, above, angle=tray_yaw) and link(above, positions, angle=tray_yaw):
                        return dict(tray_yaw=tray_yaw, lift=lifts[1:], transfer=[above], release=[low],
                                    retreat=[above, positions], item=dict(item, release=release.tolist()),
                                    edge_probe=[probe], probe_only=False)
            return None

        if start_yaw is not None:
            sweep_steps = 1 + int(np.ceil(abs(yaw - start_yaw) / .07))
            if any(not clear(positions, open_mesh[receiver], angle, keep_outside=False)
                   for angle in np.linspace(start_yaw, yaw, sweep_steps)):
                return None

        for bias in ((-.55, .82, .44, .25, .40, .38, 0.), (-.8, .55, -.35, .40, .45, .35, 0.),
                     (0.,) * 7):
            seeds = {side: positions[side] + np.asarray(bias) * [1., SIGNS[side], SIGNS[side],
                                                                1., SIGNS[side], 1., SIGNS[side]]
                     for side in SIDES}
            if flat_receiver:
                seeds[receiver] = np.deg2rad([-4., -4., -63., -19., 112., -6., -75.]) * [
                    1., SIGNS[receiver], SIGNS[receiver], 1., SIGNS[receiver], 1., SIGNS[receiver]]
            low = solve(origins, rotations, seeds, precise=True)
            if low is not None and clear(low, open_mesh[receiver]):
                break
        else:
            return None
        steps = [low]
        for push_at, push_rotation in pushes[1:]:
            goal = {receiver: receiver_origin, pusher: push_at}
            q = solve(goal, {receiver: receive_rotation, pusher: push_rotation}, steps[-1], precise=True)
            if not link(steps[-1], q):
                return None
            steps.append(q)
        pusher_clear = pushes[-1][0] + .055 * up
        if not outside_receiver:
            pusher_clear -= .025 * shift / np.linalg.norm(shift)
        final_rotations = {receiver: receive_rotation, pusher: pushes[-1][1]}
        final = solve({receiver: receiver_origin, pusher: pusher_clear}, final_rotations,
                      steps[-1], precise=True)
        if not link(steps[-1], final):
            return None
        if not all(clear(final, shape) for shape in closing_meshes):
            return None

        probe = None
        for height in ((.030, .012) if flat_receiver else (.030,)):
            raised = solve({receiver: receiver_origin + height * up, pusher: pusher_clear},
                           final_rotations, final, precise=True)
            if link(final, raised, closing_meshes[-1]):
                probe = raised
                break
        if probe is None and not flat_receiver:
            return None

        for raise_receiver, raise_pusher in ((.075, .045), (.050, .027), (.090, .058)):
            hover_at = {receiver: origins[receiver] + raise_receiver * up,
                        pusher: origins[pusher] + raise_pusher * up}
            hover = solve(hover_at, rotations, low, precise=True)
            if link(low, hover):
                break
        else:
            return None
        # Keep the pusher high while it crosses the front of the torso.
        ready_rotation = routes.world(routes.arms[pusher].tool(positions[pusher])[:3, :3], yaw)
        raised_rotation = Slerp([0., 1.], Rotation.from_matrix(
            np.stack((ready_rotation, rotations[pusher]))))(.64).as_matrix()
        raised_at = hover_at[pusher] - .120 * forward + .055 * SIGNS[pusher] * lateral + .110 * up
        raised_q = solve_arm(pusher, raised_at, raised_rotation,
                             .36 * positions[pusher] + .64 * hover[pusher], False)
        raised = dict(hover, **{pusher: raised_q}) if raised_q is not None else None
        if link(positions, raised) and link(raised, hover):
            path = [positions, raised, hover]
        else:
            receiver_out = SIGNS[receiver] * lateral
            offsets = {
                receiver: [spread * receiver_out + lift * up for spread in (.07, .12, .17)
                           for lift in (0., .07, -.045)],
                pusher: [-back * forward + lift * up + inset * receiver_out
                         for back in (.15, .23) for lift in (.07, .13) for inset in (.02, .06)],
            }

            def staged_arm(before, after, side):
                if link(before, after):
                    return [after]
                if side == receiver:
                    fraction, bias = .40, np.array([0., 0., 0., -.24, 0., 0., 0.])
                else:
                    fraction, bias = .55, np.array([-.24, 0., SIGNS[side] * .24, -.48, 0., 0., 0.])
                lower, upper = routes.arms[side].limits + [[.020], [-.020]]
                middle = np.clip(before[side] + fraction * (after[side] - before[side]) + bias, lower, upper)
                waypoint = dict(before, **{side: middle})
                if link(before, waypoint) and link(waypoint, after):
                    return [waypoint, after]
                for offset in offsets[side]:
                    middle = solve_arm(side, hover_at[side] + offset, rotations[side], after[side], False)
                    if middle is None:
                        continue
                    waypoint = dict(before, **{side: middle})
                    if link(before, waypoint) and link(waypoint, after):
                        return [waypoint, after]
                return None

            for first, second in ((receiver, pusher), (pusher, receiver)):
                apart = dict(positions, **{first: hover[first]})
                first_path = staged_arm(positions, apart, first)
                if first_path is None:
                    continue
                second_path = staged_arm(apart, hover, second)
                if second_path is not None:
                    path = [positions, *first_path, *second_path]
                    break
            else:
                return None
        close = [dict(phase='scoop_open', q=q, command=preshape) for q in steps[1:]]
        close.append(dict(phase='pusher_clear', q=final, command=preshape))
        close.append(dict(phase='close', q=final, command=grip))
        abort = [dict(phase='open', q=final, command=preshape)]
        abort.extend(dict(phase='scoop_open', q=q, command=preshape) for q in reversed(steps))
        raised_receiver = dict(final, **{receiver: hover[receiver]})
        high_exit = None
        if link(final, raised_receiver):
            if link(raised_receiver, hover):
                high_exit = [final, raised_receiver, *reversed(path)]
            else:
                for height in (.070, .105):
                    q = solve_arm(pusher, pusher_clear + height * up, pushes[-1][1], final[pusher], True)
                    if q is None:
                        continue
                    raised_pusher = dict(raised_receiver, **{pusher: q})
                    if link(raised_receiver, raised_pusher) and link(raised_pusher, hover):
                        high_exit = [final, raised_receiver, raised_pusher, *reversed(path)]
                        break
        if flat_receiver and high_exit is None:
            for pusher_goal in (hover[pusher], path[1][pusher], positions[pusher]):
                left_clear = dict(final, **{pusher: pusher_goal})
                raised_receiver = dict(left_clear, **{receiver: hover[receiver]})
                if link(final, left_clear) and link(left_clear, raised_receiver) and link(raised_receiver, hover):
                    high_exit = [final, left_clear, raised_receiver, *reversed(path)]
                    break
        plan = dict(pick_yaw=yaw, tray_yaw=yaw, approach=[*path[1:], low], grasp=[final],
                    lift=[probe] if probe is not None else [],
                    transfer=[], release=[], retreat=[*reversed(path)], close=close, abort_close=abort,
                    probe_return=[final], pickup_retreat=[*reversed(path)], grip=grip, preshape=preshape,
                    item=item, probe_only=True, edge_receive=True, grip_side=receiver, pusher_side=pusher,
                    edge_slide_m=float(np.linalg.norm(shift)))
        if high_exit is None and not flat_receiver:
            return None
        if high_exit is not None:
            plan['abort_after_push'] = high_exit
        else:
            plan['hold_on_miss'] = True
        plan['receive_point'] = routes.arms[receiver].pocket.copy()
        if outside_receiver:
            plan['receiver_edge_x_m'] = edge - .003
            plan['top_slide'] = True
        if not probe_only and probe is not None:
            route = delivery(probe)
            if route is not None:
                plan.update(route)
        if outside_receiver and not probe_only and plan['probe_only']:
            return None
        return plan

    heading = routes.yaw + np.arctan2(center[1], center[0])
    shift_sign = -SIGNS[receiver]
    # Fold the pusher's fingers behind the contact face to clear the neighboring box.
    profiles = (([.28] * 4 + [.30, 1.], (.42, -1., .82, .32, .070 * shift_sign, .035)),
                ([.38] * 4 + [.60, .75], (.42, -1., .90, .32, .085 * shift_sign, .026)))
    if box_removed:
        profiles = (([.46, .57, .74, 1., 1., 0.], (.42, -1., .95, .32, .040 * shift_sign)), *profiles)
    yaw_offsets = (.48, .57, .62)
    if flat_receiver:
        sideways = min(.041, max(0., center[0] - edge + .017) * .175) * shift_sign
        pusher_shape = [.28] * 4 + [.30, 1.]
        profiles = ((pusher_shape, (.22, -1., .98, .56, sideways, .017, 0.)),
                    (pusher_shape, (.22, -1., .98, .56, sideways, .017, np.deg2rad(2.))),
                    (pusher_shape, (.22, -1., .98, .56, sideways, .028, np.deg2rad(2.))))
        yaw_offsets = (.31, .25, .38)
    if outside_receiver:
        sideways = .012 * shift_sign
        pusher_shape = [1.] * 4 + [0., 1.]
        profiles = tuple((pusher_shape, (angle, -1., .32, 0., sideways, None, 0.))
                         for angle in (-1.58, 1.48, -1.28, 1.02, -.84, .62))
        yaw_offsets = (-.50, -.34, -.18, 0., .24, .42)
    for push_command, profile in profiles:
        preshape = {receiver: [1.] * 6 if flat_receiver or outside_receiver else list(OPEN), pusher: push_command}
        open_hands = {side: routes.hands[side].vertices(routes.hands[side].nominal_angles(preshape[side]))
                      for side in SIDES}
        open_mesh = {side: np.concatenate(list(parts.values())) for side, parts in open_hands.items()}
        frame = geometry(*profile)
        if frame is None:
            continue
        for offset in yaw_offsets:
            offset *= SIGNS[receiver]
            plan = try_route(frame, heading + offset)
            if plan is not None:
                return plan
    return None

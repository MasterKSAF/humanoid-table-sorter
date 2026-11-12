"""Paired scoop of a cable coil from one frozen RGB-D survey."""

from time import monotonic

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation, Slerp

from .edge_pair_delivery import edge_pair_delivery
from .mouse_scoop import DISTAL, SIDES, SIGNS, body_geometry
from .table_contact import finger_distance, fit_contact, object_frame, split, surface_points


OPEN = np.array([1., 1., 1., 1., .86, 0.])
CLOSED = np.array([.73, .73, .73, .73, .86, 0.])
EDGE_CLOSED = np.array([.47, .47, .47, .44, .70, .24])
EDGE_RETRY_PUSH = np.array([.57, .57, .57, .57, .80, .20])
CABLE_PINCH_OPEN = np.array([1., 1., 1., 1., 1., 0.])
CABLE_PINCH_CLOSED = np.array([1., 1., 1., 0., 0., 0.])
SCOOP_IN = .025
STAGES = (0., .25, .50, .75, 1.)
Z = np.array([0., 0., 1.])


def _hand_profile(hand, command):
    solid, tips = split(hand.vertices(hand.nominal_angles(command)), cable=True)
    return solid, np.concatenate(list(tips.values()))


def _section_hits_table(section, fore, normal, floor, edge, underneath):
    x = section @ fore
    if x.max() < edge:
        return False
    height = (section @ normal - floor) / np.linalg.norm(normal)
    bottom = -.048 if underneath else height.min() - 1.
    top = .005
    if height.min() >= top or height.max() <= bottom:
        return False
    if np.any((x >= edge) & (height > bottom) & (height < top)):
        return True
    outline = ConvexHull(np.column_stack((x, height))).vertices
    a = outline
    b = np.roll(outline, -1)
    for limit in (bottom, top):
        across = (height[a] < limit) != (height[b] < limit)
        first, second = a[across], b[across]
        fraction = (limit - height[first]) / (height[second] - height[first])
        if np.any(x[first] + fraction * (x[second] - x[first]) >= edge):
            return True
    return False


def _paired_geometry(routes, box_removed, profiles, *, edge=None, lower_body=None, occupied_boxes=()):
    arms = routes.arms
    body_planes, body_low, body_high, body_links = body_geometry()
    solid_objects = [np.asarray(obj['bounds'], dtype=float) for obj in routes.objects
                     if obj['kind'] not in ('cable', 'tray_red', 'tray_blue', 'tray_pink')
                     and not (box_removed and obj['kind'] == 'drive')]
    solid_objects.extend(occupied_boxes)
    if edge is not None:
        for obj in routes.objects:
            if not obj['kind'].startswith('tray_'):
                continue
            low, high = np.asarray(obj['bounds'], dtype=float)
            for axis in (0, 1):
                near, far = low.copy(), high.copy()
                near[axis], far[axis] = high[axis] - .012, low[axis] + .012
                solid_objects.extend(((low, far), (near, high)))

    def solve(origins, rotations, yaw, seed, *, precise=False):
        pair = {}
        for side in SIDES:
            arm = arms[side]
            orient = rotations[side]
            goal = routes.local(origins[side] + orient @ arm.pocket, yaw)
            direction = routes.local(orient, yaw)
            q = (fit_contact(arm, goal, direction, seed[side]) if precise else
                 routes.fit(side, goal, direction, seed[side]))
            if q is None:
                return None
            pair[side] = q
        return pair

    def in_body(points):
        near = points[np.all((points >= body_low) & (points <= body_high), axis=1)]
        return bool(len(near) and np.any((near @ body_planes[:, :3].T + body_planes[:, 3]).max(1) < .002))

    def clear_edge(pair, yaw, stage, underneath):
        limbs = {}
        for side in SIDES:
            arm = arms[side]
            frames = arm.frames(pair[side])
            hand = frames[-1] @ arm.hand
            sections = [mesh @ frame[:3, :3].T + frame[:3, 3]
                        for mesh, frame in zip(body_links[side], frames[2:])]
            sections.extend(part @ hand[:3, :3].T + hand[:3, 3] for part in profiles[side][stage].values())
            local = np.concatenate(sections)
            if in_body(local) or lower_body.collides(local, yaw, clearance=.006):
                return False
            normal = routes.local(routes.normal, yaw)
            fore = routes.local(np.array([1., 0., 0.]), yaw)
            if any(_section_hits_table(section, fore, normal, routes.floor, edge - .003, underneath)
                   for section in sections):
                return False
            world = routes.world(local.T, yaw).T
            for low, high in solid_objects:
                if np.any(np.all((world >= low - .003) & (world <= high + .003), axis=1)):
                    return False
            limbs[side] = local
        return not np.any(cKDTree(limbs['left']).query(limbs['right'], distance_upper_bound=.003)[0] < .003)

    def clear(pair, yaw, shapes, *, underneath=False):
        if edge is not None:
            return clear_edge(pair, yaw, shapes, underneath)
        samples = {}
        for side in SIDES:
            arm = arms[side]
            solid, tips = profiles[side][shapes]
            frames = arm.frames(pair[side])
            frame = frames[-1] @ arm.hand
            fingers = np.concatenate((solid, tips)) @ frame[:3, :3].T + frame[:3, 3]
            if in_body(fingers) or any(in_body(vertices @ frame[:3, :3].T + frame[:3, 3])
                                       for vertices, frame in zip(body_links[side], frames[2:])):
                return False
            forearm, radii = arm.samples(pair[side])
            world = routes.world(np.concatenate((forearm[:112], fingers)).T, yaw).T
            radius = np.r_[radii[:112], np.zeros(len(fingers))]
            over = world[:, 0] >= routes.edge - radius
            table_gaps = np.r_[np.full(len(world) - len(tips), .002), np.full(len(tips), -.002)]
            if np.any(world[over] @ routes.normal - routes.floor - radius[over] < table_gaps[over]):
                return False
            for low, high in solid_objects:
                occupied = world[32:]
                gap = np.maximum(0., np.maximum(low - occupied, occupied - high))
                if np.any(np.einsum('ij,ij->i', gap, gap) < (radius[32:] + .002) ** 2):
                    return False
            samples[side] = ((forearm[32:112], .038), (fingers, 0.))
        for first, ra in samples['left']:
            for second, rb in samples['right']:
                margin = ra + rb + .002
                if np.any(np.maximum(first.min(0) - second.max(0), second.min(0) - first.max(0)) > margin):
                    continue
                if np.any(cKDTree(second).query(first, distance_upper_bound=margin)[0] < margin):
                    return False
        return True

    def link(start, goal, yaw, shapes=0, *, carried=None, underneath=False):
        if goal is None:
            return False
        delta = {side: goal[side] - start[side] for side in SIDES}
        count = 2 + int(max(np.max(abs(q)) for q in delta.values()) / .065)
        for f in np.linspace(0., 1., count):
            pair = {side: start[side] + f * delta[side] for side in SIDES}
            if not clear(pair, yaw, shapes, underneath=underneath) or (carried is not None and not carried(pair, yaw)):
                return False
        return True

    return solve, clear, link


def check_cable_hook_transport(routes, plan, lower_body):
    """Validate a right-hand tray-yaw fallback with the observed tray contents."""
    item = plan['item']
    side = plan['grip_side']
    tray_yaw = plan['tray_yaw']
    if lower_body is None or not -2.967 <= tray_yaw <= np.pi:
        return False
    placement = np.asarray(plan['release_footprint'])
    floor = item['tray']['bounds'][0][2]
    heights = item.get('occupied_heights', {})
    boxes = []
    for kind, footprint in item['occupied_footprints'].items():
        low, high = np.asarray(footprint)
        if kind not in heights or not np.any(np.maximum(low - placement[1], placement[0] - high) >= .005):
            return False
        boxes.append(np.array([np.r_[low, floor], np.r_[high, heights[kind]]]))

    profiles = {}
    for hand_side, command in plan['grip'].items():
        hand = routes.hands[hand_side]
        opened = plan['release_hand'].get(hand_side, command)
        profiles[hand_side] = [hand.vertices(hand.nominal_angles(state)) for state in (command, opened)]
    _, clear, link = _paired_geometry(routes, False, profiles, edge=routes.edge + .015,
                                      lower_body=lower_body, occupied_boxes=boxes)
    arm = routes.arms[side]
    pick_yaw = plan['pick_yaw']

    def frame(q, yaw):
        return routes.world((arm.wrist(q) @ arm.hand)[:3], yaw)

    original = plan['grasp'][-1][side].copy()
    # The frozen cloud touched the fingers before the halfway curl to the wrist limit.
    original[-1] = 2 * plan['hook_half_curl_rad'] - arm.limits[1, -1]
    contact = frame(original, pick_yaw)
    target = item['target']
    center, axis, cross, up = object_frame(target, routes.normal, routes.floor)
    points = surface_points(target, True, center, axis, cross, up)
    held = (points + (plan['edge_object_center'] - center) - contact[:, 3]) @ contact[:, :3]

    def carried(pair, yaw):
        pose = frame(pair[side], yaw)
        moved = held @ pose[:, :3].T + pose[:, 3]
        over = moved[:, 0] >= routes.edge
        if np.any(moved[over] @ routes.normal - routes.floor < .005):
            return False
        for low, high in boxes:
            gap = np.maximum(0., np.maximum(low - moved, moved - high))
            if np.any(np.einsum('ij,ij->i', gap, gap) < .005 ** 2):
                return False
        return True

    lift = plan['lift'][-1]
    steps = 2 + int(abs(tray_yaw - pick_yaw) / .085)
    if not all(clear(lift, yaw, 0) and carried(lift, yaw) for yaw in np.linspace(pick_yaw, tray_yaw, steps)):
        return False
    transport = [lift, *plan['transfer'], *plan['release']]
    if not all(link(a, b, tray_yaw, 0, carried=carried) for a, b in zip(transport, transport[1:])):
        return False
    exit_path = [transport[-1], *plan['retreat'], *plan['cable_withdraw_high']]
    return all(link(a, b, tray_yaw, 1) for a, b in zip(exit_path, exit_path[1:]))


def cable_bimanual_plan(routes, item, positions, *, box_removed=False, probe_only=False):
    """Plan an open scoop, closure, and a two-hand lift to the pink tray."""
    target = item['target']
    center, axis, cross, up = object_frame(target, routes.normal, routes.floor)
    points = surface_points(target, True, center, axis, cross, up)
    if len(points) < 20:
        return None
    commands = [OPEN + f * (CLOSED - OPEN) for f in STAGES]
    profiles = {side: [_hand_profile(routes.hands[side], command) for command in commands]
                for side in SIDES}
    closed_frames = {side: hand.frames(hand.nominal_angles(CLOSED)) for side, hand in routes.hands.items()}
    solve, clear, link = _paired_geometry(routes, box_removed, profiles)
    arms = routes.arms
    heading = routes.yaw + np.arctan2(center[1], center[0])
    lowest = target['bounds'][0][2]
    highest = max(obj['bounds'][1][2] for obj in routes.objects)

    def commands_at(index):
        return {side: commands[index].tolist() for side in SIDES}

    def geometry(yaw, sweep, lateral, pitches, rolls, setback, gap, inside=.010):
        forward = routes.world(np.array([1., 0., 0.]), yaw)
        forward -= (forward @ up) * up
        forward /= np.linalg.norm(forward)
        normal = Rotation.from_rotvec(sweep * up).apply(np.cross(up, forward))
        tangent = np.cross(normal, up)
        origins, orientations, directions, heights_at = {}, {}, {}, {}
        heights = (points - center) @ up
        along = (points - center) @ tangent
        for side, angle, pitch, roll, shift in zip(SIDES, lateral, pitches, rolls, setback):
            out = SIGNS[side] * normal
            flat = -np.cos(angle) * out + np.sin(angle) * tangent
            finger_axis = np.cos(pitch) * flat - np.sin(pitch) * up
            palm = np.sin(pitch) * flat + np.cos(pitch) * up
            palm = Rotation.from_rotvec(roll * finger_axis).apply(palm)
            orient = np.column_stack((palm, np.cross(finger_axis, palm), finger_axis))
            hand = routes.hands[side]
            frames = closed_frames[side]
            distal = np.concatenate([hand.shapes[link]['vertices'] @ frames[link][:3, :3].T
                                     + frames[link][:3, 3] for link in DISTAL])
            world = distal @ orient.T
            near = (abs(along - shift) < .032) & (heights < .029)
            flank = (points - center) @ out
            edge = np.quantile(flank[near], .89) if np.count_nonzero(near) > 8 else np.quantile(flank, .85)
            vertical = orient.T @ up
            clearance = np.array([max(.005 - (solid @ vertical).min(), .001 - (tips @ vertical).min())
                                  for solid, tips in profiles[side]])
            height = max(gap - (world @ up).min(), clearance[-1])
            origin = (center + (edge - inside - np.quantile(world @ out, .10)) * out
                      + (shift - world.mean(0) @ tangent) * tangent + height * up)
            local = (points - origin) @ orient
            near_pads = sum(finger_distance(hand, frames[link], link, local) <= .004 for link in DISTAL)
            if near_pads < 2:
                return None
            origins[side] = origin
            orientations[side] = orient
            directions[side] = out * [1., 1., 0.]
            # Keep the tips near the table as curling retracts them into the palm.
            heights_at[side] = np.maximum.accumulate(np.maximum(clearance - height, 0.)[::-1])[::-1]
        return origins, orientations, directions, heights_at

    def pickup(yaw, origins, orientations, directions, heights_at):
        closed = len(STAGES) - 1
        for bias in ((-.65, .85, .45, .25, .45, .45, 0.),
                     (-.95, .52, -.35, .40, .45, .35, 0.),
                     (-.65, .50, .70, .10, .25, .25, 0.), (0.,) * 7):
            seeds = {side: positions[side] + np.asarray(bias) * [1, SIGNS[side], SIGNS[side],
                                                               1, SIGNS[side], 1, SIGNS[side]]
                     for side in SIDES}
            end = solve(origins, orientations, yaw, seeds, precise=True)
            if end is not None and clear(end, yaw, closed):
                break
        else:
            return None
        closing = [end]
        for i in reversed(range(closed)):
            at = {side: origins[side] + heights_at[side][i] * up for side in SIDES}
            last = closing[-1]
            if all(heights_at[side][i] == heights_at[side][i + 1] for side in SIDES):
                q = last
            else:
                q = solve(at, orientations, yaw, last, precise=True)
            if q is None or not clear(q, yaw, i) or not link(q, last, yaw, i + 1):
                return None
            closing.append(q)
        closing.reverse()
        open_at = {side: origins[side] + heights_at[side][0] * up for side in SIDES}
        reverse = [closing[0]]
        for i in reversed(range(len(STAGES) - 1)):
            fraction = STAGES[i]
            at = {side: open_at[side] + (1. - fraction) * SCOOP_IN * directions[side]
                  for side in SIDES}
            q = solve(at, orientations, yaw, reverse[-1], precise=True)
            if q is None or not link(q, reverse[-1], yaw):
                return None
            reverse.append(q)
        steps = list(reversed(reverse))
        hover_at = {side: open_at[side] + .100 * Z + .035 * directions[side] for side in SIDES}
        hover = solve(hover_at, orientations, yaw, steps[0], precise=True)
        if hover is None or not link(hover, steps[0], yaw):
            return None
        if link(positions, hover, yaw):
            return [hover, steps[0]], steps, closing
        initial = {side: routes.world((arms[side].wrist(positions[side]) @ arms[side].hand)[:3], yaw)
                   for side in SIDES}
        rotations = {side: Slerp([0., 1.], Rotation.from_matrix([initial[side][:, :3], orientations[side]]))
                     for side in SIDES}
        approach = [positions]
        for f in (.25, .50, .75):
            goals = {side: ((1. - f) * initial[side][:, 3] + f * hover_at[side]
                            + np.sin(np.pi * f) * (.12 * Z + .065 * directions[side]))
                     for side in SIDES}
            poses = {side: rotations[side](f).as_matrix() for side in SIDES}
            q = solve(goals, poses, yaw, approach[-1])
            if not link(approach[-1], q, yaw):
                return None
            approach.append(q)
        if not link(approach[-1], hover, yaw):
            return None
        return [*approach[1:], hover, steps[0]], steps, closing

    def delivery(yaw, origins, orientations, end):
        closed = len(STAGES) - 1
        height = max(.110, highest + .026 - lowest)
        lift_at = {side: origins[side] + height * Z for side in SIDES}
        lift = solve(lift_at, orientations, yaw, end, precise=True)
        if not link(end, lift, yaw, closed):
            return None
        bounds = np.asarray(item['tray']['bounds'])
        for site in [item['release'], *item['alternatives']]:
            site = np.array(site)
            aim = routes.yaw + np.arctan2(site[1], site[0])
            for tray_yaw in (aim, aim - .12, aim + .12, yaw):
                turn = Rotation.from_rotvec((tray_yaw - yaw) * Z).as_matrix()
                footprint = (points - center) @ turn.T
                lo = bounds[0, :2] - footprint[:, :2].min(0) + .006
                hi = bounds[1, :2] - footprint[:, :2].max(0) - .006
                if np.any(hi < lo):
                    continue
                selected = site.copy()
                selected[:2] = np.clip(site[:2], lo, hi)
                if np.linalg.norm(selected[:2] - site[:2]) > .025:
                    continue
                final = selected.copy()
                final[2] = center[2] + bounds[1, 2] + .025 - lowest
                high = final.copy()
                high[2] = max(final[2] + .070, center[2] + height)
                orientations_tray = {side: turn @ orient for side, orient in orientations.items()}
                offsets = {side: turn @ (origins[side] - center) for side in SIDES}
                above = solve({side: high + offsets[side] for side in SIDES}, orientations_tray, tray_yaw,
                              lift, precise=True)
                if not link(lift, above, tray_yaw, closed):
                    continue
                low = solve({side: final + offsets[side] for side in SIDES}, orientations_tray, tray_yaw,
                            above, precise=True)
                if not link(above, low, tray_yaw, closed) or not link(low, above, tray_yaw):
                    continue
                if not link(above, positions, tray_yaw):
                    continue
                if not all(clear(lift, turn_yaw, closed) for turn_yaw in np.linspace(yaw, tray_yaw, 7)):
                    continue
                return dict(lift=[lift], transfer=[above], release=[low], retreat=[above, positions],
                            tray_yaw=tray_yaw, item=dict(item, release=selected.tolist()))
        return None

    settings = ((.28, (.40, .72), (.64, .64), (0., 0.), (.011, -.035), .003, .024),
                (.28, (.95, .72), (.64, .64), (0., 0.), (.011, -.035), .003, .024),
                (.28, (.95, .72), (.64, .64), (0., -.55), (.011, -.035), .003, .014),
                (0., (1.35, 1.55), (.52, .22), (.25, -.78), (.035, .035), .003, .014),
                (0., (1.35, 1.55), (.52, .22), (.25, 0.), (.035, -.012), .003, .014),
                (.30, (1.35, 1.42), (.38, .57), (.45, -.55), (.025, .030), .003, .012),
                (.55, (1.12, 1.23), (.38, .62), (.40, -.65), (.015, .035), .003, .014),
                (.42, (1.05, .80), (.42, .62), (.40, -.68), (.010, .018), .003, .014),
                (0., (1.00, 1.00), (.45, .45), (0., 0.), (0., 0.), .003),
                (0., (.55, 1.15), (.40, .40), (0., -.75), (-.010, -.010), .002),
                (.38, (.75, 1.15), (.40, .52), (0., -.55), (-.010, .025), .002))
    probe_plan = None
    for offset in (0., -.18, .18):
        yaw = heading + offset
        for settings_at_yaw in settings:
            frame = geometry(yaw, *settings_at_yaw)
            if frame is None:
                continue
            approach = pickup(yaw, *frame)
            if approach is None:
                continue
            path, scoop, closing = approach
            end = closing[-1]
            opening = commands_at(0)
            states = [(scoop[0], opening)]
            forward_phases, reverse_phases = [], []
            for q in scoop[1:]:
                states.append((q, opening))
                forward_phases.append('scoop_open')
                reverse_phases.append('withdraw_open')
            for i, q in enumerate(closing[1:], 1):
                command = commands_at(i)
                previous = closing[i - 1]
                states.append((previous, command))
                forward_phases.append('close')
                reverse_phases.append('open')
                if q is not previous:
                    states.append((q, command))
                    forward_phases.append('settle_close')
                    reverse_phases.append('raise_closed')
            base = dict(pick_yaw=yaw, approach=path, grasp=[end],
                        close=[dict(q=q, command=command, phase=phase)
                               for (q, command), phase in zip(states[1:], forward_phases)],
                        abort_close=[dict(q=q, command=command, phase=phase)
                                     for (q, command), phase in zip(reversed(states[:-1]), reversed(reverse_phases))],
                        pickup_retreat=[*reversed(path[:-1]), positions],
                        preshape=opening, grip=commands_at(-1), scoop_drop_m=0.)
            origins, orientations, _, _ = frame
            if probe_plan is None:
                probe = solve({side: origin + .030 * Z for side, origin in origins.items()},
                              orientations, yaw, end, precise=True)
                if link(end, probe, yaw, len(STAGES) - 1):
                    probe_plan = dict(base, item=item, tray_yaw=yaw, lift=[probe],
                                      probe_only=True, probe_return=[end],
                                      transfer=[], release=[], retreat=[])
            if probe_only:
                if probe_plan is not None:
                    return probe_plan
                continue
            transfer = delivery(yaw, origins, orientations, end)
            if transfer is not None:
                return dict(base, **transfer)
    return probe_plan


def _left_cable_delivery(routes, item, held, yaw, points, center, solve, clear, *,
                         start_ready, edge, withdraw, obstacles):
    """Seed a tilted lift and a release over the near half of the pink tray."""
    profile = []
    for height in (.01, .02, .03, .04, .05, .06, .07, .08, .09, .10, .11, .12, .13, .14):
        fraction = min(1., height / .065)
        shift = np.array([.018 * fraction, 0., height])
        tilt = Rotation.from_euler('zyx', [0., .20 * fraction, .18 * fraction]).as_matrix()
        profile.append((shift, tilt))

    tray_low, tray_high = np.asarray(item['tray']['bounds'])
    release_turn = Rotation.from_euler('zyx', [-.1264377198, .0972963101, .7142625060]).as_matrix()
    footprint = (points - center) @ release_turn.T
    low = tray_low[:2] + .0073 - footprint[:, :2].min(0)
    high = tray_high[:2] - .0073 - footprint[:, :2].max(0)
    if np.any(low > high):
        return None
    xy = np.clip(tray_low[:2] + [.0913807465, .0667364339], low, high)
    release = np.r_[xy, tray_high[2] + .1174979948 - footprint[:, 2].min()]
    release_seed = {
        'left': np.array([-.9583389492, -.0810409698, -.2439694039, -.4644546396,
                          .9662784337, .3480761846, -.0715312991]),
        'right': np.array([-1.0791091134, .2437990725, .4357521899, -.2362038954,
                           -.9965106167, -.3822634687, -.3568011479]),
    }
    # The open fingertips need to separate before the final return to ready.
    separated = {
        'left': np.array([.793741, -.001978, -.176328, -1.970123, .456218, -.459342, .328413]),
        'right': np.array([.586803, -.110969, .293151, -1.745327, -.035633, -.483704, -.574860]),
    }
    return edge_pair_delivery(
        routes, item, held, yaw, points, center, solve, clear,
        start_ready=[separated, start_ready], edge=edge, withdraw=withdraw, obstacles=obstacles,
        lift_profile=profile, placements=[(routes.yaw + 1.0930541947, release, release_turn, .022, release_seed)])


def _cable_under_edge_plan(routes, item, positions, lower_body, start_yaw, box_removed, *, hook=False,
                           leave_hand_after_release=False, receiver_side=None, deadline=None):
    if lower_body is None or routes.edge < 0.:
        return None
    target = item['target']
    center, axis, cross, up = object_frame(target, routes.normal, routes.floor)
    cloud = surface_points(target, True, center, axis, cross, up)
    if len(cloud) < 20:
        return None
    edge = routes.edge + .015
    floor = routes.floor / np.linalg.norm(routes.normal)
    inward = np.array([1., 0., 0.]) - up[0] * up
    inward /= np.linalg.norm(inward)
    along = np.cross(up, inward)
    receiver = receiver_side or ('left' if center[1] >= 0. else 'right')
    pusher = 'right' if receiver == 'left' else 'left'
    mirror = SIGNS[receiver]
    receiving_arm = routes.arms[receiver]
    # An explicit hook receiver opts into the mirrored shallow grasp.
    shallow = receiver == 'left' or (hook and receiver_side is not None)
    if hook and not shallow:
        return None
    if shallow:
        # Present curled fingers to the cable rim while keeping the thumb out.
        open_receive = np.array([.66, .66, .66, .66, 0., 1.])
        close_receive = (np.array([.52, .52, .54, .52, 0., 1.]) if hook else
                         np.array([.56, .56, .56, .54, .78, .12]))
        receive_point = np.array([.0543, mirror * .0286, .1694])
        if hook:
            open_receive[4:] = [1., 0.]
            # Support the coil on the same bent fingers, with no opposing thumb
            # and no squeezing step after it reaches the receiving hand.
            close_receive = open_receive.copy()
    else:
        open_receive = OPEN
        close_receive = (OPEN + EDGE_CLOSED) * .5
        receive_point = receiving_arm.pocket.copy()
    obstacles = [obj for obj in routes.objects if not (box_removed and obj['kind'] == 'drive')]
    occupied = [np.asarray(obj['bounds']) for obj in obstacles if obj['kind'] != item['kind']]
    start_yaw = routes.yaw if start_yaw is None else start_yaw
    receiver_seed = np.array([-.276, -.075, -.522, -.440, 1.637, .168, -.383])
    pusher_seed = np.array([-.755, -.056, 1.128, -.669, -1.250, -.004, -.706])
    handed = np.array([1., mirror, mirror, 1., mirror, 1., mirror])

    def shape(side, command):
        hand = routes.hands[side]
        return hand.vertices(hand.nominal_angles(command))

    def solve(side, origin, rotation, yaw, seed):
        if deadline is not None and monotonic() >= deadline:
            return None
        arm = routes.arms[side]
        orient = routes.local(rotation, yaw)
        goal = routes.local(origin, yaw) + orient @ arm.pocket
        return fit_contact(arm, goal, orient, seed)

    receive_mesh = shape(receiver, open_receive)
    tips = np.concatenate([receive_mesh[name] for name in DISTAL])

    def receiving_frame(azimuth, pitch, tip_height):
        forward = np.cos(azimuth) * inward + mirror * np.sin(azimuth) * along
        fingers = np.cos(pitch) * forward + np.sin(pitch) * up
        palm = np.cos(pitch) * up - np.sin(pitch) * forward
        rotation = np.column_stack((palm, np.cross(fingers, palm), fingers))
        projected = tips @ rotation.T
        origin = center - .170 * fingers
        lip_target = routes.edge - .007 if shallow else edge + .034
        origin += (lip_target - origin[0] - projected[:, 0].max()) / inward[0] * inward
        origin += (tip_height - ((origin + projected) @ up - floor).max()) * up
        if shallow and any((part @ rotation[0] + origin[0]).max() > edge - .012
                           for part in receive_mesh.values()):
            return None
        support = origin + rotation @ receive_point
        slide = support - center
        slide -= (slide @ up) * up
        if slide[0] >= 0. or not .055 < np.linalg.norm(slide) < .30:
            return None
        if any(np.any(np.all((cloud + fraction * slide >= low - .004)
                            & (cloud + fraction * slide <= high + .004), axis=1))
               for fraction in np.linspace(0., 1., 8) for low, high in occupied):
            return None
        return origin, rotation, support, slide

    def pushing_frame(parts, slide, lateral, setback, pitch, roll):
        direction = slide / np.linalg.norm(slide)
        sideway = np.cross(direction, up)
        fingers = mirror * lateral * sideway - setback * direction + pitch * up
        fingers /= np.linalg.norm(fingers)
        palm = direction - (direction @ fingers) * fingers
        palm = Rotation.from_rotvec(mirror * roll * fingers).apply(palm / np.linalg.norm(palm))
        rotation = np.column_stack((palm, np.cross(fingers, palm), fingers))
        mesh = np.concatenate(list(parts.values())) @ rotation.T
        base = parts['base_link']
        inner = base[base[:, 0] >= np.quantile(base[:, 0], .70)]
        inner_world = inner @ rotation.T
        low = (mesh @ up).min()
        cloud_local = cloud - center
        top = np.percentile(cloud_local @ up, 97.)
        height = inner_world @ up - low + .009
        pad = (height >= .003) & (height <= top + .002)
        if np.count_nonzero(pad) < 7:
            return None
        front = np.quantile(inner_world[pad] @ direction, .98)
        patch = pad & (inner_world @ direction > front - .007)
        back = np.quantile(cloud_local @ direction, .025)
        origin = (center + (.009 - low) * up - np.median(inner_world[patch] @ sideway) * sideway
                  + (back - .002 - front) * direction)
        digits = np.concatenate([part for name, part in parts.items() if name != 'base_link'])
        samples = digits @ rotation.T + origin - center
        span = samples @ sideway
        lo, hi = np.percentile(cloud_local @ sideway, (2., 98.))
        facing = (span > lo - .001) & (span < hi + .001) & (samples @ up <= top + .003)
        if np.any(samples[facing] @ direction > back - .001):
            return None
        return origin, rotation, np.median(inner[patch], axis=0)

    def pickup(yaw, receive, rotate_receive, support, slide, push, rotate_push, push_point, commands, profiles):
        _, clear, link = _paired_geometry(routes, box_removed, profiles, edge=edge, lower_body=lower_body)

        def path_clear(waypoints, stage=0):
            return all(link(a, b, yaw, stage, underneath=True) for a, b in zip(waypoints, waypoints[1:]))

        def pair(a, b):
            return {receiver: a, pusher: b}

        low = solve(receiver, receive, rotate_receive, yaw, receiver_seed * handed)
        if low is None:
            return None

        behind = solve(pusher, push, rotate_push, yaw, pusher_seed * handed)
        if behind is None:
            return None
        first = pair(positions[receiver], behind)
        arm = routes.arms[pusher]
        start = routes.world((arm.wrist(positions[pusher]) @ arm.hand)[:3], yaw)
        orientations = Slerp([0., 1.], Rotation.from_matrix([start[:, :3], rotate_push]))
        arc = .065 * inward - mirror * .070 * along + .100 * up
        approach = [positions]
        for fraction in (.26, .52, .76, 1.):
            origin = (start[:, 3] * (1. - fraction) + push * fraction
                      + np.sin(np.pi * fraction) * arc + .110 * fraction * up)
            q = solve(pusher, origin, orientations(fraction).as_matrix(), yaw, approach[-1][pusher])
            if q is None or not path_clear([approach[-1], pair(positions[receiver], q)]):
                return None
            approach.append(pair(positions[receiver], q))
        if not path_clear([approach[-1], first]):
            return None
        approach.append(first)

        pushes = [first]
        # Keep the coil's center on the table until the receiving hand is in place.
        receive_after = min(5, int(np.floor(6 * (center[0] - edge - .020) / -slide[0])))
        if receive_after < 1:
            return None
        for index, fraction in enumerate(np.linspace(1. / 6, 1., 6)):
            if index == receive_after:
                pushing = pushes[-1][pusher]
                for setback, rise in ((.076, .100), (.065, .075), (.095, .110)):
                    outside = solve(receiver, receive - setback * inward, rotate_receive, yaw, low)
                    if outside is None:
                        continue
                    hover = solve(receiver, receive - setback * inward + rise * up, rotate_receive, yaw, outside)
                    if hover is None:
                        continue
                    entry = [pair(q, pushing) for q in (hover, outside, low)]
                    if path_clear([pushes[-1], *entry]):
                        entry_start = len(pushes)
                        pushes.extend(entry)
                        break
                else:
                    return None
            q = solve(pusher, push + fraction * slide, rotate_push, yaw, pushes[-1][pusher])
            next_pair = pair(pushes[-1][receiver], q)
            if q is None or not path_clear([pushes[-1], next_pair]):
                return None
            pushes.append(next_pair)
        held = pushes[-1]
        if hook:
            # Bring the receiver to the lip while the pusher approaches and begins its slide.
            opening = [pair(left, q[pusher]) for left, q in
                       zip(np.linspace(positions[receiver], hover, len(approach) - 1), approach[:-1])]
            opening.append(pair(outside, approach[-1][pusher]))
            early_pushes = [pair(low, q[pusher]) for q in pushes[1:entry_start]]
            parallel = opening + early_pushes + pushes[entry_start + 3:]

            def approach_clear(first, goal):
                if not path_clear([first, goal]):
                    return False
                delta_left = goal[receiver] - first[receiver]
                delta_right = goal[pusher] - first[pusher]
                if not np.any(delta_left) or not np.any(delta_right):
                    return True
                return all(path_clear([first, pair(first[receiver] + left * delta_left,
                                                   first[pusher] + (1. - left) * delta_right), goal])
                           for left in (.4, .6))

            if all(approach_clear(a, b) for a, b in zip(parallel, parallel[1:])):
                approach = opening
                pushes = [opening[-1], *parallel[len(opening):]]

            def with_arm(pair, side, origin, rotation):
                q = solve(side, origin, rotation, yaw, pair[side])
                return None if q is None else dict(pair, **{side: q})

            def checked(first, goals, stage, underneath=False):
                return all(b is not None and link(a, b, yaw, stage, underneath=underneath)
                           for a, b in zip([first, *goals], goals))

            def wrist_frame(side, pair, angle):
                arm = routes.arms[side]
                return routes.world((arm.wrist(pair[side]) @ arm.hand)[:3], angle)

            closed = len(commands) - 1
            opened = len(profiles[receiver]) - 1
            if not all(clear(held, yaw, stage, underneath=True) for stage in range(closed + 1)):
                return None
            if not all(clear(positions, angle, 0) for angle in np.linspace(start_yaw, yaw, 12)):
                return None
            # Clear the lip vertically before returning the pushing arm to READY.
            for height in (.055, .085, .115):
                raised = with_arm(held, pusher, push + slide + height * up, rotate_push)
                if raised is None or not checked(held, [raised], closed, True):
                    continue
                for takeoff in (approach[-2:-1], approach[-2:0:-1], ()):
                    pusher_retreat = [raised, *[pair(held[receiver], q[pusher]) for q in takeoff],
                                      pair(held[receiver], positions[pusher])]
                    if checked(held, pusher_retreat, closed, True) and checked(held, pusher_retreat, 0, True):
                        break
                else:
                    continue
                break
            else:
                return None
            departed = pusher_retreat[-1]
            for depth in (.060, .090, .115):
                outside = receive - depth * inward
                retracted = with_arm(departed, receiver, outside, rotate_receive)
                high = (with_arm(retracted, receiver, outside + .10 * up, rotate_receive)
                        if retracted is not None else None)
                hook_abort = [retracted, high, positions]
                if checked(departed, hook_abort, 0, True) and checked(departed, hook_abort, closed, True):
                    break
            else:
                return None

            frame = wrist_frame(receiver, departed, yaw)
            held_points = cloud + (support - center)
            hand_points = (held_points - frame[:, 3]) @ frame[:, :3]
            hand_center = (support - frame[:, 3]) @ frame[:, :3]

            def carried(pair, angle, lip_clearance=.005):
                frame = wrist_frame(receiver, pair, angle)
                moved = hand_points @ frame[:, :3].T + frame[:, 3]
                if np.any((moved[:, 0] >= edge - .003) & (moved @ up - floor < lip_clearance)):
                    return False
                return not any(np.any(np.all((moved >= lo - .004) & (moved <= hi + .004), axis=1))
                               for lo, hi in occupied)

            withdrawn_origin = receive - .020 * inward
            withdrawn = with_arm(departed, receiver, withdrawn_origin, rotate_receive)
            if not link(departed, withdrawn, yaw, closed, underneath=True,
                        carried=lambda pair, angle: carried(pair, angle, lip_clearance=.004)):
                return None
            lifts = [withdrawn]
            for height in (.035, .070, .120, .175, .225, .270, .310):
                first_lift = height == .035
                # Clear the lip before returning inward to the transport line.
                origin = withdrawn_origin if first_lift else receive
                q = with_arm(lifts[-1], receiver, origin + height * up, rotate_receive)
                if not link(lifts[-1], q, yaw, closed, underneath=first_lift, carried=carried):
                    return None
                lifts.append(q)
            high = lifts[-1]
            departure = None
            # Once the right hand is above the lip, withdraw the cable while that arm returns.
            for fraction in (.50, .25, .75):
                right = raised[pusher] + fraction * (positions[pusher] - raised[pusher])
                overlapping = pair(withdrawn[receiver], right)
                if all(link(a, b, yaw, closed, underneath=True,
                            carried=lambda q, at: carried(q, at, lip_clearance=.004))
                       for a, b in ((raised, overlapping), (overlapping, lifts[1]))):
                    departure = [raised, overlapping, lifts[1]]
                    break

            # Curl the wrist around the supported coil as the carrying arm rises.
            carry_high = high
            carry_lifts = []
            for height, flex in ((.310, .13), (.335, .24), (.350, .34)):
                orientation = Rotation.from_rotvec(flex * rotate_receive[:, 1]).as_matrix() @ rotate_receive
                origin = receive + height * up + (rotate_receive - orientation) @ hand_center
                q = with_arm(carry_high, receiver, origin, orientation)
                if not link(carry_high, q, yaw, closed, carried=carried):
                    break
                carry_high = q
                carry_lifts.append(q)

            tray_low, tray_high = np.asarray(item['tray']['bounds'])
            open_vertices = np.concatenate(list(profiles[receiver][opened].values()))

            def above_tray(pair, angle):
                frame = wrist_frame(receiver, pair, angle)
                moved = hand_points @ frame[:, :3].T + frame[:, 3]
                return (np.all(moved[:, :2] >= tray_low[:2] + .005)
                        and np.all(moved[:, :2] <= tray_high[:2] - .005)
                        and moved[:, 2].min() >= tray_high[2] + .018
                        and carried(pair, angle))

            def tray_height(goals, angle):
                hand = wrist_frame(receiver, goals, angle)
                vertices = open_vertices @ hand[:, :3].T + hand[:, 3]
                over = np.all((vertices[:, :2] >= tray_low[:2] - .008)
                              & (vertices[:, :2] <= tray_high[:2] + .008), axis=1)
                return vertices[over, 2].min(initial=np.inf)

            def retreat_above_tray(release, angle):
                frame = wrist_frame(receiver, release, angle)

                def shifted(offset, seed):
                    q = solve(receiver, frame[:, 3] + offset, frame[:, :3], angle, seed[receiver])
                    return None if q is None else pair(q, positions[pusher])

                initial_height = tray_height(release, angle)
                safe_height = max(tray_high[2] + .085, initial_height + .025)
                required_rise = (.080 if not np.isfinite(initial_height) else
                                 max(.080, safe_height - initial_height + .015))
                rises = sorted({.080, .105, .140, .180, .220, .260, .300,
                                min(.340, required_rise), min(.340, required_rise + .030)})

                def clear_tray(goals, yaw):
                    return tray_height(goals, yaw) >= safe_height

                # Raise the open hand before moving toward the table edge or turning the wrist.
                for rise in rises:
                    lifted = shifted(rise * Z, release)
                    if not link(release, lifted, angle, opened,
                                carried=lambda q, at: tray_height(q, at) >= initial_height - .002):
                        continue
                    path = [lifted]
                    for setback in (.060, .120, .180, .240):
                        outside = shifted(np.array([-setback, 0., rise]), path[-1])
                        if not link(path[-1], outside, angle, opened, carried=clear_tray):
                            break
                        path.append(outside)
                        if link(outside, positions, angle, opened, carried=clear_tray):
                            return [*path, positions]
                        for return_pose in (carry_high, high):
                            if (link(outside, return_pose, angle, opened, carried=clear_tray)
                                    and link(return_pose, positions, angle, opened, carried=clear_tray)):
                                return [*path, return_pose, positions]
                return None

            for site in [item['release'], *item['alternatives']]:
                site = np.asarray(site)
                aim = routes.yaw + np.arctan2(site[1],site[0])
                for offset in (-.22, -.32, -.10, 0., .10):
                    if deadline is not None and monotonic() >= deadline:
                        return None
                    tray_yaw = aim + mirror * offset
                    turn = np.linspace(yaw, tray_yaw, 11)
                    folded = bool(carry_lifts) and (leave_hand_after_release or
                                                     all(clear(carry_high, at, closed) and carried(carry_high, at)
                                                         for at in turn))
                    if (not leave_hand_after_release and not folded
                            and not all(clear(high, at, closed) and carried(high, at) for at in turn)):
                        continue
                    for azimuth in (0., .22, -.22):
                        forward = routes.world(np.array([np.cos(azimuth), np.sin(azimuth), 0.]), tray_yaw)
                        for pitch in (-.65, -.78, -.50, -.90, -.35, -.25, -1.00):
                            fingers = np.cos(pitch) * forward + np.sin(pitch) * Z
                            palm = np.cos(pitch) * Z - np.sin(pitch) * forward
                            orient = np.column_stack((palm, np.cross(fingers, palm), fingers))
                            footprint = (hand_points - hand_center) @ orient.T
                            lo = tray_low[:2] - footprint[:, :2].min(0) + .006
                            hi = tray_high[:2] - footprint[:, :2].max(0) - .006
                            if np.any(lo > hi):
                                continue
                            xy = np.clip(site[:2], lo, hi)
                            if np.linalg.norm(xy - site[:2]) > .04:
                                continue
                            # Set the coil down near the tray floor before opening the hand.
                            bottom = tray_low[2] + .005 - footprint[:, 2].min()
                            for drop in ((0., .025, .050, .075, .100, .125, .150, .175, .200, .220)
                                         if leave_hand_after_release else (0., .006, .012)):
                                low_point = np.r_[xy, bottom + drop] - orient @ hand_center
                                high_point = low_point + (.105 if leave_hand_after_release else .045)*Z
                                qa = solve(receiver, high_point, orient, tray_yaw, high[receiver])
                                if qa is None:
                                    continue
                                above = pair(qa, positions[pusher])
                                transfer = [above]
                                if leave_hand_after_release:
                                    pass
                                elif not folded or not link(carry_high, above, tray_yaw, closed, carried=carried):
                                    if not link(high, above, tray_yaw, closed, carried=carried):
                                        continue
                                    if folded:
                                        if not link(carry_high, high, tray_yaw, closed, carried=carried):
                                            continue
                                        transfer.insert(0, high)
                                qb = solve(receiver, low_point, orient, tray_yaw, qa)
                                if qb is None:
                                    continue
                                release = pair(qb, positions[pusher])
                                if (not leave_hand_after_release
                                        and not link(above, release, tray_yaw, closed, carried=carried)):
                                    continue
                                if (not leave_hand_after_release
                                        and not all(clear(release, tray_yaw, stage)
                                                    for stage in range(closed, opened + 1))):
                                    continue
                                if (not leave_hand_after_release
                                        and (not link(release, above, tray_yaw, opened)
                                             or not link(above, positions, tray_yaw, opened))):
                                    continue
                                release_path = [release]
                                release_center = np.r_[xy, bottom + drop]
                                release_cloud = footprint + release_center
                                retreat = [release] if leave_hand_after_release else None
                                # Leave the arm at the tray after opening when the operator will
                                # demonstrate the withdrawal by hand.
                                flat_forward = routes.world(np.array([1., 0., 0.]), tray_yaw)
                                flat = np.column_stack((-Z, np.cross(flat_forward, -Z), flat_forward))
                                release_frame = wrist_frame(receiver, release, tray_yaw)
                                rises = (() if leave_hand_after_release else
                                         (.08, .10, .12, .14, .16, .18, .20, .22, .24, .27, .30))
                                for rise in rises:
                                    raised_origin = release_frame[:, 3] + rise * Z
                                    raised_q = solve(receiver, raised_origin, orient, tray_yaw, qb)
                                    if raised_q is None:
                                        continue
                                    raised = pair(raised_q, positions[pusher])
                                    initial_height = tray_height(release, tray_yaw)
                                    if not link(release, raised, tray_yaw, opened,
                                                carried=lambda q, at: tray_height(q, at) >= initial_height - .002):
                                        continue
                                    if tray_height(raised, tray_yaw) < tray_high[2] + .018:
                                        continue
                                    flip_q = solve(receiver, raised_origin, flat, tray_yaw, raised_q)
                                    if flip_q is None:
                                        continue
                                    flip = pair(flip_q, positions[pusher])
                                    if (tray_height(flip, tray_yaw) < tray_high[2] + .018
                                            or not link(raised, flip, tray_yaw, opened,
                                                        carried=lambda q, at: tray_height(q, at) >= tray_high[2] + .018)):
                                        continue
                                    after_flip = retreat_above_tray(flip, tray_yaw)
                                    if after_flip is None:
                                        continue
                                    retreat = [raised, flip, *after_flip]
                                    break
                                if retreat is None:
                                    continue
                                push_steps = [dict(phase='scoop_open', q=q, command=commands[0]) for q in pushes[1:]]
                                close_steps = [dict(phase='close', q=held, command=cmd) for cmd in commands[1:]]
                                return dict(item=dict(item, release=release_center.tolist()), pick_yaw=yaw,
                                            tray_yaw=tray_yaw, approach=approach[1:], grasp=[held],
                                            close=push_steps + close_steps, preshape=commands[0], grip=commands[-1],
                                            under_table=True, paired_carry=False, edge_receive=True, hook_carry=True,
                                            hook_open_fingers=True,
                                            grip_side=receiver, pusher_side=pusher, probe_only=False,
                                            pusher_retreat=pusher_retreat, edge_probe=lifts[:2],
                                            lift=lifts[2:] + (carry_lifts if folded else []),
                                            transfer=transfer, release=release_path, retreat=retreat,
                                            release_hand={receiver: [1.] * 6}, edge_object_center=support,
                                            receive_point=receive_point, push_point=push_point, receiver_edge_x_m=edge,
                                            probe_return=[withdrawn, departed], hook_abort=hook_abort,
                                            abort_after_push=pusher_retreat + hook_abort,
                                            abort_close=[], pickup_retreat=list(reversed(approach[:-1])),
                                            hook_departure=departure, hook_roll=False,
                                            hook_flip_after_open=not leave_hand_after_release,
                                            hook_leave_hand_after_release=leave_hand_after_release,
                                            hook_lower_to_floor_m=float(drop) if leave_hand_after_release else 0.,
                                            release_footprint=[release_cloud[:, :2].min(0).tolist(),
                                                               release_cloud[:, :2].max(0).tolist()])
            return None
        if not all(clear(held, yaw, stage, underneath=True) for stage in range(1, len(commands))):
            return None
        if not all(clear(positions, angle, 0) for angle in np.linspace(start_yaw, yaw, 12)):
            return None
        # Withdraw the still-low hands horizontally before either can rise.
        catch_shift = support - center
        minimum_depth = .020 if shallow else .055
        depth = max(minimum_depth, ((cloud + catch_shift)[:, 0].max() - edge + .009) / inward[0])
        delivery = None
        for lateral in ((0., -.026) if receiver == 'left' else (-.026, 0.)):
            travel = -depth * inward + mirror * lateral * along
            withdrawal = [held]
            for fraction in (.25, .50, .75, 1.):
                displacement = fraction * travel
                q_receive = solve(receiver, receive + displacement, rotate_receive, yaw, withdrawal[-1][receiver])
                q_push = solve(pusher, push + slide + displacement, rotate_push, yaw, withdrawal[-1][pusher])
                if q_receive is None or q_push is None:
                    break
                q = pair(q_receive, q_push)
                if not path_clear([withdrawal[-1], q], len(commands) - 1):
                    break
                withdrawal.append(q)
            if len(withdrawal) != 5:
                continue
            shift = catch_shift + travel
            plan_delivery = _left_cable_delivery if receiver == 'left' else edge_pair_delivery
            delivery = plan_delivery(
                routes, item, withdrawal[-1], yaw, cloud + shift, np.asarray(target['xyz']) + shift, solve,
                lambda q, angle, *, opened=False, underneath=False: clear(
                    q, angle, 0 if opened else len(commands) - 1, underneath=underneath),
                start_ready=positions, edge=edge, withdraw=0., obstacles=obstacles)
            if delivery is not None and all(
                    clear(delivery['release'][-1], delivery['tray_yaw'], stage)
                    for stage in range(len(commands))):
                break
            delivery = None
        if delivery is None:
            return None
        # Reach outside the lip before the probe can rise.
        lifts = withdrawal[1:] + delivery['lift']

        def receiver_height(pair_q):
            hand = receiving_arm.wrist(pair_q[receiver]) @ receiving_arm.hand
            point = routes.world(hand[:3, 3] + hand[:3, :3] @ receive_point, yaw)
            return (point - support) @ up

        probe_count = next((n for n, pair_q in enumerate(lifts, 1)
                            if receiver_height(pair_q) >= .020), None)
        if probe_count is None:
            return None
        push_steps = [dict(phase='scoop_open', q=q, command=commands[0]) for q in pushes[1:]]
        closing_steps = [dict(phase='close', q=held, command=command) for command in commands[1:]]
        reverse = list(reversed(pushes[:-1]))
        retreat = list(reversed(approach[:-1]))
        return dict(delivery, under_table=True, paired_carry=True, edge_receive=True, probe_only=False,
                    grip_side=receiver, pusher_side=pusher, pick_yaw=yaw, approach=approach[1:], grasp=[held],
                    receive_point=receive_point, receiver_edge_x_m=edge, push_point=push_point,
                    edge_object_center=support, edge_probe=lifts[:probe_count], lift=lifts[probe_count:],
                    probe_return=[*reversed(lifts[:probe_count - 1]), held],
                    close=push_steps + closing_steps, preshape=commands[0], grip=commands[-1],
                    abort_close=[dict(phase='withdraw_open', q=q, command=commands[0]) for q in reverse],
                    abort_after_push=[*reverse, *retreat], pickup_retreat=retreat)

    open_push = np.array([.55] * 4 + [.60, 1.])
    close_push = open_push if hook else (open_push + np.array([.22] * 4 + [.44, .73])) * .5
    pushing_mesh = shape(pusher, open_push)
    commands = [{receiver: (open_receive + fraction * (close_receive - open_receive)).tolist(),
                 pusher: (open_push + fraction * (close_push - open_push)).tolist()}
                for fraction in (0., .50, 1.)]
    profiles = {side: [shape(side, command[side]) for command in commands] for side in SIDES}
    if hook:
        for fraction in (.33, .67, 1.):
            profiles[receiver].append(shape(receiver, close_receive + fraction * (1. - close_receive)))
            profiles[pusher].append(profiles[pusher][-1])
    heading = routes.yaw + np.arctan2(center[1], center[0])
    pitches = ((-.65, (.292,), -.009),) if shallow else ((-.58, (.182, .112), -.071), (-.65, (.292,), -.071))
    for pitch, offsets, tip_height in pitches:
        frame = receiving_frame(.35, pitch, tip_height)
        if frame is None:
            continue
        push_frame = pushing_frame(pushing_mesh, frame[-1], 1., 1.2, -1.2, .65)
        if push_frame is None:
            continue
        for offset in offsets:
            plan = pickup(heading + mirror * offset, *frame, *push_frame, commands, profiles)
            if plan is not None:
                return plan
    return None


def cable_edge_receive_plan(routes, item, positions, *, box_removed=False, probe_only=False, edge_retry=False,
                            under_table=False, lower_body=None, start_yaw=None, hook=False,
                            leave_hand_after_release=False, receiver_side=None, deadline=None):
    """Slide the coil to a support palm for a catch at the observed table edge."""
    if under_table or hook:
        if deadline is not None and monotonic() >= deadline:
            return None
        return _cable_under_edge_plan(routes, item, positions, lower_body, start_yaw, box_removed, hook=hook,
                                      leave_hand_after_release=leave_hand_after_release,
                                      receiver_side=receiver_side, deadline=deadline)
    target = item['target']
    center, axis, cross, up = object_frame(target, routes.normal, routes.floor)
    points = surface_points(target, True, center, axis, cross, up)
    landing_x = routes.edge - .021
    if len(points) < 20 or not .035 < center[0] - landing_x < .31:
        return None
    across = np.array([1., 0., 0.]) - up[0] * up
    across /= np.linalg.norm(across)
    along = np.cross(up, across)
    slide = (landing_x - center[0]) / across[0] * across
    landing = center + slide
    receiver, pusher = SIDES if landing[1] >= 0. else SIDES[::-1]
    mirror = SIGNS[receiver]
    push_command = EDGE_RETRY_PUSH if edge_retry else OPEN
    commands = [{side: (OPEN + fraction * (EDGE_CLOSED - OPEN) if side == receiver else push_command).tolist()
                 for side in SIDES} for fraction in STAGES]
    profiles = {side: [_hand_profile(routes.hands[side], stage[side]) for stage in commands] for side in SIDES}
    _, clear, link = _paired_geometry(routes, box_removed, profiles)
    closed = len(STAGES) - 1
    hand = routes.hands[receiver]
    parts = hand.vertices(hand.nominal_angles(OPEN))
    nails = np.concatenate([parts[name] for name in DISTAL])
    pushing_hand = np.concatenate(profiles[pusher][0])

    def wrist(side, origin, orient, yaw, seed):
        arm = routes.arms[side]
        return fit_contact(arm, routes.local(origin + orient @ arm.pocket, yaw), routes.local(orient, yaw), seed)

    def receive_pose(azimuth, pitch, shift):
        flat = np.cos(azimuth) * across + mirror * np.sin(azimuth) * along
        fingers = np.cos(pitch) * flat + np.sin(pitch) * up
        palm = np.cos(pitch) * up - np.sin(pitch) * flat
        orient = np.column_stack((palm, np.cross(fingers, palm), fingers))
        tips = nails @ orient.T
        origin = landing + mirror * shift * along - .187 * fingers
        origin += (routes.edge - .004 - origin[0] - tips[:, 0].max()) / across[0] * across
        origin += (-.003 - ((origin - center + tips) @ up).max()) * up
        return origin, orient

    def push_pose(azimuth, pitch, roll, shift, gap):
        flat = np.cos(azimuth) * across + mirror * np.sin(azimuth) * along
        fingers = np.cos(pitch) * flat - np.sin(pitch) * up
        palm = Rotation.from_rotvec(mirror * roll * fingers).apply(mirror * np.cross(up, flat))
        orient = np.column_stack((palm, np.cross(fingers, palm), fingers))
        shape = pushing_hand @ orient.T
        bottom = (shape @ up).min()
        height = shape @ up - bottom
        pad = shape[(height > .002) & (height < .023)]
        if len(pad) < 4:
            return None
        contact_x = np.quantile(pad[:, 0], .06)
        contact_y = (pad[pad[:, 0] < contact_x + .011] @ along).mean()
        rear = np.quantile(points[:, 0], .98) + .005 if edge_retry else points[:, 0].max() + .010
        origin = (center + (rear - center[0] - contact_x) / across[0] * across
                  + (mirror * shift - contact_y) * along + (gap - bottom) * up)
        return origin, orient

    def pickup(yaw, receive, receive_rot, push, push_rot):
        receive_q = wrist(receiver, receive, receive_rot, yaw, positions[receiver])
        push_q = wrist(pusher, push, push_rot, yaw, positions[pusher])
        if receive_q is None or push_q is None:
            return None
        first = {receiver: receive_q, pusher: push_q}
        if not clear(first, yaw, 0):
            return None
        hover = {
            receiver: wrist(receiver, receive + .120 * Z, receive_rot, yaw, receive_q),
            pusher: wrist(pusher, push + .120 * Z, push_rot, yaw, push_q),
        }
        if any(q is None for q in hover.values()) or not link(hover, first, yaw):
            return None
        if not link(positions, hover, yaw):
            return None

        slide_steps = [first]
        stroke = slide - (.033 if edge_retry else .008) / across[0] * across
        for fraction in np.linspace(1. / 6, 1., 6):
            q = wrist(pusher, push + fraction * stroke, push_rot, yaw, slide_steps[-1][pusher])
            pair = {receiver: receive_q, pusher: q}
            if q is None or not link(slide_steps[-1], pair, yaw):
                return None
            slide_steps.append(pair)
        end = slide_steps[-1]
        # Lift the pusher out of the coil before closing the receiving hand.
        raised_q = wrist(pusher, push + stroke + .095 * Z, push_rot, yaw, end[pusher])
        if raised_q is None:
            return None
        raised = {receiver: receive_q, pusher: raised_q}
        depart_q = wrist(pusher, push + stroke + .095 * Z + .035 * across, push_rot, yaw, raised_q)
        probe_q = wrist(receiver, receive + .030 * Z, receive_rot, yaw, receive_q)
        departed = {receiver: receive_q, pusher: depart_q}
        probe = {receiver: probe_q, pusher: depart_q}
        if (depart_q is None or probe_q is None or not link(end, raised, yaw) or not link(raised, departed, yaw)
                or not all(clear(departed, yaw, stage) for stage in range(1, closed + 1))
                or not link(departed, probe, yaw, closed)):
            return None

        high = {receiver: hover[receiver], pusher: depart_q}
        if not link(departed, high, yaw):
            return None
        if link(high, positions, yaw):
            abort_after_push = [raised, departed, high, positions]
        elif link(high, hover, yaw):
            abort_after_push = [raised, departed, high, hover, positions]
        else:
            return None

        phases = ([dict(q=q, command=commands[0], phase='scoop_open') for q in slide_steps[1:]]
                  + [dict(q=q, command=commands[0], phase='pusher_clear') for q in (raised, departed)]
                  + [dict(q=departed, command=cmd, phase='close') for cmd in commands[1:]])
        reverse = [dict(q=q, command=commands[0], phase='withdraw_open') for q in reversed(slide_steps[:-1])]
        return dict(item=item, pick_yaw=yaw, tray_yaw=yaw, approach=[hover, first], grasp=[departed],
                    close=phases, abort_close=reverse, abort_after_push=abort_after_push, pickup_retreat=[hover, positions],
                    preshape=commands[0], grip=commands[-1], edge_receive=True,
                    grip_side=receiver, pusher_side=pusher, edge_probe=[probe],
                    probe_return=[departed], probe_only=True, lift=[probe],
                    transfer=[], release=[], retreat=[])

    def delivery(plan, receive, receive_rot):
        yaw = plan['pick_yaw']
        raised = plan['edge_probe'][-1]
        hold_center = (landing + .003 * up - receive) @ receive_rot
        hold_points = (points + slide - receive) @ receive_rot
        arm = routes.arms[receiver]
        obstacles = [np.asarray(obj['bounds']) for obj in routes.objects
                     if obj['kind'] not in ('cable', 'tray_blue', 'tray_red', 'tray_pink')
                     and not (box_removed and obj['kind'] == 'drive')]

        def carried(pair, at_yaw):
            frame = routes.world((arm.wrist(pair[receiver]) @ arm.hand)[:3], at_yaw)
            cloud = hold_points @ frame[:, :3].T + frame[:, 3]
            if np.min(cloud @ routes.normal - routes.floor) < .010:
                return False
            return not any(np.any(np.all((cloud >= low - .007) & (cloud <= high + .007), axis=1))
                           for low, high in obstacles)

        lift_height = max(.260, max(obj['bounds'][1][2] for obj in routes.objects)
                          + .028 - (points[:, 2].min() + slide[2]))
        lift = [raised]
        for height in (.060, .120, .180, .215, .240, lift_height):
            q = wrist(receiver, receive + height * Z, receive_rot, yaw, lift[-1][receiver])
            pair = {receiver: q, pusher: raised[pusher]}
            if q is None or not link(lift[-1], pair, yaw, closed, carried=carried):
                return None
            lift.append(pair)
        raised = lift[-1]
        bounds = np.asarray(item['tray']['bounds'])
        for site in [item['release'], *item['alternatives']]:
            site = np.asarray(site)
            aim = routes.yaw + np.arctan2(site[1], site[0])
            for tray_yaw in (aim - .22, aim - .32, aim - .10, aim):
                if not all(clear(raised, at, closed) and carried(raised, at)
                           for at in np.linspace(yaw, tray_yaw, 8)):
                    continue
                forward = routes.world(np.array([1., 0., 0.]), tray_yaw)
                for pitch in (-.90, -.75, -.60, -.48, -.35):
                    fingers = np.cos(pitch) * forward + np.sin(pitch) * Z
                    palm = np.cos(pitch) * Z - np.sin(pitch) * forward
                    orient = np.column_stack((palm, np.cross(fingers, palm), fingers))
                    footprint = (hold_points - hold_center) @ orient.T
                    lo = bounds[0, :2] - footprint[:, :2].min(0) + .006
                    hi = bounds[1, :2] - footprint[:, :2].max(0) - .006
                    if np.any(lo > hi):
                        continue
                    xy = np.clip(site[:2], lo, hi)
                    if np.linalg.norm(xy - site[:2]) > .025:
                        continue
                    # Leave room for the coil to slide down the fingers after opening.
                    hand_xy = np.clip(xy - .016 * forward[:2], lo, hi)
                    low_z = max(.015, bounds[1, 2] - footprint[:, 2].min() + .030)
                    high = np.r_[hand_xy, low_z + .045] - orient @ hold_center
                    low = np.r_[hand_xy, low_z] - orient @ hold_center
                    high_q = wrist(receiver, high, orient, tray_yaw, raised[receiver])
                    above = {receiver: high_q, pusher: raised[pusher]}
                    if high_q is None or not link(raised, above, tray_yaw, closed, carried=carried):
                        continue
                    low_q = wrist(receiver, low, orient, tray_yaw, high_q)
                    release = {receiver: low_q, pusher: raised[pusher]}
                    if low_q is None or not link(above, release, tray_yaw, closed, carried=carried):
                        continue
                    if not link(release, above, tray_yaw) or not link(above, positions, tray_yaw):
                        continue
                    placement = dict(item, release=np.r_[xy, site[2]].tolist())
                    return dict(plan, item=placement, tray_yaw=tray_yaw, probe_only=False,
                                lift=lift[1:], transfer=[above], release=[release], retreat=[above, positions])
        return None

    heading = routes.yaw + np.arctan2(center[1], center[0])
    profiles_at_edge = (
        ((.35, -.20, 0.), (1.55, .43, 0., 0., .012)),
        ((.35, -.20, -.026), (1.55, .43, 0., 0., .012)),
        ((.60, -.20, 0.), (1.20, .43, 0., -.035, .012)),
        ((.35, -.20, .026), (1.20, .22, -.35, -.035, .012)),
    )
    probe = None
    if edge_retry:
        profiles_at_edge = (((.35, -.20, 0.), (1.15, .65, .35, 0., .014)),)
    yaw_offsets = (.10, -.025, .22) if edge_retry else (-.025, -.145, .10)
    for offset in yaw_offsets:
        yaw = heading + offset
        for receive_config, push_config in profiles_at_edge:
            push_frame = push_pose(*push_config)
            if push_frame is None:
                continue
            receive_frame = receive_pose(*receive_config)
            plan = pickup(yaw, *receive_frame, *push_frame)
            if plan is None:
                continue
            if probe_only:
                return plan
            if probe is None:
                probe = plan
            full = delivery(plan, *receive_frame)
            if full is not None:
                return full
    return probe

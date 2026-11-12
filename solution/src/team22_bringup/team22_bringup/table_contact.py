"""Top pinches based on the frozen RGB-D survey and the official hand meshes."""

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


TIP_LINKS = {'thumb': 'Link48', 'index': 'Link4'}
CABLE_TIPS = {**TIP_LINKS, 'middle': 'Link14', 'ring': 'Link24', 'little': 'Link34'}
CABLE_DIGIT_LINKS = (('Link33', 'Link34'), ('Link23', 'Link24'), ('Link13', 'Link14'),
                     ('Link3', 'Link4'), ('Link47', 'Link48'))
CONTACT_GAP = .0006
SOLID_GAP = .002
CLOSE_STEPS = (.25, .50, .75, 1.)
CABLE_SCOOP_DROP = .020
CABLE_SCOOP_LOW = .006
CABLE_SCOOP_HIGH = CABLE_SCOOP_LOW + CABLE_SCOOP_DROP
CABLE_SCOOP_PROFILE = ((.20, .0256), (.40, .025), (.60, .022), (.78, .015),
                       (1., CABLE_SCOOP_LOW))
CABLE_OPEN = np.array([1., 1., 1., 1., .84, 0.])
CABLE_CLOSING_RATE = np.array([1., 1., 1., 1.12, 1.12, 1.])


def split(parts, *, cable=False):
    links = CABLE_TIPS if cable else TIP_LINKS
    solid = np.concatenate([vertices for name, vertices in parts.items() if name not in links.values()])
    tips = {finger: parts[link] for finger, link in links.items()}
    return solid, tips


def fit_contact(arm, point, orient, seed):
    """Refine surface waypoints more tightly than ordinary transport IK."""
    lower, upper = arm.limits + [[.018], [-.018]]

    def residual(q):
        pose = arm.tool(q)
        turn = Rotation.from_matrix(orient.T @ pose[:3, :3]).as_rotvec()
        return np.r_[pose[:3, 3] - point, .24 * turn, .00003 * (q - seed)]

    result = least_squares(residual, np.clip(seed, lower + 1e-7, upper - 1e-7),
                           bounds=(lower, upper), max_nfev=45, ftol=1e-6, gtol=1e-8)
    if np.linalg.norm(result.fun[:3]) < .0008 and np.linalg.norm(result.fun[3:6]) < .0005:
        return result.x
    return None


def support_open(hand, command, beta, *, cable=False):
    joints = hand.nominal_angles(command)
    parts = hand.vertices(joints)
    solid, tips = split(parts, cable=cable)

    def normal(alpha):
        return np.array([-np.cos(alpha) * np.cos(beta), np.sin(beta), -np.sin(alpha) * np.cos(beta)])

    def difference(alpha):
        n = normal(alpha)
        return (tips['thumb'] @ n).min() - (tips['index'] @ n).min()

    if difference(-.4) * difference(1.3) > 0.:
        return None
    n = normal(brentq(difference, -.4, 1.3))
    thumb, index = (tips[finger][np.argmin(tips[finger] @ n)] for finger in TIP_LINKS)
    if (solid @ n).min() - thumb @ n < .004 - (CABLE_SCOOP_HIGH if cable else 0.):
        return None
    along = index - thumb
    along /= np.linalg.norm(along)
    return parts, solid, tips, thumb, index, np.column_stack((along, np.cross(n, along), n))


def object_frame(target, normal, floor):
    center = np.array(target['xyz'], dtype=float)
    center -= (center @ normal - floor) * normal / (normal @ normal)
    geometry = target.get('geometry', {})
    long = np.asarray(geometry.get('long_axis_xy', ()), dtype=float)
    low, high = np.asarray(target['bounds'], dtype=float)
    if long.shape != (2,) or np.linalg.norm(long) < .5:
        long = np.eye(2)[np.argmax((high - low)[:2])]
    axis = np.r_[long, 0.]
    axis -= (axis @ normal) * normal / (normal @ normal)
    axis /= np.linalg.norm(axis)
    up = normal / np.linalg.norm(normal)
    return center, axis, np.cross(up, axis), up


def surface_points(target, cable, center, axis, cross, up):
    points = np.asarray(target.get('contact_points', ()), dtype=float)
    if points.ndim == 2 and points.shape[1] == 3 and len(points) >= 20:
        return points[np.isfinite(points).all(axis=1)]
    if cable:
        return np.empty((0, 3))
    low, high = np.asarray(target['bounds'])
    size = np.asarray(target.get('geometry', {}).get('dimensions_mm', ())) * .001
    if size.shape != (3,):
        size = np.r_[np.sort((high - low)[:2])[::-1], high[2] - low[2]]
    # Legacy mouse snapshots have the oriented body dimensions, but no samples.
    # Approximate the rounded sides at several heights for nominal contact.
    angles = np.linspace(0., 2. * np.pi, 36, endpoint=False)
    rim = size[0] * .5 * np.cos(angles)[:, None] * axis + size[1] * .5 * np.sin(angles)[:, None] * cross
    return np.concatenate([center + scale * rim + z * size[2] * up
                           for z, scale in ((.25, .95), (.48, 1.), (.72, .86), (.90, .57))])


def contact_sites(cable, center, axis, cross, up, surface, *, centered=False):
    basis = np.column_stack((axis, cross))
    tree = cKDTree((surface - center) @ basis)
    if not cable:
        if centered:
            for direction in (cross, -cross):
                yield center, direction, center, tree, basis
            return
        # The fingertips travel along the long axis as they curl. Start behind
        # the pinch section so the closing pads meet the wider part of the shell.
        for shift in (.040, .032, .022, .014, 0.):
            for angle in (0., -.20, .20, -.4, .4):
                along = Rotation.from_rotvec(up * angle).apply(cross)
                for sign in (1., -1.):
                    direction = sign * along
                    site = center + sign * shift * axis
                    for bias in (.004, 0., -.004):
                        yield site + bias * direction, direction, site, tree, basis
        return

    # A radial line through a visible section of the winding. Bias the midpoint
    # outward so the fully open thumb reaches the hole, not the opposite turn.
    for degrees in (60, 30, -60, 90, -90, -30, 120, -120, 0, 180, 150, -150, 45, 75, -45, -75):
        angle = np.deg2rad(degrees)
        radial = np.cos(angle) * axis + np.sin(angle) * cross
        for radius in (.046, .057, .062, .070, .080):
            rim = center + radius * radial
            if tree.query((rim - center) @ basis)[0] > .014:
                continue
            for turn in (.38, -.38, -.20, 0., .22, -.22):
                direction = Rotation.from_rotvec(up * turn).apply(radial)
                for bias in (.024, .019, .029):
                    yield rim + bias * radial, direction, rim, tree, basis


def finger_distance(hand, frame, link, points, count=1):
    local = (points - frame[:3, 3]) @ frame[:3, :3]
    shape = hand.shapes[link]
    vertices = shape['vertices']
    nearby = local[np.all((local >= vertices.min(axis=0) - .007)
                          & (local <= vertices.max(axis=0) + .007), axis=1)]
    planes = shape['planes']
    distance = (nearby @ planes[:, :3].T + planes[:, 3]).max(axis=1)
    return np.partition(distance, count - 1)[count - 1] if len(distance) >= count else 1.


def fingertip_distances(hand, joints, surfaces, count=1):
    frames = hand.frames(joints)
    return [finger_distance(hand, frames[link], link, points, count)
            for link, points in zip(TIP_LINKS.values(), surfaces)]


def close_shapes(hand, opened, command, origin, orient, normal, floor):
    axis = orient.T @ normal
    plane_at_origin = origin @ normal - floor
    stages = []
    previous = 0.
    minimum_height = 0.
    for fraction in CLOSE_STEPS:
        states = [hand.vertices(hand.nominal_angles(opened + t * (command - opened)))
                  for t in np.linspace(previous, fraction, 5)]
        profiles = [split(parts) for parts in states]
        solid = np.concatenate([parts[0] for parts in profiles])
        tips = np.concatenate([v for _, finger_parts in profiles for v in finger_parts.values()])
        required = max(SOLID_GAP - plane_at_origin - (solid @ axis).min(),
                       CONTACT_GAP - plane_at_origin - (tips @ axis).min()) / np.linalg.norm(normal)
        # A loaded finger can remain in any earlier pose while closing.
        minimum_height = max(minimum_height, required)
        height = minimum_height
        stages.append((opened + fraction * (command - opened), height, solid, tips, states[-1]))
        previous = fraction
    return stages


def cable_command(fraction):
    # Lead with the index and thumb so they close around the coil.
    return CABLE_OPEN * np.maximum(0., 1. - fraction * CABLE_CLOSING_RATE)


def cable_grasp_contacts(hand, frames, surface, origin, orient):
    points = (surface - origin - [0., 0., CABLE_SCOOP_LOW]) @ orient
    return all(any(finger_distance(hand, frames[link], link, points) <= .003 for link in digit)
               for digit in CABLE_DIGIT_LINKS)


def close_candidates(hand, opened, surface, origin, orient, normal, floor, up, rim, direction,
                     *, recenter=None):
    if recenter is not None:
        settings = ((.48, .33, .72), (.48, .36, .72), (.48, .39, .72), (.46, .39, .72))
    else:
        settings = ((.46, .40, .76), (.43, .38, .86), (.45, .41, .82), (.42, .37, .80),
                    (.44, .41, 1.), (.44, .38, .72), (.44, .33, .73), (.47, .45, .63),
                    (.40, .36, .72), (.39, .34, .88), (.46, .37, .70))
    transverse = (surface - rim) @ direction
    half_width = min(.018, np.quantile(abs(transverse), .85) * .70)
    flanks = (surface[transverse < -half_width], surface[transverse > half_width])
    closing_origin = origin if recenter is None else origin + recenter
    for index, thumb, opposition in settings:
        command = np.array([1., 1., 1., index, thumb, opposition])
        joints = hand.nominal_angles(command)
        parts = hand.vertices(joints)
        th, ix = [closing_origin + orient @ parts[link].mean(axis=0) for link in TIP_LINKS.values()]
        if min((rim - th) @ direction, (ix - rim) @ direction) < .011:
            continue
        separation = (ix - th) @ direction
        if not .022 < separation < .044:
            continue
        stages = close_shapes(hand, opened, command, origin, orient, normal, floor)
        last_height = stages[-1][1]
        points = tuple((flank - closing_origin - last_height * up) @ orient for flank in flanks)
        distances = fingertip_distances(hand, joints, points, count=3)
        # A single nearby sample did not give the actuators preload in simulation.
        if max(distances) > -.002 or last_height >= .035:
            continue
        yield command, stages


def surface_link(routes, side, start, goal, yaw, solid, tips):
    count = 2 + int(np.max(abs(goal - start)) / .035)
    return all(routes.clear(side, q, solid, yaw, touch=tips, gap=SOLID_GAP)
               for q in np.linspace(start, goal, count))


def cable_scoop(routes, side, yaw, start, orient, origin, hand, fit):
    # Enter above the table and keep lowering as the fingers wrap the coil.
    steps = []
    previous_q, previous_hand = start, CABLE_OPEN
    for fraction, height in CABLE_SCOOP_PROFILE:
        values = cable_command(fraction)
        q = fit(origin + [0., 0., height], orient, previous_q, precise=True)
        if q is None:
            return None
        for t in np.linspace(0., 1., 9):
            command = previous_hand + t * (values - previous_hand)
            solid, tips = split(hand.vertices(hand.nominal_angles(command)), cable=True)
            if not routes.clear(side, previous_q + t * (q - previous_q), solid, yaw,
                                touch=np.concatenate(list(tips.values())), gap=SOLID_GAP):
                return None
        steps.append(dict(q=q, command=values.tolist()))
        previous_q, previous_hand = q, values
    closed = cable_command(1.)
    return closed, steps, previous_q, hand.vertices(hand.nominal_angles(closed))


def touch_route(routes, side, yaw, start, orient, origin, up, hand, opening, closed, cable,
                *, recenter=None, surface=None):
    parts, solid, tips, thumb, index, _ = opening
    stacked = np.concatenate(list(tips.values()))
    arm = routes.arms[side]

    def fit(at, rotation, seed, precise=False):
        goal = routes.local(at + rotation @ arm.pocket, yaw)
        direction = routes.local(rotation, yaw)
        if precise:
            return fit_contact(arm, goal, direction, seed)
        return routes.fit(side, goal, direction, seed)

    lift_axis = np.array([0., 0., 1.]) if cable else up
    entry_origin = origin + CABLE_SCOOP_HIGH * lift_axis if cable else origin
    rough = fit(entry_origin, orient, start)
    if rough is None:
        return None
    both = fit(entry_origin, orient, rough, precise=True)
    if both is None:
        return None
    hover = fit(origin + (.100 if cable else .070) * lift_axis, orient, both, precise=True)
    if hover is None:
        return None
    touches = [both]
    prefix = [hover]
    if not all(surface_link(routes, side, a, b, yaw, solid, stacked)
               for a, b in zip([prefix[-1], *touches], touches)):
        return None

    open_shape = np.concatenate(list(parts.values()))
    path = [*prefix, *touches]
    if not routes.link(side, start, hover, open_shape, yaw):
        raised = routes.raised_ready(side, start, open_shape, yaw)
        if raised is None or not routes.link(side, raised, hover, open_shape, yaw):
            return None
        path.insert(0, raised)

    if cable:
        scoop = cable_scoop(routes, side, yaw, both, orient, origin, hand, fit)
        if scoop is None:
            return None
        command, close, final, closed_parts = scoop
        state = dict(solid=solid, tips=tips, waypoints=touches, close=close, scoop=True,
                     scoop_drop_m=CABLE_SCOOP_DROP)
        closed_solid, closed_tips = split(closed_parts, cable=True)
        return path, state, final, closed_solid, np.concatenate(list(closed_tips.values())), command

    command, stages = closed
    close = []
    previous = both
    previous_parts = parts
    for index, (values, height, sweep_solid, sweep_tips, next_parts) in enumerate(stages):
        at = origin + height * up
        if recenter is not None and index:
            at += recenter
        q = fit(at, orient, previous, precise=True)
        previous_solid, previous_tips = split(previous_parts)
        if (q is None or not routes.clear(side, q, sweep_solid, yaw, touch=sweep_tips, gap=SOLID_GAP)
                or not surface_link(routes, side, previous, q, yaw, previous_solid,
                                    np.concatenate(list(previous_tips.values())))):
            return None
        if recenter is not None and index == 1:
            # Check the held finger pose while the raised wrist moves to the shell flanks.
            joints = hand.nominal_angles(stages[0][0])
            for qi in np.linspace(previous, q, 11):
                pose = routes.world((arm.wrist(qi) @ arm.hand)[:3, :], yaw)
                if hand.collision((surface - pose[:, 3]) @ pose[:, :3], joints, margin=-.001):
                    return None
        close.append(dict(q=q, command=values.tolist()))
        previous, previous_parts = q, next_parts

    state = dict(solid=solid, tips=tips, waypoints=touches, close=close)
    closed_solid, closed_tips = split(previous_parts)
    return path, state, previous, closed_solid, np.concatenate(list(closed_tips.values())), command


def table_pickups(routes, item, side, start):
    """Yield overhead pinch routes with explicit tabletop contact waypoints."""
    cable = item['kind'] == 'cable'
    target = item['target']
    center, axis, cross, up = object_frame(target, routes.normal, routes.floor)
    surface = surface_points(target, cable, center, axis, cross, up)
    if len(surface) < 20:
        return
    hand = routes.hands[side]
    arm = routes.arms[side]
    bottom = target['bounds'][0][2]
    yaw_center = routes.heading(center, side)
    sites = list(contact_sites(cable, center, axis, cross, up, surface))
    centered_sites = [] if cable else list(contact_sites(False, center, axis, cross, up, surface, centered=True))
    configurations = ((0., -.20, False), (0., -.10, False), (0., 0., False)) if cable else (
        (1., .33, True), (1., .43, False), (1., .53, False), (1., .63, False),
        (1., .73, False), (1., .33, False), (.75, .53, False))
    headings = (-.27, 0., .27, -.55, .55, -.8, .8) if cable else (
        -.55, -.8, -1.08, -.27, 0., .27, .55, .8)
    cable_frames = hand.frames(hand.nominal_angles(cable_command(1.))) if cable else None
    yielded = 0
    attempts = 0
    for opposition, beta, centered in configurations:
        opened = CABLE_OPEN if cable else np.r_[np.ones(5), opposition]
        opening = support_open(hand, opened, beta, cable=cable)
        if opening is None:
            continue
        _, _, _, thumb, index, basis_hand = opening
        open_joints = hand.nominal_angles(opened)
        first_attempt = attempts
        for mid, direction, rim, tree, basis_table in centered_sites if centered else sites:
            if attempts - first_attempt >= 24:
                break
            orient = np.column_stack((direction, np.cross(up, direction), up)) @ basis_hand.T
            origin = mid - orient @ ((thumb + index) * .5) + CONTACT_GAP * up
            contacts = origin + np.array([thumb, index]) @ orient.T
            if np.min(contacts[:, 0]) < routes.edge + .012:
                continue
            if cable and np.linalg.norm(contacts[0] - center) > .051:
                continue
            if np.min(tree.query((contacts - center) @ basis_table)[0]) < (.005 if cable else .007):
                continue
            entry = origin + [0., 0., CABLE_SCOOP_HIGH] if cable else origin
            if hand.collision((surface - entry) @ orient, open_joints, margin=-.001):
                continue
            recenter = .022 * np.cross(direction, up) + .004 * direction if centered else None
            if cable:
                if not cable_grasp_contacts(hand, cable_frames, surface, origin, orient):
                    continue
                closing_options = ((cable_command(1.), None),)
            else:
                closing_options = close_candidates(hand, opened, surface, origin, orient,
                                                   routes.normal, routes.floor, up, rim, direction,
                                                   recenter=recenter)
            for closed in closing_options:
                shape = routes.shape(side, item['family'], closed[0], opened)
                for offset in headings:
                    yaw = yaw_center + offset
                    attempts += 1
                    if attempts > 90:
                        return
                    route = touch_route(routes, side, yaw, start, orient, origin, up, hand, opening, closed, cable,
                                        recenter=recenter, surface=surface)
                    if route is None:
                        continue
                    path, touch, q_close, solid, tips, command = route
                    carry_orientation = routes.local(orient, yaw)
                    grasp = arm.tool(q_close)[:3, 3]
                    lift = grasp.copy()
                    lift[2] = max(grasp[2] + .105, routes.highest + grasp[2] - bottom + .022)
                    q_lift = routes.fit(side, lift, carry_orientation, q_close)
                    if (q_lift is None or not surface_link(routes, side, q_close, q_lift, yaw, solid, tips)
                            or not routes.clear(side, q_lift, shape, yaw)):
                        continue
                    yield yaw, dict(approach=path, lift=q_lift, orient=carry_orientation, shape=shape,
                                    grip=command.tolist(), preshape=opened.tolist(),
                                    bottom_offset=grasp[2] - bottom, height=lift[2],
                                    point=routes.world(grasp, yaw), touch=touch)
                    yielded += 1
                    if yielded >= 8:
                        return

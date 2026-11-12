"""Plan empty-hand transitions from the frozen survey, before the batch starts."""
import numpy as np
from scipy.spatial import cKDTree

from .cable_bimanual import _section_hits_table
from .mouse_scoop import body_geometry


SIDES = ('left', 'right')


def empty_transition(routes, start, start_yaw, end, end_yaw, commands, lower_body):
    """Return a simultaneous arm/waist bridge, or retain the original exit.

    Grasp approaches and contact poses are outside this function. Both hands
    are empty; their opened and next preshape meshes bound the transition.
    """
    torso, torso_low, torso_high, links = body_geometry()
    shapes = {side: np.concatenate([
        np.concatenate(list(routes.hands[side].vertices(
            routes.hands[side].nominal_angles(command)).values()))
        for command in commands[side]]) for side in SIDES}
    up = routes.normal / np.linalg.norm(routes.normal)
    floor = routes.floor / np.linalg.norm(routes.normal)
    obstacles = []
    for obj in routes.objects:
        lo, hi = np.asarray(obj['bounds'], dtype=float)
        if obj['kind'].startswith('tray_'):
            for axis in (0, 1):
                a, b = hi.copy(), lo.copy()
                a[axis] = lo[axis] + .012
                b[axis] = hi[axis] - .012
                obstacles.extend(((lo, a), (b, hi)))
        else:
            obstacles.append((lo, hi))

    def clear(pair, yaw):
        fore = routes.local(np.array([1., 0., 0.]), yaw)
        normal = routes.local(up, yaw)
        clouds = {}
        for side in SIDES:
            arm = routes.arms[side]
            frames = arm.frames(pair[side])
            hand = frames[-1] @ arm.hand
            parts = [mesh @ frame[:3, :3].T + frame[:3, 3]
                     for mesh, frame in zip(links[side], frames[2:])]
            parts.append(shapes[side] @ hand[:3, :3].T + hand[:3, 3])
            if any(_section_hits_table(part, fore, normal, floor,
                                       routes.edge - .003, False) for part in parts):
                return False
            cloud = np.concatenate(parts)
            near = cloud[np.all((cloud >= torso_low) & (cloud <= torso_high), axis=1)]
            if len(near) and np.any((near @ torso[:, :3].T + torso[:, 3]).max(1) < .002):
                return False
            if lower_body.collides(cloud, yaw, clearance=.008):
                return False
            world = routes.world(cloud.T, yaw).T
            if any(np.any(np.all((world >= lo - .006) & (world <= hi + .006), axis=1))
                   for lo, hi in obstacles):
                return False
            clouds[side] = cloud
        return not np.any(cKDTree(clouds['left']).query(
            clouds['right'], distance_upper_bound=.008)[0] < .008)

    delta = max(abs(end_yaw - start_yaw),
                max(float(np.max(abs(end[side] - start[side]))) for side in SIDES))
    count = max(3, int(delta / .035) + 2)
    if not all(clear({side: (1. - f) * start[side] + f * end[side] for side in SIDES},
                     (1. - f) * start_yaw + f * end_yaw)
               for f in np.linspace(0., 1., count)):
        return None
    # Intermediate targets bound tracking errors while avoiding a stop there.
    return [dict(q={side: (1. - f) * start[side] + f * end[side] for side in SIDES},
                 yaw=(1. - f) * start_yaw + f * end_yaw)
            for f in np.linspace(0., 1., max(2, int(delta / .18) + 2))[1:]]

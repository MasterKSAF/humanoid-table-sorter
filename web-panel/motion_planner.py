"""Bounded OMPL search over the measured scene. Never publishes robot commands.

Uses the existing sampled arm envelope, NOT the unavailable full robot meshes.
All returned edges are independently checked after OMPL at <= .015 rad spacing.
"""
import time
import numpy as np
from scipy.spatial import cKDTree
from team22_bringup.grasp import Arm
from team22_bringup.posture import preparation_routes, ready_angles


STEP = .015
MARGIN = .002


def edge_samples(a, b, step=STEP):
    return np.linspace(a, b, max(2, int(np.ceil(np.max(abs(np.asarray(b)-a))/step))+1))


def rrt_connect(start, goal, limits, valid, seconds=3.):
    """Return only an exact, revalidated path; missing OMPL is a hard failure."""
    from ompl import base as ob, geometric as og
    begin = time.monotonic()
    start, goal, limits = map(np.asarray, (start, goal, limits))
    n = len(start)
    if (not np.isfinite([*start, *goal, *limits.ravel()]).all()
            or np.any(start < limits[0]) or np.any(start > limits[1])
            or np.any(goal < limits[0]) or np.any(goal > limits[1])
            or not valid(start) or not valid(goal)):
        raise ValueError('Начало или цель маршрута недопустимы')
    if all(valid(q) for q in edge_samples(start, goal)):
        return [start.copy(), goal.copy()]
    space = ob.RealVectorStateSpace(n)
    bounds = ob.RealVectorBounds(n)
    for i in range(n):
        bounds.setLow(i, float(limits[0, i])); bounds.setHigh(i, float(limits[1, i]))
    space.setBounds(bounds)
    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(lambda s: bool(valid(np.array([s[i] for i in range(n)]))))
    # OMPL measures Euclidean length; this also bounds every joint increment.
    ss.getSpaceInformation().setStateValidityCheckingResolution(STEP/space.getMaximumExtent())
    a, b = space.allocState(), space.allocState()
    for i in range(n): a[i], b[i] = float(start[i]), float(goal[i])
    ss.setStartAndGoalStates(a, b)
    planner = og.RRTConnect(ss.getSpaceInformation()); planner.setRange(.22)
    ss.setPlanner(planner)
    # Mesh validation before solve can exceed this search budget on the simulator
    # host. Give the sampling search its own bounded time, not the leftover 1 ms.
    ss.solve(seconds)
    if not ss.haveExactSolutionPath():
        raise ValueError('За отведённое время полный обход не найден')
    path = ss.getSolutionPath()
    states = [np.array([path.getState(j)[i] for i in range(n)]) for j in range(path.getStateCount())]
    # Deterministic shortcut with the SAME dense checker. No unbounded simplify().
    shortcut_deadline=time.monotonic()+min(seconds,6.)
    result = [states[0]]; index = 0
    while index < len(states)-1:
        next_index = index+1
        if time.monotonic()<shortcut_deadline:
            for candidate in range(len(states)-1, index, -1):
                if all(valid(q) for q in edge_samples(states[index], states[candidate])):
                    next_index = candidate; break
                if time.monotonic()>=shortcut_deadline: break
        result.append(states[next_index]); index = next_index
    if not np.allclose(result[-1], goal, atol=1e-8):
        raise ValueError('План не достигает требуемой цели')
    if not all(valid(q) for a, b in zip(result, result[1:]) for q in edge_samples(a, b)):
        raise ValueError('Обход не прошёл независимую проверку')
    return result


def witness(model, q, table, other_model, other_q):
    points, radii = model.samples(q)
    plane, edge, tree = table
    dist, index = tree.query(points)
    gap = dist-radii; at = int(np.argmin(gap))
    best = dict(kind='obstacle', gap_mm=float(gap[at]*1000),
                robot_xyz=points[at].tolist(), obstacle_xyz=tree.data[index[at]].tolist())
    over = points[:, 0] >= edge-radii
    height = points[:, 2]-points[:, :2]@plane[:2]-plane[2]-radii
    height[~over] = np.inf; at = int(np.argmin(height))
    if height[at]*1000 < best['gap_mm']:
        p = points[at].copy(); p[2] = p[:2]@plane[:2]+plane[2]
        best = dict(kind='table', gap_mm=float(height[at]*1000), robot_xyz=points[at].tolist(), obstacle_xyz=p.tolist())
    parked, _ = other_model.samples(other_q)
    dist, index = cKDTree(parked).query(points); gap = dist-radii-.038; at = int(np.argmin(gap))
    if gap[at]*1000 < best['gap_mm']:
        best = dict(kind='other_arm', gap_mm=float(gap[at]*1000), robot_xyz=points[at].tolist(), obstacle_xyz=parked[index[at]].tolist())
    return best


def skeleton(model, q):
    return [f[:3, 3].tolist() for f in model.frames(q)]+[(model.wrist(q)@model.hand)[:3, 3].tolist(), model.tool(q)[:3, 3].tolist()]


def preview(models, initial, route, table, blocked=None):
    poses = {s: np.array(initial[s]) for s in models}
    frames = [{s: skeleton(m, poses[s]) for s, m in models.items()}]
    for side, _, goal in route:
        for q in edge_samples(poses[side], np.asarray(goal), .08)[1:]:
            poses[side] = q
            frames.append({s: skeleton(m, poses[s]) for s, m in models.items()})
    if blocked and blocked.get('frame'): frames.append(blocked['frame'])
    points = table[2].data
    return dict(frames=frames, obstacles=points[::max(1, int(np.ceil(len(points)/900)))].tolist(),
                plane=np.asarray(table[0]).tolist(), edge=float(table[1]), blocked=blocked,
                model='sampled_arm_envelopes', margin_mm=MARGIN*1000,
                warning='Расчёт по упрощённым объёмам рук; полная геометрия корпуса и пальцев не подтверждена.')


def plan_ready(packet):
    from transfer_trial import body_clear
    begin = time.monotonic()
    models = {s: Arm(s) for s in ('left', 'right')}
    initial = {s: np.array([packet['angles'][k] for k in m.names]) for s, m in models.items()}
    table = np.asarray(packet['plane']), float(packet['edge']), cKDTree(packet['points'])
    deadline = begin+float(packet.get('budget_s', 8.))
    failures = []; result = dict(stamp=packet['stamp'], start=packet['angles'], route=[], planner='validated_templates')
    sides=tuple(packet.get('sides',('left','right')))
    if not sides or len(set(sides))!=len(sides) or any(s not in models for s in sides):
        raise ValueError('Недопустимый набор рук для подготовки')
    orders=(sides,tuple(reversed(sides))) if len(sides)>1 else (sides,)
    for order in orders:
        future = {s: q.copy() for s, q in initial.items()}; route = []; blocked = None
        for side in order:
            model = models[side]; opposite = 'right' if side == 'left' else 'left'
            other = cKDTree(models[opposite].samples(future[opposite])[0])
            def valid(q):
                return bool(np.isfinite(q).all() and np.all(q >= model.limits[0]+.035)
                    and np.all(q <= model.limits[1]-.035) and body_clear(model, q)
                    and model.clearance(q, table, other) >= MARGIN)
            accepted = None
            # Known safe templates are inexpensive; the fallback searches new routes.
            for candidate in preparation_routes(side, future[side], packet.get('repeat', False)):
                current = future[side]
                for label, goal in candidate:
                    bad = next((q for q in edge_samples(current, goal) if not valid(q)), None)
                    if bad is not None:
                        blocked = dict(side=side, stage=label, angles=bad.tolist(),
                            **witness(model, bad, table, models[opposite], future[opposite]))
                        if not body_clear(model, bad):
                            blocked['stage'] += ': пересечение корпуса'
                        blocked['frame'] = {s: skeleton(m, bad if s == side else future[s]) for s, m in models.items()}
                        break
                    current = goal
                else: accepted = candidate; break
            if accepted is None:
                if time.monotonic() >= deadline:
                    failures.append('Исчерпан бюджет поиска'); break
                try:
                    path = rrt_connect(future[side], ready_angles(side), model.limits+np.array([[.035], [-.035]]),
                                       valid, max(.05, (deadline-time.monotonic())/2))
                    accepted = [(f'Обход препятствия {i}', q) for i, q in enumerate(path[1:], 1)]
                    result['planner'] = 'OMPL RRTConnect'
                except (ValueError, ImportError) as error:
                    failures.append(f'{side}: {error}'); break
            route.extend((side, label, np.asarray(q).tolist()) for label, q in accepted)
            future[side] = np.asarray(accepted[-1][1])
        else:
            result['route'] = route
            result['preview'] = preview(models, initial, route, table)
            result['planning_ms'] = (time.monotonic()-begin)*1000
            return result
    result['error'] = '; '.join(failures) or 'Полный маршрут обеих рук не найден'
    result['preview'] = preview(models, initial, [], table, blocked)
    result['planning_ms'] = (time.monotonic()-begin)*1000
    return result


def start_matches(result, angles, tolerance=.009):
    # Also bind to the fixed body and head orientation used during the scan.
    return all(k in angles and np.isfinite(angles[k]) and abs(angles[k]-v) <= tolerance
               for k, v in result['start'].items() if '_joint' in k)

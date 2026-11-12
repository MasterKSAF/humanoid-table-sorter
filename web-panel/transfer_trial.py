"""Bounded diagnostic pick/place trial. Uses camera geometry, never scene truth.

This is experimental lab code, not a competition entry. An unsuccessful grasp
is opened at its source; transport requires both finger and visual evidence.
"""
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from team22_bringup.grasp import Arm, CLEARANCE_M
from team22_bringup.scene_geometry import planning_edge

KINDS = ('cable', 'gold_jar', 'mouse', 'blue_jar', 'drive', 'lotion')
NAMES = {
    'lotion': 'Marc_Anthony_True_Professional_Strictly_Curls_Curl_Defining_Lotion',
    'gold_jar': 'Wilton_Pearlized_Sugar_Sprinkles_525_oz_Gold',
    'cable': 'Ubisoft_RockSmith_Real_Tone_Cable_Xbox_360',
    'mouse': 'Razer_Taipan_Black_Ambidextrous_Gaming_Mouse',
    'blue_jar': 'Beta_Glucan', 'drive': 'Deskstar_Desk_Top_Hard_Drive_1_TB'}
from team22_bringup.table_vision import CATALOG
NAMES.update({k:name for k,(_,name) in CATALOG.items() if name})
KINDS=('lotion','gold_jar','blue_jar','drive','mouse','cable','figure_lion','figure_unicorn',
       'figure_eagle','white_mouse','gelatin_box','alarm_clock','green_speaker')
CLOSE = np.array([.25, .25, .25, .25, .25, .45])
CLOSE[4] = float(os.environ.get('TEAM22_THUMB_CLOSE','.25'))
if not .05 <= CLOSE[4] <= .30:raise ValueError('Invalid thumb closing target')
CYLINDER_CLOSE = np.array([.25, .25, .25, .25, .10, .45])
OPERATOR_PRESHAPE=None
if os.environ.get('TEAM22_OPERATOR_CYLINDER')=='1':
    from operator_preshape import cylinder_ready
    OPERATOR_PRESHAPE=cylinder_ready()
    if OPERATOR_PRESHAPE is None:raise ValueError('Operator preshape is missing')
CYLINDER_APPROACH=(list(OPERATOR_PRESHAPE['command']) if OPERATOR_PRESHAPE else
                   [.9512,.9501,.9455,.9591,.5842,.0593])
HAND_FORCE=float(os.environ.get('TEAM22_HAND_FORCE','.45'))
HAND_SPEED=float(os.environ.get('TEAM22_HAND_SPEED','1.0'))
if not .25<=HAND_FORCE<=.5 or not .05<=HAND_SPEED<=1.:raise ValueError('Invalid hand settings')
ARM_SPEED=float(os.environ.get('TEAM22_ARM_SPEED','.8'))
if not np.isfinite(ARM_SPEED) or not .05<=ARM_SPEED<=.8:raise ValueError('Invalid arm speed')
DIRECT_START_MODE=os.environ.get('TEAM22_DIRECT_START','auto').strip().lower()
if DIRECT_START_MODE not in ('auto','0','1'):raise ValueError('Invalid direct-start mode')

def finger_settlement(history):
    """Use a time window of measured positions, not a noisy instantaneous velocity."""
    if not history:return dict(settled=False)
    end=history[-1][0];window=[s for s in history if end-s[0]<=.55]
    if len(window)<6 or end-window[0][0]<.45:return dict(settled=False)
    positions=np.asarray([s[2]['position'][:6] for s in window]);times=np.array([s[0] for s in window])
    spread=np.percentile(positions,90,axis=0)-np.percentile(positions,10,axis=0)
    a=times<=times[0]+.15;b=times>=times[-1]-.15
    drift=(np.median(positions[b],axis=0)-np.median(positions[a],axis=0))/(np.mean(times[b])-np.mean(times[a]))
    return dict(settled=bool(np.max(spread)<.05 and np.max(abs(drift))<.05),
                position_spread=spread.tolist(),position_drift=drift.tolist(),window_s=float(times[-1]-times[0]))


def destination(task, kind):
    category = task['objects'][NAMES[kind]]
    return 'tray_' + task['trays'][task['categories'][category]]


def access_blockers(scene,kinds):
    """Find near-side objects whose measured footprints overlap others' approach lanes."""
    visible={o['kind']:o for o in scene.get('objects',[])
             if o.get('kind') in kinds and o.get('status')=='observed'}
    scores={kind:0. for kind in visible};blocked_by={kind:[] for kind in visible}
    for kind,obj in visible.items():
        a=np.asarray(obj.get('bounds',()),float)
        if a.shape!=(2,3) or not np.all(np.isfinite(a)):continue
        center=np.asarray(obj['xyz'],float);half=np.ptp(a,axis=0)[:2]/2
        for other_kind,other in visible.items():
            if other_kind==kind:continue
            b=np.asarray(other.get('bounds',()),float)
            if b.shape!=(2,3) or not np.all(np.isfinite(b)):continue
            other_center=np.asarray(other['xyz'],float);other_half=np.ptp(b,axis=0)[:2]/2
            dx=float(other_center[0]-center[0]);dy=abs(float(other_center[1]-center[1]))
            lane=half[1]+other_half[1]+.035-dy
            if .015<dx<.32 and lane>0:
                scores[kind]+=lane*(.32-dx)/.32
                blocked_by[kind].append(other_kind)
            overlap_x=min(a[1,0],b[1,0])-max(a[0,0],b[0,0])
            overlap_y=min(a[1,1],b[1,1])-max(a[0,1],b[0,1])
            if dx>0 and overlap_x>0 and overlap_y>0:
                scores[kind]+=.025+min(overlap_x,overlap_y)
                if other_kind not in blocked_by[kind]:blocked_by[kind].append(other_kind)
    return scores,blocked_by


def body_clear(model, angles):
    samples, radii = model.samples(angles)
    # The first two segments attach inside the torso and must not collide with
    # their own mounting volume. Check the upper arm and all more distal links.
    samples, radii = samples[32:], radii[32:]
    return not np.any((samples[:, 0] < .10+radii) &
                      (samples[:, 0] > -.19-radii) &
                      (abs(samples[:, 1]) < .17+radii) &
                      (samples[:, 2] > -.22-radii) & (samples[:, 2] < .43+radii))


def checked(model, start, goal, table, other, minimum=CLEARANCE_M):
    if not model.path_clear(start, goal, table, other, minimum):
        return False
    return all(body_clear(model, q) for q in np.linspace(start, goal,
               int(np.ceil(np.max(abs(goal-start))/.04))+1))


def validate_preshape(current, angles, minimum):
    """Keep the multi-mesh finger sweep out of the ROS executor."""
    from grasp_shape_checks import closure_clear
    model=Arm(current['side'])
    table=np.array(current['plane']),current.get('table_edge',-100.),cKDTree(current['obstacles'])
    return closure_clear(model,[angles[n] for n in model.names],table,angles,
                         current['approach_preshape'],minimum=minimum)


def validate_measured_route(current, angles, path):
    """Expensive mesh checks run outside the ROS sensor/command callbacks."""
    side=current['side'];model=Arm(side)
    if current.get('pocket') is not None:model.pocket=np.array(current['pocket'])
    model.extra_hand_points=current.get('extra_hand_points',[])
    if current.get('grasp')=='geometry_pinch' or current.get('palm_roll') or current.get('operator_preshape') or (current.get('grasp')=='cylinder_side' and current['kind']!='gold_jar'):
        from team22_bringup.hand_geometry import HandGeometry
        model.hand_geometry=HandGeometry(side);model.finger_angles=angles
    parked=Arm('left' if side=='right' else 'right')
    other=cKDTree(parked.samples([angles[k] for k in parked.names])[0])
    table=np.array(current['plane']),current.get('table_edge',-100.),cKDTree(current['obstacles'])
    seed=np.array([angles[k] for k in model.names])
    for waypoint in path:
        goal=np.asarray(waypoint)
        if current.get('thin_object') or current.get('hand_object_margin') is not None:
            from grasp_shape_checks import checked_hand_table
            valid=checked_hand_table(model,seed,goal,table,other,hand_minimum=CLEARANCE_M,
                                     arm_minimum=CLEARANCE_M,hand_object_minimum=CLEARANCE_M)
        else:valid=checked(model,seed,goal,table,other,current.get('clearance_m',CLEARANCE_M))
        if not valid:return False
        seed=goal
    return True


def plan_carried_lotion_route(current, angles, obj, tray, points, plane, table_edge, offset, held_cloud=None):
    """Plan a fresh, checked carry route from the measured held pose to a live tray."""
    from side_grasp import solve_side
    from tray_placement import release_sites

    bounds=np.asarray(obj.get('bounds'),float)
    if bounds.shape!=(2,3) or not np.all(np.isfinite(bounds)):
        return dict(error='Нет свежих границ удерживаемого флакона')
    points=np.asarray(points,float).reshape((-1,3))
    if len(points)<30:return dict(error='Недостаточно точек для карты переноса')
    margin=float(current.get('clearance_m',CLEARANCE_M))
    # Remove only the held bottle cloud; arm links were already self-filtered by
    # the RGB-D map builder and remain collision checked by the arm model.
    held=np.all((points>=bounds[0]-.016)&(points<=bounds[1]+.016),axis=1)
    bbox_removed=int(np.count_nonzero(held))
    points=points[~held]
    cloud_removed=0
    if held_cloud is not None:
        held_cloud=np.asarray(held_cloud,float).reshape((-1,3))
        if len(held_cloud):
            distance,_=cKDTree(held_cloud).query(points)
            cloud_removed=int(np.count_nonzero(distance<=.012))
            points=points[distance>.012]
    if len(points)<30:return dict(error='После маски флакона карта стала слишком разреженной')
    sites=release_sites(obj,tray,points)
    if not sites:return dict(error='В текущем лотке нет подтверждённого свободного места')

    side=current['side'];model=Arm(side)
    if current.get('pocket') is not None:model.pocket=np.asarray(current['pocket'],float)
    model.extra_hand_points=current.get('extra_hand_points',[])
    from team22_bringup.hand_geometry import HandGeometry
    model.hand_geometry=HandGeometry(side);model.finger_angles=angles
    other_model=Arm('left' if side=='right' else 'right')
    other=cKDTree(other_model.samples([angles[k] for k in other_model.names])[0])
    table=(np.asarray(plane,float),float(table_edge),cKDTree(points))
    start=np.asarray([angles[k] for k in model.names],float)
    start_pose=model.tool(start);rotation=start_pose[:3,:3]
    offset=np.asarray(offset,float)
    center=start_pose[:3,3]+rotation@offset
    object_center=np.asarray(obj['xyz'],float)
    # Binding is refreshed from the current camera pair immediately before
    # planning. Use its measured vertical offset within the bottle envelope.
    bottom_offset=float(object_center[2]-bounds[0,2])
    bottom_offset=float(np.clip(bottom_offset,.025,max(.025,bounds[1,2]-bounds[0,2])))
    radius=float(max(np.ptp(bounds,axis=0)[:2])/2+margin)
    tray_bounds=np.asarray(tray['bounds'],float)
    high=float(max(center[2]+.08,np.max(points[:,2])+bottom_offset+margin))
    failures=[]
    for site in sites:
        release=np.asarray(site,float);release[2]=tray_bounds[0,2]+bottom_offset+margin
        centers=[center.copy()];lift=center.copy();lift[2]=high;centers.append(lift)
        for fraction in (.25,.5,.75,1.):
            waypoint=lift*(1.-fraction)+np.array([release[0],release[1],high])*fraction
            centers.append(waypoint)
        centers.append(release)
        route=[];seed=start.copy();failure=None
        for index,goal_center in enumerate(centers[1:]):
            tool_goal=goal_center-rotation@offset
            fitted=solve_side(model,tool_goal,rotation,seed)
            if fitted is None:
                failure=f'IK не нашёл позу сегмента {index}';break
            count=int(np.ceil(np.max(abs(fitted-seed))/.04))+1
            sweep=np.linspace(seed,fitted,count)
            clearances=np.asarray([model.clearances(q,table,other) for q in sweep])
            limits_ok=bool(np.all((fitted>=model.limits[0]+.03)&(fitted<=model.limits[1]-.03)))
            arms_clear=bool(limits_ok and np.all(clearances>margin))
            body_ok=all(body_clear(model,q) for q in sweep)
            if not arms_clear or not body_ok:
                minimum=np.min(clearances,axis=0)
                failure=(f'запас руки на сегменте {index}: старт {model.clearances(seed,table,other)}, '
                         f'минимум {minimum.tolist()}, конец {model.clearances(fitted,table,other)}, '
                         f'запас {margin*1000:.1f} мм, корпус {body_ok}')
                break
            for sample_index,q in enumerate(np.linspace(seed,fitted,15)):
                pose=model.tool(q)
                carried=pose[:3,3]+pose[:3,:3]@offset
                near=points[np.linalg.norm(points[:,:2]-carried[:2],axis=1)<radius]
                if len(near):
                    obstacle_top=float(near[:,2].max())
                    gap=float(carried[2]-bottom_offset-obstacle_top-margin)
                    if gap<0:
                        blocker=near[int(np.argmax(near[:,2]))]
                        outside=np.maximum(np.asarray(bounds[0]-.016)-blocker,
                                           blocker-np.asarray(bounds[1]+.016))
                        failure=(f'недостаточный зазор флакона на сегменте {index}: '
                                 f'образец {sample_index}, низ {carried[2]-bottom_offset:.3f} м, '
                                 f'поверхность препятствия {obstacle_top:.3f} м, запас {gap*1000:.1f} мм, '
                                 f'точка {blocker.tolist()}, вне маски {np.maximum(outside,0).tolist()}')
                        break
            if failure:break
            route.append(fitted.tolist());seed=fitted
        if not failure:
            return dict(route=route,release_site=np.asarray(site,float).tolist(),
                        release_center=release.tolist(),high=high,
                        obstacle_count=int(len(points)),obstacles=points.tolist(),
                        plane=np.asarray(plane,float).tolist(),table_edge=float(table_edge),failures=failures)
        failures.append(dict(site=np.asarray(site,float).tolist(),reason=failure,
                             held_bbox_points_removed=bbox_removed,
                             held_rgbd_points_removed=cloud_removed))
        # The first segment is the same vertical egress for every tray site.
        # Avoid repeating the same expensive IK and collision sweep nine times.
        if index==0:
            return dict(error='Начальный подъём не прошёл проверку',failures=failures)
    return dict(error='Безопасный маршрут до текущего лотка не найден',failures=failures)


def solve(model, goal, rotation, seed):
    def residual(q):
        pose = model.tool(q)
        return np.r_[pose[:3, 3]-goal,
                     .15*Rotation.from_matrix(rotation.T@pose[:3, :3]).as_rotvec(),
                     .0015*(q-seed)]
    fit = least_squares(residual, seed, bounds=(model.limits[0]+.04, model.limits[1]-.04), max_nfev=100)
    error = residual(fit.x)
    if np.linalg.norm(error[:3]) > .004 or np.linalg.norm(error[3:6]) > .0075:
        return None
    return fit.x


def plan_once(packet):
    if packet['kind']=='tube_cap':
        from tube_grasp import plan_cap
        return plan_cap(packet)
    if packet.get('release_sites') is not None and not packet['release_sites']:
        return dict(kind=packet['kind'],error='В целевом лотке не нашлось свободной точки с учётом габарита предмета')
    if packet.get('profile') in ('dual','complete') and packet['kind'] in ('drive','green_speaker'):
        from bimanual import plan_two_hands
        return plan_two_hands(packet)
    if packet['kind'] in ('gold_jar','blue_jar','lotion'):
        from side_grasp import plan_cylinder_candidates
        return plan_cylinder_candidates(packet)
    if packet.get('profile')=='complete':
        from geometry_grasp import plan_geometry_candidates
        return plan_geometry_candidates(packet)
    started = time.monotonic()
    scene, kind, angles = packet['scene'], packet['kind'], packet['angles']
    objects = {o['kind']: o for o in scene['objects'] if o.get('status') == 'observed'}
    if kind not in objects:
        return dict(error='Предмет не виден сейчас', kind=kind)
    obj = objects[kind]
    tray_kind = destination(packet['task'], kind)
    if tray_kind not in objects:
        return dict(error='Не виден нужный лоток', kind=kind)
    source_center = np.array(obj['xyz'], float)
    target = source_center.copy()
    bounds = np.array(obj['bounds'])
    # Contact pocket sits below the visible upper surface, not at a hidden base.
    target[2] = bounds[1, 2] - min(.025, (bounds[1, 2]-bounds[0, 2])*.25)
    side = 'left' if target[1] > .12 else 'right'
    other_side = 'right' if side == 'left' else 'left'
    model, parked = Arm(side), Arm(other_side)
    start = np.array([angles[k] for k in model.names])
    other = cKDTree(parked.samples([angles[k] for k in parked.names])[0])
    points = np.array(packet['obstacles'])
    inside = np.all((points >= bounds[0]-.012) & (points <= bounds[1]+.012), axis=1)
    points = points[~inside]
    table = np.array(scene['plane']), planning_edge(scene), cKDTree(points)
    failures = dict(ik=0, collision=0)
    selected = None
    for yaw in (-1.25, -.65, 0., .65, 1.25, -1.65, 1.65, -2., 2., -2.45, 2.45, np.pi):
        fingers = np.array([np.cos(yaw), np.sin(yaw)*(1 if side=='left' else -1), 0.])
        palm = np.array([0., 0., -1.])
        rotation = np.column_stack((palm, np.cross(fingers, palm), fingers))
        seed = start.copy(); route = []
        for height in (.20, .12, .06, .04, 0.):
            fitted = solve(model, target+[0, 0, height], rotation, seed)
            if fitted is None:
                failures['ik'] += 1; break
            if not checked(model, seed, fitted, table, other):
                failures['collision'] += 1; break
            route.append(fitted); seed = fitted
        if len(route) == 5:
            selected = route, rotation, yaw; break
    if selected is None:
        return dict(error='Нет подхода с запасом до стола, соседей, корпуса и пределов суставов',
                    kind=kind, failures=failures, planning_ms=1000*(time.monotonic()-started))
    approach, rotation, yaw = selected
    tray = objects[tray_kind]
    end = np.asarray(packet.get('release_site',tray['xyz']),float).copy()
    bottom_offset = float(source_center[2]-bounds[0, 2])
    high_z = max(float(source_center[2]+.16),float(points[:,2].max())+bottom_offset+CLEARANCE_M)
    release = end.copy(); release[2] = np.array(tray['bounds'])[0, 2]+bottom_offset+CLEARANCE_M
    goals = [np.array([source_center[0],source_center[1],high_z])]
    goals += [goals[0]*(1-t)+np.array([end[0],end[1],high_z])*t for t in (.25,.5,.75,1.)]
    goals.append(release)
    transfer_reason = None
    payload_radius = np.linalg.norm((bounds[1]-bounds[0])[:2])/2+CLEARANCE_M
    for end_yaw in (1.25, .65, 0., -1.25, 2.):
        transport = []; seed = approach[1]
        transfer_reason = None
        object_offset = rotation.T@(source_center-target)
        for index, goal in enumerate(goals):
            fraction = min(index/4, 1.)
            heading = yaw+(end_yaw-yaw)*fraction
            fingers = np.array([np.cos(heading), np.sin(heading)*(1 if side=='left' else -1), 0.])
            orient = np.column_stack((palm, np.cross(fingers, palm), fingers))
            fitted = solve(model, goal-orient@object_offset, orient, seed)
            if fitted is None or not checked(model, seed, fitted, table, other):
                transfer_reason = 'Нет проверенного пути этой рукой в назначенный лоток; нужен другой хват или передача'; break
            # Check the carried object's conservative envelope along the entire
            # joint interpolation, not only at waypoint endpoints.
            if index < len(goals)-1:
                for intermediate in np.linspace(seed, fitted, 15):
                    pose=model.tool(intermediate)
                    carried = pose[:3, 3]+pose[:3, :3]@object_offset
                    neighbors = points[np.linalg.norm(points[:, :2]-carried[:2], axis=1)<payload_radius]
                    if len(neighbors) and carried[2]-bottom_offset < neighbors[:, 2].max()+CLEARANCE_M:
                        transfer_reason = 'Недостаточный зазор переносимого предмета над препятствиями'; break
                if transfer_reason: break
            transport.append(fitted); seed = fitted
        if not transfer_reason: break
    if transfer_reason:
        transport = []
    return dict(kind=kind, side=side, target=target.tolist(), original=obj,clearance_m=CLEARANCE_M,
                tray=tray_kind, tray_observation=tray,release_site=end.tolist(),release_center=release.tolist(),yaw=yaw,
                approach=[q.tolist() for q in approach], transport=[q.tolist() for q in transport],
                transfer_reason=transfer_reason, obstacles=points.tolist(), plane=list(scene['plane']),table_edge=table[1],
                start=start.tolist(), failures=failures, planning_ms=1000*(time.monotonic()-started))


def remove_movable_obstacles(packet):
    """Keep the table and tray collision map, but allow pushing other task items."""
    from mission_scene import observed_objects
    scene_objects=observed_objects(packet)
    points=np.asarray(packet['obstacles'],float)
    keep=np.ones(len(points),dtype=bool);ignored=[]
    for kind in KINDS:
        if kind==packet['kind']:continue
        obj=scene_objects.get(kind)
        if obj is None:continue
        bounds=np.asarray(obj.get('bounds',()),float)
        if bounds.shape!=(2,3) or not np.all(np.isfinite(bounds)):continue
        inside=np.all((points>=bounds[0]-.025)&(points<=bounds[1]+.025),axis=1)
        if np.any(inside):
            keep &= ~inside;ignored.append(kind)
    relaxed=dict(packet,obstacles=points[keep].tolist(),allow_object_interference=False,
                 ignored_object_obstacles=ignored)
    if packet.get('release_sites') is not None:
        from tray_placement import release_sites
        kind=packet['kind'];tray_kind=destination(packet['task'],kind)
        obj=scene_objects.get(kind);tray=scene_objects.get(tray_kind)
        sites=release_sites(obj,tray,relaxed['obstacles']) if obj is not None and tray is not None else []
        relaxed['release_sites']=sites
        relaxed['release_site']=sites[0] if sites else None
    return relaxed


def plan(packet):
    kind=packet.get('kind')
    if kind in ('gold_jar','blue_jar','lotion'):
        packet=dict(packet)
        # The gold jar has a proven side-wrap grasp, but its approach often
        # needs extra yaw/IK candidates. Keep the shared collision deadline
        # generous enough to reach the remembered candidate and alternatives.
        packet.setdefault('_deadline',time.monotonic()+(36. if kind=='gold_jar' else 22.))
    result=plan_once(packet)
    failed=bool(result.get('error') or result.get('transfer_reason') or
                ('transport' in result and not result.get('transport')))
    force_relaxed=(packet.get('kind')=='mouse' and int(packet.get('retry_number',1))>1)
    if not packet.get('allow_object_interference') or (not failed and not force_relaxed):
        return result
    if packet.get('kind') in ('gold_jar','blue_jar','lotion'):
        reasons=[]
        for attempt in result.get('candidate_attempts',[]):
            reasons.extend(str(row[1]) for row in (attempt.get('failures') or [])
                           if isinstance(row,(list,tuple)) and len(row)>1)
        if reasons and all(reason.startswith('terminal IK') for reason in reasons):
            result['object_interference_fallback_skipped']='Все варианты конечной IK-позы недостижимы; повтор с удалёнными предметами результата не изменит.'
            return result
    relaxed=remove_movable_obstacles(packet)
    if not relaxed.get('ignored_object_obstacles'):
        return result
    if packet.get('kind') in ('gold_jar','blue_jar','lotion'):
        # One short, separately bounded retry can help only when a movable
        # object intersects the route; terminal IK failures already returned.
        relaxed['_deadline']=time.monotonic()+10.
    retry=plan_once(relaxed)
    retry_failed=bool(retry.get('error') or retry.get('transfer_reason') or
                      ('transport' in retry and not retry.get('transport')))
    if not retry_failed:
        retry['object_interference_fallback']=True
        retry['ignored_object_obstacles']=relaxed['ignored_object_obstacles']
        return retry
    if failed:
        result['object_interference_fallback_failures']=retry.get('failures',retry.get('error'))
        result['ignored_object_obstacles']=relaxed['ignored_object_obstacles']
    else:
        result['object_interference_fallback_attempted']=True
        result['object_interference_fallback_failures']=retry.get('failures',retry.get('error'))
        result['ignored_object_obstacles']=relaxed['ignored_object_obstacles']
    return result


def warm_cylinder_planner():
    """Load the cylinder planner dependency graph off the ROS tick thread."""
    import importlib
    for module in ('mission_scene', 'tray_placement', 'cylinder_candidates',
                   'grasp_shape_checks', 'geometry_grasp', 'side_grasp',
                   'team22_bringup.grasp_priors', 'team22_bringup.hand_geometry'):
        importlib.import_module(module)


class TransferTrial:
    def __init__(self, lab):
        self.lab = lab
        # Keep planning off the ROS tick thread. Warm lazy cylinder imports in
        # this worker before the first attempt, without process startup/IPC cost.
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='transfer-plan')
        self.planner_warmup_started = time.monotonic()
        self.planner_warmup_ms = None
        self.planner_warmup = self.pool.submit(warm_cylinder_planner)
        self.planner_ready = False
        self.wait_for_scene_reload = os.environ.get('TEAM22_TRIAL_WAIT_FOR_RELOAD') == '1'
        self.scene_reload_seen = False
        self.future = None
        self.plan_submit_wall = None
        self.planning_close_command=None
        self.plan_preshape_targets=[]
        self.stage = 'prepare'
        self.queue = deque(k for k in KINDS if k not in os.environ.get('TEAM22_TRIAL_SKIP','').split(','))
        self.current = None
        self.events = []
        self.entered = 0.
        self.wall_start = time.monotonic()
        self.max_wall = float(os.environ.get('TEAM22_TRIAL_MAX_WALL', '1250'))
        self.empty = {}
        self.confirmed = None
        self.result = None
        self.want_map = False
        self.attempt = 0
        self.tracker = None
        self.carry_scan_requested=False;self.carry_scan_started=None
        self.carry_scan_samples={};self.carry_scan_reconciled=False
        self.carry_plan_future=None
        self.grip_pose = None
        self.grip_object_offset = None
        self.grip_reference_rotation = None
        self.bad_visual_since = None
        self.released_expected = None
        self.maintain_grip=False
        self.last_head_command=0.
        self.lift_visual_count=0
        self.lift_visual_stamp=None
        self.unsupported=[]
        self.deferred=[];self.scene_changed=False;self.retry_count={};self.released=False
        self.retrying_kind=None
        self.blockers={}
        self.scene_cache={};self.cached_scene_logged=False
        self.inspect_step=-1
        self.aligned_objects=set()
        self.alignment_pending_kind=None
        self.align_ready_requested=False
        self.recovery_open_started=None
        self.profile=os.environ.get('TEAM22_GRASP_PROFILE','legacy')
        self.direct_start_mode=DIRECT_START_MODE
        self.direct_start={'auto':None,'0':False,'1':True}[DIRECT_START_MODE]
        self.scan_available=False
        self.scan_stamp=None
        self.tube_pick=os.environ.get('TEAM22_TUBE_PICK')=='1'
        self.tube_reference=None
        resume=os.environ.get('TEAM22_TRIAL_RESUME')
        self.short_retreat=bool(resume)
        if resume:
            from pathlib import Path
            saved=json.loads(Path(resume).read_text())
            self.current=saved['plan'];self.empty=saved['empty'];self.grip_pose=np.array(saved['grip_pose'])
            if self.current.get('grasp')=='cylinder_side' and self.current.get('kind')=='lotion':
                from object_tracker import ObjectTracker
                self.tracker=ObjectTracker(self.current['original'],self.current['plane'])
                # Reacquire a fresh RGB-D baseline around the currently held
                # bottle; the saved pickup pose alone can drift during a stop.
                self.tracker.carrying=True
                self.tracker.resume_reacquire=True
                if saved.get('grip_object_offset') is not None:
                    self.grip_object_offset=np.asarray(saved['grip_object_offset'],float)
                if saved.get('grip_reference_rotation') is not None:
                    self.grip_reference_rotation=np.asarray(saved['grip_reference_rotation'],float)
                self.resume_baseline_bound=False
            else:
                from jar_tracker import JarTracker
                self.tracker=JarTracker(self.current['original'],self.current['plane'])
                self.tracker.baseline=saved['baseline']
            self.maintain_grip=True;self.stage='resume';self.attempt=1
            self.queue=deque(k for k in self.queue if k!=self.current.get('kind'))
            self.want_map=True
            self.lab.map_stamp=None;self.lab.map_geometry={};self.lab.scan_views=[]
            model=lab.controller.models[self.current['side']]
            model.pocket=np.array(self.current['pocket']);model.extra_hand_points=self.current['extra_hand_points']

    def event(self, now, message, **data):
        row = dict(sim=now, wall=time.monotonic(), stage=self.stage,
                   attempt=self.attempt, kind=self.current and self.current.get('kind'), message=message, **data)
        self.events.append(row)
        if self.lab.output:
            with (self.lab.output/'attempts.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
        self.lab.controller.stage = message
        self.lab.controller.note(now, message)

    def working_pose(self):
        """Return whether both measured arms and hands match the working pose."""
        samples={key:self.lab.timeline.latest(key) for key in ('joints','hand_left','hand_right')}
        if any(sample is None or time.monotonic()-sample[1]>1.5 for sample in samples.values()):return None
        from team22_bringup.posture import ready_angles, working_hand_command
        q=samples['joints'][2]
        for side in ('left','right'):
            model=self.lab.controller.models[side]
            arm=np.asarray([q.get(name,np.nan) for name in model.names],float)
            hand=np.asarray(samples['hand_'+side][2].get('position',()),float)
            desired=np.asarray(working_hand_command(side),float)
            if (not np.all(np.isfinite(arm)) or len(hand)!=6 or not np.all(np.isfinite(hand))
                    or np.max(abs(arm-ready_angles(side)))>.055
                    or np.max(abs(hand-desired))>.10):return False
        return True

    def requires_ready(self):
        task=self.lab.state.task or {}
        if not task:return False
        if getattr(self,'direct_start_mode','0')=='auto' and getattr(self,'direct_start',None) is None:
            measured=self.working_pose()
            if measured is not None:self.direct_start=measured
        if getattr(self,'direct_start',False):return False
        if getattr(self,'profile','legacy')=='complete' and task.get('task')=='tubes':return True
        return task.get('task')!='tubes' and any(NAMES[k] in (task.get('objects') or {}) for k in self.queue)

    def enter(self, stage, now, message, **data):
        self.stage, self.entered, self.confirmed = stage, now, None
        if stage=='next':
            self.tracker=None;self.grip_pose=None;self.grip_object_offset=None;self.grip_reference_rotation=None;self.released_expected=None
            if hasattr(self.lab,'center_head') and os.environ.get('TEAM22_FIXED_HEAD')!='1':self.lab.center_head()
        self.event(now, message, **data)

    def fingers(self, side, value):
        if np.all(np.array(value)[:5]>.9):self.maintain_grip=False
        from sensor_msgs.msg import JointState
        msg = JointState()
        msg.header.stamp = self.lab.node.get_clock().now().to_msg()
        msg.name, msg.position = [str(i) for i in range(1, 7)], list(map(float, value))
        self.lab.hand_pubs[side].publish(msg)

    @staticmethod
    def planning_preshape(kind):
        if kind in ('gold_jar','blue_jar','lotion'):
            target=(OPERATOR_PRESHAPE['command'] if kind=='lotion' and OPERATOR_PRESHAPE
                    else CYLINDER_APPROACH)
            # The planner may select either arm when the preferred arm cannot
            # reach the terminal pose. Prepare both so its chosen hand already
            # has the thumb abducted before the route is computed.
            return [('right',target),('left',target)]
        return None

    def move(self, path, now, stage, message):
        q=dict(self.lab.timeline.latest('joints')[2])
        path=[np.asarray(v).tolist() for v in path]
        self.pending_route=dict(path=path,angles=q,stage=stage,message=message)
        self.route_future=self.pool.submit(validate_measured_route,self.current,q,path)
        self.enter('route_validation',now,'Проверка маршрута по текущим датчикам: '+message)

    def prepare_fingers(self, now, minimum=CLEARANCE_M):
        minimum=self.current.get('clearance_m',minimum)
        self.preshape_angles=dict(self.lab.timeline.latest('joints')[2])
        self.preshape_future=self.pool.submit(validate_preshape,self.current,self.preshape_angles,minimum)
        self.enter('preshape_validation',now,'Проверяю свободное место для изменения формы кисти')

    def start_validated_route(self, now):
        c=self.lab.controller;pending=self.pending_route;side=self.current['side']
        from readiness import NO_PROGRESS_TIMEOUT
        c.progress_timeout=8. if pending['stage']=='retreat' else NO_PROGRESS_TIMEOUT
        q=self.lab.timeline.latest('joints')[2]
        arm_keys=[k for model in c.models.values() for k in model.names]
        finger_keys=[k for k in pending['angles'] if k.startswith(side+'_') and any(f in k for f in ('thumb','index','middle','ring','little'))]
        if max((abs(q[k]-pending['angles'][k]) for k in arm_keys),default=0.)>.025 or max((abs(q[k]-pending['angles'][k]) for k in finger_keys),default=0.)>.08:
            c.stop('Углы изменились за время повторной проверки маршрута',now);return
        from readiness import observation_is_fresh
        if not observation_is_fresh(self.lab.observation,now,self.lab.processed_wall,time.monotonic()):return
        c.table=np.array(self.current['plane']),self.current.get('table_edge',-100.),cKDTree(self.current['obstacles'])
        c.synchronized = True
        # Mesh-based narrow approaches must pause the reference when a loaded
        # joint lags; otherwise integral compensation cuts across checked edges.
        c.planned_route = (self.current.get('grasp') in ('geometry_pinch','cylinder_side')
                           or bool(self.current.get('operator_preshape')))
        c.arm_speed_override=ARM_SPEED
        c.motion_speed=ARM_SPEED
        c.tracking_joint_tolerance=.006
        if c.planned_route:
            c.goal={k:pending['angles'][k] for k in arm_keys}
            c.reference=dict(c.goal)
        c.route = deque((side, pending['message'], np.array(v)) for v in pending['path'])
        c.status = 'moving'
        c.next_stage(now)
        self.enter(pending['stage'], now, pending['message'])
        self.pending_route=None;self.route_future=None

    def scene(self, now):
        scene=next((p['scene'] for p in reversed(self.lab.pairs)
                     if p.get('scene') and 0 <= now-p['stamp'] < .45), None)
        if scene and getattr(self,'tube_pick',False):
            from tube_grasp import select_cap
            cap=select_cap(scene,getattr(self,'tube_reference',None))
            if cap:scene=dict(scene,objects=[*scene['objects'],cap])
        if scene:
            objects=[dict(obj) for obj in scene.get('objects',())]
            present={obj.get('kind') for obj in objects if obj.get('status')=='observed'}
            for obj in objects:
                if obj.get('status')=='observed' and obj.get('kind') and 'xyz' in obj:
                    self.scene_cache[obj['kind']]=dict(obj)
            pending=set(self.queue)
            if self.current and self.current.get('kind'):pending.add(self.current['kind'])
            task=scene.get('task') or self.lab.state.task or {}
            for kind in list(pending):
                try:tray_kind=destination(task,kind)
                except Exception:tray_kind=None
                if tray_kind:pending.add(tray_kind)
            for kind in pending:
                if kind not in present and kind in self.scene_cache:
                    cached=dict(self.scene_cache[kind],status='observed',cached_from_scan=True)
                    objects.append(cached);present.add(kind)
            scene=dict(scene,objects=objects)
        return scene

    def remove_cached_object_points(self,bounds):
        bounds=np.asarray(bounds,float)
        if bounds.shape!=(2,3) or not np.all(np.isfinite(bounds)):return 0
        removed=0;views=list(getattr(self.lab,'scan_views',()))
        for index,view in enumerate(views):
            points=np.asarray(view,float)
            if points.ndim!=2 or points.shape[1]!=3:continue
            inside=np.all((points>=bounds[0]-.025)&(points<=bounds[1]+.025),axis=1)
            removed+=int(np.count_nonzero(inside));self.lab.scan_views[index]=points[~inside]
        return removed

    def rotate_scene_cache(self,delta):
        from body_turn import rotation
        r=rotation(-float(delta))
        def rotate_object(source):
            obj=dict(source)
            for key in ('xyz','visible_xyz'):
                value=np.asarray(obj.get(key,()),float)
                if value.shape==(3,):obj[key]=(r@value).tolist()
            value=np.asarray(obj.get('xyz_mm',()),float)
            if value.shape==(3,):obj['xyz_mm']=(r@value).tolist()
            bounds=np.asarray(obj.get('bounds',()),float)
            if bounds.shape==(2,3) and np.all(np.isfinite(bounds)):
                corners=np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
                corners=corners@r.T
                obj['bounds']=[corners.min(0).tolist(),corners.max(0).tolist()]
            geometry=obj.get('geometry')
            if isinstance(geometry,dict):
                geometry=dict(geometry)
                axis=np.asarray(geometry.get('long_axis_xy',()),float)
                if axis.shape==(2,):geometry['long_axis_xy']=(r[:2,:2]@axis).tolist()
                corners=np.asarray(geometry.get('corners_xy_m',()),float)
                if corners.ndim==2 and corners.shape[1]==2:
                    geometry['corners_xy_m']=(corners@r[:2,:2].T).tolist()
                obj['geometry']=geometry
            observations=obj.get('camera_observations')
            if isinstance(observations,dict):
                obj['camera_observations']={camera:rotate_object(value) if isinstance(value,dict) else value
                                            for camera,value in observations.items()}
            return obj
        self.scene_cache={kind:rotate_object(obj) for kind,obj in self.scene_cache.items()}

    def contact(self):
        side = self.current['side']; sample = self.lab.timeline.latest('hand_'+side)[2]
        empty = self.empty[side]
        pos, effort = np.array(sample['position']), abs(np.array(sample['effort']))
        blocked = pos-np.array(empty['position']) > .045
        loaded = effort-abs(np.array(empty['effort'])) > .05
        # Official order is little, ring, middle, index, thumb flexion,
        # thumb opposition. Either loaded thumb axis can resist the object;
        # requiring flexion alone falsely rejected an opposed, lifted jar.
        # Both channels actuate the same thumb. In recorded transfer Z the
        # flexion stayed blocked while load moved to opposition for one sample;
        # the object remained visibly held. Require deflection and load on that
        # thumb, without demanding they appear on the identical motor axis.
        thumb_contact=bool(np.any(blocked[4:6]) and np.any(loaded[4:6]))
        active=np.array(self.current.get('close_command',CLOSE))[:4]<.9
        if self.current.get('contact_fingers') is not None:
            active &= np.isin(np.arange(4),self.current['contact_fingers'])
        ok = bool(np.any(blocked[:4]&loaded[:4]&active) and thumb_contact)
        velocity=np.asarray(sample.get('velocity',[float('inf')]*6))
        settlement=finger_settlement(self.lab.timeline.data.get('hand_'+side,()))
        return ok, dict(position=pos.tolist(), effort=effort.tolist(), blocked=blocked.tolist(), loaded=loaded.tolist(),
                        **settlement,velocity=velocity.tolist(),thumb_contact=thumb_contact)

    def finish(self, now, reason):
        self.enter('done', now, reason)
        self.want_map = False
        self.lab.exercise = 0
        if self.lab.output:
            (self.lab.output/'complete.json').write_text(json.dumps(dict(
                diagnostic=True, reason=reason, attempted=self.attempt, remaining=list(self.queue),
                blockers=self.blockers), ensure_ascii=False), encoding='utf-8')

    def recover_after_drop(self, now, reason):
        self.recovery_open_started=None
        if self.lab.controller.status!='stopped':
            self.lab.controller.stop(reason,now)
        previous_stage=self.stage
        kind=(self.current or {}).get('kind')
        retry_queued=bool(kind and not self.released and self.retry_count.get(kind,0)<3)
        if kind and not self.released:
            self.blockers.setdefault(kind,[]).append(dict(
                attempt=self.retry_count.get(kind,0),reason=reason,object_may_have_dropped=True))
            if retry_queued:self.queue.appendleft(kind)
        self.event(now,'Контакт потерян; отпускаю кисти и возвращаюсь к рабочей позе',
                   blocker=reason,retry_queued=retry_queued)
        if self.released:self.scene_changed=True
        if (kind and previous_stage in ('carry_replan','transport','dual_lift','dual_transport','dual_release')
                and self.current.get('original')):
            removed=self.remove_cached_object_points(self.current['original'].get('bounds'))
            self.scene_cache.pop(kind,None)
            self.event(now,'Убираю старую точку предмета из кэшированной карты',kind=kind,removed_points=removed)
        self.want_map=True
        self.released_expected=None
        self.fingers('left',np.ones(6));self.fingers('right',np.ones(6))
        try:
            if self.lab.map is not None:self.lab.controller.start(now,self.lab.map,wall=time.monotonic())
            else:self.lab.request('ready',now)
        except Exception as error:
            self.event(now,'Ожидаю свежие датчики для возврата в рабочую позу',error=str(error))
        self.enter('recover_drop',now,'Возвращаю руки в рабочую позу после потери предмета')

    def held_center(self, q):
        model=self.lab.controller.models[self.current['side']]
        pose=model.tool([q[k] for k in model.names])
        offset=getattr(self,'grip_object_offset',None)
        if offset is not None:return pose[:3,3]+pose[:3,:3]@offset
        return self.tracker.original+pose[:3,3]-self.grip_pose

    def paired_center(self,q):
        offsets=self.current['grip_offsets'];centers=[]
        for side in ('left','right'):
            model=self.lab.controller.models[side]
            angles=[q[name] for name in model.names]
            centers.append(model.tool(angles)[:3,3]-np.asarray(offsets[side],float))
        return np.mean(centers,axis=0)

    def release_gate(self,now,q):
        center=self.current.get('release_center')
        tray=self.current.get('tray_observation')
        obs=self.tracker.latest if self.tracker is not None else None
        if center is None or tray is None:return False,dict(reason='Нет проверенной точки сброса или границ лотка')
        center=np.asarray(center,float);held=self.held_center(q)
        if np.linalg.norm(held-center)>.030:
            return False,dict(reason='Кисть и предмет ещё не над выбранной точкой',held_error_mm=round(float(np.linalg.norm(held-center))*1000,1))
        if (obs is None or not (obs.get('valid') or obs.get('partial_visible'))
                or not 0<=now-obs.get('stamp',-1)<.45):
            return False,dict(reason='Нет свежего наблюдения предмета перед раскрытием')
        measured=np.asarray(obs.get('xyz',()),float)
        if measured.shape!=(3,) or not np.all(np.isfinite(measured)):
            return False,dict(reason='Трекер не выдал координату предмета')
        error=float(np.linalg.norm(measured-center))
        if error>.030:
            return False,dict(reason='Предмет вне допуска 30 мм от точки сброса',tracker_error_mm=round(error*1000,1))
        tray_bounds=np.asarray(tray['bounds'],float)
        source_bounds=np.asarray(self.current['original']['bounds'],float)
        radius=float(np.max(np.ptp(source_bounds,axis=0)[:2])/2+CLEARANCE_M)
        lo=tray_bounds[0,:2]+radius;hi=tray_bounds[1,:2]-radius
        if np.any(lo>hi) or not np.all((measured[:2]>=lo)&(measured[:2]<=hi)):
            return False,dict(reason='Предмет не помещается внутри границ лотка',tracker_xy=measured[:2].tolist(),safe_xy=[lo.tolist(),hi.tolist()])
        return True,dict(held_error_mm=round(float(np.linalg.norm(held-center))*1000,1),
                         tracker_error_mm=round(error*1000,1),release_center=center.tolist())

    def bind_visible_object(self, pose, now, force_rotation=False):
        observation=self.tracker and self.tracker.latest
        if (observation and (observation.get('valid') or observation.get('partial_visible'))
                and observation.get('prediction_error_mm',0.)<=30. and 0<=now-observation['stamp']<.4):
            self.grip_object_offset=pose[:3,:3].T@(np.array(observation['xyz'])-pose[:3,3])
            if force_rotation or getattr(self,'grip_reference_rotation',None) is None:
                self.grip_reference_rotation=pose[:3,:3].copy()
            self.event(now,'Привязка измеренного центра предмета к кисти',
                       offset=self.grip_object_offset.tolist(),vision=observation)

    def reconcile_lotion_views(self,now,q):
        tracker=self.tracker
        if tracker is None or tracker.baseline is None:return False,dict(reason='Нет исходной RGB-D поверхности')
        chosen=[]
        for yaw,samples in self.carry_scan_samples.items():
            if not samples:continue
            chosen.append(max(samples,key=lambda row:row['points']))
        if len(chosen)<3:return False,dict(reason='Не собрались три ракурса',views=len(chosen))
        xyz=np.asarray([row['xyz'] for row in chosen],float)
        diameter=max(float(np.linalg.norm(a-b)) for i,a in enumerate(xyz) for b in xyz[i+1:])
        if diameter>.030:return False,dict(reason='Три ракурса расходятся более чем на 30 мм',spread_mm=round(diameter*1000,1))
        center=np.median(xyz,axis=0);base=np.asarray(tracker.baseline['xyz'],float);delta=center-base
        predicted=self.held_center(q);expected_delta=predicted-base
        if not .022<expected_delta[2]<.075 or np.linalg.norm(expected_delta[:2])>.04:
            return False,dict(reason='Кинематика не подтверждает подъём предмета',
                              predicted_lift_mm=(expected_delta*1000).round(1).tolist())
        tracking_error=float(np.linalg.norm(center-predicted))
        if tracking_error>.030:
            return False,dict(reason='Три ракурса расходятся с положением в руке более чем на 30 мм',
                              tracker_error_mm=round(tracking_error*1000,1),
                              delta_mm=(delta*1000).round(1).tolist(),
                              predicted_center=predicted.tolist())
        representative=max(chosen,key=lambda row:row['points'])
        latest=dict(tracker.latest or {})
        latest.update(stamp=now,xyz=center.tolist(),bounds=representative['bounds'],
            points=sum(row['points'] for row in chosen),valid=True,partial_visible=True,
            shape_ok=True,carried_shape_ok=True,carried_motion_candidate=True,
            prediction_error_mm=0.,delta=delta.tolist(),method='three_view_lotion_reacquisition',
            seen_by=sorted({camera for row in chosen for camera in row['seen_by']}))
        tracker.latest=latest
        model=self.lab.controller.models[self.current['side']]
        pose=model.tool([q[k] for k in model.names])
        self.bind_visible_object(pose,now,force_rotation=True)
        return True,dict(center=center.tolist(),spread_mm=round(diameter*1000,1),
                         delta_mm=(delta*1000).round(1).tolist(),
                         tracker_error_mm=round(tracking_error*1000,1),views=len(chosen),
                         head_yaws=[round(row['yaw'],2) for row in chosen])

    def start_lotion_replan(self,now,q):
        from mission_scene import observed_objects
        scene=self.scene(now)
        if (not scene or self.lab.map is None or self.lab.map_stamp is None
                or now-self.lab.map_stamp>1.0 or not self.lab.map_geometry.get('observed')):
            self.lab.controller.stop('Нет свежей карты после трёхракурсного измерения флакона',now);return
        scene=dict(scene,table_geometry=self.lab.map_geometry)
        points=self.lab.map[2].data.copy();packet=dict(scene=scene,obstacles=points)
        objects=observed_objects(packet);tray=objects.get(self.current.get('tray'))
        if tray is None:
            self.lab.controller.stop('Не вижу текущий лоток для перепланирования переноса',now);return
        obs=self.tracker.latest
        if not obs or not (obs.get('valid') or obs.get('partial_visible')):
            self.lab.controller.stop('Три ракурса не восстановили координаты флакона',now);return
        source=np.asarray(self.current['original']['bounds'],float)
        observed=np.asarray(obs['bounds'],float)
        size=np.maximum(np.ptp(source,axis=0),np.ptp(observed,axis=0))
        center=np.asarray(obs['xyz'],float)
        tracked=dict(self.current['original'],xyz=center.tolist(),
                     bounds=[(center-size/2).tolist(),(center+size/2).tolist()])
        self.current['tray_observation']=tray
        edge=planning_edge(scene)
        self.carry_plan_future=self.pool.submit(plan_carried_lotion_route,dict(self.current),dict(q),tracked,tray,
            points,np.asarray(scene['plane'],float),edge,np.asarray(self.grip_object_offset,float).copy(),None)
        self.enter('carry_replan',now,'Перестраиваю перенос по трём ракурсам уже захваченного флакона')

    def observe(self,frames,q,stamp):
        if self.tracker is None:return None
        expected=self.tracker.original.copy()
        if self.released_expected is not None:
            expected=self.released_expected.copy()
        elif (self.current and self.current.get('grasp')=='two_palm_support'
              and self.current.get('grip_offsets')
              and (self.stage in ('dual_lift','dual_transport','dual_release') or
                   (self.stage=='dual_validation' and getattr(self,'pending_pair',{}).get('stage') in ('dual_lift','dual_transport')))):
            expected=self.paired_center(q)
        elif self.grip_pose is not None and self.stage in ('resume','resume_map','resume_look','resume_replan','lift','verify','carry_replan','transport','release','retreat','route_validation'):
            expected=self.held_center(q)
            if hasattr(self.tracker,'expected_rotation') and self.grip_reference_rotation is not None:
                model=self.lab.controller.models[self.current['side']]
                self.tracker.expected_rotation=model.tool([q[k] for k in model.names])[:3,:3]@self.grip_reference_rotation.T
        result=self.tracker.observe(frames,q,stamp,expected)
        if (self.current and self.current.get('kind')=='lotion' and self.carry_scan_requested
                and self.stage=='verify'):
            yaw=float(q.get('head_yaw_joint',0.))
            poses=getattr(self.lab,'scan_poses',())
            target=min((p[0] for p in poses),key=lambda value:abs(value-yaw)) if poses else yaw
            bounds=np.asarray(result.get('bounds',()),float)
            dimensions=np.ptp(bounds,axis=0) if bounds.shape==(2,3) else np.zeros(3)
            quality=(result.get('points',0)>=150 and .07<dimensions[2]<.30 and
                .025<max(dimensions[:2])<.20 and result.get('tilt_deg',90)<55 and
                result.get('shape_axis_ratio',0)>1.15 and abs(yaw-target)<.055 and 'xyz' in result)
            if quality:
                self.carry_scan_samples.setdefault(round(target,2),[]).append(dict(
                    yaw=yaw,stamp=stamp,xyz=result['xyz'],bounds=result['bounds'],points=int(result['points']),
                    seen_by=list(result.get('seen_by',[]))))
        if (self.short_retreat and self.current.get('kind')=='lotion'
                and self.current.get('grasp')=='cylinder_side'
                and not getattr(self,'resume_baseline_bound',False) and result.get('valid')):
            model=self.lab.controller.models[self.current['side']]
            pose=model.tool([q[k] for k in model.names])
            self.grip_object_offset=pose[:3,:3].T@(np.asarray(result['xyz'])-pose[:3,3])
            self.grip_reference_rotation=pose[:3,:3].copy()
            self.resume_baseline_bound=True
            self.event(stamp,'Повторная привязка видимого флакона к кисти',
                       offset=self.grip_object_offset.tolist(),vision=result)
        if (self.stage=='transport' and self.current.get('kind')=='lotion'
                and getattr(self.tracker,'carrying',False)
                and result.get('carried_motion_candidate')):
            anchor=getattr(self,'resume_motion_anchor',None)
            if anchor and stamp-anchor['stamp']>=.5:
                expected_delta=expected-anchor['expected']
                observed_delta=np.asarray(result['xyz'])-anchor['xyz']
                expected_motion=float(np.linalg.norm(expected_delta))
                observed_motion=float(np.linalg.norm(observed_delta))
                residual=float(np.linalg.norm(observed_delta-expected_delta))
                if expected_motion>=.006:
                    if observed_motion>=.004 and residual<=.010:
                        self.resume_motion_concordance=getattr(self,'resume_motion_concordance',0)+1
                        self.resume_motion_failures=0
                    else:
                        self.resume_motion_concordance=0
                        self.resume_motion_failures=getattr(self,'resume_motion_failures',0)+1
                    self.event(stamp,'Сравнение движения флакона с кистью',
                        expected_mm=round(expected_motion*1000,1),observed_mm=round(observed_motion*1000,1),
                        residual_mm=round(residual*1000,1),consecutive=self.resume_motion_concordance)
                    self.resume_motion_anchor=dict(expected=expected.copy(),xyz=np.asarray(result['xyz']).copy(),stamp=stamp)
                    if self.resume_motion_concordance>=2:
                        model=self.lab.controller.models[self.current['side']]
                        pose=model.tool([q[k] for k in model.names])
                        self.bind_visible_object(pose,stamp,force_rotation=True)
                        self.resume_motion_concordance=0;self.resume_motion_failures=0
                        self.resume_motion_anchor=dict(expected=self.held_center(q).copy(),
                            xyz=np.asarray(result['xyz']).copy(),stamp=stamp)
                    elif self.resume_motion_failures>=2:
                        result['valid']=False;result['partial_visible']=False
                        result['resume_reacquire_partial']=False;result['carried_motion_partial']=False
                        self.event(stamp,'Предмет не повторил движение кисти',vision=result)
        if self.lab.output:
            with (self.lab.output/'jar-observations.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(stage=self.stage,**result),ensure_ascii=False)+'\n')
        return result

    def tick(self, now, wall):
        c = self.lab.controller
        if self.stage == 'done': return
        if c.status == 'stopped' and self.stage not in ('recover_drop','recover_ready'):
            reason='Защитная остановка: '+str(c.reason)
            if self.current is None:
                self.enter('recover_ready',now,'Жду свежие датчики перед первым предметом',reason=reason)
            else:self.recover_after_drop(now,reason)
            return
        if wall-self.wall_start > self.max_wall:
            c.stop('Истёк лимит диагностического сеанса', now)
            self.finish(now, c.reason); return
        if self.stage in ('verify','carry_replan','transport') and self.lab.preparing:return
        if self.wait_for_scene_reload:
            scene = self.lab.state.scene.snapshot()
            if scene.get('busy') or scene.get('status') in ('starting','reloading'):
                self.scene_reload_seen = True
            current = scene.get('current') or {}
            expected_seed = int(os.environ.get('TEAM22_TRIAL_EXPECTED_SCENE_SEED','1610067433'))
            if (self.scene_reload_seen and scene.get('status') == 'ready'
                    and current.get('id') == os.environ.get('TEAM22_TRIAL_EXPECTED_SCENE_ID','sort2:2')
                    and int(current.get('seed',-1)) == expected_seed):
                self.wait_for_scene_reload = False
                self.event(now, 'Сохраненная сцена перезагружена перед пробой', scene=current)
            else:
                return
        if not self.planner_ready:
            if not self.planner_warmup.done(): return
            self.planner_warmup_ms = round(1000*(time.monotonic()-self.planner_warmup_started),1)
            try:
                self.planner_warmup.result()
            except Exception as error:
                self.event(now, 'Предварительная загрузка планировщика не удалась',
                           planner_warmup_ms=self.planner_warmup_ms,error=str(error))
            self.planner_ready = True
            self.planner_warmup = None
            self.event(now, 'Планировщик предварительно загружен',
                       planner_warmup_ms=self.planner_warmup_ms)
        if self.stage=='route_validation':
            if self.route_future.done():
                if not self.route_future.result():
                    c.stop('Повторная проверка маршрута по фактическим углам не пройдена',now)
                else:self.start_validated_route(now)
            return
        if self.stage=='preshape_validation':
            if not self.preshape_future.done():return
            safe,detail=self.preshape_future.result()
            q=self.lab.timeline.latest('joints')[2]
            keys=[k for k in self.preshape_angles if k.startswith(self.current['side']+'_')]
            if not safe:c.stop('Переход к форме кисти: '+detail,now)
            elif max((abs(q[k]-self.preshape_angles[k]) for k in keys),default=0.)>.02:
                c.stop('Кисть сместилась во время проверки формы',now)
            else:
                self.fingers(self.current['side'],self.current['approach_preshape'])
                self.enter('pinch_preshape',now,'Формирую кисть после проверки свободного места')
            return
        if self.stage=='align_prepare':
            if self.lab.preparing or self.lab.planning or c.status!='holding':return
            q=self.lab.timeline.latest('joints')[2]
            if not all(c.arm_matches(side,q) for side in ('left','right')):
                if not self.align_ready_requested:
                    if self.lab.map is None or self.lab.map_stamp is None or now-self.lab.map_stamp>.5:return
                    self.align_ready_requested=True
                    try:c.start(now,self.lab.map,wall=wall)
                    except Exception as error:
                        self.event(now,'Не удалось вернуть руки в безопасную позу; поворот пропускаю',reason=str(error))
                        self.enter('next',now,'Продолжаю без разворота корпуса')
                return
            for side in ('left','right'):
                if min(self.lab.timeline.latest('hand_'+side)[2]['position'])<=.9:
                    self.fingers(side,np.ones(6))
            self.enter('align_open',now,'Раскрываю пустые кисти перед поворотом корпуса')
            return
        if self.stage=='align_open':
            if c.status!='holding' or self.lab.preparing or self.lab.planning:return
            samples={s:self.lab.timeline.latest('hand_'+s) for s in ('left','right')}
            settled={s:finger_settlement(self.lab.timeline.data.get('hand_'+s,())) for s in samples}
            opened=all(sample is not None and sample[0]-self.entered>=.55
                       and settled[side].get('settled')
                       and len(sample[2].get('position',()))==6
                       and min(sample[2]['position'])>.9
                       for side,sample in samples.items())
            if opened:
                if self.lab.map is None or self.lab.map_stamp is None or now-self.lab.map_stamp>.5:return
                c.table=self.lab.map
                try:self.lab.body_turn.start(self.alignment_goal,now)
                except ValueError as error:
                    kind=(self.queue[0] if self.queue else None)
                    if kind:self.blockers.setdefault(kind,[]).append(dict(
                        attempt=self.retry_count.get(kind,0),reason='Поворот корпуса пропущен: '+str(error)))
                    self.event(now,'Поворот корпуса пропущен по проверке безопасности; продолжаю с текущей ориентации',
                               reason=str(error))
                    self.enter('next',now,'Проверяю доступность хвата с текущей ориентации')
                    return
                self.enter('align_turn',now,'Поворачиваю корпус к крупному предмету с пустыми поднятыми руками')
            elif now-self.entered>6.:
                c.stop('Кисти не раскрылись перед поворотом корпуса',now)
            return
        if self.stage=='align_turn':
            if self.lab.body_turn.active:return
            delta=float(getattr(self.lab.body_turn,'last_delta',0.))
            if abs(delta)>1e-6:self.rotate_scene_cache(delta)
            self.want_map=True
            self.enter('align_scan',now,'Переношу сохранённую карту в новую рамку корпуса')
            return
        if self.stage=='align_scan':
            scene=self.scene(now)
            if (self.lab.preparing or not scene or self.lab.map_stamp is None
                    or now-self.lab.map_stamp>.5):return
            self.scan_available=True;self.scan_stamp=self.lab.map_stamp
            self.enter('next',now,'Текущий кадр обновлён; использую сохранённые координаты остальных предметов')
            return
        if self.stage.startswith('dual_'):
            if c.status=='stopped':
                self.recover_after_drop(now,c.reason or 'Двухручная попытка остановлена');return
            from bimanual import tick
            tick(self,now)
            if c.status=='stopped':self.recover_after_drop(now,c.reason or 'Двухручная попытка остановлена')
            return
        if self.stage in ('recover_drop','recover_ready'):
            if self.lab.preparing or self.lab.planning:return
            if c.status=='stopped':
                try:
                    if self.lab.map is not None:c.start(now,self.lab.map,wall=wall)
                    else:self.lab.request('ready',now)
                except Exception:
                    if now-self.entered>45.:self.finish(now,'Не удалось восстановить рабочую позу после потери предмета')
                    return
                return
            if c.status=='holding':
                if self.recovery_open_started is None:
                    self.recovery_open_started=now
                    self.fingers('left',np.ones(6));self.fingers('right',np.ones(6))
                samples={side:self.lab.timeline.latest('hand_'+side) for side in ('left','right')}
                settled={side:finger_settlement(self.lab.timeline.data.get('hand_'+side,())) for side in samples}
                opened=(now-self.recovery_open_started>=.55 and all(
                    sample is not None and 0<=now-sample[0]<.18
                    and len(sample[2].get('position',()))==6 and min(sample[2]['position'])>.9
                    and settled[side].get('settled') for side,sample in samples.items()))
                if opened:
                    self.recovery_open_started=None
                    if self.alignment_pending_kind is not None:
                        self.aligned_objects.discard(self.alignment_pending_kind)
                    self.scene_changed=self.scene_changed or self.released;self.scan_available=True
                    self.enter('next',now,'Датчики, поза и открытые кисти восстановлены; продолжаю по сохранённой карте')
                elif now-self.recovery_open_started>6.:
                    kind=self.alignment_pending_kind or (self.current or {}).get('kind')
                    if kind:self.blockers.setdefault(kind,[]).append(dict(reason='Кисти не раскрылись после восстановления'))
                    self.finish(now,'Кисти не раскрылись после восстановления')
            elif now-self.entered>45.:
                self.finish(now,'Не удалось восстановить рабочую позу после потери предмета')
            return
        if (os.environ.get('TEAM22_FIXED_HEAD')!='1'
                and self.stage in ('approach','opposition','contact','lift','verify','transport','release','retreat')
                and getattr(getattr(self,'tracker',None),'rig',None) is not None and now-self.last_head_command>.25):
            from bodyctrl_msgs.msg import CmdSetMotorPosition,SetMotorPosition
            model=c.models[self.current['side']];q=self.lab.timeline.latest('joints')[2]
            target=(self.released_expected if self.released_expected is not None else
                    self.held_center(q) if self.grip_pose is not None else self.tracker.original)
            from grasp_gaze import gaze_angles
            yaw,pitch=gaze_angles(target,[q['head_yaw_joint'],q['head_pitch_joint']],self.tracker.rig)
            self.lab.head_pub.publish(CmdSetMotorPosition(cmds=[SetMotorPosition(name=3,pos=float(yaw),spd=self.lab.head_speed),SetMotorPosition(name=2,pos=float(pitch),spd=self.lab.head_speed),SetMotorPosition(name=1,pos=0.,spd=self.lab.head_speed)]))
            self.last_head_command=now
        tracker=getattr(self,'tracker',None)
        if tracker and self.stage in ('approach','opposition','contact') and tracker.latest and tracker.baseline:
            observation=tracker.latest
            moved=(observation.get('valid') and np.linalg.norm(np.array(observation.get('delta',[0,0,0]))[:2])>.014)
            tilted=observation.get('points',0)>70 and observation.get('tilt_deg',0)>40
            if 0<=now-observation['stamp']<.4 and (moved or tilted):
                if self.bad_visual_since is None:self.bad_visual_since=now
                if now-self.bad_visual_since>.25:
                    self.event(now,'Банка смещается или наклоняется до подъёма',vision=observation)
                    c.stop('Боковой подход прерван: банка сдвинулась или наклонилась',now);return
            else:self.bad_visual_since=None
        if self.stage=='resume':
            if c.health(now,wall):return
            ok,data=self.contact()
            if not ok:
                c.stop('Перед обновлением карты нет подтверждённого удержания',now);return
            self.lab.map_stamp=None;self.lab.map_geometry={};self.lab.scan_views=[]
            self.want_map=True;self.lab.center_head()
            self.enter('resume_map',now,'Обновляю карту, удерживая флакон и не двигая руки')
        elif self.stage=='resume_map':
            if c.health(now,wall):return
            ok,data=self.contact()
            if not ok:
                c.stop('Во время обновления карты потерян контакт с флаконом',now);return
            fresh_map=(self.lab.map_stamp is not None and 0<=now-self.lab.map_stamp<.25
                       and self.lab.map_geometry.get('observed'))
            if fresh_map:
                from bodyctrl_msgs.msg import CmdSetMotorPosition,SetMotorPosition
                self.want_map=False
                q=self.lab.timeline.latest('joints')[2];model=c.models[self.current['side']]
                pos=model.tool([q[k] for k in model.names])[:3,3]
                yaw_offset=float(os.environ.get('TEAM22_RESUME_LOOK_YAW_OFFSET','.35'))
                yaw=float(np.clip(np.arctan2(pos[1],pos[0])+yaw_offset,-1.35,1.35))
                pitch=float(np.clip(float(os.environ.get('TEAM22_RESUME_LOOK_PITCH','.20')),.05,.65))
                self.lab.head_pub.publish(CmdSetMotorPosition(cmds=[SetMotorPosition(name=3,pos=yaw,spd=self.lab.head_speed),
                    SetMotorPosition(name=2,pos=pitch,spd=self.lab.head_speed),SetMotorPosition(name=1,pos=0.,spd=self.lab.head_speed)]))
                self.resume_valid_count=0;self.resume_last_observation_stamp=None
                self.enter('resume_look',now,'Восстанавливаю обзор флакона после свежего обзора лотка')
            elif now-self.entered>8.:c.stop('Не удалось обновить карту при удерживаемом флаконе',now)
        elif self.stage=='resume_look':
            ok,data=self.contact();obs=self.tracker.latest
            if not ok:
                c.stop('Перед продолжением нет противоположного контакта',now);return
            lotion_resume=self.current.get('kind')=='lotion' and self.current.get('grasp')=='cylinder_side'
            visible=bool(obs and (obs.get('valid') if lotion_resume else
                                  (obs.get('valid') or obs.get('partial_visible'))))
            fresh=bool(obs and 0<=now-obs['stamp']<.4)
            if visible and fresh:
                if obs['stamp']!=self.resume_last_observation_stamp:
                    self.resume_valid_count+=1
                    self.resume_last_observation_stamp=obs['stamp']
            else:
                self.resume_valid_count=0;self.resume_last_observation_stamp=None
            if (now-self.entered>1. and visible and fresh
                    and (not lotion_resume or self.resume_valid_count>=2)):
                q=self.lab.timeline.latest('joints')[2]
                if lotion_resume and getattr(self.tracker,'resume_reacquire',False):
                    model=c.models[self.current['side']]
                    pose=model.tool([q[k] for k in model.names])
                    self.bind_visible_object(pose,now,force_rotation=True)
                    self.grip_pose=pose[:3,3].copy()
                    self.resume_baseline_bound=True
                    self.tracker.original=np.asarray(obs['xyz'],float).copy()
                    self.tracker.bounds=np.asarray(obs['bounds'],float).copy()
                    self.tracker.resume_reacquire=False
                    self.resume_motion_anchor=dict(expected=self.held_center(q).copy(),
                        xyz=np.asarray(obs['xyz']).copy(),stamp=obs['stamp'])
                    self.resume_motion_concordance=0;self.resume_motion_failures=0
                c.goal={k:q[k] for model in c.models.values() for k in model.names}
                c.reference=c.goal.copy();c.command=c.goal.copy();c.integral={k:0. for k in c.goal}
                c.last_tick=now;c.reason=None;c.status='holding'
                self.event(now,'Повторный захват подтверждён; строю маршрут по текущим данным',vision=obs,fingers=data)
                scene=self.scene(now)
                if lotion_resume:
                    if (not scene or self.lab.map is None or self.lab.map_stamp is None
                            or now-self.lab.map_stamp>4. or not self.lab.map_geometry.get('observed')):
                        c.stop('Нет свежей карты стола для нового маршрута',now);return
                    task=self.lab.state.task or {}
                    tray_kind=self.current.get('tray') or destination(task,'lotion')
                    tray=next((o for o in scene['objects'] if o.get('kind')==tray_kind),None)
                    if tray is None:
                        c.stop('Не вижу назначенный лоток на актуальном кадре',now);return
                    scene=dict(scene,table_geometry=self.lab.map_geometry)
                    obj=dict(obs,kind='lotion',id='lotion')
                    points=self.lab.map[2].data.copy()
                    self.current['original']=obj;self.current['tray']=tray_kind
                    self.current['tray_observation']=tray
                    self.resume_plan_future=self.pool.submit(plan_carried_lotion_route,
                        dict(self.current),dict(q),obj,tray,points,np.asarray(scene['plane'],float),
                        planning_edge(scene),np.asarray(self.grip_object_offset,float).copy(),
                        None if self.tracker.reference is None else np.asarray(self.tracker.reference,float).copy())
                    self.enter('resume_replan',now,'Пересчитываю перенос к текущему свободному месту в лотке')
                else:
                    self.move(self.current['transport'][-2:],now,'transport','Завершение переноса в назначенный лоток')
            elif now-self.entered>7.:c.stop('Камера не восстановила наблюдение банки',now)
        elif self.stage=='resume_replan':
            ok,data=self.contact();obs=self.tracker.latest
            if not ok:
                c.stop('Во время перепланирования потерян контакт с флаконом',now);return
            if not obs or not obs.get('valid') or not 0<=now-obs['stamp']<.5:
                if self.bad_visual_since is None:self.bad_visual_since=now
                elif now-self.bad_visual_since>.8:
                    c.stop('Во время перепланирования потеряно свежее наблюдение флакона',now)
                return
            self.bad_visual_since=None
            future=getattr(self,'resume_plan_future',None)
            if future is not None and future.done():
                try:result=future.result()
                except Exception as exc:
                    c.stop('Ошибка перепланирования: '+str(exc),now);return
                self.resume_plan_future=None
                if result.get('error') or not result.get('route'):
                    self.event(now,'Новый маршрут отклонён; причины по точкам',
                               plan_error=result.get('error'),plan_failures=result.get('failures'))
                    c.stop('Новый маршрут не прошёл проверку: '+result.get('error','пустой путь'),now);return
                self.current['transport']=result['route']
                self.current['release_site']=result['release_site']
                self.current['release_center']=result['release_center']
                self.current['obstacles']=result['obstacles']
                self.current['plane']=result['plane']
                self.current['table_edge']=result['table_edge']
                self.event(now,'Путь переноса проверен по свежей карте',waypoints=len(result['route']),
                           release_site=result['release_site'],high=result['high'],obstacle_count=result['obstacle_count'])
                self.move(result['route'],now,'transport','Перенос флакона в актуальный синий лоток')
            elif now-self.entered>25.:c.stop('Пересчёт безопасного маршрута занял слишком долго',now)
        elif self.stage in ('initial_scan','scan_next'):
            if self.lab.preparing or self.lab.planning:return
            if self.lab.map_stamp is None or not self.lab.map_geometry.get('observed'):return
            self.scan_available=True;self.scan_stamp=self.lab.map_stamp
            destination_stage='prepare' if self.stage=='initial_scan' else 'next'
            self.enter(destination_stage,now,'Обзор стола завершён; использую одну карту для предмета')
        elif self.stage == 'prepare':
            task=self.lab.state.task
            if not task:return
            from readiness import observation_is_fresh
            observation=self.lab.observation
            if (not observation or 'stamp' not in observation or
                    not observation_is_fresh(observation,now,self.lab.processed_wall,wall) or c.health(now,wall)):return
            needs_ready=self.requires_ready()
            if needs_ready:
                if c.status!='holding':return
                if self.lab.map_stamp is not None and self.lab.map_geometry.get('observed'):
                    self.scan_available=True;self.scan_stamp=self.lab.map_stamp
            if (getattr(self,'direct_start',False) or needs_ready) and not self.scan_available:
                if self.lab.preparing:return
                self.lab.request('scan',now)
                self.enter('initial_scan',now,'Один обзор стола перед первым предметом')
                return
            if getattr(self,'direct_start',False) and c.status=='idle':
                q=self.lab.timeline.latest('joints')[2]
                c.goal={k:q[k] for model in c.models.values() for k in model.names}
                c.reference=c.goal.copy();c.command=c.goal.copy()
                c.integral={k:0. for k in c.goal};c.last_tick=now
                c.status='holding';c.reason=None
                c.stage='Удерживаю фактическую позу перед проверенным подходом'
            self.lab.exercise = 0
            names=task.get('objects') or {}
            self.unsupported=[name for name in names if name not in NAMES.values()]
            self.queue=deque(k for k in self.queue if NAMES.get(k) in names)
            if getattr(self,'tube_pick',False) and task.get('task')=='tubes' and task.get('variant')!='lying':
                from tube_grasp import select_cap
                cap=select_cap(self.scene(now))
                if cap:
                    self.tube_reference=cap;self.queue=deque(['tube_cap'])
            self.event(now,'Проверка поддержки текущего задания',task=task,
                       supported=list(self.queue),unsupported=self.unsupported)
            if (task.get('task')=='tubes' and not getattr(self,'tube_reference',None)) or not self.queue:
                self.enter('inspect_task',now,'Осматриваю задание: подходящего распознавания и хвата пока нет')
                return
            from bodyctrl_msgs.srv import SetForce, SetSpeed
            self.settings = []
            for side in ('left', 'right'):
                for kind, typ, ratio in (('force', SetForce, HAND_FORCE), ('speed', SetSpeed, HAND_SPEED)):
                    client = self.lab.node.create_client(typ, '/inspire_hand/set_'+kind+'/'+side+'_hand')
                    req = typ.Request()
                    for i in range(6): setattr(req, kind+str(i)+'_ratio', ratio)
                    self.settings.append([kind, side, client, None, req])
            self.enter('settings', now, f'Настройка кистей: усилие {HAND_FORCE:.0%}, скорость {HAND_SPEED:.0%}',
                       close=CLOSE.tolist(),force=HAND_FORCE,speed=HAND_SPEED)
        elif self.stage=='inspect_task':
            if self.profile=='complete' and self.lab.state.task.get('task')=='tubes':
                if now-self.entered<3.:return
                self.event(now,'Пробирки на столе: видимые крышки и размеры',scene=self.scene(now))
                self.lab.body_turn.start(1.57,now);self.lab.center_head()
                self.enter('inspect_hood',now,'Поворачиваю пустой корпус к шкафу для измерения штативов')
                return
            from bodyctrl_msgs.msg import CmdSetMotorPosition,SetMotorPosition
            step=min(3,int((now-self.entered)/4.))
            if step!=self.inspect_step:
                self.event(now,'Камерный осмотр варианта задания',scene=self.scene(now),view=step)
                yaw=(0.,.65,1.2,0.)[step]
                self.lab.head_pub.publish(CmdSetMotorPosition(cmds=[SetMotorPosition(name=3,pos=yaw,spd=self.lab.head_speed),
                    SetMotorPosition(name=2,pos=.42,spd=self.lab.head_speed),SetMotorPosition(name=1,pos=0.,spd=self.lab.head_speed)]))
                self.inspect_step=step
            if now-self.entered>16.:
                reason=('Нет детектора пробирок и гнёзд штативов; перенос в шкаф не спланирован'
                        if self.lab.state.task.get('task')=='tubes' else
                        'Предметы этого пула не поддержаны текущим распознаванием и хватами')
                self.finish(now,reason)
        elif self.stage=='inspect_hood':
            if self.lab.body_turn.active:return
            if self.confirmed is None:self.confirmed=now
            if now-self.confirmed<3.:return
            self.event(now,'Осмотр штативов в шкафу',scene=self.scene(now))
            self.lab.body_turn.start(0.,now)
            self.enter('inspect_return',now,'Возвращаю пустой корпус к столу')
        elif self.stage=='inspect_return' and not self.lab.body_turn.active:
            self.finish(now,'Измерены стол и шкаф; перенос пробирок не выполнен без проверенного гнезда и хвата')
        elif self.stage == 'settings':
            for setting in self.settings:
                if setting[3] is None and setting[2].service_is_ready():
                    setting[3] = setting[2].call_async(setting[4])
            if not all(future is not None and future.done() for _,_,_,future,_ in self.settings):
                if now-self.entered>4: c.stop('Нет ответа настройки кистей', now)
                return
            if not all(getattr(f.result(), kind+'_accepted') for kind,_,_,f,_ in self.settings):
                c.stop('Настройки кистей не приняты', now); return
            for side in ('left', 'right'): self.fingers(side, np.ones(6))
            self.enter('next', now, 'Настройки кистей приняты; выбираю следующий предмет')
        elif self.stage == 'calibrate_grasp':
            sides=self.calibration_sides
            samples={s:self.lab.timeline.latest('hand_'+s) for s in sides}
            settlement={s:finger_settlement(self.lab.timeline.data.get('hand_'+s,())) for s in sides}
            def ready(side):
                sample=samples[side]
                if sample is None or sample[0]-self.entered<.55 or not settlement[side].get('settled'):return False
                position=np.asarray(sample[2].get('position',()),float)
                return bool(len(position)==6 and np.all(np.isfinite(position)) and np.max(abs(position-CLOSE))<.10)
            if all(ready(side) for side in sides):
                self.empty.update({s:samples[s][2] for s in sides})
                self.event(now,'Базовые показания пустых кистей',
                           empty={s:self.empty[s] for s in sides},settlement=settlement)
                for side in sides:self.fingers(side,np.ones(6))
                self.enter('calibration_open',now,'Проверяю раскрытие и остановку выбранной кисти',
                           calibration_sides=list(sides))
            elif now-self.entered>6.:
                c.stop('Пустое закрытие кисти перед хватом не установилось по датчикам',now)
        elif self.stage == 'calibration_open':
            sides=self.calibration_sides
            samples={s:self.lab.timeline.latest('hand_'+s) for s in sides}
            settlement={s:finger_settlement(self.lab.timeline.data.get('hand_'+s,())) for s in sides}
            def opened(side):
                sample=samples[side]
                if sample is None or sample[0]-self.entered<.55 or not settlement[side].get('settled'):return False
                position=np.asarray(sample[2].get('position',()),float)
                return bool(len(position)==6 and np.all(np.isfinite(position)) and np.min(position)>.9)
            if all(opened(side) for side in sides):
                result=self.current
                if result.get('grasp')=='two_palm_support':
                    self.enter('dual_open',now,'Раскрываю обе кисти для бокового охвата коробки')
                elif result.get('grasp')=='cylinder_side' and not result.get('approach_preshape'):
                    self.enter('preshape',now,'Все пальцы полностью раскрыты для низкого бокового подхода')
                elif result.get('preparation_route'):
                    self.move(result['preparation_route'],now,'pregrasp_raise','Поднимаю раскрытую кисть; пальцы сформируются над столом')
                elif result.get('approach_preshape'):
                    self.prepare_fingers(now)
                else:
                    self.move(result['approach'],now,'approach','Подвод раскрытой кисти: '+result['kind'])
            elif now-self.entered>6.:
                c.stop('Кисть после калибровки не раскрылась и не успокоилась',now)
        elif self.stage == 'next':
            self.want_map = True
            self.current=None
            if not self.queue and self.deferred:
                self.queue.extend(k for k in self.deferred if self.retry_count.get(k,0)<3)
                self.deferred=[];self.scene_changed=False
            if not self.queue:
                self.finish(now, 'Выбранные попытки завершены; результаты в attempts.jsonl'); return
            if not self.scan_available:
                if self.lab.preparing or self.lab.planning or c.status!='holding':return
                self.lab.request('scan',now)
                self.enter('scan_next',now,'Один новый обзор перед следующим предметом')
                return
            scene = self.scene(now)
            if not scene or self.lab.map_stamp is None or now-self.lab.map_stamp > .2: return
            if not all(min(self.lab.timeline.latest('hand_'+s)[2]['position'][:4])>.9 for s in ('left','right')): return
            alignment_resume=(self.alignment_pending_kind is not None
                              and self.alignment_pending_kind in self.queue)
            if alignment_resume:
                pending=self.alignment_pending_kind
                self.queue=deque([pending]+[kind for kind in self.queue if kind!=pending])
                self.alignment_pending_kind=None
            auto_order=os.environ.get('TEAM22_AUTO_ORDER')
            if not alignment_resume and ((self.attempt == 0 and auto_order in ('nearest','align_first'))
                                         or auto_order=='blockers_first'):
                q=self.lab.timeline.latest('joints')[2]
                tools=[model.tool([q[n] for n in model.names])[:3,3]
                       for model in self.lab.controller.models.values()]
                ranked=[]
                for candidate in self.queue:
                    obj=next((o for o in scene['objects']
                              if o['kind']==candidate and o.get('status')=='observed'),None)
                    if obj is None: continue
                    position=np.asarray(obj['xyz'],float)
                    distance=min(float(np.linalg.norm(position-tool)) for tool in tools)
                    ranked.append((distance,candidate))
                if ranked:
                    blocker_scores,blocked_by=access_blockers(scene,[kind for _,kind in ranked])
                    if auto_order=='blockers_first':
                        ranked.sort(key=lambda item:(-blocker_scores.get(item[1],0.),item[0]))
                    else:
                        ranked.sort(key=lambda item:((item[1] not in ('drive','green_speaker'))
                                                     if auto_order=='align_first' else False,item[0]))
                    ordered=[kind for _,kind in ranked]
                    if auto_order=='blockers_first':
                        ordered=([kind for kind in ('lotion','gold_jar') if kind in ordered]+
                                 [kind for kind in ordered if kind not in ('lotion','gold_jar')])
                    self.queue=deque(ordered+[kind for kind in self.queue if kind not in ordered])
                    message=('Сначала выбираю предмет, перекрывающий доступ к другим'
                             if auto_order=='blockers_first' else
                             'Сначала выбираю предмет, которому нужно выравнивание корпуса'
                             if auto_order=='align_first' else
                             'Выбираю ближайший видимый предмет по текущей карте')
                    self.event(now,message,
                               candidates=[dict(kind=kind,hand_distance_m=round(distance,3))
                                           for distance,kind in ranked],
                               ordered_kinds=ordered,
                               blocker_scores={kind:round(score,4) for kind,score in blocker_scores.items()},
                               blocked_by=blocked_by)
            if self.retrying_kind is not None:
                retry=self.retrying_kind;self.retrying_kind=None
                if retry in self.queue:
                    self.queue=deque([retry]+[kind for kind in self.queue if kind!=retry])
            kind = self.queue.popleft()
            if kind not in self.aligned_objects:
                obj=next((o for o in scene['objects'] if o['kind']==kind and o.get('status')=='observed'),None)
                if obj:
                    relative_yaw=float(np.arctan2(obj['xyz'][1],obj['xyz'][0]))
                    alignment_offset=0.
                    tray_kind=next((tray for source,tray in {
                        'lotion':'tray_blue','gold_jar':'tray_blue','blue_jar':'tray_blue',
                        'drive':'tray_red','mouse':'tray_red','cable':'tray_red'}.items()
                        if source==kind),None)
                    tray=next((o for o in scene['objects'] if o.get('kind')==tray_kind and o.get('status')=='observed'),None)
                    if tray:
                        options=[]
                        for offset in (-.20,0.,.20):
                            delta=float(np.clip(relative_yaw+offset,-1.1,1.1))
                            source_y=-np.sin(delta)*obj['xyz'][0]+np.cos(delta)*obj['xyz'][1]
                            tray_y=-np.sin(delta)*tray['xyz'][0]+np.cos(delta)*tray['xyz'][1]
                            same_side=((source_y>.08 and tray_y>.08) or (source_y<-.08 and tray_y<-.08))
                            score=(1. if same_side else 0.)+min(abs(source_y),abs(tray_y))-.01*abs(offset)
                            options.append((score,delta,offset,source_y,tray_y))
                        _,relative_yaw,alignment_offset,source_y,tray_y=max(options)
                    else:source_y=tray_y=None
                    self.aligned_objects.add(kind)
                    if abs(relative_yaw)>.12:
                        self.alignment_pending_kind=kind
                        self.queue.appendleft(kind)
                        q=self.lab.timeline.latest('joints')[2]
                        self.alignment_goal=float(q['body_yaw_joint']+relative_yaw)
                        self.align_ready_requested=False
                        self.enter('align_prepare',now,'Подвожу корпус к предмету и его лотку',
                                   alignment_delta=relative_yaw,alignment_offset=alignment_offset,
                                   pickup_side_y=source_y,tray_side_y=tray_y)
                        return
            self.current = dict(kind=kind); self.attempt += 1
            self.tracker=None;self.grip_pose=None;self.grip_object_offset=None
            self.grip_reference_rotation=None;self.released_expected=None
            self.release_scan_requested=False
            self.carry_scan_requested=False;self.carry_scan_started=None
            self.carry_scan_samples={};self.carry_scan_reconciled=False;self.carry_plan_future=None
            self.dual_lift_scan_requested=False;self.dual_transport_scan_requested=False
            self.dual_release_scan_requested=False
            self.released=False;self.retry_count[kind]=self.retry_count.get(kind,0)+1
            self.enter('plan_ready',now,'Планирую хват без предварительной калибровки кистей',
                       scan_stamp=self.scan_stamp)
        elif self.stage == 'plan_ready':
            scene=self.scene(now)
            if not scene or self.lab.map_stamp is None or now-self.lab.map_stamp>.2:return
            kind=self.current['kind']
            if not all(min(self.lab.timeline.latest('hand_'+s)[2]['position'][:4])>.9 for s in ('left','right')):return
            preshapes=self.planning_preshape(kind)
            if preshapes:
                pending=[]
                for side,target in preshapes:
                    sample=self.lab.timeline.latest('hand_'+side)
                    if (sample is None or now-sample[0]>.2
                            or np.max(abs(np.asarray(sample[2]['position'])-target))>.025):
                        pending.append((side,target))
                if pending:
                    self.plan_preshape_targets=preshapes
                    for side,target in pending:self.fingers(side,target)
                    self.enter('plan_preshape',now,'Отвожу большие пальцы обеих кистей и подтверждаю датчиками',
                               sides=[side for side,_ in preshapes],targets={side:list(target) for side,target in preshapes})
                    return
            q = self.lab.timeline.latest('joints')[2]
            scene=dict(scene,table_geometry=getattr(self.lab,'map_geometry',{}),
                       scan_view_count=len(getattr(self.lab,'scan_views',())))
            packet=dict(scene=scene, kind=kind, angles=q,obstacles=self.lab.map[2].data.tolist(),task=self.lab.state.task,
                        profile=self.profile,clearance_m=CLEARANCE_M,scan_stamp=self.scan_stamp,
                        scan_view_count=len(getattr(self.lab,'scan_views',())))
            if kind!='tube_cap':
                from mission_scene import observed_objects
                from tray_placement import release_sites
                visible=observed_objects(packet)
                tray_kind=destination(packet['task'],kind)
                planned_sites=(release_sites(visible[kind],visible[tray_kind],packet['obstacles'])
                               if kind in visible and tray_kind in visible else [])
                packet['release_sites']=planned_sites
                packet['release_site']=planned_sites[0] if planned_sites else None
                self.event(now,'Точка укладки выбрана до подхода к предмету',tray=tray_kind,
                           release_site=packet['release_site'],candidates=len(planned_sites),
                           scan_views=packet['scan_view_count'])
            if kind in ('gold_jar','blue_jar','lotion'):
                packet['grip_only']=os.environ.get('TEAM22_CYLINDER_TRANSFER','1')=='0'
            self.planning_close_command=None
            if kind in ('blue_jar','lotion') and os.environ.get('TEAM22_LOW_CYLINDER_GRASP')=='1':
                packet['low_cylinder_grasp']=True
                packet['low_contact_height_m']=.025
                packet['retry_number']=self.retry_count.get(kind,1)
                if kind=='lotion' and OPERATOR_PRESHAPE:
                    packet['arm_side']=OPERATOR_PRESHAPE['side']
                    packet['approach_preshape']=list(OPERATOR_PRESHAPE['command'])
                    self.planning_close_command=CYLINDER_CLOSE.tolist()
                elif kind in ('gold_jar','blue_jar'):
                    packet['arm_side']='right' if kind=='gold_jar' else 'left'
                    packet['approach_preshape']=list(CYLINDER_APPROACH)
                    self.planning_close_command=CYLINDER_CLOSE.tolist()
                elif OPERATOR_PRESHAPE:
                    packet['arm_side']=OPERATOR_PRESHAPE['side']
                    packet['approach_preshape']=list(OPERATOR_PRESHAPE['command'])
                    self.planning_close_command=CYLINDER_CLOSE.tolist()
                else:
                    packet['approach_preshape']=[.9512,.9501,.9455,.9591,.5842,.0593]
                    self.planning_close_command=CYLINDER_CLOSE.tolist()
                packet['close_command']=list(self.planning_close_command)
            elif kind=='gold_jar':
                # Gold jar has a verified successful side-wrap prior. Preserve
                # its reachable contact height while explicitly abducting the
                # thumb before approach; low-contact search was unreachable.
                packet['arm_side']='right'
                packet['approach_preshape']=list(CYLINDER_APPROACH)
                self.planning_close_command=CYLINDER_CLOSE.tolist()
                packet['close_command']=list(self.planning_close_command)
            elif OPERATOR_PRESHAPE and kind in ('blue_jar','lotion'):
                packet['operator_preshape']=OPERATOR_PRESHAPE
                operator_close=CYLINDER_CLOSE.tolist()
                packet['close_command']=operator_close
                self.planning_close_command=operator_close
            elif kind in ('mouse','white_mouse'):
                self.planning_close_command=[.24,.24,.24,.24,.10,.45]
                packet['close_command']=list(self.planning_close_command)
            packet['allow_object_interference']=os.environ.get('TEAM22_ALLOW_OBJECT_INTERFERENCE')=='1'
            packet['retry_number']=self.retry_count.get(kind,1)
            if kind=='mouse' and packet['retry_number']>1:
                packet['arm_side']='left' if packet['retry_number']%2 else 'right'
            elif kind=='mouse':
                packet['arm_side']='left'
            if kind in ('gold_jar','blue_jar') and self.lab.latest_frames:
                from jar_tracker import JarTracker
                obj=next((o for o in scene['objects'] if o['kind']==kind and o.get('status')=='observed'),None)
                if obj:
                    frames,frame_q,frame_t=self.lab.latest_frames
                    tracker=JarTracker(obj,scene['plane']);obs=tracker.observe(frames,frame_q,frame_t,obj['xyz'])
                    if obs.get('valid'):packet['cylinder_xy']=obs['xyz'][:2]
                    self.event(now,'Оценка оси цилиндра по цвету и глубине',vision=obs)
            if self.lab.output:
                np.savez_compressed(self.lab.output/f'planning-input-{self.attempt}.npz',
                    obstacles=np.array(packet['obstacles']),
                    metadata=json.dumps({k:v for k,v in packet.items() if k!='obstacles'},ensure_ascii=False))
            self.future = self.pool.submit(plan, packet)
            self.plan_submit_wall=time.monotonic()
            self.want_map = False
            self.enter('planning', now, 'Расчёт подхода и переноса: '+kind, scene=scene, angles=q)
            self.event(now, 'Назначение из /sim/task', task=self.lab.state.task)
        elif self.stage == 'plan_preshape':
            confirmed={};targets=self.plan_preshape_targets or self.planning_preshape(self.current['kind']) or []
            for side,target in targets:
                sample=self.lab.timeline.latest('hand_'+side)
                if (sample is not None and now-sample[0]<.2 and sample[0]-self.entered>=.35
                        and np.max(abs(np.asarray(sample[2]['position'])-target))<=.025):
                    confirmed[side]=sample[2]['position']
            if len(confirmed)==len(targets):
                self.enter('plan_ready',now,'Отведение больших пальцев подтверждено датчиками; строю маршрут',
                           actual=confirmed)
            elif now-self.entered>4.:
                self.lab.controller.stop('Не подтвердилось отведение большого пальца по датчикам',now)
        elif self.stage == 'planning' and self.future.done():
            result = self.future.result(); self.future = None
            if result.get('grasp')=='cylinder_side' and self.planning_close_command is not None:
                result['close_command']=list(self.planning_close_command)
            if (result.get('grasp')=='cylinder_side'
                    and os.environ.get('TEAM22_LOW_CYLINDER_GRASP')=='1'
                    and OPERATOR_PRESHAPE):
                result['operator_preshape']=OPERATOR_PRESHAPE['id']
            elapsed_ms=None if self.plan_submit_wall is None else 1000*(time.monotonic()-self.plan_submit_wall)
            self.plan_submit_wall=None
            self.event(now, 'Результат планирования',
                       planner_wall_ms=None if elapsed_ms is None else round(elapsed_ms,1),
                       planner_compute_ms=result.get('planning_ms'),
                       plan={k:v for k,v in result.items() if k!='obstacles'})
            if 'error' in result:
                if os.environ.get('TEAM22_STOP_AFTER_ATTEMPT')=='1':
                    self.finish(now,result['error']);return
                self.blockers.setdefault(self.current['kind'],[]).append(dict(
                    attempt=self.retry_count.get(self.current['kind'],0),reason=result['error'],
                    failures=result.get('failures'),
                    clutter_fallback=result.get('object_interference_fallback_failures')))
                # A rejected plan did not move an object or either arm, so the
                # existing multi-view scan remains valid for the next candidate.
                self.scan_available=True
                self.deferred.append(result.get('kind',self.current['kind']))
                self.enter('next', now, result['error']); return
            self.current = result
            if result['kind']=='tube_cap':self.tube_reference=result['original']
            self.lab.controller.models={s:Arm(s) for s in ('left','right')}
            if result.get('pocket') is not None:
                self.lab.controller.models[result['side']].pocket=np.array(result['pocket'])
            if result.get('grasp')=='geometry_pinch' or result.get('palm_roll',0.) or result.get('operator_preshape') or (result.get('grasp')=='cylinder_side' and result['kind']!='gold_jar'):
                from team22_bringup.hand_geometry import HandGeometry
                model=self.lab.controller.models[result['side']]
                model.hand_geometry=HandGeometry(result['side'])
                model.finger_angles=self.lab.timeline.latest('joints')[2]
                if result.get('thin_object'):model.minimum_table_gap=CLEARANCE_M
                if result.get('hand_object_margin') is not None:model.minimum_object_gap=CLEARANCE_M
            if result.get('clearance_m') is not None and result.get('side') in self.lab.controller.models:
                model=self.lab.controller.models[result['side']]
                model.minimum_table_gap=model.minimum_object_gap=result['clearance_m']
            if result.get('grasp')=='cylinder_side':
                from jar_tracker import JarTracker
                self.lab.controller.models[result['side']].pocket=np.array(result['pocket'])
                self.lab.controller.models[result['side']].extra_hand_points=result['extra_hand_points']
                if result['kind']=='lotion':
                    from object_tracker import ObjectTracker
                    self.tracker=ObjectTracker(result['original'],result['plane'])
                else:self.tracker=JarTracker(result['original'],result['plane'])
                self.grip_pose=None
                self.released_expected=None
            elif result.get('grasp')=='geometry_pinch' and (result['kind'] in ('mouse','cable','lotion','tube_cap') or result['kind'].startswith('figure_')):
                from object_tracker import ObjectTracker
                self.tracker=ObjectTracker(result['original'],result['plane'])
                self.grip_pose=None
                if self.lab.latest_frames:
                    frames,frame_q,frame_t=self.lab.latest_frames
                    self.tracker.observe(frames,frame_q,frame_t,result['original']['xyz'])
            if self.tracker is not None and not self.tracker.baseline and self.lab.latest_frames:
                frames,frame_q,frame_t=self.lab.latest_frames
                self.tracker.observe(frames,frame_q,frame_t,result['original']['xyz'])
            # No object is moved unless a fresh observation still matches the planned source.
            scene = self.scene(now)
            obj = next((o for o in (scene or {}).get('objects',[]) if o['kind']==result['kind'] and o.get('status')=='observed'), None)
            if obj is None or np.linalg.norm(np.array(obj['xyz'])-result['original']['xyz'])>.018:
                self.enter('next', now, 'Предмет исчез или сместился за время планирования'); return
            if result.get('grasp')=='two_palm_support':
                from bimanual import models
                self.lab.controller.models=models(self.lab.timeline.latest('joints')[2])
                from object_tracker import ObjectTracker
                self.tracker=ObjectTracker(result['original'],result['plane'])
                self.grip_pose=None;self.grip_object_offset=None
                if self.lab.latest_frames:
                    frames,frame_q,frame_t=self.lab.latest_frames
                    self.tracker.observe(frames,frame_q,frame_t,result['original']['xyz'])
            self.calibration_sides=(('left','right') if result.get('grasp')=='two_palm_support'
                                    else (result['side'],))
            for side in self.calibration_sides:self.fingers(side,CLOSE)
            self.enter('calibrate_grasp',now,'Измеряю пустую кисть после успешного плана',
                       calibration_sides=list(self.calibration_sides))
        elif self.stage=='pregrasp_raise' and c.status=='holding':
            self.prepare_fingers(now,minimum=CLEARANCE_M)
        elif self.stage=='pinch_preshape':
            hand=self.lab.timeline.latest('hand_'+self.current['side'])[2]
            if np.max(abs(np.asarray(hand['position'])-self.current['approach_preshape']))>.025:
                if now-self.entered>4.:c.stop('Не достигнута форма кисти перед подходом',now)
                return
            if self.tracker is not None and not self.tracker.baseline:
                if now-self.entered>4.:c.stop('Нет исходного наблюдения поверхности предмета',now)
                return
            self.move(self.current['approach'],now,'approach','Подход к тонкому предмету с заранее развёрнутым большим пальцем')
        elif self.stage=='preshape' and now-self.entered>1.:
            if min(self.lab.timeline.latest('hand_'+self.current['side'])[2]['position'])<.9:
                if now-self.entered>4.:c.stop('Кисть не раскрылась полностью перед подходом',now)
                return
            if not self.tracker.baseline:
                if now-self.entered>3.:c.stop('Нет надёжного исходного наблюдения банки',now)
                return
            self.move(self.current['approach'],now,'approach','Боковой подход к вертикальной банке')
        elif self.stage in ('approach','opposition') and c.status == 'holding':
            if self.stage=='opposition':
                hand=self.lab.timeline.latest('hand_'+self.current['side'])[2]
                if now-self.entered<.7:return
                opposition=self.current.get('approach_preshape') or CYLINDER_APPROACH
                if np.max(abs(np.array(hand['position'])-opposition))>.025:
                    if now-self.entered>4.:c.stop('Большой палец не вышел в положение бокового обхвата',now)
                    return
            if self.current.get('grasp')=='cylinder_side':
                obs=self.tracker.latest
                if not obs or not obs.get('valid') or not 0<=now-obs['stamp']<.4:
                    c.stop('Перед сжатием банка не подтверждается камерой',now);return
                model=c.models[self.current['side']];q=self.lab.timeline.latest('joints')[2]
                error=np.linalg.norm(model.tool([q[k] for k in model.names])[:3,3]-self.current['target'])
                if error>.002:
                    c.stop('Кисть отклонилась от места бокового обхвата более чем на 2 мм',now);return
                if (self.current.get('operator_preshape') or
                        (self.current.get('kind') in ('gold_jar','blue_jar') and
                         self.current.get('grasp')=='cylinder_side')):
                    from operator_preshape import palm_entry
                    entered,detail=palm_entry(self.current,model,q,obs)
                    self.event(now,'Проверка входа цилиндра в ладонь',accepted=entered,**detail)
                    if not entered:
                        c.stop('Предмет ещё не вошёл в ладонь или его верх ниже большого пальца',now);return
                self.event(now,'Кисть у боковой поверхности банки',pocket_error_mm=float(error*1000),vision=obs)
                if self.stage=='approach':
                    self.event(now,'Низкий подход завершён раскрытой кистью',
                        geometry=self.current.get('grasp_geometry'),
                        fingers=self.lab.timeline.latest('hand_'+self.current['side'])[2])
                    self.fingers(self.current['side'],self.current.get('approach_preshape') or CYLINDER_APPROACH)
                    self.enter('opposition',now,'Подвожу большой палец сбоку; остальные пока раскрыты')
                    return
            if self.stage=='approach':
                self.fingers(self.current['side'],self.current.get('approach_preshape') or CYLINDER_APPROACH)
                self.enter('opposition',now,'Сначала разворачиваю большой палец; остальные раскрыты')
                return
            self.fingers(self.current['side'], self.current.get('close_command',CLOSE))
            self.enter('contact', now, 'Сжатие и проверка противоположных пальцев')
        elif self.stage == 'contact':
            ok, data = self.contact()
            if ok and data.get('settled',False):
                if self.confirmed is None: self.confirmed = now
                elif now-self.confirmed > .35:
                    self.event(now, 'Противоположный контакт подтверждён', fingers=data)
                    self.maintain_grip=True
                    if tracker is not None:
                        model=c.models[self.current['side']];q=self.lab.timeline.latest('joints')[2]
                        actual=np.array([q[k] for k in model.names]);pose=model.tool(actual)
                        self.grip_pose=pose[:3,3].copy()
                        self.bind_visible_object(pose,now)
                        from side_grasp import solve_side
                        lift=solve_side(model,self.grip_pose+[0,0,.04],pose[:3,:3],actual)
                        if lift is None:
                            if hasattr(self.tracker,'carrying'):self.tracker.carrying=False
                            c.stop('Не найден подъём от фактического положения кисти',now);return
                        self.event(now,'Измеренное положение перед подъёмом',pocket=self.grip_pose.tolist(),vision=self.tracker.latest)
                    else:lift=self.current['approach'][-2]
                    self.lift_visual_count=0;self.lift_visual_stamp=None
                    if tracker is not None and hasattr(tracker,'carrying'):tracker.carrying=True
                    self.move([lift], now, 'lift', 'Пробный подъём на 4 см'); return
            else: self.confirmed = None
            if now-self.entered > 8.:
                self.event(now, 'Захват не подтверждён; подъём запрещён', fingers=data)
                self.fingers(self.current['side'], self.current.get('approach_preshape') or np.ones(6))
                self.enter('open_source', now, 'Раскрытие кисти у исходного предмета')
        elif self.stage == 'lift' and c.status == 'holding':
            self.lift_visual_count=0;self.lift_visual_stamp=None
            self.enter('verify', now, 'Проверяю камерой подъём предмета вместе с кистью')
        elif self.stage == 'verify':
            ok, data = self.contact(); scene = self.scene(now)
            if ok and tracker is not None and hasattr(tracker,'carrying'):
                tracker.carrying=True
            obj = next((o for o in (scene or {}).get('objects',[]) if o['kind']==self.current['kind'] and o.get('status')=='observed'), None)
            delta = None if obj is None else np.array(obj['xyz'])-self.current['original']['xyz']
            visual = delta is not None and .022<delta[2]<.070 and np.linalg.norm(delta[:2])<.04
            if tracker is not None:
                from jar_tracker import lift_visible
                visual=lift_visible(self.tracker.latest,now,allow_partial=self.current['kind']=='lotion')
                delta=np.array(self.tracker.latest['delta']) if self.tracker.latest and 'delta' in self.tracker.latest else None
            if visual:
                stamp=self.tracker.latest.get('stamp') if tracker is not None and self.tracker.latest else self.lab.map_stamp
                if stamp is not None and stamp != self.lift_visual_stamp:
                    self.lift_visual_count+=1
                    self.lift_visual_stamp=stamp
            else:self.lift_visual_count=0
            visual_confirmed=(visual and self.lift_visual_count>=2) if tracker is not None else visual
            if ok and visual_confirmed:
                self.event(now, 'Подъём подтверждён камерой и пальцами', delta=delta.tolist(),
                           visual_frames=self.lift_visual_count,fingers=data)
                if tracker is not None:
                    model=c.models[self.current['side']];q=self.lab.timeline.latest('joints')[2]
                    self.bind_visible_object(model.tool([q[k] for k in model.names]),now)
                if self.current['transport']:
                    if hasattr(self.tracker,'carrying'):self.tracker.carrying=True
                    if self.current.get('kind')=='lotion' and self.tracker is not None:
                        q=self.lab.timeline.latest('joints')[2];observation=self.tracker.latest
                        if observation and observation.get('valid'):
                            self.resume_motion_anchor=dict(expected=self.held_center(q).copy(),
                                xyz=np.asarray(observation['xyz']).copy(),stamp=observation['stamp'])
                            self.resume_motion_concordance=0;self.resume_motion_failures=0
                    path=self.current['transport'] if self.current.get('grasp') in ('cylinder_side','geometry_pinch') else [self.current['approach'][1]]+self.current['transport']
                    self.move(path, now, 'transport', 'Перенос в '+self.current['tray'])
                else:
                    self.event(now, self.current['transfer_reason'])
                    self.move([self.current['approach'][-1]], now, 'return_source', 'Возврат на исходное место: путь в лоток не проверен')
            elif not ok or now-self.entered>2.5:
                self.event(now, 'Подъём не подтверждён обоими каналами',
                           fingers=data,delta=None if delta is None else delta.tolist(),
                           visual_frames=self.lift_visual_count,vision=tracker.latest if tracker else None)
                self.blockers.setdefault(self.current['kind'],[]).append(dict(
                    attempt=self.retry_count.get(self.current['kind'],0),
                    reason='Подъём не подтверждён контактами и камерой',
                    delta_mm=None if delta is None else (delta*1000).round(1).tolist(),
                    visual_frames=self.lift_visual_count))
                if not ok:
                    self.fingers(self.current['side'], self.current.get('approach_preshape') or np.ones(6))
                    self.enter('open_source', now, 'Контакт потерян: раскрываю кисть над исходной точкой без опускания')
                else:
                    self.move([self.current['approach'][-1]], now, 'return_source', 'Опускаю удерживаемый предмет обратно перед раскрытием')
        elif self.stage=='carry_replan':
            ok,data=self.contact()
            if not ok:
                self.recover_after_drop(now,'Потерян контакт при пересчёте маршрута удерживаемого флакона');return
            future=self.carry_plan_future
            if future is None:
                c.stop('Нет расчёта нового пути для удерживаемого флакона',now);return
            if future.done():
                try:result=future.result()
                except Exception as error:
                    c.stop('Ошибка нового маршрута флакона: '+str(error),now);return
                self.carry_plan_future=None
                if result.get('error') or not result.get('route'):
                    self.event(now,'Свежий путь флакона отклонён',plan_error=result.get('error'),failures=result.get('failures'))
                    c.stop('Новый путь флакона не прошёл проверку: '+result.get('error','пустой маршрут'),now);return
                self.current['transport']=result['route'];self.current['release_site']=result['release_site']
                self.current['release_center']=result['release_center'];self.current['obstacles']=result['obstacles']
                self.current['plane']=result['plane'];self.current['table_edge']=result['table_edge']
                self.event(now,'Новый маршрут проверен по карте после захвата',waypoints=len(result['route']),
                           release_site=result['release_site'],release_center=result['release_center'],
                           obstacle_count=result['obstacle_count'])
                self.move(result['route'],now,'transport','Перенос флакона по свежему трёхракурсному измерению')
            elif now-self.entered>60.:
                c.stop('Истёк расчёт свежего маршрута удерживаемого флакона',now)
        elif self.stage == 'transport':
            ok, data = self.contact()
            if not ok:
                self.blockers.setdefault(self.current['kind'],[]).append(dict(
                    attempt=self.retry_count.get(self.current['kind'],0),
                    reason='Потерян контакт при переносе',fingers=data))
                self.recover_after_drop(now,'Потерян контакт при переносе');return
            elif (self.current.get('grasp')=='cylinder_side' and c.status!='holding' and
                  (not self.tracker.latest or not (self.tracker.latest.get('valid') or self.tracker.latest.get('partial_visible')) or now-self.tracker.latest['stamp']>.5)):
                if self.bad_visual_since is None:self.bad_visual_since=now
                elif now-self.bad_visual_since>1.:
                    c.stop('При переносе потеряно наблюдение удерживаемой банки',now)
                    self.event(now,c.reason,vision=self.tracker.latest)
            elif c.status == 'holding':
                q=self.lab.timeline.latest('joints')[2]
                allowed,detail=self.release_gate(now,q)
                if not allowed:
                    if not getattr(self,'release_scan_requested',False):
                        self.release_scan_requested=True;self.release_gate_wait_started=now
                        self.event(now,'Жду свежий текущий кадр перед решением об отпускании',**detail)
                        return
                    if now-self.release_gate_wait_started<1.:
                        return
                    self.event(now,'Проверка точки сброса не пройдена; кисть остаётся сжатой',**detail)
                    c.stop('Не раскрываю кисть: '+detail['reason'],now);return
                self.released_expected=np.asarray(self.current['release_center'],float)
                self.event(now,'Кисть и предмет подтверждены над выбранной точкой в лотке',**detail)
                self.fingers(self.current['side'], np.ones(6))
                self.enter('release', now, 'Раскрытие над назначенным лотком')
            else:self.bad_visual_since=None
        elif self.stage == 'return_source' and c.status == 'holding':
            self.fingers(self.current['side'], self.current.get('approach_preshape') or np.ones(6))
            self.enter('open_source', now, 'Раскрытие у исходной точки')
        elif self.stage == 'open_source' and now-self.entered>1.5:
            target=self.current.get('approach_preshape') or np.ones(6)
            if np.max(abs(np.array(self.lab.timeline.latest('hand_'+self.current['side'])[2]['position'])-target))>.1:
                if now-self.entered>4: c.stop('Кисть не раскрылась перед отходом', now)
                return
            if tracker is not None and hasattr(tracker,'carrying'):tracker.carrying=False
            self.move(self.current.get('recovery_route',list(reversed(self.current['approach'][:-1]))+[self.current['start']]), now, 'retreat', 'Отход раскрытой кисти по проверенному пути')
        elif self.stage == 'release' and now-self.entered>2.:
            if min(self.lab.timeline.latest('hand_'+self.current['side'])[2]['position'])<.9:
                if now-self.entered>4.:c.stop('Кисть не раскрылась перед отходом от лотка',now)
                return
            self.released=True
            if self.tracker is not None and hasattr(self.tracker,'carrying'):self.tracker.carrying=False
            path=[self.current['transport'][-2]] if self.short_retreat else self.current.get('retreat',list(reversed(self.current['transport'][:-1]))+list(reversed(self.current['approach'][:-1]))+[self.current['start']])
            self.move(path, now, 'retreat', 'Отход после отпускания предмета')
        elif self.stage == 'retreat' and c.status == 'holding':
            if self.released:self.scene_changed=True
            self.event(now, 'Кисть вернулась; проверка сцены после попытки', scene=self.scene(now))
            placement_verified=None
            if getattr(self,'released_expected',None) is not None:
                obs=self.tracker.latest;bounds=np.array(self.current['tray_observation']['bounds'])
                inside=None
                if obs and (obs.get('valid') or obs.get('partial_visible')) and now-obs['stamp']<.4:
                    xy=np.array(obs['xyz'])[:2];release=np.asarray(self.current.get('release_center',self.released_expected))
                    source=np.asarray(self.current['original']['bounds']);radius=float(np.max(np.ptp(source,axis=0)[:2])/2+CLEARANCE_M)
                    inside=bool(np.linalg.norm(np.asarray(obs['xyz'])-release)<=.030 and
                        np.all(xy>bounds[0,:2]+radius) and np.all(xy<bounds[1,:2]-radius))
                self.event(now,'Проверяю камерой укладку в лотке',inside=inside,vision=obs,
                           selected_release_center=self.current.get('release_center'))
                placement_verified=inside
                if inside is not True:
                    self.blockers.setdefault(self.current['kind'],[]).append(dict(
                        attempt=self.retry_count.get(self.current['kind'],0),
                        reason='Камера не подтвердила укладку внутри лотка',inside=inside,
                        release_center=self.current.get('release_center')))
            # Pinch-only trials retreat with the prepared thumb/fingers. Restore
            # a fully open hand only after reaching the checked clear pose.
            self.fingers(self.current['side'],np.ones(6))
            if os.environ.get('TEAM22_STOP_AFTER_ATTEMPT')=='1':
                self.finish(now,'Попытка закончена. Ожидаю разбора перед следующим движением.');return
            kind=self.current.get('kind')
            self.aligned_objects.discard(kind)
            if self.released and self.current.get('original'):
                removed=self.remove_cached_object_points(self.current['original'].get('bounds'))
                self.scene_cache.pop(kind,None)
                self.event(now,'Убираю прежнюю точку уложенного предмета из сохранённой карты',
                           kind=kind,removed_points=removed)
            retry_needed=(not self.released or placement_verified is not True)
            if retry_needed and self.retry_count.get(kind,0)<3:
                self.queue.appendleft(kind)
                self.retrying_kind=kind
                self.event(now,'После отхода повторно беру этот предмет другим вариантом',
                           retry=self.retry_count.get(kind,0)+1,placed=placement_verified,
                           blocker=self.blockers.get(kind))
            self.enter('next', now, 'Следующий предмет')

    def snapshot(self):
        return dict(stage=self.stage, attempt=self.attempt, kind=self.current and self.current.get('kind'),
                    planner_ready=self.planner_ready,
                    planner_warmup_ms=self.planner_warmup_ms,
                    waiting_for_scene_reload=self.wait_for_scene_reload,
                    requires_ready=self.requires_ready(),direct_start=getattr(self,'direct_start',None),
                    preshape=self.current and self.current.get('operator_preshape'),
                    palm_entry=next(({k:e[k] for k in ('accepted','pocket_error_mm','top_above_thumb_mm','pocket_inside_object_xy') if k in e}
                                     for e in reversed(self.events) if e.get('message')=='Проверка входа цилиндра в ладонь' and e.get('kind')==(self.current or {}).get('kind')),None),
                    jar_tracking=self.tracker.latest if self.tracker else None,
                    events=[{k:e[k] for k in ('sim','stage','kind','message')} for e in self.events[-12:]], remaining=list(self.queue))

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)

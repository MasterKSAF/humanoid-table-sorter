"""Camera-derived upright-cylinder side grasp; no simulator object transforms."""
import time
import numpy as np

CYLINDER_APPROACH=[.9512,.9501,.9455,.9591,.5842,.0593]
CYLINDER_CLOSE=[.25,.25,.25,.25,.10,.45]
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from team22_bringup.scene_geometry import planning_edge
from team22_bringup.grasp import Arm, CLEARANCE_M


def orientation(yaw,side='right'):
    # Right hand: fingers forward, palm inward, finger row vertical (thumb up).
    sign=1 if side=='right' else -1
    fingers=np.array([np.cos(yaw),sign*np.sin(yaw),0.])
    palm=np.array([-np.sin(yaw),sign*np.cos(yaw),0.])
    return np.column_stack((palm,np.cross(fingers,palm),fingers))


def solve_side(model,goal,rotation,seed):
    """Avoid the folded-behind-torso IK branch while solving the redundant arm."""
    def residual(q):
        pose=model.tool(q);ss,rr=model.samples(q)
        stride=1 if getattr(model,'strict_body',False) else 4
        ss,rr=ss[32::stride],rr[32::stride]
        distances=np.column_stack((.11+rr-ss[:,0],ss[:,0]+.19+rr,
            .18+rr-abs(ss[:,1]),ss[:,2]+.22+rr,.43+rr-ss[:,2]))
        penetration=np.maximum(0.,distances.min(axis=1))
        return np.r_[pose[:3,3]-goal,.15*Rotation.from_matrix(rotation.T@pose[:3,:3]).as_rotvec(),
                     .001*(q-seed),2*penetration]
    generic_a=np.array([-.5,-.4,0.,-1.4,.5,.3,.5])
    generic_b=np.array([-.9,-.7,-1.,-1.2,1.2,.3,-.3])
    if model.side=='left':
        mirror=np.array([1,-1,-1,1,-1,1,-1])
        generic_a=generic_a*mirror;generic_b=generic_b*mirror
    live_seed=getattr(model,'live_seed',None)
    guesses=[generic_a,seed]
    if live_seed is not None:guesses.append(np.asarray(live_seed,dtype=float))
    guesses.append(generic_b)
    if getattr(model,'prefer_seed',False):
        guesses=[seed]+([np.asarray(live_seed,dtype=float)] if live_seed is not None else [])+[generic_a,generic_b]
    for initial in guesses:
        fit=least_squares(residual,np.clip(initial,model.limits[0]+.09,model.limits[1]-.09),
                          bounds=(model.limits[0]+.08,model.limits[1]-.08),max_nfev=130)
        error=residual(fit.x)
        if np.linalg.norm(error[:3])<.004 and np.linalg.norm(error[3:6])<.0075 and max(error[13:],default=0)<.003:
            return fit.x
    return None


def plan_cylinder(packet):
    from mission_scene import observed_objects
    from transfer_trial import destination
    from tray_placement import release_sites
    packet=dict(packet)
    packet.setdefault('_deadline',time.monotonic()+18.)
    objects=observed_objects(packet);kind=packet['kind'];tray_kind=destination(packet['task'],kind)
    if kind not in objects or tray_kind not in objects:return _plan_cylinder_at_site(packet)
    sites=release_sites(objects[kind],objects[tray_kind],packet['obstacles'])
    preferred=packet.get('release_site')
    if preferred is not None and sites:
        index=min(range(len(sites)),key=lambda i:np.linalg.norm(np.asarray(sites[i])[:2]-np.asarray(preferred)[:2]))
        if np.linalg.norm(np.asarray(sites[index])[:2]-np.asarray(preferred)[:2])<.002:
            sites=[sites[index]]+[site for i,site in enumerate(sites) if i!=index]
    if not sites:
        if packet.get('allow_lift_only') or packet.get('grip_only'):
            return _plan_cylinder_at_site(dict(packet,grip_only=True))
        return dict(kind=kind,error='В назначенном лотке не найдено свободного места с запасом под размер предмета')
    # A tray gives up to nine grid points. Try the best two under one shared
    # deadline so a bad grasp cannot spend minutes repeating the same route.
    exact_preference=(preferred is not None and
        np.linalg.norm(np.asarray(sites[0])[:2]-np.asarray(preferred)[:2])<.002)
    site_limit=int(packet.get('max_release_sites',2))
    if exact_preference and not packet.get('search_alternate_sites'):
        site_limit=1
    sites=sites[:site_limit]
    attempts=[];lift_only=None;result=None
    for site in sites:
        if time.monotonic()>=packet['_deadline']:break
        result=_plan_cylinder_at_site(dict(packet,release_site=site))
        attempts.append(dict(xyz=site,error=result.get('error'),failures=result.get('failures')))
        if 'error' not in result:
            if result.get('transport') or packet.get('grip_only'):
                result['placement_attempts']=attempts;return result
            if lift_only is None:lift_only=result
        elif result.get('failures') and all(not any(word in str(reason) for word in ('transport','payload')) for _,reason in result['failures']):
            break  # An approach failure cannot be fixed by another release site.
    result=lift_only or result or dict(kind=kind,error='Исчерпан бюджет планирования безопасного переноса')
    result['placement_attempts']=attempts
    return result


def _plan_cylinder_at_site(packet):
    from transfer_trial import checked,destination,body_clear
    solve=solve_side
    begin=time.monotonic();deadline=float(packet.get('_deadline',begin+18.))
    scene=packet['scene'];kind=packet['kind'];angles=packet['angles']
    profile=packet.get('profile','legacy')
    lowering={'wide-low':.008,'wide-mid':.004,'wide-high':0.,'complete':0.}.get(profile,.008)
    margin=CLEARANCE_M
    from mission_scene import observed_objects
    objects=observed_objects(packet)
    tray_kind=destination(packet['task'],kind)
    if kind not in objects or tray_kind not in objects:
        return dict(kind=kind,error='Банка или назначенный лоток не видны')
    obj=objects[kind];tray=objects[tray_kind];bounds=np.array(obj['bounds'])
    if bounds[1,2]-bounds[0,2]<.085:
        return dict(kind=kind,error='Банка не стоит вертикально или её высота не измерена')
    side=packet.get('arm_side','left' if obj['xyz'][1]>.12 else 'right')
    sign=1 if side=='right' else -1
    model=Arm(side);model.pocket=np.array(packet.get('pocket',[.035,0.,.134 if kind=='gold_jar' else .140]))
    # Enclose the thumb fully spread, opposed, and halfway through its sweep.
    model.extra_hand_points=[(.031,sign*.096,.136),(.057,sign*.087,.137),(.076,sign*.070,.137)]
    parked=Arm('left' if side=='right' else 'right');other=cKDTree(parked.samples([angles[k] for k in parked.names])[0])
    start=np.array([angles[k] for k in model.names]);model.live_seed=start.copy();points=np.array(packet['obstacles'])
    transit_table=np.array(scene['plane']),planning_edge(scene),cKDTree(points)
    inside=np.all((points>=bounds[0]-.014)&(points<=bounds[1]+.014),axis=1)
    target_points=cKDTree(points[inside])
    points=points[~inside];table=np.array(scene['plane']),planning_edge(scene),cKDTree(points)
    # Lower than the previous successful grasp, retaining all collision margins.
    target=np.array(obj['xyz'])
    object_height=float(bounds[1,2]-bounds[0,2])
    if packet.get('low_cylinder_grasp'):
        contact_height=float(np.clip(packet.get('low_contact_height_m',.035),.025,
                                     max(.025,object_height-.020)))
    else:
        contact_height=.078 if kind=='gold_jar' else float(np.clip(object_height*.55,.062,.115))
        contact_height+=packet.get('height_adjustment',0.)
        contact_height=min(contact_height,object_height-.020)
    target[2]=bounds[0,2]+contact_height-lowering
    if 'cylinder_xy' in packet:target[:2]=packet['cylinder_xy']
    tray_xyz=np.array(tray['xyz']);tray_bounds=np.array(tray['bounds'])
    payload_radius=max(np.ptp(bounds,axis=0)[:2])/2+margin
    inner=payload_radius
    if tray_bounds[1,1]-tray_bounds[0,1]>2*inner:
        # Any fully contained placement is valid. Prefer the nearer end of the
        # observed tray instead of requiring an often unreachable tray centre.
        tray_xyz[1]=np.clip(target[1],tray_bounds[0,1]+inner,tray_bounds[1,1]-inner)
    if packet.get('release_site') is not None:tray_xyz=np.asarray(packet['release_site'])
    failures=[];lift_only=None
    # Use horizontal insertion along the palm normal, never descend onto the jar.
    roll=float(packet.get('palm_roll',0.))
    if roll or kind!='gold_jar' or packet.get('approach_preshape'):
        from team22_bringup.hand_geometry import HandGeometry
        model.hand_geometry=HandGeometry(side);model.finger_angles=angles
        if packet.get('approach_preshape'):
            model.finger_angles=dict(angles,**model.hand_geometry.nominal_angles(packet['approach_preshape']))
    for yaw in packet.get('cylinder_yaws',(1.1,1.25,1.,.9,.8,.75,1.4,0.,.4,1.8,2.2,-.4,-.8,-1.2)):
        if time.monotonic()>=deadline:
            failures.append([yaw,'planner time budget']);break
        rot=orientation(yaw,side)@Rotation.from_euler('z',roll).as_matrix();normal=rot[:,0]
        if packet.get('terminal_first'):
            terminal_seed=np.asarray(packet.get('terminal_seed',start))
            if 'terminal_seed' in packet:model.prefer_seed=True
            terminal=solve(model,target,rot,terminal_seed)
            terminal_error=None
            if terminal is None:
                terminal_error='terminal IK'
            elif model.clearance(terminal,table,other)<margin:
                terminal_error=f'terminal clearance {model.clearances(terminal,table,other)}'
            elif not body_clear(model,terminal):
                terminal_error='terminal body clearance'
            if terminal_error:
                failures.append([yaw,terminal_error]);continue
            if hasattr(model,'hand_geometry'):
                from geometry_grasp import open_target_clear
                if not open_target_clear(model,terminal,terminal,target_points,minimum=margin,hand_minimum=margin):
                    failures.append([yaw,'open target intersection at terminal']);continue
            from grasp_shape_checks import closure_clear
            safe,detail=closure_clear(model,terminal,table,angles,packet.get('close_command',[.25]*4+[.10,.45]),minimum=margin,initial_command=packet.get('approach_preshape'))
            if not safe:
                failures.append([yaw,'closure sweep: '+detail]);continue
        near=target-normal*.065;near[2]+=(near[:2]-target[:2])@table[0][:2]
        insertion=[]
        for gap in (.055,.045,.035,.025,.015,.005,0.):
            pos=target-normal*gap;pos[2]+=(pos[:2]-target[:2])@table[0][:2]
            insertion.append(pos)
        air_entry=[] if packet.get('direct_object_entry',kind!='gold_jar') else [np.array([.40,-sign*.34,.22])]
        goals=air_entry+[near+[0,0,z] for z in (.20,.15,.10,.05,0.)]+insertion
        route=[];seed=start.copy();failure=None;entry_index=0
        if packet.get('backward_approach'):
            model.prefer_seed=True
            backward=[terminal];seed=terminal
            for backward_index,goal in enumerate(list(reversed(insertion[:-1]))+[near+[0,0,z] for z in (0.,.05,.10,.15,.20)]):
                if time.monotonic()>=deadline:
                    failure='planner time budget';break
                fit=solve(model,goal,rot,seed)
                if fit is None or not checked(model,seed,fit,table,other,margin):
                    failure=f'backward insertion {backward_index}: '+('IK' if fit is None else str(model.clearances(fit,table,other)));break
                backward.append(fit);seed=fit
            if failure:
                failures.append([yaw,failure]);continue
            from grasp_transit import transit
            entry=transit(model,start,backward[-1],transit_table,other,margin,
                          max_rrt_seconds=max(1.,min(5.,deadline-time.monotonic())))
            if entry is None:
                failures.append([yaw,'backward entry unavailable']);continue
            route=entry+list(reversed(backward[:-1]));entry_index=len(entry)-1;seed=route[-1]
        for i,goal in enumerate([] if packet.get('backward_approach') else goals):
            if time.monotonic()>=deadline:
                failure='planner time budget';break
            fit=solve(model,goal,rot,seed)
            if i==0 and fit is not None:
                from grasp_transit import transit
                entry=transit(model,seed,fit,transit_table,other,margin,
                              max_rrt_seconds=max(1.,min(5.,deadline-time.monotonic())))
                if entry is not None:
                    route.extend(entry);entry_index=len(route)-1;seed=fit;continue
            if fit is None or not checked(model,seed,fit,table,other,margin):
                failure=f'approach {i} '+('ik' if fit is None else
                    f'clearance {model.clearances(fit,table,other)} body {body_clear(model,fit)}');break
            route.append(fit);seed=fit
        if failure:
            failures.append([yaw,failure]);continue
        approach_waypoints_before=len(route)
        if time.monotonic()<deadline:
            route,entry_index=_shortcut_open_route(model,start,route,entry_index,table,other,
                                                    target_points,margin,deadline)
        approach_waypoints_after=len(route)
        lift=solve(model,target+[0,0,.04],rot,seed)
        if lift is None or not checked(model,seed,lift,table,other,margin):
            failures.append([yaw,'lift']);continue
        if (packet.get('allow_lift_only') or packet.get('grip_only')) and lift_only is None:
            recovery=list(reversed(route[entry_index:-1]))
            lift_only=dict(kind=kind,side=side,grasp='cylinder_side',pocket=model.pocket.tolist(),
                grasp_height_m=contact_height,
                height_adjustment=float(packet.get('height_adjustment',0.)),clearance_m=margin,
                approach_preshape=packet.get('approach_preshape'),
                extra_hand_points=model.extra_hand_points,target=target.tolist(),original=obj,
                tray=tray_kind,tray_observation=tray,yaw=yaw,palm_roll=roll,
                grasp_geometry=dict(profile=profile,palm_roll=roll,planning_margin_mm=margin*1000,
                    pocket_above_table_mm=1000*(target[2]-target[:2]@table[0][:2]-table[0][2]),
                    minimum_table_gap_mm=1000*model.clearances(route[-1],table,other)[0]),
                approach=[q.tolist() for q in route],lift=lift.tolist(),transport=[],
                approach_waypoints_before=approach_waypoints_before,
                approach_waypoints_after=approach_waypoints_after,
                retreat=[q.tolist() for q in recovery],recovery_route=[q.tolist() for q in recovery],
                transfer_reason='Учебный подъём: полного пути в лоток нет; возвращаю предмет на исходное место',
                obstacles=points.tolist(),plane=list(scene['plane']),table_edge=table[1],start=start.tolist())
            if packet.get('grip_only'):
                return dict(lift_only,failures=failures,planning_ms=1000*(time.monotonic()-begin))
        # Keep the cylinder upright. Move above all neighbours before crossing.
        source_center=np.asarray(obj['xyz'],float)
        bottom_offset=float(source_center[2]-bounds[0,2])
        high=max(source_center[2]+.08,float(points[:,2].max())+bottom_offset+margin)
        release_z=tray_bounds[0,2]+bottom_offset+margin
        release_center=np.array([tray_xyz[0],tray_xyz[1],release_z])
        source_rotation=orientation(yaw,side)@Rotation.from_euler('z',roll).as_matrix()
        object_offset=source_rotation.T@(source_center-target)
        default_end_yaws=(yaw,0.) if packet.get('low_cylinder_grasp') else (-1.1,-1.4,-.8,-1.7)
        default_crossing_xs=(.46,.50) if packet.get('low_cylinder_grasp') else (.46,.50,.42,.38,.34,.30)
        for end_yaw in packet.get('end_yaws',default_end_yaws):
            if time.monotonic()>=deadline:
                failure='planner time budget';break
            for crossing_x in packet.get('crossing_xs',default_crossing_xs):
                if time.monotonic()>=deadline:
                    failure='planner time budget';break
                centers2=[np.array([source_center[0],source_center[1],high]),
                          np.array([crossing_x,source_center[1],high]),
                          np.array([crossing_x,(source_center[1]+tray_xyz[1])/2,high]),
                          np.array([tray_xyz[0],tray_xyz[1],high]),release_center]
                transfer=[];seed=lift;failure=None
                for i,goal_center in enumerate(centers2):
                    if time.monotonic()>=deadline:
                        failure='planner time budget';break
                    fraction=min(1.,i/3.)
                    rotation=orientation(yaw+(end_yaw-yaw)*fraction,side)@Rotation.from_euler('z',roll).as_matrix()
                    fit=solve(model,goal_center-rotation@object_offset,rotation,seed)
                    if fit is None or not checked(model,seed,fit,table,other,margin):
                        failure=f'transport {i}';break
                    # A cylinder stays upright; radius conservatively includes RGB-D extent.
                    radius=payload_radius
                    for q in np.linspace(seed,fit,15):
                        pose=model.tool(q);pos=pose[:3,3]+pose[:3,:3]@object_offset
                        near=points[np.linalg.norm(points[:,:2]-pos[:2],axis=1)<radius]
                        if len(near) and pos[2]-bottom_offset<near[:,2].max()+margin:
                            failure='payload clearance';break
                    if failure:break
                    transfer.append(fit);seed=fit
                if not failure:
                    transfer_waypoints_before=len(transfer)
                    if time.monotonic()<deadline:
                        transfer=_shortcut_carried_route(model,lift,transfer,table,other,points,
                                                          bottom_offset,payload_radius,margin,deadline)
                    transfer_waypoints_after=len(transfer)
                    recovery=list(reversed(route[entry_index:-1]))
                    retreat=[transfer[-2],route[entry_index]]
                    if not all(checked(model,a,b,table,other,margin) for a,b in zip([transfer[-1],*retreat],retreat)):
                        retreat=list(reversed(transfer[:-1]))+recovery
                    return dict(kind=kind,side=side,grasp='cylinder_side',pocket=model.pocket.tolist(),
                        grasp_height_m=contact_height,
                        height_adjustment=float(packet.get('height_adjustment',0.)),clearance_m=margin,
                        approach_preshape=packet.get('approach_preshape'),
                        extra_hand_points=model.extra_hand_points,
                        target=target.tolist(),original=obj,tray=tray_kind,tray_observation=tray,
                        release_site=tray_xyz.tolist(),release_center=release_center.tolist(),yaw=yaw,palm_roll=roll,
                        grasp_geometry=dict(lowered_mm=lowering*1000,profile=profile,planning_margin_mm=margin*1000,
                            pocket_above_table_mm=1000*(target[2]-target[:2]@table[0][:2]-table[0][2]),
                            minimum_table_gap_mm=1000*model.clearances(route[-1],table,other)[0],
                            approach_fingers=[1.]*6,opposed_fingers=CYLINDER_APPROACH),
                        approach=[q.tolist() for q in route],lift=lift.tolist(),
                        approach_waypoints_before=approach_waypoints_before,
                        approach_waypoints_after=approach_waypoints_after,
                        transport_waypoints_before=transfer_waypoints_before,
                        transport_waypoints_after=transfer_waypoints_after,
                        retreat=[q.tolist() for q in retreat],
                        recovery_route=[q.tolist() for q in recovery],
                        transport=[q.tolist() for q in transfer],transfer_reason=None,
                        obstacles=points.tolist(),plane=list(scene['plane']),table_edge=table[1],start=start.tolist(),
                        failures=failures,planning_ms=1000*(time.monotonic()-begin))
        failures.append([yaw,failure])
    if lift_only is not None:
        return dict(lift_only,failures=failures,planning_ms=1000*(time.monotonic()-begin))
    return dict(kind=kind,error='Не найден полный безопасный боковой подход и перенос',failures=failures,
                planning_ms=1000*(time.monotonic()-begin))


def _shortcut_open_route(model,start,route,entry_index,table,other,target_points,margin,deadline=None):
    """Greedily skip waypoints only when the full joint sweep stays clear."""
    from transfer_trial import checked
    from geometry_grasp import open_target_clear

    def shorten(seed, points):
        result=[];index=0;seed=np.asarray(seed)
        while index<len(points):
            selected=index
            for candidate in range(len(points)-1,index,-1):
                if deadline is not None and time.monotonic()>=deadline:break
                goal=np.asarray(points[candidate])
                if (checked(model,seed,goal,table,other,margin)
                        and open_target_clear(model,seed,goal,target_points,
                                              minimum=margin,hand_minimum=margin)):
                    selected=candidate;break
            goal=np.asarray(points[selected])
            result.append(goal);seed=goal;index=selected+1
        return result

    entry=shorten(start,route[:entry_index+1])
    seed=entry[-1] if entry else np.asarray(start)
    approach=shorten(seed,route[entry_index+1:])
    return entry+approach,len(entry)-1


def _shortcut_carried_route(model,start,route,table,other,points,bottom_offset,radius,margin,deadline=None):
    """Skip carried-waypoints only after rechecking arm and object clearance."""
    from transfer_trial import checked

    def clear(seed,goal):
        if not checked(model,seed,goal,table,other,margin):return False
        count=max(15,int(np.ceil(np.max(abs(goal-seed))/.035))+1)
        for q in np.linspace(seed,goal,count):
            pos=model.tool(q)[:3,3]
            near=points[np.linalg.norm(points[:,:2]-pos[:2],axis=1)<radius]
            if len(near) and pos[2]-bottom_offset<near[:,2].max()+margin:return False
        return True

    # Preserve the final descent and the preceding high retreat waypoint.
    path=list(route[:-1]);result=[];seed=np.asarray(start);index=0
    while index<len(path):
        selected=index
        for candidate in range(len(path)-1,index,-1):
            if deadline is not None and time.monotonic()>=deadline:return list(route)
            goal=np.asarray(path[candidate])
            if clear(seed,goal):selected=candidate;break
        goal=np.asarray(path[selected]);result.append(goal);seed=goal;index=selected+1
    if not clear(seed,np.asarray(route[-1])):
        return list(route)
    return result+[np.asarray(route[-1])]


def plan_cylinder_candidates(packet):
    attempts=[];started=time.monotonic()
    deadline=float(packet.get('_deadline',started+22.))
    from mission_scene import observed_objects
    from team22_bringup.grasp_priors import verified_prior
    grip_only=bool(packet.get('grip_only',False))
    allow_lift_only=bool(packet.get('allow_lift_only',grip_only))
    obj=observed_objects(packet).get(packet['kind'])
    operator_ready=packet.get('operator_preshape')
    operator_prior=(verified_prior(obj,operator_ready['side'])
                    if operator_ready and obj else None)
    if operator_ready and obj and not operator_prior:
        from cylinder_candidates import rank_cylinder
        ready=operator_ready;side=ready['side']
        clearance=CLEARANCE_M
        # The operator record is an open approach shape, not a verified
        # closure. Use the saved successful cylinder close after palm entry.
        command=packet.get('close_command') or CYLINDER_CLOSE
        sides=(side,'left' if side=='right' else 'right')
        candidate_count=0
        preferred_side_deadline=min(deadline,started+(deadline-started)*.55)
        for side_index,candidate_side in enumerate(sides):
            side_deadline=preferred_side_deadline if side_index==0 else deadline
            if time.monotonic()>side_deadline:break
            candidates=rank_cylinder(obj,packet['obstacles'],packet['scene']['plane'],candidate_side,command,
                                     preshape=ready['command'],thumb_below_top=True,minimum=clearance,
                                     table_minimum=clearance,max_candidates=2)
            # Search new approach directions first; diversify before revisiting
            # hundreds of nearly identical high-overlap but obstructed candidates.
            groups={}
            for candidate in candidates:
                groups.setdefault(tuple(candidate['cylinder_yaws']),[]).append(candidate)
            order=sorted(groups,key=lambda k:abs(k[0]-1.2))
            candidates=[groups[key][i] for i in range(2) for key in order if len(groups[key])>i]
            candidate_count+=len(candidates)
            for candidate in candidates:
                if time.monotonic()>side_deadline:break
                result=plan_cylinder(dict(packet,**candidate,arm_side=candidate_side,close_command=command,_deadline=side_deadline,
                    profile='complete',terminal_first=True,backward_approach=True,
                    allow_lift_only=allow_lift_only,grip_only=grip_only,direct_object_entry=True))
                attempts.append(dict(side=candidate_side,candidate=candidate,error=result.get('error'),failures=result.get('failures')))
                if 'error' not in result:
                    result.update(operator_preshape=ready['id'],approach_preshape=ready['command'],
                                  close_command=command,candidate_attempts=attempts,clearance_m=clearance)
                    return result
        # Keep searching with the learned opposed-thumb and alternate-yaw
        # strategies below when the ranked operator candidates cannot solve IK.
        # Returning here used to discard the only blue-jar candidates whose
        # wrist angles reached a complete grasp-and-carry route.
        attempts.append(dict(strategy='operator_presweep_exhausted',candidate_count=candidate_count))
    prior=None
    if obj and not packet.get('ignore_learned_prior'):
        preferred=packet.get('arm_side') or (
            operator_ready['side'] if operator_ready else
            ('left' if obj['xyz'][1]>.12 else 'right'))
        for prior_side in (preferred,'left' if preferred=='right' else 'right'):
            prior=verified_prior(obj,prior_side)
            if prior:break
    if prior:
        low=bool(packet.get('low_cylinder_grasp'))
        prior_deadline=min(deadline,started+(6. if low else 18.))
        learned=dict(packet,arm_side=prior['side'],pocket=prior['pocket'],_deadline=prior_deadline,
                     palm_roll=prior['palm_roll'],cylinder_yaws=[prior['yaw']],
                     height_adjustment=prior.get('height_adjustment',0.),max_release_sites=4,
                     search_alternate_sites=True,
                     end_yaws=[prior['yaw']],crossing_xs=[.46])
        learned.update(terminal_first=True,backward_approach=True,
                       allow_lift_only=allow_lift_only,grip_only=grip_only)
        # Seed the terminal IK from the verified grasp configuration; the
        # free-space entry still starts at live joints and every edge is checked.
        if prior.get('terminal_seed') is not None:
            learned['terminal_seed']=prior['terminal_seed']
        if packet.get('approach_preshape') is not None:
            # The current operator-demonstrated open shape takes precedence over
            # an older prior which kept the thumb only half opposed.
            learned['approach_preshape']=list(packet['approach_preshape'])
        elif operator_ready is not None:
            learned['approach_preshape']=list(operator_ready['command'])
        elif prior.get('approach_preshape') is not None:
            learned['approach_preshape']=prior['approach_preshape']
        elif packet['kind'] in ('gold_jar','blue_jar'):
            learned['approach_preshape']=list(packet.get('approach_preshape') or CYLINDER_APPROACH)
        if packet['kind'] in ('gold_jar','blue_jar'):
            learned['close_command']=prior.get('close_command') or prior.get('control',{}).get('close') or CYLINDER_CLOSE
        result=plan_cylinder(learned)
        attempts.append(dict(evidence=prior['evidence'],error=result.get('error'),failures=result.get('failures')))
        if 'error' not in result:
            result['candidate_attempts']=attempts;result['prior_evidence']=prior['evidence'];return result
        reasons=[str(row[1]) for row in (result.get('failures') or [])
                 if isinstance(row,(list,tuple)) and len(row)>1]
        if reasons and all(reason.startswith('terminal IK') for reason in reasons):
            # Continuation IK can reach a known grasp even when a direct solve
            # from the saved end pose cannot. Approach incrementally from live joints.
            forward=dict(learned,_deadline=min(deadline,time.monotonic()+12.))
            forward.pop('terminal_first',None);forward.pop('backward_approach',None)
            forward.pop('terminal_seed',None)
            result=plan_cylinder(forward)
            attempts.append(dict(evidence=prior['evidence'],approach='forward continuation',
                                error=result.get('error'),failures=result.get('failures')))
            if 'error' not in result:
                result['candidate_attempts']=attempts;result['prior_evidence']=prior['evidence'];return result
        alternate_side='left' if prior['side']=='right' else 'right'
        alternate_yaws=list(dict.fromkeys((float(prior['yaw']),float(prior['yaw'])-.15,
                                           float(prior['yaw'])+.15,float(prior['yaw'])-.30,
                                           float(prior['yaw'])+.30)))
        mirrored=dict(learned,arm_side=alternate_side,cylinder_yaws=alternate_yaws,
                      _deadline=min(deadline,time.monotonic()+10.))
        mirrored.pop('terminal_seed',None)
        result=plan_cylinder(mirrored)
        attempts.append(dict(evidence=prior['evidence'],approach='opposite-arm fallback',
                            side=alternate_side,error=result.get('error'),failures=result.get('failures')))
        if 'error' not in result:
            result['candidate_attempts']=attempts;result['prior_evidence']=prior['evidence'];return result
    if (packet.get('low_cylinder_grasp') and
            packet['kind'] in ('gold_jar','blue_jar','lotion') and obj):
        from cylinder_candidates import rank_cylinder
        side=packet.get('arm_side','left' if obj['xyz'][1]>.12 else 'right')
        close=packet.get('close_command',[.25]*4+[.10,.45])
        low=bool(packet.get('low_cylinder_grasp'))
        candidates=rank_cylinder(obj,packet['obstacles'],packet['scene']['plane'],
            side,close,preshape=packet.get('approach_preshape') if low else None,
            thumb_below_top=low,max_candidates=2)
        if low and packet.get('retry_number',1)>1:
            candidates=list(reversed(candidates))
        deadline=deadline if low else min(deadline,started+3.)
        for candidate_index,candidate in enumerate(candidates):
            if time.monotonic()>deadline:break
            heights=((.025,) if candidate_index==0 else (.045,)) if low else (None,)
            for height in heights:
                if time.monotonic()>deadline:break
                request=dict(packet,**candidate,_deadline=deadline,terminal_first=True,backward_approach=True,
                    allow_lift_only=allow_lift_only,grip_only=grip_only)
                if low:
                    request['low_contact_height_m']=height
                    request['approach_preshape']=list(packet['approach_preshape'])
                    request['close_command']=list(close)
                    candidate_yaw=(candidate.get('cylinder_yaws') or [1.2])[0]
                    yaw_centers=([float(prior['yaw'])] if prior and prior['side']==side else [])
                    yaw_centers.append(float(candidate_yaw))
                    yaws=[]
                    for center in yaw_centers:
                        for delta in (0.,-.3,.3):
                            value=round(center+delta,2)
                            if value not in yaws:yaws.append(value)
                    request['cylinder_yaws']=yaws
                    if prior and prior['side']==side:
                        # Reuse the demonstrated wrist roll after trying its exact
                        # pocket and yaw above; all current clearances still apply.
                        request['palm_roll']=float(prior['palm_roll'])
                result=plan_cylinder(request)
                attempts.append(dict(candidate=candidate,height_m=height,error=result.get('error'),failures=result.get('failures')))
                if 'error' not in result:
                    result['candidate_attempts']=attempts;return result
        # Low contacts are tried first. If they are unreachable, retry the
        # verified grasp at its measured contact height with the abducted thumb
        # still required; all 2 mm collision and palm-entry checks remain active.
        if prior and prior['side']==side:
            prior_yaw=float(prior['yaw'])
            verified_yaws=list(dict.fromkeys(round(prior_yaw+delta,2)
                                             for delta in (0.,-.15,.15,-.30,.30)))
            fallback=dict(packet,low_cylinder_grasp=False,arm_side=prior['side'],
                pocket=prior['pocket'],palm_roll=prior['palm_roll'],
                cylinder_yaws=verified_yaws,end_yaws=verified_yaws,
                height_adjustment=float(prior.get('height_adjustment',0.)),
                terminal_seed=prior.get('terminal_seed'),terminal_first=True,
                backward_approach=True,max_release_sites=4,search_alternate_sites=True,
                _deadline=time.monotonic()+30.)
            fallback['approach_preshape']=list(packet.get('approach_preshape') or CYLINDER_APPROACH)
            fallback['close_command']=prior.get('close_command') or prior.get('control',{}).get('close') or close
            result=plan_cylinder(fallback)
            attempts.append(dict(evidence=prior['evidence'],strategy='verified_height_after_low',
                                 error=result.get('error'),failures=result.get('failures')))
            if 'error' not in result:
                result['candidate_attempts']=attempts;result['prior_evidence']=prior['evidence'];return result
        return dict(kind=packet['kind'],error='Нет доступного бокового хвата с двумя встречными контактами',candidate_attempts=attempts)
    if packet['kind']!='gold_jar':
        if packet['kind']=='blue_jar':
            opposed=dict(packet,palm_roll=.6,terminal_first=True,backward_approach=True,
                height_adjustment=0.,allow_lift_only=allow_lift_only,grip_only=grip_only,
                approach_preshape=packet.get('approach_preshape') or CYLINDER_APPROACH,
                close_command=packet.get('close_command',[.24,.24,.24,.24,.10,.45]),
                pocket=[.055,0.,.140],cylinder_yaws=[-.3,0.,-.6,.3])
            result=plan_cylinder(opposed)
            attempts.append(dict(strategy='thumb_opposed_before_approach',error=result.get('error'),failures=result.get('failures')))
            if 'error' not in result:
                result['candidate_attempts']=attempts;return result
        for pocket_x in (.035,.045,.050):
            vertical=dict(packet,palm_roll=0.,terminal_first=True,backward_approach=True,
                height_adjustment=.012,allow_lift_only=allow_lift_only,grip_only=grip_only,
                pocket=[pocket_x,0.,.140],cylinder_yaws=[-.3,0.,-.6,.3,.6,.9])
            result=plan_cylinder(vertical)
            attempts.append(dict(pocket_x=pocket_x,palm_roll=0.,error=result.get('error'),failures=result.get('failures')))
            if 'error' not in result:
                result['candidate_attempts']=attempts;return result
        tilted=dict(packet,palm_roll=.6,terminal_first=True,height_adjustment=.020,allow_lift_only=allow_lift_only,
                    pocket=[.060,0.,.140],grip_only=grip_only,
                    cylinder_yaws=[-.3,0.,-.6,-.9,.3,.6,.9],
                    crossing_xs=[.34,.30,.38,.42],end_yaws=[0.,-.6,-1.1,.4,.8,1.1])
        result=plan_cylinder(tilted)
        attempts.append(dict(palm_roll=.6,error=result.get('error'),failures=result.get('failures')))
        if 'error' not in result:
            result['candidate_attempts']=attempts;return result
    offsets=(0.,) if packet['kind']=='gold_jar' else (.016,.028,0.,.008)
    for offset in offsets:
        result=plan_cylinder(dict(packet,height_adjustment=offset,_deadline=deadline))
        attempts.append(dict(height_adjustment_mm=offset*1000,error=result.get('error'),failures=result.get('failures')))
        if 'error' not in result:
            result['candidate_attempts']=attempts;return result
    result['candidate_attempts']=attempts
    return result

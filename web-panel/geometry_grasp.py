"""Candidate shape-aware pinch with a complete, collision-checked carry route."""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation,Slerp
from team22_bringup.grasp import Arm, CLEARANCE_M
from side_grasp import solve_side
from team22_bringup.scene_geometry import planning_edge
from team22_bringup.grasp_priors import narrow_side_yaws, contact_strategy


def open_target_clear(model,start,goal,target_points,minimum=CLEARANCE_M,hand_minimum=CLEARANCE_M):
    """Before closure, neither rigid links nor open fingers may enter the target."""
    count=max(2,int(np.ceil(np.max(abs(goal-start))/.035))+1)
    for q in np.linspace(start,goal,count):
        samples,radii=model.samples(q)
        gaps=target_points.query(samples)[0]-radii
        # Seven arm/wrist segments end at the palm base. Contact starts with
        # the separate closing command.
        if np.min(gaps[:112],initial=np.inf)<minimum:return False
        if hasattr(model,'hand_geometry'):
            hand=model.wrist(q)@model.hand
            local=(target_points.data-hand[:3,3])@hand[:3,:3]
            if model.hand_geometry.collision(local,model.finger_angles,hand_minimum):return False
        elif np.min(gaps[112:],initial=np.inf)<hand_minimum:return False
    return True


def plan_geometry_candidates(packet):
    if packet['kind'] not in ('mouse','white_mouse'):
        if packet['kind'].startswith('figure_'):
            attempts=[]
            for height in (0.,.008,.020,.028):
                result=plan_geometry(dict(packet,grip_only=True,backward_approach=True,
                    hand_table_margin=CLEARANCE_M,approach_preshape=[1.]*5+[.45],pinch_height_adjustment=height))
                attempts.append(dict(height_adjustment=height,error=result.get('error'),failures=result.get('failures')))
                if 'error' not in result:
                    result['candidate_attempts']=attempts;return result
            result['candidate_attempts']=attempts;return result
        return plan_geometry(packet)
    attempts=[]
    from mission_scene import observed_objects
    from pinch_candidates import ranked_candidates
    obj=observed_objects(packet).get(packet['kind'])
    if obj:
        side=packet.get('arm_side','left' if obj['xyz'][1]>.08 else 'right')
        close_command=packet.get('close_command',[.24,.24,.24,.24,.10,.45])
        candidates=ranked_candidates(obj,packet['obstacles'],packet['scene']['plane'],
            side,close_command,table_reserve=max(CLEARANCE_M,.011))
        for candidate in candidates:
            if candidate['score']<.0008:
                attempts.append(dict(candidate=candidate,error='Недостаточный запас противоположного контакта',failures=[]))
                continue
            result=plan_geometry(dict(packet,**candidate,thin_object=True,
                                      close_command=close_command))
            attempts.append(dict(candidate=candidate,error=result.get('error'),failures=result.get('failures')))
            if 'error' not in result:
                result['candidate_attempts']=attempts;return result
    return dict(kind=packet['kind'],error='Нет доступного хвата с противоположными контактами по объёму предмета',candidate_attempts=attempts)


def plan_geometry(packet):
    from transfer_trial import checked,destination
    begin=time.monotonic();kind=packet['kind'];scene=packet['scene']
    from mission_scene import observed_objects
    objects=observed_objects(packet)
    cap_trial=kind=='tube_cap' and packet.get('grip_only')
    tray_kind=None if cap_trial else destination(packet['task'],kind)
    if kind not in objects or (not cap_trial and tray_kind not in objects):return dict(kind=kind,error='Предмет или целевой лоток не наблюдается')
    obj,tray=objects[kind],objects.get(tray_kind);bounds=np.asarray(obj['bounds']);plane=np.asarray(scene['plane'])
    if kind in ('drive','green_speaker'):
        return dict(kind=kind,error='Для крупного предмета требуется совместный двухручный маршрут')
    source_center=np.asarray(obj['xyz'],float).copy();target=source_center.copy();base=float(target[:2]@plane[:2]+plane[2])
    target[2]=max(base+.034,bounds[1,2]-min(.010,np.ptp(bounds,axis=0)[2]*.20))
    if 'pinch_table_height' in packet:target[2]=base+float(packet['pinch_table_height'])
    target[2]+=float(packet.get('pinch_height_adjustment',0.))
    side=packet.get('arm_side','left' if target[1]>.08 else 'right');other_side='right' if side=='left' else 'left'
    model,parked=Arm(side),Arm(other_side);model.prefer_seed=True
    if 'pocket' in packet:model.pocket=np.array(packet['pocket'])
    from team22_bringup.hand_geometry import HandGeometry
    model.hand_geometry=HandGeometry(side)
    model.finger_angles={k:v for k,v in packet['angles'].items() if k.startswith(side+'_')}
    preshape=packet.get('approach_preshape')
    thin=packet.get('thin_object',False)
    if preshape is not None:model.finger_angles.update(model.hand_geometry.nominal_angles(preshape))
    from grasp_shape_checks import checked_hand_table
    def route_clear(a,b):
        return (checked_hand_table(model,a,b,table,other,hand_minimum=CLEARANCE_M,
                                   arm_minimum=CLEARANCE_M,hand_object_minimum=CLEARANCE_M)
                if thin or 'hand_table_margin' in packet else checked(model,a,b,table,other,CLEARANCE_M))
    start=np.array([packet['angles'][k] for k in model.names])
    other=cKDTree(parked.samples([packet['angles'][k] for k in parked.names])[0])
    points=np.asarray(packet['obstacles']);inside=((points>=bounds[0]-.012)&(points<=bounds[1]+.012)).all(1)
    target_points=cKDTree(points[inside])
    if not np.any(inside):return dict(kind=kind,error='Нет текущей глубины поверхности предмета')
    transit_table=plane,planning_edge(scene),cKDTree(points)
    points=points[~inside];table=plane,planning_edge(scene),cKDTree(points)
    failures=[];sign=1 if side=='left' else -1
    for yaw in packet.get('pinch_yaws',narrow_side_yaws(obj, side)):
        fingers=np.array([np.cos(yaw),sign*np.sin(yaw),0.]);palm=np.array([0.,0.,-1.])
        tilt=float(packet.get('finger_down_tilt',0.))
        roll=float(packet.get('pinch_roll',0.))
        orient=(np.column_stack((palm,np.cross(fingers,palm),fingers))@
                Rotation.from_euler('y',tilt).as_matrix()@Rotation.from_euler('z',roll).as_matrix())
        terminal=solve_side(model,target,orient,start)
        if terminal is None:
            failures.append([yaw,'Контактная поза: ИК']);continue
        if not route_clear(terminal,terminal):
            gaps=model.clearances(terminal,table,other)
            failures.append([yaw,f'Контактная поза: зазор стол/предметы {gaps[0]*1000:.1f}/{gaps[1]*1000:.1f} мм']);continue
        from grasp_shape_checks import closure_clear
        safe,detail=closure_clear(model,terminal,table,packet['angles'],packet.get('close_command',[.25]*4+[.10,.45]),
                                  minimum=CLEARANCE_M,initial_command=preshape)
        if not safe:failures.append([yaw,detail]);continue
        initial=model.tool(start);turn=Slerp([0,1],Rotation.from_matrix([initial[:3,:3],orient]))
        seed=start.copy();route=[];reason=None;preparation=[]
        backward=thin or packet.get('backward_approach',False)
        if backward:
            # Preserve the reachable terminal IK branch by building the last
            # vertical centimetres backwards before planning free-space entry.
            raised=[terminal];seed=terminal;raise_reason=None
            for height in (.04,.08,.12):
                fitted=solve_side(model,target+[0,0,height],orient,seed)
                if fitted is None:raise_reason=f'{height:.2f} m: IK';break
                if not route_clear(seed,fitted):raise_reason=f'{height:.2f} m: clearance';break
                if not open_target_clear(model,seed,fitted,target_points):raise_reason=f'{height:.2f} m: target clearance';break
                raised.append(fitted);seed=fitted
            if len(raised)<(2 if cap_trial else 3):
                failures.append([yaw,'Нет обратного вертикального подхода: '+str(raise_reason)]);continue
            from grasp_transit import transit
            # Include the cap itself in the checked free-space entry.
            cap_entry=(lambda a,b:checked_hand_table(model,a,b,transit_table,other,
                hand_minimum=CLEARANCE_M,arm_minimum=CLEARANCE_M,
                hand_object_minimum=CLEARANCE_M)) if cap_trial else None
            delayed=cap_trial and packet.get('delayed_preshape',False)
            shaped_angles=model.finger_angles
            if delayed:model.finger_angles=model.hand_geometry.nominal_angles(np.ones(6))
            entry=transit(model,start,raised[-1],transit_table,other,CLEARANCE_M,edge_check=cap_entry,prefer_compact=delayed)
            entry_valid=entry is not None and all(open_target_clear(model,a,b,target_points) for a,b in zip([start,*entry],entry))
            model.finger_angles=shaped_angles
            if not entry_valid:
                failures.append([yaw,'Нет пути к обратному вертикальному подходу']);continue
            if delayed:
                safe,detail=closure_clear(model,raised[-1],transit_table,packet['angles'],preshape,
                    minimum=CLEARANCE_M,initial_command=np.ones(6))
                if not safe:
                    failures.append([yaw,'Форма кисти над столом: '+detail]);continue
                preparation=entry
                route=list(reversed(raised[:-1]))
            else:route=entry+list(reversed(raised[:-1]))
        stages=[(target+[0,0,h],orient) for h in (.18,.12,.08,.04,0.)]
        for index,(point,rotation) in enumerate([] if backward else stages):
            fitted=solve_side(model,point,rotation,seed)
            if index==0 and fitted is not None:
                from grasp_transit import transit
                entry=transit(model,seed,fitted,transit_table,other,CLEARANCE_M)
                if entry is not None:
                    if not all(open_target_clear(model,a,b,target_points) for a,b in zip([seed,*entry],entry)):
                        reason='Выход к предмету: точная геометрия кисти задевает предмет';break
                    route.extend(entry);seed=fitted;continue
            if fitted is None or not route_clear(seed,fitted):
                reason=f'Подход {index}: '+('ИК' if fitted is None else 'зазор');break
            if not open_target_clear(model,seed,fitted,target_points):
                reason=f'Подход {index}: раскрытая кисть или запястье задевает сам предмет';break
            route.append(fitted);seed=fitted
        if reason:failures.append([yaw,reason]);continue
        lift=raised[1] if backward else route[-2]
        bottom=float(source_center[2]-bounds[0,2]);radius=np.linalg.norm(np.ptp(bounds,axis=0)[:2])/2+CLEARANCE_M
        if packet.get('grip_only'):
            return dict(kind=kind,side=side,grasp='geometry_pinch',target=target.tolist(),original=obj,
                thin_object=thin,approach_preshape=preshape,clearance_m=CLEARANCE_M,
                close_command=packet.get('close_command'),
                pinch_roll=roll,contact_fingers=packet.get('contact_fingers'),
                hand_object_margin=packet.get('hand_object_margin'),
                pocket=model.pocket.tolist(),finger_down_tilt=tilt,tray=tray_kind,tray_observation=tray,yaw=yaw,
                approach=[q.tolist() for q in route],preparation_route=[q.tolist() for q in preparation],lift=lift.tolist(),transport=[],
                transfer_reason='Учебный хват по форме: подъём и возврат на стол',
                retreat=[q.tolist() for q in reversed(route[:-1])],
                recovery_route=[q.tolist() for q in list(reversed(route[:-1]))+([preparation[-1]] if preparation else [])],
                plane=plane.tolist(),table_edge=table[1],obstacles=points.tolist(),start=start.tolist(),failures=failures,
                planning_ms=(time.monotonic()-begin)*1000)
        high=max(.08,float(points[:,2].max())+bottom+CLEARANCE_M,float(source_center[2]+.12))
        destination_xyz=np.asarray(packet.get('release_site',tray['xyz']),float).copy();tray_bounds=np.asarray(tray['bounds'])
        release_z=tray_bounds[0,2]+bottom+CLEARANCE_M
        high=max(high,release_z+CLEARANCE_M)
        for end_yaw in (yaw,0.,-1.2,1.2,2.4,-2.4):
            seed=lift;transport=[];reason=None
            release_center=np.array([destination_xyz[0],destination_xyz[1],release_z])
            centers=[np.array([source_center[0],source_center[1],high]),
                     np.array([.46,source_center[1],high]),np.array([.46,destination_xyz[1],high]),
                     np.array([destination_xyz[0],destination_xyz[1],high]),release_center]
            object_offset=orient.T@(source_center-target)
            for index,goal_center in enumerate(centers):
                heading_fraction=min(max(0.,(index-1)/3.),1.)
                heading=yaw+(end_yaw-yaw)*heading_fraction
                fingers=np.array([np.cos(heading),sign*np.sin(heading),0.])
                rotation=np.column_stack((palm,np.cross(fingers,palm),fingers))@Rotation.from_euler('y',tilt).as_matrix()
                point=goal_center-rotation@object_offset
                fitted=solve_side(model,point,rotation,seed)
                if fitted is None or not checked(model,seed,fitted,table,other,CLEARANCE_M):
                    reason=f'Перенос {index}: '+('ИК' if fitted is None else 'зазор');break
                for q in np.linspace(seed,fitted,max(3,int(np.max(abs(fitted-seed))/.035)+2)):
                    pose=model.tool(q);p=pose[:3,3]+pose[:3,:3]@object_offset
                    near=points[np.linalg.norm(points[:,:2]-p[:2],axis=1)<radius]
                    if len(near) and p[2]-bottom<near[:,2].max()+CLEARANCE_M:reason='Габарит груза задевает препятствие';break
                if reason:break
                transport.append(fitted);seed=fitted
            if reason:continue
            return dict(kind=kind,side=side,grasp='geometry_pinch',target=target.tolist(),original=obj,pocket=model.pocket.tolist(),finger_down_tilt=tilt,
                thin_object=thin,approach_preshape=preshape,clearance_m=CLEARANCE_M,
                tray=tray_kind,tray_observation=tray,release_site=destination_xyz.tolist(),
                release_center=[float(destination_xyz[0]),float(destination_xyz[1]),float(release_z)],yaw=yaw,approach=[q.tolist() for q in route],
                lift=lift.tolist(),transport=[q.tolist() for q in transport],transfer_reason=None,
                retreat=[q.tolist() for q in list(reversed(transport[:-1]))+list(reversed(route[:-1]))+[start]],
                plane=plane.tolist(),table_edge=table[1],obstacles=points.tolist(),start=start.tolist(),failures=failures,
                contact_strategy=contact_strategy(obj),planning_ms=(time.monotonic()-begin)*1000)
        failures.append([yaw,reason or 'Нет переноса'])
    return dict(kind=kind,error='Нет полного маршрута для захвата по геометрии',failures=failures,
                planning_ms=(time.monotonic()-begin)*1000)

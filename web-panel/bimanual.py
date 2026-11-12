"""Two opposing palms: measured geometry, coupled paths, both-hand feedback."""
import time
import os
import numpy as np
from team22_bringup.scene_geometry import planning_edge
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation,Slerp
from team22_bringup.grasp import Arm

SIDES=('left','right')
CLEARANCE_M=float(os.environ.get('TEAM22_CLEARANCE_M','0.002'))

def models(angles=None):
    result={s:Arm(s) for s in SIDES}
    for side,m in result.items():
        m.pocket=np.array([.035,0.,.140])
        m.prefer_seed=True
        m.strict_body=True
        sign=-1 if side=='left' else 1
        m.extra_hand_points=[(.031,sign*.096,.136),(.057,sign*.087,.137),(.076,sign*.070,.137)]
        if angles is not None:
            from team22_bringup.hand_geometry import HandGeometry
            m.hand_geometry=HandGeometry(side);m.finger_angles=angles
    return result

def interarm_gap(arms,q):
    a,ra=arms['left'].samples(q['left']);b,rb=arms['right'].samples(q['right'])
    a,ra,b,rb=a[32:],ra[32:],b[32:],rb[32:]
    d,i=cKDTree(b).query(a)
    return float(np.min(d-ra-rb[i]))

def check_pair(arms,start,end,table,margin=None,table_margin=None):
    margin=CLEARANCE_M if margin is None else margin
    table_margin=margin if table_margin is None else table_margin
    from transfer_trial import body_clear
    count=max(2,int(np.ceil(max(np.max(abs(end[s]-start[s])) for s in SIDES)/.035))+1)
    minimum=np.inf
    for t in np.linspace(0,1,count):
        q={s:start[s]+t*(end[s]-start[s]) for s in SIDES}
        for s in SIDES:
            if not body_clear(arms[s],q[s]):return False,f'{s}: корпус, доля пути {t:.2f}'
        distances={'between_hands':interarm_gap(arms,q)}
        for s in SIDES:
            table_gap,object_gap=arms[s].clearances(q[s],table)
            distances[s+'_table']=table_gap;distances[s+'_objects']=object_gap
        source=min(distances,key=distances.get);gap=distances[source]
        minimum=min(minimum,gap)
        threshold=table_margin if source.endswith('_table') else margin
        if gap<threshold:return False,f'{source}: зазор {gap*1000:.1f} мм, доля пути {t:.2f}'
    return True,f'минимум {minimum*1000:.1f} мм'

def plan_two_hands(packet):
    from side_grasp import solve_side
    from transfer_trial import destination
    from mission_scene import observed_objects
    from tray_placement import release_sites
    begin=time.monotonic();scene=packet['scene'];kind=packet['kind']
    objects=observed_objects(packet)
    obj=objects.get(kind)
    if obj is None:return dict(kind=kind,error='Предмет не виден для двух рук')
    tray_kind=destination(packet['task'],kind);tray=objects.get(tray_kind)
    if tray is None:return dict(kind=kind,error='Не вижу назначенный лоток до подхода к предмету')
    bounds=np.asarray(obj['bounds']);plane=np.asarray(scene['plane'])
    scan_views=int(packet.get('scan_view_count',scene.get('scan_view_count',0)))
    dimensions=np.ptp(bounds,axis=0)
    multi_view_shape=(scan_views>=3 and obj.get('points',0)>=250 and
        dimensions[0]>.065 and dimensions[1]>.065 and .015<dimensions[2]<.28)
    if len(obj.get('seen_by',[]))<2 and not multi_view_shape:
        return dict(kind=kind,error='Для двух ладоней нужны две камеры или три проверенных ракурса и полный объём')
    sites=release_sites(obj,tray,packet['obstacles'])
    preferred=packet.get('release_site')
    if preferred is not None and sites:
        ix=min(range(len(sites)),key=lambda i:np.linalg.norm(np.asarray(sites[i])[:2]-np.asarray(preferred)[:2]))
        if np.linalg.norm(np.asarray(sites[ix])[:2]-np.asarray(preferred)[:2])<.002:
            sites=[sites[ix]]+[site for i,site in enumerate(sites) if i!=ix]
    if not sites:return dict(kind=kind,error='До подхода не найдено свободное место для предмета в назначенном лотке')
    arms=models(packet['angles']);start={s:np.array([packet['angles'][k] for k in arms[s].names]) for s in SIDES}
    points=np.asarray(packet['obstacles']);inside=((points>=bounds[0]-.012)&(points<=bounds[1]+.012)).all(1)
    target_points=cKDTree(points[inside])
    points=points[~inside];table=plane,planning_edge(scene),cKDTree(points)
    center=np.asarray(obj['xyz']);base=float(center[:2]@plane[:2]+plane[2])
    geometry=obj.get('geometry',{});axis=np.asarray(geometry.get('long_axis_xy',[1.,0.]))
    normals=[np.array([0.,1.]),np.array([-axis[1],axis[0]]),axis]
    failures=[]
    for direction in normals:
        if direction[1]<0:direction=-direction
        if abs(direction[1])<.25:continue
        normal=np.r_[direction,0.]
        # Project the actually observed surface along this contact normal. A
        # rotated world-aligned box exaggerates the required palm separation.
        half=float(np.max(abs((target_points.data[:,:2]-center[:2])@direction)))+.004
        for height in (.160,.170,.150,.135,.140,.125,.105):
            target=center.copy();target[2]=base+height
            if target[2]>bounds[1,2]-.020:continue
            rotation={}
            for s,sign in (('left',1),('right',-1)):
                palm=-sign*normal
                fingers=np.array([1.,0.,0.])-normal*normal[0]
                fingers/=np.linalg.norm(fingers)
                tilt=float(packet.get('finger_elevation',0.))
                fingers=np.cos(tilt)*fingers+np.sin(tilt)*np.array([0.,0.,1.])
                rotation[s]=np.column_stack((palm,np.cross(fingers,palm),fingers))
            seed={s:q.copy() for s,q in start.items()};route=[];failure=None
            # Raise one arm at a time along a collision-checked joint route.
            # Cartesian interpolation from hanging hands crosses the tabletop.
            from grasp_transit import transit
            full_table=plane,table[1],cKDTree(packet['obstacles'])
            for s,sign in (('left',1),('right',-1)):
                other_side='right' if s=='left' else 'left'
                other=cKDTree(arms[other_side].samples(seed[other_side])[0])
                fitted=solve_side(arms[s],np.array([.42,sign*.42,.20]),rotation[s],seed[s])
                entry=None if fitted is None else transit(arms[s],seed[s],fitted,full_table,other,CLEARANCE_M)
                if entry is None:failure=f'Подъём {s}: нет проверенного пути';break
                for waypoint in entry:
                    end={k:q.copy() for k,q in seed.items()};end[s]=np.array(waypoint)
                    ok,detail=check_pair(arms,seed,end,full_table)
                    if not ok:failure=f'Подъём {s}: {detail}';break
                    route.append({k:q.tolist() for k,q in end.items()});seed=end
                if failure:break
            if failure:
                failures.append(dict(normal=normal.tolist(),height=height,reason=failure));continue
            approach_complete=False;approach_failure=None
            approach_levels=((.05,.07),(.025,.07),(0.,.05),(0.,.025),(0.,.010))
            for level,(elevation,gap) in enumerate(approach_levels):
                goals={s:target+sign*normal*(half+gap)+[0,0,elevation] for s,sign in (('left',1),('right',-1))}
                fitted={s:solve_side(arms[s],goals[s],rotation[s],seed[s]) for s in SIDES}
                if any(q is None for q in fitted.values()):
                    approach_failure=f'ИК {[s for s in SIDES if fitted[s] is None]}, высота {elevation}, зазор {gap}';continue
                ok,detail=check_pair(arms,seed,fitted,table)
                if not ok:approach_failure=f'Подход {len(route)}: {detail}';continue
                from geometry_grasp import open_target_clear
                if not all(open_target_clear(arms[s],seed[s],fitted[s],target_points) for s in SIDES):
                    approach_failure=f'Подход {len(route)}: раскрытая кисть пересекает предмет';continue
                route.append({s:q.tolist() for s,q in fitted.items()});seed=fitted
                if level==len(approach_levels)-1:approach_complete=True
            if not approach_complete:
                failures.append(dict(normal=normal.tolist(),height=height,
                                     reason=approach_failure or 'Не достигнут безопасный зазор захвата'));continue
            from grasp_shape_checks import closure_clear
            sweep={s:closure_clear(arms[s],seed[s],table,packet['angles'],[.45]*6) for s in SIDES}
            if not all(v[0] for v in sweep.values()):
                failures.append(dict(reason='Траектория пальцев',details=sweep));continue
            lift={s:solve_side(arms[s],goals[s]+[0,0,.04],rotation[s],seed[s]) for s in SIDES}
            if any(q is None for q in lift.values()):failures.append(dict(reason='ИК подъёма'));continue
            ok,detail=check_pair(arms,seed,lift,table)
            if not ok:failures.append(dict(reason='Подъём: '+detail));continue
            relative=arms['left'].tool(seed['left'])[:3,3]-arms['right'].tool(seed['right'])[:3,3]
            deformation=max(np.linalg.norm(
                arms['left'].tool(seed['left']+t*(lift['left']-seed['left']))[:3,3]-
                arms['right'].tool(seed['right']+t*(lift['right']-seed['right']))[:3,3]-relative)
                for t in np.linspace(0,1,16))
            if deformation>.006:failures.append(dict(reason=f'Расхождение ладоней при подъёме {deformation*1000:.1f} мм'));continue
            offsets={s:arms[s].tool(seed[s])[:3,3]-center for s in SIDES}
            bottom_offset=float(center[2]-bounds[0,2]);payload_radius=float(np.max(np.ptp(bounds,axis=0)[:2])/2+CLEARANCE_M)
            tray_bounds=np.asarray(tray['bounds'],float);transport=None;release_center=None;chosen_site=None
            high=max(float(center[2]+.12),float(points[:,2].max())+bottom_offset+CLEARANCE_M)
            for site in sites[:5]:
                release=np.asarray(site,float).copy()
                release[2]=tray_bounds[0,2]+bottom_offset+CLEARANCE_M
                centers=[center.copy()];raised=center.copy();raised[2]=high;centers.append(raised)
                for fraction in (.25,.5,.75,1.):
                    waypoint=raised*(1.-fraction)+np.array([release[0],release[1],high])*fraction
                    centers.append(waypoint)
                centers.append(release)
                pair_path=[];pair_seed={s:q.copy() for s,q in lift.items()};failure=None
                for segment,goal_center in enumerate(centers[1:]):
                    fitted={s:solve_side(arms[s],goal_center+offsets[s],rotation[s],pair_seed[s]) for s in SIDES}
                    if any(q is None for q in fitted.values()):
                        failure=f'Перенос {segment}: ИК';break
                    ok,detail=check_pair(arms,pair_seed,fitted,table)
                    if not ok:
                        failure=f'Перенос {segment}: {detail}';break
                    for t in np.linspace(0,1,10):
                        q={s:pair_seed[s]+t*(fitted[s]-pair_seed[s]) for s in SIDES}
                        measured=np.mean([arms[s].tool(q[s])[:3,3]-offsets[s] for s in SIDES],axis=0)
                        near=points[np.linalg.norm(points[:,:2]-measured[:2],axis=1)<payload_radius]
                        if len(near) and measured[2]-bottom_offset<float(near[:,2].max())+CLEARANCE_M:
                            failure=f'Предмет цепляет препятствие на переносе {segment}';break
                    if failure:break
                    pair_path.append({s:q.tolist() for s,q in fitted.items()});pair_seed=fitted
                if not failure:
                    transport=pair_path;release_center=release;chosen_site=site;break
                failures.append(dict(release_site=site,reason=failure))
            if transport is None:
                failures.append(dict(reason='Не найден безопасный двухручный маршрут до лотка'))
                continue
            return dict(kind=kind,grasp='two_palm_support',side='both',original=obj,
                tray=tray_kind,tray_observation=tray,release_site=list(chosen_site),release_center=release_center.tolist(),
                plane=plane.tolist(),table_edge=table[1],obstacles=points.tolist(),approach=route,
                lift={s:q.tolist() for s,q in lift.items()},transport=transport,
                grip_offsets={s:q.tolist() for s,q in offsets.items()},start={s:q.tolist() for s,q in start.items()},
                target=target.tolist(),normal=normal.tolist(),contact_gap_mm=10.,
                scan_view_count=scan_views,failures=failures,planning_ms=1000*(time.monotonic()-begin),
                transfer_reason=None)
    return dict(kind=kind,error=f'Не найден совместный подход двух рук с зазором {CLEARANCE_M*1000:.0f} мм',
                failures=failures,planning_ms=1000*(time.monotonic()-begin))

def validate_pair_route(current,measured,path):
    arms=models(measured)
    seed={s:np.array([measured[k] for k in arms[s].names]) for s in SIDES}
    table=np.asarray(current['plane']),current.get('table_edge',-100.),cKDTree(current['obstacles'])
    for waypoint in path:
        end={s:np.asarray(waypoint[s]) for s in SIDES}
        ok,reason=check_pair(arms,seed,end,table,CLEARANCE_M)
        if not ok:return False,reason
        seed=end
    return True,None


def move_pair(trial,path,now,stage,message):
    measured=dict(trial.lab.timeline.latest('joints')[2])
    trial.pending_pair=dict(path=path,angles=measured,stage=stage,message=message)
    trial.pair_future=trial.pool.submit(validate_pair_route,trial.current,measured,path)
    trial.enter('dual_validation',now,'Проверяю обе траектории по текущим датчикам')

def contact_checks(trial,settlement=False):
    from transfer_trial import finger_settlement
    checks={}
    for side in SIDES:
        state=trial.lab.timeline.latest('hand_'+side)[2];empty=trial.empty[side]
        # Empty calibration closes to .25; the supporting grip commands .45.
        blocked=np.asarray(state['position'])-np.asarray(empty['position'])>.245
        loaded=np.abs(state['effort'])-np.abs(empty['effort'])>.05
        checks[side]=dict(contact=bool(np.count_nonzero(blocked[:5]&loaded[:5])>=2),state=state)
        if settlement:checks[side].update(finger_settlement(trial.lab.timeline.data.get('hand_'+side,())))
    return checks


def tick(trial,now):
    """Transport only while both contacts and the carried object remain measured."""
    c=trial.lab.controller;stage=trial.stage
    if not stage.startswith('dual_'):return False
    holding_load=stage in ('dual_lift','dual_transport') or (stage=='dual_validation' and
        getattr(trial,'pending_pair',{}).get('stage') in ('dual_lift','dual_transport'))
    if holding_load:
        checks=contact_checks(trial)
        if not all(v['contact'] for v in checks.values()):
            trial.event(now,'Потерян контакт одной кисти при двухручном движении',hands=checks)
            c.stop('Двухручное движение остановлено: удержание одной кистью потеряно',now)
            return True
    if stage in ('dual_lift','dual_transport','dual_release') and trial.lab.preparing:return True
    if stage=='dual_validation':
        if not trial.pair_future.done():return True
        ok,reason=trial.pair_future.result()
        if not ok:c.stop('Повторная проверка двух рук: '+str(reason),now);return True
        pending=trial.pending_pair;q=trial.lab.timeline.latest('joints')[2]
        names=[k for m in c.models.values() for k in m.names]
        fingers=[k for k in pending['angles'] if any(f in k for f in ('thumb','index','middle','ring','little'))]
        if max(abs(q[k]-pending['angles'][k]) for k in names)>.025 or max((abs(q[k]-pending['angles'][k]) for k in fingers),default=0.)>.08:
            c.stop('Руки изменили положение за время расчёта двухручного пути',now);return True
        from readiness import observation_is_fresh
        if not observation_is_fresh(trial.lab.observation,now,trial.lab.processed_wall,time.monotonic()):return True
        from collections import deque
        c.table=np.asarray(trial.current['plane']),trial.current.get('table_edge',-100.),cKDTree(trial.current['obstacles'])
        c.synchronized=True;c.interarm_guard=True;c.planned_route=True
        c.goal={k:pending['angles'][k] for k in names};c.reference=dict(c.goal)
        c.route=deque(('both',pending['message'],{s:np.asarray(w[s]) for s in SIDES}) for w in pending['path'])
        c.status='moving';c.next_stage(now)
        trial.enter(pending['stage'],now,pending['message'])
        return True
    if stage=='dual_approach':
        scene=trial.scene(now)
        obj=next((o for o in (scene or {}).get('objects',[]) if o['kind']==trial.current['kind'] and o.get('status')=='observed'),None)
        if obj and np.linalg.norm(np.array(obj['xyz'])-trial.current['original']['xyz'])>.020:
            c.stop('Коробка сместилась до двухручного захвата',now);return True
        if obj is None and not (trial.tracker and trial.tracker.baseline):
            c.stop('Нет исходной RGB-D поверхности для проверки коробки перед захватом',now);return True
    if stage=='dual_open' and now-trial.entered>1.:
        if not all(min(trial.lab.timeline.latest('hand_'+s)[2]['position'])>.9 for s in SIDES):return True
        move_pair(trial,trial.current['approach'],now,'dual_approach','Подвод двух раскрытых ладоней с противоположных сторон')
    elif stage=='dual_approach' and c.status=='holding':
        trial.dual_baseline=(obj if obj is not None else trial.tracker.baseline)
        if trial.dual_baseline is None:
            c.stop('Нет исходного RGB-D центра коробки для проверки подъёма',now);return True
        trial.dual_confirmed=None
        for s in SIDES:trial.fingers(s,[.45,.45,.45,.45,.45,.45])
        trial.enter('dual_contact',now,'Пробую удержание обеими кистями, проверяю пальцы каждой')
    elif stage=='dual_contact' and now-trial.entered>2.:
        checks=contact_checks(trial,settlement=True)
        if all(v['contact'] and v['settled'] for v in checks.values()):
            if getattr(trial,'dual_confirmed',None) is None:trial.dual_confirmed=now
            if now-trial.dual_confirmed<.35:return True
            observation=trial.tracker.latest if trial.tracker is not None else None
            if observation and observation.get('valid') and 0<=now-observation.get('stamp',-1)<.45:
                measured_center=np.asarray(observation['xyz'],float)
            elif trial.tracker and trial.tracker.baseline:
                measured_center=np.asarray(trial.tracker.baseline['xyz'],float)
            else:
                trial.event(now,'Не подтверждён исходный RGB-D центр коробки перед подъёмом',vision=observation)
                for s in SIDES:trial.fingers(s,np.ones(6))
                trial.enter('dual_open_failed',now,'Не двигаю коробку без исходного наблюдения')
                return True
            q=trial.lab.timeline.latest('joints')[2]
            measured_offsets={}
            for s in SIDES:
                model=c.models[s]
                measured_offsets[s]=(model.tool([q[k] for k in model.names])[:3,3]-measured_center).tolist()
            planned=trial.current.get('grip_offsets',{})
            errors={s:float(np.linalg.norm(np.asarray(measured_offsets[s])-np.asarray(planned[s]))) for s in SIDES}
            if max(errors.values())>.030:
                trial.event(now,'Фактический охват отличается от проверенного плана',offset_error_mm={s:round(v*1000,1) for s,v in errors.items()})
                c.stop('Двухручный охват отклонился от проверенного плана более чем на 30 мм',now)
                return True
            trial.current['grip_offsets']=measured_offsets
            trial.tracker.carrying=True
            trial.event(now,'Контакт при двухручном захвате',hands=checks)
            trial.maintain_grip=True
            move_pair(trial,[trial.current['lift']],now,'dual_lift','Согласованный подъём обеих кистей на 4 см')
        else:
            trial.dual_confirmed=None
            if now-trial.entered<8.:return True
            trial.event(now,'Контакт при двухручном захвате',hands=checks)
            for s in SIDES:trial.fingers(s,np.ones(6))
            trial.enter('dual_open_failed',now,'Контакт обеих кистей не подтверждён; раскрываю на исходном месте')
    elif stage=='dual_lift' and c.status=='holding':
        obs=trial.tracker.latest if trial.tracker is not None else None
        base=trial.tracker.baseline if trial.tracker is not None else None
        delta=(np.asarray(obs['xyz'])-np.asarray(base['xyz']) if obs and base and obs.get('xyz') else None)
        views_ok=bool(obs and (len(obs.get('seen_by',[]))>=2 or trial.current.get('scan_view_count',0)>=3))
        success=bool(delta is not None and (obs.get('valid') or obs.get('partial_visible')) and views_ok
            and 0<=now-obs.get('stamp',-1)<.45 and .022<delta[2]<.075 and np.linalg.norm(delta[:2])<.030)
        trial.event(now,'Камерная проверка двухручного подъёма',verified=success,
            delta=delta.tolist() if delta is not None else None,vision=obs,views_ok=views_ok)
        if success:
            trial.dual_lift_scan_requested=False
            trial.current['scan_view_count']=len(getattr(trial.lab,'scan_views',()))
            trial.tracker.carrying=True
            move_pair(trial,trial.current['transport'],now,'dual_transport','Переношу коробку обеими ладонями над лотком')
        elif not getattr(trial,'dual_lift_scan_requested',False):
            try:
                trial.lab.request('scan',now)
                trial.dual_lift_scan_requested=True
                trial.event(now,'Повторяю RGB-D проверку с тремя небольшими поворотами камеры')
            except Exception as error:
                trial.event(now,'Повторный обзор после подъёма недоступен',error=str(error))
                move_pair(trial,[trial.current['approach'][-1]],now,'dual_lower','Возвращаю коробку на исходную высоту')
        elif now-trial.entered>15.:
            move_pair(trial,[trial.current['approach'][-1]],now,'dual_lower','Возвращаю коробку на исходную высоту: подъём не подтверждён')
    elif stage=='dual_transport' and c.status=='holding':
        obs=trial.tracker.latest if trial.tracker is not None else None
        q=trial.lab.timeline.latest('joints')[2]
        release=np.asarray(trial.current.get('release_center',()),float)
        held=trial.paired_center(q)
        if release.shape!=(3,) or np.linalg.norm(held-release)>.030:
            c.stop('Не раскрываю ладони: предмет не над выбранной точкой сброса',now)
            return True
        fresh=bool(obs and (obs.get('valid') or obs.get('partial_visible')) and 0<=now-obs.get('stamp',-1)<.45)
        if not fresh:
            if not getattr(trial,'dual_transport_scan_requested',False):
                try:
                    trial.lab.request('scan',now);trial.dual_transport_scan_requested=True
                    trial.event(now,'Повторяю обзор предмета над точкой сброса тремя ракурсами')
                    return True
                except Exception as error:
                    c.stop('Не раскрываю ладони без свежей камеры: '+str(error),now);return True
            c.stop('Не раскрываю ладони: трекер не подтвердил предмет над лотком',now);return True
        measured=np.asarray(obs['xyz'],float);tray_bounds=np.asarray(trial.current['tray_observation']['bounds'],float)
        source_bounds=np.asarray(trial.current['original']['bounds'],float)
        radius=float(np.max(np.ptp(source_bounds,axis=0)[:2])/2+CLEARANCE_M)
        lo=tray_bounds[0,:2]+radius;hi=tray_bounds[1,:2]-radius
        tracker_error=float(np.linalg.norm(measured-release))
        if tracker_error>.030 or np.any(lo>hi) or not np.all((measured[:2]>=lo)&(measured[:2]<=hi)):
            if not getattr(trial,'dual_transport_scan_requested',False):
                try:
                    trial.lab.request('scan',now);trial.dual_transport_scan_requested=True
                    trial.event(now,'Повторяю измерение лотка и предмета перед раскрытием',
                        tracker_error_mm=round(tracker_error*1000,1),tracker=measured.tolist(),
                        release_center=release.tolist(),safe_xy=[lo.tolist(),hi.tolist()])
                    return True
                except Exception:
                    pass
            c.stop('Не раскрываю ладони: RGB-D предмет вне выбранной точки или границ лотка',now)
            trial.event(now,'Предсбросовая проверка двухручного переноса',tracker_error_mm=round(tracker_error*1000,1),
                tracker=measured.tolist(),release_center=release.tolist(),safe_xy=[lo.tolist(),hi.tolist()])
            return True
        trial.event(now,'Две ладони и предмет подтверждены над точкой внутри лотка',
            tracker_error_mm=round(tracker_error*1000,1),held_center=held.tolist(),release_center=release.tolist())
        trial.released_expected=release.copy();trial.tracker.carrying=False
        for s in SIDES:trial.fingers(s,np.ones(6))
        trial.enter('dual_release',now,'Раскрываю обе ладони над выбранной точкой в лотке')
    elif stage=='dual_release' and now-trial.entered>1.5:
        if not all(min(trial.lab.timeline.latest('hand_'+s)[2]['position'])>.9 for s in SIDES):
            if now-trial.entered>4.:c.stop('Обе кисти не раскрылись внутри лотка',now)
            return True
        obs=trial.tracker.latest if trial.tracker is not None else None
        fresh=bool(obs and (obs.get('valid') or obs.get('partial_visible')) and 0<=now-obs.get('stamp',-1)<.5)
        verified=False
        if fresh:
            tray_bounds=np.asarray(trial.current['tray_observation']['bounds'],float)
            source_bounds=np.asarray(trial.current['original']['bounds'],float)
            radius=float(np.max(np.ptp(source_bounds,axis=0)[:2])/2+CLEARANCE_M)
            center=np.asarray(obs['xyz'],float);release=np.asarray(trial.current['release_center'],float)
            verified=(np.linalg.norm(center-release)<=.030 and
                      np.all(center[:2]>=tray_bounds[0,:2]+radius) and
                      np.all(center[:2]<=tray_bounds[1,:2]-radius))
        if not fresh and not getattr(trial,'dual_release_scan_requested',False):
            try:
                trial.lab.request('scan',now);trial.dual_release_scan_requested=True
                trial.event(now,'Проверяю укладку тремя ракурсами после отпускания')
                return True
            except Exception as error:
                trial.event(now,'Повторный обзор после отпускания недоступен',error=str(error))
        trial.event(now,'Проверка предмета после отпускания',verified=verified,vision=obs,
                    release_center=trial.current['release_center'])
        trial.released=True;trial.scene_changed=True
        retreat=list(reversed(trial.current['transport'][:-1]))+list(reversed(trial.current['approach'][:-2]))+[trial.current['start']]
        move_pair(trial,retreat,now,'dual_retreat','Отвожу раскрытые ладони по проверенному пути')
    elif stage=='dual_lower' and c.status=='holding':
        for s in SIDES:trial.fingers(s,np.ones(6))
        trial.tracker.carrying=False
        trial.enter('dual_open_failed',now,'Отпускаю предмет над исходным местом после проверки')
    elif stage=='dual_open_failed' and now-trial.entered>1.5:
        if not all(min(trial.lab.timeline.latest('hand_'+s)[2]['position'])>.9 for s in SIDES):
            c.stop('Одна из кистей не раскрылась; отвод двух рук запрещён',now);return True
        move_pair(trial,list(reversed(trial.current['approach'][:-1]))+[trial.current['start']],now,'dual_retreat','Отвожу обе руки по проверенному пути')
    elif stage=='dual_retreat' and c.status=='holding':
        trial.tracker=None;trial.released_expected=None
        trial.enter('next',now,'Двухручная попытка завершена; перехожу к следующему предмету')
    return True

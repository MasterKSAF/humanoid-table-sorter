"""Measured cap pinch candidates. First validate extraction/return, before insertion."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from team22_bringup.hand_geometry import HandGeometry
from team22_bringup.grasp import CLEARANCE_M


def qualified_caps(scene):
    result=[]
    for o in (scene or {}).get('objects',[]):
        if o['kind']!='tube_standing' or o.get('status')!='observed' or len(o.get('seen_by',[]))!=2:continue
        views=list(o.get('camera_observations',{}).values())
        if len(views)!=2:continue
        a,b=(np.asarray(v['bounds']) for v in views)
        # Centroids of different visible faces need not coincide. Require
        # adjacent/overlapping surfaces and a cap-sized union instead.
        gap=np.maximum(np.maximum(a[0]-b[1],b[0]-a[1]),0.)
        union=np.maximum(a[1],b[1])-np.minimum(a[0],b[0])
        if np.linalg.norm(gap)>.006 or max(union[:2])>.045:continue
        if np.linalg.norm(np.asarray(views[0]['xyz'])-views[1]['xyz'])>=.020:continue
        result.append(o)
    return result


def select_cap(scene, reference=None):
    caps=qualified_caps(scene)
    if reference is not None:
        caps=[o for o in caps if o['color']==reference['color'] and
              np.linalg.norm(np.array(o['xyz'])-reference['xyz'])<.018]
    if not caps:return None
    # Front cap offers the most free space. This uses observed positions only.
    cap=min(caps,key=lambda o:o['xyz'][0])
    return dict(cap,kind='tube_cap',source_cap_id=cap['id'])


def template_candidate(obj,points,plane,side='right',refined=False):
    """Offline-reachable hand shape; verify contacts on this observed cap."""
    plane=np.array(plane);center=np.array(obj['xyz']);bounds=np.array(obj['bounds'])
    yaw,tilt,dz=(-.6,.6,.018) if refined else (-1.2,.35,.012)
    c=dict(arm_side=side,pocket=[.055,.03,.12] if refined else [.055,.04,.13],pinch_table_height=float(center[2]-center[:2]@plane[:2]-plane[2]+dz),
        approach_preshape=[.05,.05,.05,.65,1.,.45],close_command=[.05,.05,.05,.25,.10,.45],
        finger_down_tilt=tilt,pinch_roll=0.,pinch_yaws=[yaw],contact_fingers=[3])
    points=np.asarray(points);own=points[np.all((points>=bounds[0]-.012)&(points<=bounds[1]+.012),axis=1)]
    surface=points[np.all((points>=bounds[0]-.004)&(points<=bounds[1]+.004),axis=1)]
    if len(surface)<15:return None
    eq=ConvexHull(surface).equations;hand=HandGeometry(side);sign=1 if side=='left' else -1
    direction=np.array([np.cos(yaw),sign*np.sin(yaw),0]);palm=np.array([0,0,-1])
    rot=np.column_stack((palm,np.cross(direction,palm),direction))@Rotation.from_euler('y',tilt).as_matrix()
    origin=center+[0,0,dz]-rot@c['pocket']
    if hand.collision((own-origin)@rot,hand.nominal_angles(c['approach_preshape']),CLEARANCE_M):return None
    closed=hand.vertices(hand.nominal_angles(c['close_command']));overlaps=[]
    for finger in ('thumb','index'):
        links={j['child'] for j in hand.joints if finger in j['name']}
        vertices=np.concatenate([v[::max(1,len(v)//70)] for n,v in closed.items() if n in links])@rot.T+origin
        overlaps.append(-float(np.min(np.max(vertices@eq[:,:3].T+eq[:,3],axis=1))))
    if min(overlaps)<.0005:return None
    c['predicted_thumb_overlap_mm'],c['predicted_finger_overlap_mm']=[v*1000 for v in overlaps]
    return c


def candidates(obj,points,plane,side,command,unused=.3):
    bounds=np.asarray(obj['bounds']);points=np.asarray(points)
    points=points[np.all((points>bounds[0]-.004)&(points<bounds[1]+.004),axis=1)]
    if len(points)<15:return []
    hull=ConvexHull(points);eq=hull.equations
    hand=HandGeometry(side);preshape=[unused,unused,unused,.65,1.,command[5]]
    opened=hand.vertices(hand.nominal_angles(preshape));closed=hand.vertices(hand.nominal_angles(command))
    thumb_links={j['child'] for j in hand.joints if 'thumb' in j['name']}
    finger_links={j['child'] for j in hand.joints if 'index' in j['name']}
    def sparse(v):return v[::max(1,len(v)//50)]
    thumb=np.concatenate([sparse(v) for n,v in closed.items() if n in thumb_links])
    fingers=np.concatenate([sparse(v) for n,v in closed.items() if n in finger_links])
    open_points=np.concatenate([sparse(v) for v in opened.values()])
    center=np.asarray(obj['xyz']);plane=np.asarray(plane);base=center[:2]@plane[:2]+plane[2]
    sign=1 if side=='left' else -1;result=[]
    for yaw in (-.3,0.,-.6,.6,-1.2,1.2,2.,-2.):
        direction=np.array([np.cos(yaw),sign*np.sin(yaw),0.]);palm=np.array([0.,0.,-1.])
        for tilt,roll in ((t,r) for t in (.35,.75,1.25) for r in (0.,-.65,.65,-1.3,1.3)):
            rotation=(np.column_stack((palm,np.cross(direction,palm),direction))@
                      Rotation.from_euler('y',tilt).as_matrix()@Rotation.from_euler('z',roll).as_matrix())
            rt=rotation.T@eq[:,:3].T
            for px in (.035,.045,.055,.065):
                for py in (0.,.02,.04):
                    for pz in (.17,.19,.15,.13):
                        pocket=np.array([px,py,pz])
                        for dz in (0.,.006,.012,-.006):
                            position=center+[0,0,dz];origin=position-rotation@pocket
                            shift=origin@eq[:,:3].T+eq[:,3]
                            if np.min(np.max(open_points@rt+shift,axis=1))<CLEARANCE_M:continue
                            t=float(np.min(np.max(thumb@rt+shift,axis=1)))
                            f=float(np.min(np.max(fingers@rt+shift,axis=1)))
                            if max(t,f)>-.0005:continue
                            result.append(dict(score=min(-t,-f),pocket=pocket.tolist(),
                                pinch_table_height=float(position[2]-base),approach_preshape=preshape,
                                finger_down_tilt=tilt,pinch_roll=roll,pinch_yaws=[yaw],arm_side=side,
                                close_command=command,contact_fingers=[3],
                                predicted_thumb_overlap_mm=-t*1000,predicted_finger_overlap_mm=-f*1000))
    return sorted(result,key=lambda a:a['score'],reverse=True)


def cap_sweep_clear(candidate,obj,points,plane):
    """Reject impossible finger sweeps before spending time on arm IK."""
    side=candidate['arm_side'];hand=HandGeometry(side)
    yaw=candidate['pinch_yaws'][0];sign=1 if side=='left' else -1
    direction=np.array([np.cos(yaw),sign*np.sin(yaw),0.]);palm=np.array([0.,0.,-1.])
    rotation=(np.column_stack((palm,np.cross(direction,palm),direction))@
              Rotation.from_euler('y',candidate['finger_down_tilt']).as_matrix()@
              Rotation.from_euler('z',candidate['pinch_roll']).as_matrix())
    center=np.array(obj['xyz']);plane=np.array(plane)
    center[2]=center[:2]@plane[:2]+plane[2]+candidate['pinch_table_height']
    origin=center-rotation@candidate['pocket'];bounds=np.array(obj['bounds'])
    points=np.asarray(points);own=np.all((points>=bounds[0]-.012)&(points<=bounds[1]+.012),axis=1)
    local=(points[~own]-origin)@rotation
    for step,command in enumerate(np.linspace(candidate['approach_preshape'],candidate['close_command'],13)):
        angles=hand.nominal_angles(command)
        collision=hand.collision(local,angles,CLEARANCE_M)
        if collision:return f'obstacle {collision}, step {step}'
        vertices=np.concatenate(list(hand.vertices(angles).values()))@rotation.T+origin
        gap=np.min(vertices[:,2]-vertices[:,:2]@plane[:2]-plane[2])
        if gap<CLEARANCE_M:return f'table {gap*1000:.1f} mm, step {step}'
    return None


def plan_cap(packet):
    from geometry_grasp import plan_geometry
    begin=time.monotonic();obj=next(o for o in packet['scene']['objects'] if o['kind']=='tube_cap')
    attempts=[]
    # Try a small number of validated hand templates on every measured cap
    # before an expensive grid search on the front one.
    for cap in sorted(qualified_caps(packet['scene']),key=lambda o:o['xyz'][0]):
        cap=dict(cap,kind='tube_cap',source_cap_id=cap['id'])
        for side,refined in (('right',True),('left',True),('right',False),('left',False)):
            candidate=template_candidate(cap,packet['obstacles'],packet['scene']['plane'],side,refined)
            if candidate is None or cap_sweep_clear(candidate,cap,packet['obstacles'],packet['scene']['plane']):continue
            scene=dict(packet['scene'],objects=[o for o in packet['scene']['objects'] if o['kind']!='tube_cap']+[cap])
            result=plan_geometry(dict(packet,scene=scene,**candidate,grip_only=True,backward_approach=True,
                hand_table_margin=CLEARANCE_M,object_margin=CLEARANCE_M,
                hand_object_margin=CLEARANCE_M,delayed_preshape=True))
            attempts.append(dict(cap=cap['id'],candidate=candidate,error=result.get('error'),failures=result.get('failures')))
            if 'error' not in result:
                result['candidate_attempts']=attempts;result['close_command']=candidate['close_command']
                result['transfer_reason']='Проверка извлечения и возврата пробирки; перенос в шкаф ещё не проверен'
                return result
    ranked=[]
    for side in ('right','left'):
        ranked+=candidates(obj,packet['obstacles'],packet['scene']['plane'],side,[.3,.3,.3,.25,.10,.45])
    ranked.sort(key=lambda a:a['score'],reverse=True)
    groups={}
    for c in ranked:groups.setdefault((c['arm_side'],c['pinch_yaws'][0],c['finger_down_tilt'],c['pinch_roll']),[]).append(c)
    ranked=[v[i] for i in range(20) for v in groups.values() if len(v)>i]
    for candidate in ranked:
        if time.monotonic()-begin>160:break
        rejected=cap_sweep_clear(candidate,obj,packet['obstacles'],packet['scene']['plane'])
        if rejected:
            attempts.append(dict(candidate=candidate,error='Номинальное закрытие задевает соседей или стол',failures=rejected))
            continue
        command=candidate['close_command']
        # Visible-face centroids differ by viewpoint and are not calibration error.
        result=plan_geometry(dict(packet,**candidate,grip_only=True,backward_approach=True,
            hand_table_margin=CLEARANCE_M,object_margin=CLEARANCE_M,
            hand_object_margin=CLEARANCE_M,delayed_preshape=True))
        attempts.append(dict(candidate=candidate,error=result.get('error'),failures=result.get('failures')))
        if 'error' not in result:
            result['candidate_attempts']=attempts
            result['close_command']=command
            result['transfer_reason']='Проверка удержания крышки; пробирка возвращается в исходный штатив, перенос в шкаф ещё не проверен'
            return result
    return dict(kind='tube_cap',error='Не найден подвод к крышке без касания соседей и штатива',candidate_attempts=attempts,
                planning_ms=(time.monotonic()-begin)*1000)

"""Rank grasp poses by opposed finger intersection with observed object volume.

These are prospective candidates, not evidence of contact or retention.
Full mesh, IK and route checks must still follow.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from team22_bringup.hand_geometry import HandGeometry
from team22_bringup.grasp import CLEARANCE_M


def opposed_contact_quality(geometry,hull,candidate,center,plane,side,command):
    """Nominal first surface contacts, never a claim of real force closure."""
    yaw=candidate['pinch_yaws'][0];sign=1 if side=='left' else -1
    direction=np.array([np.cos(yaw),sign*np.sin(yaw),0.]);palm=np.array([0.,0.,-1.])
    rotation=np.column_stack((palm,np.cross(direction,palm),direction))@Rotation.from_euler('y',candidate['finger_down_tilt']).as_matrix()
    position=np.array(center);position[2]=position[:2]@plane[:2]+plane[2]+candidate['pinch_table_height']
    origin=position-rotation@candidate['pocket'];eq=hull.equations
    groups=[{j['child'] for j in geometry.joints if 'thumb' in j['name']},
            {j['child'] for j in geometry.joints if 'thumb' not in j['name']}]
    contacts=[None,None]
    for cmd in np.linspace(candidate['approach_preshape'],command,13):
        vertices=geometry.vertices(geometry.nominal_angles(cmd))
        for index,names in enumerate(groups):
            if contacts[index] is not None:continue
            local=np.concatenate([v[::max(1,len(v)//50)] for n,v in vertices.items() if n in names])
            world=local@rotation.T+origin;distance=world@eq[:,:3].T+eq[:,3]
            d=distance.max(axis=1);i=d.argmin()
            if d[i]<=.001:
                normal=eq[distance[i].argmax(),:3]
                contacts[index]=(world[i],normal)
        if all(c is not None for c in contacts):break
    if any(c is None for c in contacts):return dict(opposed_contact_score=-1.,contact_normals=None)
    dot=float(contacts[0][1]@contacts[1][1]);vertical=max(abs(c[1][2]) for c in contacts)
    # Same-facing top surfaces can block fingers while the object stays on table.
    quality=-dot-.5*max(0.,vertical-.65)
    return dict(opposed_contact_score=quality,contact_normals=[c[1].tolist() for c in contacts],
                predicted_first_contacts=[c[0].tolist() for c in contacts])


def ranked_candidates(obj, points, plane, side, command, table_reserve=CLEARANCE_M):
    bounds=np.asarray(obj['bounds']);points=np.asarray(points)
    points=points[np.all((points>bounds[0]-.003)&(points<bounds[1]+.003),axis=1)]
    if len(points)<30:return []
    hull=ConvexHull(points);equations=hull.equations
    geometry=HandGeometry(side)
    preshape=[1.]*5+[command[5]]
    opened=geometry.vertices(geometry.nominal_angles(preshape))
    closed=geometry.vertices(geometry.nominal_angles(command))
    thumb_links={j['child'] for j in geometry.joints if 'thumb' in j['name']}
    finger_links={j['child'] for j in geometry.joints if 'thumb' not in j['name']}
    def thin(v,n=70):return v[::max(1,len(v)//n)]
    thumb=np.concatenate([thin(v) for n,v in closed.items() if n in thumb_links])
    fingers=np.concatenate([thin(v) for n,v in closed.items() if n in finger_links])
    open_points=np.concatenate([thin(v) for v in opened.values()])
    sweep=np.concatenate([np.concatenate(list(geometry.vertices(geometry.nominal_angles(q)).values()))
                          for q in np.linspace(preshape,command,9)])
    center=np.asarray(obj['xyz']);plane=np.asarray(plane)
    base=center[:2]@plane[:2]+plane[2];sign=1 if side=='left' else -1
    results=[]
    for yaw in (-.3,0.,-.6,-1.,-1.4,.6,1.2,2.):
        direction=np.array([np.cos(yaw),sign*np.sin(yaw),0.]);palm=np.array([0.,0.,-1.])
        for tilt in (.35,.55,.75,1.):
            rotation=np.column_stack((palm,np.cross(direction,palm),direction))@Rotation.from_euler('y',tilt).as_matrix()
            normal=rotation.T@np.r_[-plane[:2],1.]
            min_local=float(np.min(sweep@normal))
            rt=rotation.T@equations[:,:3].T
            for px in (.035,.045,.055,.065):
                for py in (.0,.02,.04):
                    for pz in (.13,.15,.17):
                        pocket=np.array([px,py,pz]);position=center.copy()
                        # Thin-object finger contact requires a lower envelope
                        # than the transport clearance. Arm paths are checked next.
                        height=max(CLEARANCE_M,table_reserve-min_local+pocket@normal)
                        if height>.060:continue
                        position[2]=base+height
                        origin=position-rotation@pocket
                        shift=origin@equations[:,:3].T+equations[:,3]
                        # Open-state penetrations are rejected before expensive IK.
                        if np.min(np.max(open_points@rt+shift,axis=1))<CLEARANCE_M:continue
                        dt=np.max(thumb@rt+shift,axis=1);df=np.max(fingers@rt+shift,axis=1)
                        t=float(dt.min());f=float(df.min())
                        if max(t,f)>.001:continue
                        score=min(-t,-f)
                        results.append(dict(score=score,pocket=pocket.tolist(),pinch_table_height=height,
                                            approach_preshape=preshape,
                                            finger_down_tilt=tilt,pinch_yaws=[yaw],
                                            predicted_thumb_overlap_mm=-t*1000,predicted_finger_overlap_mm=-f*1000))
    results.sort(key=lambda r:r['score'],reverse=True)
    # Examine several directions before expensive IK, rather than repeatedly
    # choosing the deepest nominal overlap on the same rounded top surface.
    groups={}
    for c in results:groups.setdefault((c['pinch_yaws'][0],c['finger_down_tilt']),[]).append(c)
    candidates=[v[i] for i in range(5) for v in groups.values() if len(v)>i]
    from team22_bringup.grasp_priors import failed_pinch_prior
    for c in candidates:
        c.update(opposed_contact_quality(geometry,hull,c,center,plane,side,command))
        c['failed_prior']=failed_pinch_prior(obj,side,c)
    return sorted(candidates,key=lambda c:(not bool(c['failed_prior']),c['opposed_contact_score'],c['score']),reverse=True)

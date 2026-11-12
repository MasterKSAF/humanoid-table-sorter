"""Checked free-space entry for only the arm that will grasp the object."""
import numpy as np
from team22_bringup.posture import preparation_routes
from team22_bringup.grasp import CLEARANCE_M

def transit(model,start,goal,table,other,margin=CLEARANCE_M,edge_check=None,prefer_compact=False,
            max_rrt_seconds=5.):
    from transfer_trial import checked,body_clear
    clear=edge_check or (lambda a,b:checked(model,a,b,table,other,margin))
    if clear(start,goal):return [goal]
    # For mesh-checked grasps, try raising and bending together. The original
    # fully stretched sideways lift spends a long time supporting a long lever.
    # These are candidates only: both edges must pass the unchanged clearance.
    if hasattr(model,'hand_geometry') and start[3]>-.65:
        sign=1 if model.side=='left' else -1
        entries=[]
        for pitch,roll,elbow in ((.3,1.05,-.5),(0.,.85,-.9),(-.3,.65,-1.3)):
            q=np.array([pitch,sign*roll,0.,elbow,0.,0.,0.])
            if clear(start,q) and clear(q,goal):
                entries.append((max(abs(q-start))+max(abs(goal-q)),q))
        if entries:return [min(entries,key=lambda x:x[0])[1],goal]
    if prefer_compact:
        # Candidate from the recorded RGB-D cap search (cap-compact-plan.json).
        # It is only a seed: every segment is rechecked against this scene.
        if model.side=='right' and start[3]>-.65:
            candidate=[np.array([.336,-.590,-.532,-.510,.063,.024,-.129]),
                       np.array([-.144,-.673,-.073,-1.092,.198,-.053,-.034]),goal]
            previous=start
            for target in candidate:
                if not clear(previous,target):break
                previous=target
            else:return candidate
        # Avoid holding a fully stretched arm while bending its elbow. Sampling
        # finds a coupled shoulder/elbow exit, checked against this live cloud.
        from motion_planner import rrt_connect
        try:
            route=rrt_connect(start,goal,np.array([model.limits[0]+.035,model.limits[1]-.035]),
                lambda q:clear(q,q),min(4.,max_rrt_seconds))
            if all(clear(a,b) for a,b in zip(route,route[1:])):return route[1:]
        except (ValueError,ImportError):pass
    # Reuse known arm-only escape shapes before spending time on sampling search.
    for candidate in preparation_routes(model.side,start):
        path=[];previous=start
        for _,q in candidate:
            if not clear(previous,q):break
            path.append(np.asarray(q));previous=q
            if clear(previous,goal):return path+[goal]
    from motion_planner import rrt_connect
    valid=(lambda q:clear(q,q)) if edge_check else (lambda q:body_clear(model,q) and model.clearance(q,table,other)>margin)
    try:
        route=rrt_connect(start,goal,np.array([model.limits[0]+.035,model.limits[1]-.035]),valid,max_rrt_seconds)
    except (ValueError,ImportError):return None
    if not all(clear(a,b) for a,b in zip(route,route[1:])):return None
    return route[1:]

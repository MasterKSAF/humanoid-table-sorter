"""Prospective finger sweep against observed surroundings, using official meshes."""
import numpy as np
from team22_bringup.hand_geometry import HandGeometry
from team22_bringup.grasp import CLEARANCE_M

def closure_clear(model,q,table,angles,command,minimum=CLEARANCE_M,initial_command=None):
    hand=model.wrist(q)@model.hand
    geometry=HandGeometry(model.side)
    if initial_command is None:
        states=[{k:v for k,v in angles.items() if k.startswith(model.side+'_')}]
        states+=geometry.closing_sweep(command)
    else:
        states=[geometry.nominal_angles(c) for c in np.linspace(initial_command,command,17)]
    plane,edge,tree=table
    # Only observed obstacle points; the intended target has been excluded by
    # the planner. Table contact is never treated as grasp contact.
    local=(tree.data-hand[:3,3])@hand[:3,:3]
    for index,state in enumerate(states):
        vertices=np.concatenate(list(geometry.vertices(state).values()))
        world=vertices@hand[:3,:3].T+hand[:3,3]
        over=world[:,0]>=edge
        gap=np.min(world[over,2]-world[over,:2]@plane[:2]-plane[2],initial=np.inf)
        if gap<minimum:return False,f'table {gap*1000:.1f} mm, step {index}'
        collision=geometry.collision(local,state,minimum)
        if collision:return False,f'obstacle at {collision}, step {index}'
    return True,'nominal sweep clear; encoder/contact validation still required'


def checked_hand_table(model,start,goal,table,other,hand_minimum=CLEARANCE_M,
                       arm_minimum=CLEARANCE_M,hand_object_minimum=None):
    """Lower fingertip envelope for thin objects; keep arm/obstacle reserves."""
    from transfer_trial import body_clear
    plane,edge,_=table
    for q in np.linspace(start,goal,max(2,int(np.ceil(np.max(abs(np.asarray(goal)-start))/.025))+1)):
        if not body_clear(model,q):return False
        tg,og=model.clearances(q,table,other)
        if tg<hand_minimum or og<(arm_minimum if hand_object_minimum is None else hand_object_minimum):return False
        points,radii=model.samples(q);points=points[:112];radii=radii[:112]
        mask=points[:,0]+radii>=edge
        gaps=points[:,2]-points[:,:2]@plane[:2]-plane[2]-radii
        if np.min(gaps[mask],initial=np.inf)<arm_minimum:return False
        if hand_object_minimum is not None:
            obstacles=table[2].query(points)[0]-radii
            if other is not None:obstacles=np.minimum(obstacles,other.query(points)[0]-.038-radii)
            if np.min(obstacles,initial=np.inf)<arm_minimum:return False
    return True

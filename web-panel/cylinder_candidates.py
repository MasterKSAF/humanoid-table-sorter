"""Opposed-contact candidates from observed cylinder surface and robot meshes."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from team22_bringup.hand_geometry import HandGeometry
from team22_bringup.grasp import CLEARANCE_M
from side_grasp import orientation


def rank_cylinder(obj, points, plane, side, command, preshape=None, thumb_below_top=False,
                  minimum=CLEARANCE_M,table_minimum=CLEARANCE_M,max_candidates=2):
    bounds=np.asarray(obj['bounds']);points=np.asarray(points)
    points=points[np.all((points>=bounds[0]-.003)&(points<=bounds[1]+.003),axis=1)]
    if len(points)<30:return []
    eq=ConvexHull(points).equations;g=HandGeometry(side)
    preshape=list(preshape) if preshape is not None else [1.]*5+[command[5]]
    opened=g.vertices(g.nominal_angles(preshape));closed=g.vertices(g.nominal_angles(command))
    thumbs={j['child'] for j in g.joints if 'thumb' in j['name']}
    def sample(v):return v[::max(1,len(v)//80)]
    t=np.concatenate([sample(v) for n,v in closed.items() if n in thumbs])
    f=np.concatenate([sample(v) for n,v in closed.items() if n not in thumbs and n!='base_link'])
    o=np.concatenate([sample(v) for v in opened.values()])
    open_thumb=np.concatenate([v for n,v in opened.items() if n in thumbs])
    center=np.array(obj['xyz']);height=np.ptp(bounds,axis=0)[2]
    base_height=.078 if obj['kind']=='gold_jar' else float(np.clip(height*.55,.062,.115));plane=np.asarray(plane)
    result=[]
    yaws=(1.2,.9,1.5,1.8,2.2,-.3,0.,-.6,.3,.6,-.9) if thumb_below_top else (-.3,0.,-.6,.3,.6,-.9)
    for yaw in yaws:
      for roll in (.6,.3,0.,.9):
        rot=orientation(yaw,side)@Rotation.from_euler('z',roll).as_matrix()
        rt=rot.T@eq[:,:3].T
        for z_shift in (0.,.012,-.012,.024):
          target=center.copy();target[2]=bounds[0,2]+min(base_height+z_shift,height-.020)
          for px in (.035,.045,.055,.065,.075):
           for py in (-.03,-.015,0.,.015,.03):
            for pz in (.12,.14,.16):
              pocket=np.array([px,py,pz]);origin=target-rot@pocket;shift=origin@eq[:,:3].T+eq[:,3]
              if thumb_below_top and np.max((open_thumb@rot.T+origin)[:,2]) > bounds[1,2]-CLEARANCE_M:continue
              if np.min(np.max(o@rt+shift,axis=1))<minimum:continue
              dt=np.min(np.max(t@rt+shift,axis=1));df=np.min(np.max(f@rt+shift,axis=1))
              if max(dt,df)>.0005:continue
              world=o@rot.T+origin
              if np.min(world[:,2]-world[:,:2]@plane[:2]-plane[2])<table_minimum:continue
              result.append(dict(score=float(min(-dt,-df)),pocket=pocket.tolist(),palm_roll=roll,
                height_adjustment=z_shift,cylinder_yaws=[yaw],approach_preshape=preshape,
                predicted_thumb_overlap_mm=float(-dt*1000),predicted_finger_overlap_mm=float(-df*1000)))
              if len(result)>=max_candidates:return result
    return sorted(result,key=lambda x:x['score'],reverse=True)

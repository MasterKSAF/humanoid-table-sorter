"""Choose contained release sites from current tray bounds and depth obstacles."""
import numpy as np
from team22_bringup.grasp import CLEARANCE_M


def release_sites(obj,tray,points):
    bounds=np.asarray(tray['bounds']);target=np.asarray(obj['bounds'])
    radius=max(np.ptp(target,axis=0)[:2])/2
    inner=radius+CLEARANCE_M;lo=bounds[0,:2]+inner;hi=bounds[1,:2]-inner
    if np.any(lo>hi):return []
    points=np.asarray(points)
    # The 18/25 mm filters distinguish tray wall and floor returns from an
    # occupied interior; actual placement clearance is CLEARANCE_M.
    mask=(np.all((points[:,:2]>bounds[0,:2]+.018)&(points[:,:2]<bounds[1,:2]-.018),axis=1)
          &(points[:,2]>bounds[0,2]+.025))
    occupied=points[mask];sites=[]
    for x in np.linspace(lo[0],hi[0],3):
        for y in np.linspace(lo[1],hi[1],3):
            center=np.array([x,y])
            if len(occupied) and np.any(np.linalg.norm(occupied[:,:2]-center,axis=1)<radius+CLEARANCE_M):continue
            sites.append([float(x),float(y),float(tray['xyz'][2])])
    # The robot approaches from the near side (-X): fill the opposite wall first.
    return sorted(sites,key=lambda p:(-p[0],-abs(p[1])))

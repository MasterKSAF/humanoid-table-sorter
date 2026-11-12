"""Scene boundaries measured from RGB-D; no simulator position constants."""
import numpy as np
from .survey import HEAD


def table_boundary(frames,angles,rig,plane):
    candidates=[]
    for camera,(_,depth,k) in frames.items():
        pose=rig.body if camera=='torso' else rig.head_pose([angles[n] for n in HEAD])
        v,u=np.mgrid[:depth.shape[0]:2,:depth.shape[1]:2];d=depth[::2,::2]/1000.
        rays=np.stack(((u-k[2])*d/k[0],(v-k[5])*d/k[4],d),-1)
        p=rays@pose[:3,:3].T+pose[:3,3]
        h=p[:,:,2]-p[:,:,:2]@plane[:2]-plane[2];valid=(d>.15)&(d<2.5)
        # A surface ending against a measured lower surface is an observed edge.
        # The bottom image border and missing depth are not evidence of an edge.
        # A rounded or sloping front face can descend gradually across many
        # pixels. Keep all intervening depth samples and use the nearest x
        # seen anywhere in the transition, not just its tabletop endpoint.
        for step in (2, 12, 20):
            if h.shape[0] <= step: continue
            mask=(abs(h[:-step])<.008)&(h[step:]<-.025)
            nearest=p[:-step,:,0].copy()
            for offset in range(step+1):
                mask &= valid[offset:offset+len(mask)] & (h[offset:offset+len(mask)]<.012)
                nearest=np.minimum(nearest,p[offset:offset+len(mask),:,0])
            ep=np.column_stack((nearest[mask],p[:-step,:,1][mask]))
            ep=ep[(ep[:,0]>0)&(ep[:,0]<1.2)&(abs(ep[:,1])<1.2)]
            if len(ep)<30:continue
            near=ep[ep[:,0]<np.quantile(ep[:,0],.1)+.018]
            if len(near)<30 or np.ptp(near[:,1])<.12 or np.ptp(near[:,0])>.025:continue
            candidates.append(dict(camera=camera,edge_x_m=float(np.quantile(near[:,0],.05)),
                                   y_span_m=np.ptp(near[:,1]).item(),points=len(near)))
            break
    if not candidates:return dict(observed=False,reason='Ближняя граница стола не подтверждена перепадом глубины')
    edges=[x['edge_x_m'] for x in candidates]
    if np.ptp(edges)>.025:return dict(observed=False,reason='Камеры расходятся в положении границы стола')
    return dict(observed=True,frame='waist_yaw_link',edge_x_m=min(edges)-.015,
                reserve_mm=15.,observations=candidates,source='table_surface_to_lower_depth_transition')


def planning_edge(scene):
    geometry=scene.get('table_geometry') or {}
    # Without a measured boundary, conservatively extend the measured plane
    # behind the robot. This sentinel is not an assumed scene coordinate.
    return float(geometry['edge_x_m']) if geometry.get('observed') else -100.


def environment_points(points,plane):
    p=np.asarray(points);h=p[:,2]-p[:,:2]@plane[:2]-plane[2]
    # Workspace cropping is only a search range. Keep side/behind obstacles,
    # high cabinet surfaces and vertical table faces, not only tabletop objects.
    near=(np.linalg.norm(p[:,:2],axis=1)<1.6)&(p[:,2]>-.60)&(p[:,2]<1.20)
    torso=(p[:,0]>-.23)&(p[:,0]<.16)&(abs(p[:,1])<.20)&(p[:,2]>-.38)&(p[:,2]<.72)
    return p[near&~torso&(abs(h)>.012)]

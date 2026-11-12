"""Conservative removal of camera returns on sampled official robot surfaces.

This is NOT a collision checker and never fills a hull or masks an entire AABB.
Only returns within 8 mm of measured robot surfaces are removed. Finger
collision-mesh faces are sampled at 3 mm; their enclosing volume is never erased.
"""
from pathlib import Path
from functools import lru_cache
import time
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.spatial.transform import Rotation
from .grasp import origin, Arm
from .hand_geometry import HandGeometry


@lru_cache(maxsize=2)
def hand_surface_samples(side):
    """Immutable local meshes are sampled once, before live movement begins."""
    hand=HandGeometry(side);surfaces={}
    for name,shape in hand.shapes.items():
        vertices=shape['vertices'];pieces=[vertices]
        for tri in vertices[ConvexHull(vertices).simplices]:
            steps=max(1,int(np.ceil(max(np.linalg.norm(tri[i]-tri[j]) for i,j in ((0,1),(1,2),(2,0)))/.003)))
            bary=np.array([(i/steps,j/steps) for i in range(steps+1) for j in range(steps+1-i)])
            pieces.append(tri[0]+bary[:,0,None]*(tri[1]-tri[0])+bary[:,1,None]*(tri[2]-tri[0]))
        points=np.concatenate(pieces)
        _,ids=np.unique(np.round(points/.001).astype(int),axis=0,return_index=True)
        sampled=points[ids];sampled.setflags(write=False);surfaces[name]=sampled
    return surfaces


class RobotSurfaceMask:
    def __init__(self):
        root=Path(__file__).parent
        with np.load(root/'robot_surfaces.npz') as data:
            self.surfaces={k:data[k].copy() for k in data.files}
        self.hands={side:HandGeometry(side) for side in ('left','right')}
        self.arms={side:Arm(side) for side in self.hands}
        self.hand_surfaces={side:hand_surface_samples(side) for side in self.hands}
        self.joints=[]
        for j in ET.parse(root/'upper_body.urdf').findall('joint'):
            a=j.find('axis')
            self.joints.append((j.find('parent').get('link'),j.find('child').get('link'),j.get('name'),
                                origin(j),None if j.get('type')=='fixed' else np.fromstring(a.get('xyz'),sep=' ')))

    def points(self, angles):
        poses={'waist_yaw_link':np.eye(4)};pending=list(self.joints)
        while pending:
            previous=len(pending)
            for joint in pending[:]:
                parent,child,name,offset,axis=joint
                if parent not in poses:continue
                p=offset.copy()
                if axis is not None:
                    if name not in angles or not np.isfinite(angles[name]):
                        raise ValueError('Нет угла для фильтра поверхности робота: '+name)
                    p[:3,:3]=p[:3,:3]@Rotation.from_rotvec(axis*angles[name]).as_matrix()
                poses[child]=poses[parent]@p;pending.remove(joint)
            if len(pending)==previous:raise ValueError('Разрыв кинематической цепи фильтра робота')
        result=[p@poses[name][:3,:3].T+poses[name][:3,3] for name,p in self.surfaces.items()]
        self.filtered_hands=[]
        for side,hand in self.hands.items():
            names=[j['name'] for j in hand.joints]
            if any(n not in angles or not np.isfinite(angles[n]) for n in names):continue
            arm=self.arms[side];base=arm.wrist([angles[n] for n in arm.names])@arm.hand
            for name,frame in hand.frames(angles).items():
                if name not in self.hand_surfaces[side]:continue
                pose=base@frame
                result.append(self.hand_surfaces[side][name]@pose[:3,:3].T+pose[:3,3])
            self.filtered_hands.append(side)
        return np.concatenate(result)

    def filter(self, points, angles, tolerance=.008):
        begin=time.monotonic()
        distances=cKDTree(self.points(angles)).query(points,distance_upper_bound=tolerance)[0]
        keep=distances>tolerance
        # Depth also returns points from inside the open hand's collision
        # meshes. Surface samples alone leave those points in the obstacle map.
        # Remove only points inside the measured finger and palm solids.
        solid_removed=0
        world=np.asarray(points)
        for side,hand in self.hands.items():
            arm=self.arms[side]
            if any(name not in angles for name in arm.names):continue
            base=arm.wrist([angles[name] for name in arm.names])@arm.hand
            local=(world-base[:3,3])@base[:3,:3]
            frames=hand.frames(angles)
            for name,shape in hand.shapes.items():
                pose=frames[name]
                part=(local-pose[:3,3])@pose[:3,:3]
                vertices=shape['vertices']
                near=keep & np.all((part>=vertices.min(0)-tolerance)&
                                   (part<=vertices.max(0)+tolerance),axis=1)
                if not np.any(near):continue
                signed=part[near]@shape['planes'][:,:3].T+shape['planes'][:,3]
                inside=np.max(signed,axis=1)<=tolerance
                selected=np.flatnonzero(near)[inside]
                keep[selected]=False
                solid_removed+=len(selected)
        return world[keep],dict(removed=int((~keep).sum()),solid_removed=solid_removed,
                                          total=len(keep),tolerance_mm=tolerance*1000,
                                          method='official_robot_surfaces',finger_meshes=self.filtered_hands,
                                          processing_ms=(time.monotonic()-begin)*1000)

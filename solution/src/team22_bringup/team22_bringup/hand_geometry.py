"""Measured-joint hand geometry from official robot USD, in canonical hand frame."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import itertools
from scipy.spatial.transform import Rotation
from .grasp import CLEARANCE_M

@lru_cache(maxsize=1)
def _data():
    return json.loads(Path(__file__).with_name('hand_collisions.json').read_text())

class HandGeometry:
    def __init__(self,side):
        self.side=side;data=_data()['hands'][side]
        self.base=np.asarray(data['base_transform']);self.joints=data['joints']
        self.shapes={n:{k:np.asarray(v) for k,v in s.items()} for n,s in data['shapes'].items()}
        self.boxes_local={}
        for name,shape in self.shapes.items():
            v=shape['vertices'];_,axes=np.linalg.eigh(np.cov(v.T))
            projected=v@axes;low,high=projected.min(0),projected.max(0)
            self.boxes_local[name]=((low+high)/2@axes.T,axes,(high-low)/2)
        self._box_cache=None

    def boxes(self,angles):
        key=tuple(float(angles.get(j['name'],0.)) for j in self.joints)
        if self._box_cache is not None and self._box_cache[0]==key:return self._box_cache[1]
        frames=self.frames(angles);result=[]
        for name,(center,axes,extent) in self.boxes_local.items():
            f=frames[name];result.append((f[:3,:3]@center+f[:3,3],f[:3,:3]@axes,extent))
        self._box_cache=(key,result);return result

    def clearances(self,hand,angles,table,other=None):
        """Conservative enclosing boxes replace fictional straight finger tubes."""
        plane,edge,tree=table;vertical=np.inf;nearest=.08
        signs=np.array(list(itertools.product((-1.,1.),repeat=3)))
        # A planar table has an exact minimum on mesh vertices. PCA box corners
        # protrude several millimetres beyond rounded fingertips and incorrectly
        # reject otherwise valid thin-object grasps.
        for vertices in self.vertices(angles).values():
            world=vertices@hand[:3,:3].T+hand[:3,3]
            if np.max(world[:,0])>=edge:
                vertical=min(vertical,float(np.min(world[:,2]-world[:,:2]@plane[:2]-plane[2])))
        for center,axes,extent in self.boxes(angles):
            center=hand[:3,:3]@center+hand[:3,3];axes=hand[:3,:3]@axes
            corners=(signs*extent)@axes.T+center
            for cloud,radius in ((tree,0.),(other,.038)):
                if cloud is None:continue
                ids=cloud.query_ball_point(center,np.linalg.norm(extent)+nearest+radius)
                if not ids:continue
                d=np.abs((cloud.data[ids]-center)@axes)-extent
                signed=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(d.max(1),0)
                nearest=min(nearest,float(signed.min())-radius)
        return vertical,nearest

    def nominal_angles(self, command):
        """Official six-channel order; prospective geometry, not measured contact.

        Mimic joints can deflect in physics. Runtime must still use encoders.
        """
        q={};sign=1 if self.side=='right' else -1
        for i,finger in enumerate(('little','ring','middle','index')):
            first=-1.333*(1-float(command[i]))
            q[f'{self.side}_{finger}_1_joint']=first
            q[f'{self.side}_{finger}_2_joint']=max(-1.527,first*1.1169)
        q[f'{self.side}_thumb_1_joint']=-sign*1.246165*(1-float(command[5]))
        second=sign*.48*(1-float(command[4]))
        q[f'{self.side}_thumb_2_joint']=second
        third=max(-.3578,-sign*second*1.1425)
        q[f'{self.side}_thumb_3_joint']=third
        q[f'{self.side}_thumb_4_joint']=max(-.2775,third*.7508)
        return q

    def closing_sweep(self, command, steps=9):
        """Open, oppose, then flex: conservative sampled nominal sweep."""
        command=np.asarray(command);opposed=np.ones(6);opposed[5]=command[5]
        states=[np.ones(6)+t*(opposed-1) for t in np.linspace(0,1,steps)]
        states += [opposed+t*(command-opposed) for t in np.linspace(0,1,steps)[1:]]
        return [self.nominal_angles(c) for c in states]

    def frames(self,angles):
        frames={'base_link':self.base};pending=list(self.joints)
        while pending:
            progress=False
            for j in pending[:]:
                if j['parent'] not in frames:continue
                q=float(angles.get(j['name'],0.));r=np.eye(4)
                axis=np.eye(3)['XYZ'.index(j['axis'])]
                r[:3,:3]=Rotation.from_rotvec(axis*q).as_matrix()
                frames[j['child']]=frames[j['parent']]@np.asarray(j['t0'])@r@np.asarray(j['inverse_t1'])
                pending.remove(j);progress=True
            if not progress:raise ValueError('Disconnected hand model')
        return frames

    def vertices(self,angles):
        frames=self.frames(angles)
        return {n:s['vertices']@frames[n][:3,:3].T+frames[n][:3,3] for n,s in self.shapes.items()}

    def collision(self,points,angles,margin=CLEARANCE_M):
        """Conservative convex half-space check, including points inside a solid.

        Expanding all face planes by margin overestimates clearance volume at
        edges. Rejection is conservative; this is not exact Euclidean distance.
        """
        frames=self.frames(angles)
        for name,shape in self.shapes.items():
            t=frames[name];p=(points-t[:3,3])@t[:3,:3]
            v=shape['vertices'];near=np.all((p>=v.min(0)-margin)&(p<=v.max(0)+margin),axis=1)
            p=p[near]
            if not len(p):continue
            equations=shape['planes']
            for chunk in np.array_split(p,max(1,int(np.ceil(len(p)/128)))):
                d=chunk@equations[:,:3].T+equations[:,3]
                if np.any(d.max(axis=1)<=margin):return name
        return None

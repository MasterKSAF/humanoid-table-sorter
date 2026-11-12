"""Yaw-frame transforms and swept clearance from observed depth, without scene truth."""
import copy,itertools
import os
import numpy as np
from scipy.spatial import cKDTree
from team22_bringup.grasp import CLEARANCE_M


def rotation(yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def rotated_packet(packet,delta):
    result=copy.deepcopy(packet);r=rotation(-delta)
    result['obstacles']=(np.asarray(packet['obstacles'])@r.T).tolist()
    plane=np.asarray(packet['scene']['plane']);normal=r@np.r_[plane[:2],0.]
    result['scene']['plane']=[*normal[:2],plane[2]]
    result['scene']['table_geometry']=dict(observed=False,reason='После поворота требуется повторное измерение границы')
    for obj in result['scene']['objects']:
        obj['xyz']=(r@np.asarray(obj['xyz'])).tolist()
        bounds=np.asarray(obj['bounds']);corners=np.array(list(itertools.product(*zip(*bounds))))@r.T
        obj['bounds']=[corners.min(0).tolist(),corners.max(0).tolist()]
        if 'geometry' in obj and 'long_axis_xy' in obj['geometry']:
            obj['geometry']['long_axis_xy']=(r[:2,:2]@obj['geometry']['long_axis_xy']).tolist()
    result['angles']['body_yaw_joint']=packet['angles'].get('body_yaw_joint',0.)+delta
    return result


def swept_clearance(arms,angles,points,plane,delta,margin=CLEARANCE_M):
    """Keep arms fixed relative to the torso while checking every degree of yaw."""
    plane=np.asarray(plane);tree=cKDTree(points);minimum=np.inf
    parts=[m.samples([angles[k] for k in m.names]) for m in arms.values()]
    for yaw in np.linspace(0,delta,max(2,int(abs(delta)/.017)+2)):
        r=rotation(yaw)
        for samples,radii in parts:
            p=samples@r.T
            table_gap=np.min(p[:,2]-p[:,:2]@plane[:2]-plane[2]-radii)
            obstacle_gap=np.min(tree.query(p)[0]-radii)
            minimum=min(minimum,table_gap,obstacle_gap)
            if minimum<margin:return False,float(minimum)
    return True,float(minimum)


class EmptyBodyTurn:
    """Only rotate with empty, open hands held at a verified arm pose."""
    def __init__(self,lab):
        self.lab=lab;self.active=False;self.reference=0.;self.goal=0.
        self.last_delta=0.
        # The waist position command is capped at 0.16 rad/s. Keep its
        # reference ramp below that actuator limit so tracking error does not
        # trip the 0.045 rad protective stop during an otherwise clear turn.
        self.speed=float(np.clip(float(os.environ.get('TEAM22_BODY_SPEED','.12')),.05,.12))

    def start(self,goal,now):
        c=self.lab.controller;q=self.lab.timeline.latest('joints')[2]
        if c.status!='holding' or self.active:raise ValueError('Для поворота нужны неподвижные руки')
        if not all(min(self.lab.timeline.latest('hand_'+s)[2]['position'])>.9 for s in c.models):
            raise ValueError('Поворот разрешён только с раскрытыми пустыми кистями')
        if not -2.90<goal<3.05:raise ValueError('Поворот вне пределов корпуса')
        plane,_,tree=c.table
        requested=float(os.environ.get('TEAM22_BODY_TURN_CLEARANCE_M','.001'))
        margin=float(np.clip(requested,.00025,CLEARANCE_M))
        requested_delta=float(goal-q['body_yaw_joint'])
        chosen_delta=requested_delta
        valid,gap=swept_clearance(c.models,q,tree.data,plane,chosen_delta,margin)
        if not valid:
            # Swept volumes are nested as the requested yaw shrinks toward
            # zero. Find the largest safe partial turn instead of giving up
            # after trimming only 0.1 rad; keep the same clearance threshold.
            direction=np.sign(requested_delta)
            base_valid,base_gap=swept_clearance(c.models,q,tree.data,plane,0.,margin)
            if base_valid:
                low,high=0.,abs(requested_delta)
                chosen_gap=base_gap
                for _ in range(8):
                    mid=(low+high)/2
                    candidate=direction*mid
                    candidate_valid,candidate_gap=swept_clearance(c.models,q,tree.data,plane,candidate,margin)
                    if candidate_valid:
                        low=mid;chosen_gap=candidate_gap
                    else:
                        high=mid
                if low>=.08:
                    chosen_delta=direction*low;gap=chosen_gap;valid=True
                else:
                    gap=base_gap
        if not valid:raise ValueError(f'Поворот корпуса: зазор {gap*1000:.1f} мм меньше {margin*1000:.1f} мм')
        goal=float(q['body_yaw_joint']+chosen_delta)
        self.plane=np.array(plane);self.points=tree.data.copy();self.initial=q['body_yaw_joint']
        self.reference=self.initial;self.goal=goal;self.last=now;self.settled=None;self.active=True
        self.last_delta=0.
        c.body_reference=self.reference;c.body_turn_active=True
        note=f'Поворот пустого корпуса к {np.degrees(goal):.1f}°, запас {gap*1000:.1f} мм'
        if abs(chosen_delta-requested_delta)>.001:
            note+=f' (цель скорректирована с {np.degrees(q["body_yaw_joint"]+requested_delta):.1f}°)'
        c.note(now,note)

    def tick(self,now):
        if not self.active:return
        c=self.lab.controller
        if c.status=='stopped':
            sample=self.lab.timeline.latest('joints')
            if sample:c.body_reference=sample[2].get('body_yaw_joint',c.body_reference)
            c.body_turn_active=False;self.active=False;return
        sample=self.lab.timeline.latest('joints')
        if not sample or not -.03<=now-sample[0]<.18:
            c.stop('Нет свежих углов при повороте корпуса',now);return
        q=sample[2];actual=q['body_yaw_joint'];dt=min(.05,max(0.,now-self.last));self.last=now
        if abs(actual-self.reference)>.045:
            c.stop('Корпус отстаёт от заданного поворота более чем на 0,045 рад',now);return
        self.reference+=float(np.clip(self.goal-self.reference,-self.speed*dt,self.speed*dt))
        c.body_reference=self.reference
        r=rotation(-(actual-self.initial));normal=r@np.r_[self.plane[:2],0.]
        c.table=np.r_[normal[:2],self.plane[2]],-100.,cKDTree(self.points@r.T)
        self.lab.command_body(self.reference)
        if abs(actual-self.goal)<.008:
            if self.settled is None:self.settled=now
            elif now-self.settled>.6:
                self.active=False;c.body_reference=self.goal;c.body_turn_active=False
                delta=float(actual-self.initial);r=rotation(-delta)
                self.last_delta=delta
                self.lab.scan_views=[np.asarray(points)@r.T for points in self.lab.scan_views]
                self.lab.map=c.table;self.lab.map_stamp=None
                self.lab.map_geometry={}
                self.lab.scene_worker.reset();self.lab.pairs.clear()
                c.note(now,'Поворот корпуса подтверждён энкодером; карта переведена в новую рамку')
        else:self.settled=None

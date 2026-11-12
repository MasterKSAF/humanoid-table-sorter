"""RGB-D inventory of pool 2. Measurements use waist_yaw_link, never simulator poses."""
import time
import cv2
import numpy as np
from scipy.spatial import cKDTree

from .grasp import Arm
from .survey import Survey, HEAD, plane_from_torso
from .object_geometry import measure,geometry_match,finish_measurement
from .tray_tracking import tray_support

CATALOG = {
    'mouse': ('Мышь', 'Razer_Taipan_Black_Ambidextrous_Gaming_Mouse'),
    'cable': ('Кабель', 'Ubisoft_RockSmith_Real_Tone_Cable_Xbox_360'),
    'drive': ('Коробка с диском', 'Deskstar_Desk_Top_Hard_Drive_1_TB'),
    'blue_jar': ('Банка Beta Glucan', 'Beta_Glucan'),
    'gold_jar': ('Банка с посыпкой', 'Wilton_Pearlized_Sugar_Sprinkles_525_oz_Gold'),
    'lotion': ('Флакон лосьона', 'Marc_Anthony_True_Professional_Strictly_Curls_Curl_Defining_Lotion'),
    'tray_red': ('Красный лоток', None),
    'tray_blue': ('Синий лоток', None),
    'tray_green': ('Зелёный лоток', None),
    'tube_candidate': ('Лежащая пробирка — кандидат',None),
    'tube_standing': ('Стоящая пробирка — крышка',None),
    'figure_lion': ('Фигурка льва','Schleich_Lion_Action_Figure'),
    'figure_unicorn': ('Фигурка единорога','Schleich_S_Bayala_Unicorn_70432'),
    'figure_eagle': ('Фигурка орла','Schleich_Bald_Eagle'),
    'white_mouse': ('Белая мышь','Razer_Taipan_White_Ambidextrous_Gaming_Mouse'),
    'green_speaker': ('Зелёная колонка','JBL_Charge_Speaker_portable_wireless_wired_Green'),
    'alarm_clock': ('Будильник','Crosley_Alarm_Clock_Vintage_Metal'),
    'gelatin_box': ('Коробка желатина','009_gelatin_box'),
    'package_box': ('Коробка упаковки — вариант не определён',None),
}
GRID = .004
XY0 = np.array([.14, -.95])
SHAPE = (265, 475)


def classify(feature):
    """Scores express rule support, not probabilities or a trained confidence."""
    f=feature
    long,short=f['long'],f['short']
    h=f['height']
    if .045<h<.095 and long>.17 and short>.10:
        color=max(('red','blue','green'),key=lambda k:f[k])
        if f[color]>.08:
            return 'tray_'+color, .85
    if .13<h<.26 and long>.09 and short>.05 and f['pale']>.60:
        return 'drive',.85
    if .082<h<.15 and long<.08 and f['blue']>.08:
        return 'blue_jar',.85
    if .105<h<.16 and long<.075 and f['yellow']>.035:
        return 'gold_jar',.85
    if .15<h<.27 and long<.11 and f['yellow']>.06:
        return 'lotion',.85
    if h<.065 and .075<long<.24 and short>.035:
        if f['core_width']>.035 and short<.085 and f['dark']>.10:
            return 'mouse',.82
        if long>.11 and short>.09 and f['dark']>.08:
            return 'cable',.78
    return 'unknown',0.


def contextual_candidate(f,task):
    """Size/color prototypes from annotated camera images; never position templates."""
    names=task.get('objects') or {};long,short,h=f['long'],f['short'],f['height']
    proposals=[]
    if .055<h<.085 and .075<long<.115 and .025<short<.055 and f['yellow']>.25:proposals.append('figure_lion')
    if .08<h<.125 and .115<long<.17 and .03<short<.07 and f['pale']>.80:proposals.append('figure_unicorn')
    if .07<h<.11 and .045<long<.09 and .04<short<.08 and f['yellow']>.08:proposals.append('figure_eagle')
    if .025<h<.055 and .095<long<.145 and .045<short<.08 and f['pale']>.55:proposals.append('white_mouse')
    if .05<h<.10 and .145<long<.21 and .05<short<.09 and f['green']>.25:proposals.append('green_speaker')
    if .125<h<.18 and .08<long<.12 and .025<short<.07 and f['dark']>.10:proposals.append('alarm_clock')
    if .015<h<.045 and .06<long<.115 and .04<short<.09 and f['red']>.40:proposals.append('gelatin_box')
    proposals=[k for k in proposals if CATALOG[k][1] in names]
    if len(proposals)==1:return proposals[0],.65
    if ((task.get('categories') or {}).get('box') and .125<h<.18 and .065<long<.115 and .025<short<.07
            and f['dark']<.03 and f['yellow']>.12):return 'package_box',.55
    return 'unknown',0.


class TableVision:
    def __init__(self):
        cv2.setNumThreads(1)
        self.rig=Survey()
        self.arms={s:Arm(s) for s in ('left','right')}
        self.tracks={}
        self.last_stamp=None
        self.body_yaw=None
        self.task={}

    def clear(self):
        self.tracks.clear()
        self.last_stamp=None
        self.body_yaw=None

    def points(self, camera, frame, angles, *, mask_arms=True):
        bgr,depth,k=frame
        if depth.dtype!=np.uint16 or depth.shape!=bgr.shape[:2] or not np.all(np.isfinite(k)) or min(k[0],k[4])<=0:
            raise ValueError('Неверная глубина или калибровка камеры.')
        pose=self.rig.body if camera=='torso' else self.rig.head_pose([angles[k] for k in HEAD])
        v,u=np.mgrid[:depth.shape[0]:2,:depth.shape[1]:2]
        d=depth[::2,::2].astype(float)/1000
        rays=np.stack(((u-k[2])*d/k[0],(v-k[5])*d/k[4],d),axis=-1)
        xyz=rays@pose[:3,:3].T+pose[:3,3]
        valid=(d>=.15)&(d<=2.5)&np.isfinite(d)
        p=xyz[valid]
        pixels=np.column_stack((u[valid],v[valid]))
        colors=bgr[::2,::2][valid]
        inside=(p[:,0]>.14)&(p[:,0]<1.20)&(abs(p[:,1])<.95)&(p[:,2]>-.3)&(p[:,2]<.25)
        p,pixels,colors=p[inside],pixels[inside],colors[inside]
        for arm in self.arms.values() if mask_arms else ():
            if all(k in angles for k in arm.names):
                samples,radii=arm.samples([angles[k] for k in arm.names])
                padding=float(radii.max())+.022
                nearby=((p>=samples.min(0)-padding)&(p<=samples.max(0)+padding)).all(1)
                keep=np.ones(len(p),dtype=bool)
                distance,idx=cKDTree(samples).query(p[nearby],distance_upper_bound=padding)
                keep[nearby]=distance>radii[np.minimum(idx,len(radii)-1)]+.022
                p,pixels,colors=p[keep],pixels[keep],colors[keep]
        return dict(points=p,pixels=pixels,colors=colors,pose=pose,k=k,shape=depth.shape)

    def segments(self, p, colors, plane):
        heights=p[:,2]-p[:,:2]@plane[:2]-plane[2]
        mask=(heights>.009)&(heights<.28)
        p,colors,heights=p[mask],colors[mask],heights[mask]
        ij=np.floor((p[:,:2]-XY0)/GRID).astype(int)
        valid=(ij>=0).all(1)&(ij<SHAPE).all(1)
        p,colors,heights,ij=p[valid],colors[valid],heights[valid],ij[valid]
        occupancy=np.zeros(SHAPE,np.uint8)
        occupancy[tuple(ij.T)]=1
        closed=cv2.morphologyEx(occupancy,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        count,labels,stats,_=cv2.connectedComponentsWithStats(closed,8)
        hsv=cv2.cvtColor(colors[None,...],cv2.COLOR_BGR2HSV)[0] if len(colors) else np.empty((0,3))
        output=[]
        for label in range(1,count):
            if stats[label,cv2.CC_STAT_AREA]<24:
                continue
            selection=labels[tuple(ij.T)]==label
            group=p[selection]
            if len(group)<25:
                continue
            high=heights[selection]
            hue,sat,val=hsv[selection].T
            lo,hi=np.quantile(group,[.02,.98],axis=0)
            rect=cv2.minAreaRect(group[:,:2].astype(np.float32))
            long,short=sorted(rect[1],reverse=True)
            left,top,width,height=stats[label,:4]
            region=(labels[top:top+height,left:left+width]==label).astype(np.uint8)
            region=np.pad(region,1)
            distance=cv2.distanceTransform(region,cv2.DIST_L2,5)*GRID
            feature=dict(long=float(long),short=float(short),height=float(np.quantile(high,.98)),
                         core_width=float(distance.max()*2),dark=float(np.mean(val<125)),
                         pale=float(np.mean((sat<45)&(val>155))),
                         blue=float(np.mean((hue>90)&(hue<135)&(sat>40))),
                         red=float(np.mean(((hue<14)|(hue>170))&(sat>10))),
                         green=float(np.mean((hue>45)&(hue<85)&(sat>25))),
                         yellow=float(np.mean((hue>12)&(hue<40)&(sat>45))))
            kind,score=classify(feature)
            if (self.task.get('task')=='tubes' and not kind.startswith('tray_') or
                self.task.get('objects') and kind in CATALOG and CATALOG[kind][1] and
                CATALOG[kind][1] not in self.task['objects']):
                kind,score='unknown',0.
            if kind=='unknown':kind,score=contextual_candidate(feature,self.task)
            center=(lo+hi)/2
            # Mouse cord is not the body. Locate the broad solid part independently of view angle.
            if kind in ('mouse','white_mouse'):
                core=(distance>.012).astype(np.uint8)
                n,cores,areas,_=cv2.connectedComponentsWithStats(core,8)
                if n>1:
                    best=1+int(np.argmax(areas[1:,cv2.CC_STAT_AREA]))
                    core_cells=np.argwhere(cores==best)+[top-1,left-1]
                    center[:2]=np.mean(XY0+(core_cells+.5)*GRID,axis=0)
                    near=np.linalg.norm(group[:,:2]-center[:2],axis=1)<.035
                    if near.any():
                        center[2]=float(np.median(group[near,2]))
            # The geometric center can project onto an occluding forearm or empty space.
            # Anchor image labels on an actually measured surface supporting this class.
            support=np.ones(len(group),dtype=bool)
            if kind in ('mouse','cable'):
                support=val<125
            elif kind in ('lotion','gold_jar'):
                support=(hue>12)&(hue<40)&(sat>45)
            elif kind in ('blue_jar','tray_blue'):
                support=(hue>90)&(hue<135)&(sat>40)
            elif kind=='tray_red':
                support=((hue<14)|(hue>170))&(sat>10)
            elif kind=='tray_green':
                support=(hue>45)&(hue<85)&(sat>25)
            surfaces=group[support] if support.any() else group
            visible=surfaces[np.argmin(np.linalg.norm(surfaces-center,axis=1))]
            geometry_points=group
            if kind in ('mouse','white_mouse'):
                body=group[np.linalg.norm(group[:,:2]-center[:2],axis=1)<.07]
                if len(body)>25:geometry_points=body
            geometry=measure(geometry_points,plane,kind)
            if self.task.get('task')=='tubes' and geometry['shape']=='slender_lying_candidate':
                kind,score='tube_candidate',.55
                geometry['volume_candidate_ml']=50 if feature['height']>.027 else 15
                geometry['volume_verified']=False
            output.append(dict(kind=kind,label=CATALOG.get(kind,('Неизвестный предмет',))[0],
                               score=score,xyz=center.tolist(),bounds=[lo.tolist(),hi.tolist()],
                               visible_xyz=visible.tolist(),feature=feature,points=len(group),
                               geometry=geometry))
        # Thin dark cord beside a broad mouse body belongs to that mouse, not another item.
        attached=[]
        for item in output:
            f=item['feature']
            trays=[t for t in output if t['kind'].startswith('tray_')
                   and abs(t['feature']['height']-f['height'])<.008
                   and f[t['kind'].split('_',1)[1]]>.15
                   and np.linalg.norm(np.maximum(0,np.maximum(np.array(t['bounds'][0])-item['bounds'][1],np.array(item['bounds'][0])-t['bounds'][1])))<.016
                   and np.linalg.norm(np.array(t['xyz'])[:2]-item['xyz'][:2])<.19]
            if item['kind']=='unknown' and len(trays)==1:
                tray=trays[0]
                bounds=np.array([tray['bounds'],item['bounds']])
                tray['bounds']=[bounds[:,0].min(0).tolist(),bounds[:,1].max(0).tolist()]
                tray['xyz']=np.mean(tray['bounds'],axis=0).tolist()
                attached.append(id(item))
                continue
            if item['kind']!='unknown' or f['height']>.035 or f['dark']<.35 or f['short']>.065:
                continue
            mice=[m for m in output if m['kind'] in ('mouse','white_mouse') and np.linalg.norm(np.array(m['xyz'])[:2]-item['xyz'][:2])<.11]
            if len(mice)==1:
                mice[0]['attachment']='Видимый провод мыши'
                attached.append(id(item))
        return [i for i in output if id(i) not in attached]

    def observe(self, frames, angles, stamp, task=None, *, audit_views=True, frame_angles=None):
        begin=time.perf_counter()
        frame_angles=frame_angles or {}
        clouds={camera:self.points(camera,frame,frame_angles.get(camera,angles)) for camera,frame in frames.items()}
        scene=self.observe_clouds(clouds,angles,stamp,task,audit_views=audit_views)
        scene['processing_ms']=1000*(time.perf_counter()-begin)
        return scene

    def observe_clouds(self, clouds, angles, stamp, task=None, *, audit_views=True):
        """Classify depth views already registered in one waist frame."""
        begin=time.perf_counter()
        self.task=task or {}
        yaw=angles.get('body_yaw_joint',0.)
        if self.last_stamp is not None and (stamp<self.last_stamp or abs(yaw-self.body_yaw)>.03):
            self.clear()
        self.last_stamp,self.body_yaw=stamp,yaw
        p=np.concatenate([v['points'] for v in clouds.values()])
        colors=np.concatenate([v['colors'] for v in clouds.values()])
        plane=plane_from_torso(p)
        def visible_items(points,colors):
            items=self.segments(points,colors,plane)
            # A placed object changes the connected component's total height.
            # Keep locating the coloured low rim independently of its contents.
            hsv=cv2.cvtColor(colors[None],cv2.COLOR_BGR2HSV)[0] if len(colors) else np.empty((0,3))
            height=points[:,2]-points[:,:2]@plane[:2]-plane[2]
            rim=(height>.012)&(height<.090)&(hsv[:,1]>70)&(hsv[:,2]>60)
            trays=[o for o in self.segments(points[rim],colors[rim],plane) if o['kind'].startswith('tray_')]
            # Contents can connect separate coloured walls into a tall component
            # which resembles a second white box. Recover its rim from current
            # coloured depth, requiring three rectangle walls, before counting
            # duplicate object classes. Never infer a tray from colour alone.
            replacements=[]
            from .tray_tracking import colored_rim_groups
            for color,p,lo,hi,support in colored_rim_groups(points,hsv,plane):
                kind='tray_'+color;center=(lo+hi)/2
                overlapping=[]
                for item in items:
                    bounds=np.asarray(item['bounds'])
                    overlap=np.maximum(0,np.minimum(bounds[1,:2],hi[:2])-np.maximum(bounds[0,:2],lo[:2]))
                    fraction=np.prod(overlap)/max(1e-9,np.prod(bounds[1,:2]-bounds[0,:2]))
                    if fraction>.60:overlapping.append(item)
                if not overlapping:continue
                item=max(overlapping,key=lambda o:o['points'])
                tray=dict(item,kind=kind,label=CATALOG[kind][0],score=.8,xyz=center.tolist(),
                    bounds=[lo.tolist(),hi.tolist()],visible_xyz=p[np.argmin(np.linalg.norm(p-center,axis=1))].tolist(),
                    geometry=measure(p,plane,kind),points=len(p),current_depth_support=support,
                    identity_source='three_current_coloured_walls',contents_present=any(o['feature']['height']>.095 for o in overlapping))
                trays=[t for t in trays if t['kind']!=kind]+[tray]
                replacements.extend(id(o) for o in overlapping)
            kinds={o['kind'] for o in trays}
            return [o for o in items if o['kind'] not in kinds and id(o) not in replacements]+trays
        current=visible_items(p,colors)
        independent={c:visible_items(v['points'],v['colors']) for c,v in clouds.items()} if audit_views else {}
        for camera,candidates in independent.items():
            view=clouds[camera]
            pose,k=view['pose'],view['k']
            for candidate in candidates:
                optical=(np.array(candidate['xyz'])-pose[:3,3])@pose[:3,:3]
                anchor=(np.array(candidate['visible_xyz'])-pose[:3,3])@pose[:3,:3]
                candidate['pixel']=((anchor[:2]/anchor[2])*[k[0],k[4]]+[k[2],k[5]]).tolist()
                candidate['optical_xyz_m']=optical.tolist()
        for item in current:
            item['seen_by']=[]
            item['camera_observations']={}
            item['camera_disagreement_mm']=None
            for camera,candidates in independent.items():
                same=[c for c in candidates if c['kind']==item['kind'] and np.linalg.norm(np.array(c['xyz'])[:2]-item['xyz'][:2])<.07]
                if same:
                    found=min(same,key=lambda c:np.linalg.norm(np.array(c['xyz'])-item['xyz']))
                    item['seen_by'].append(camera)
                    item['camera_observations'][camera]=found
                    found['identity_source']='independent_classification'
                elif item['kind']!='unknown':
                    found=geometry_match(item,candidates)
                    if found is not None:
                        # Keep its independent class unchanged; association is explicit.
                        found=dict(found,identity_source='unique_3d_overlap')
                        item['seen_by'].append(camera)
                        item['camera_observations'][camera]=found
            if len(item['camera_observations'])==2:
                a,b=item['camera_observations'].values()
                item['camera_disagreement_mm']=float(1000*np.linalg.norm(np.array(a['xyz'])-b['xyz']))
            item['stamp']=stamp
            item['status']='observed'
            item['xyz_mm']=(1000*np.array(item['xyz'])).tolist()
            item['height_mm']=float(1000*(item['xyz'][2]-np.array(item['xyz'][:2])@plane[:2]-plane[2]))
        # A class may remain unknown; never force every visible component into the expected inventory.
        counts={kind:sum(i['kind']==kind for i in current) for kind in CATALOG}
        for index,item in enumerate(current):
            kind=item['kind']
            item['id']=kind if kind!='unknown' and counts[kind]==1 else f'unknown-{index}'
            if kind not in ('unknown','tube_candidate','package_box') and counts[kind]>1:
                item.update(kind='unknown',label='Неоднозначный предмет',score=0.)
                item['geometry']['shape']='unclassified'
            finish_measurement(item)
            if item['kind'] not in ('unknown','tube_candidate','package_box'):
                self.tracks[item['id']]=item
        result=list(current)
        found={i['id'] for i in current}
        for key,item in list(self.tracks.items()):
            age=stamp-item['stamp']
            if age>15.:
                del self.tracks[key]
            elif key not in found:
                support=None;seen=[]
                if item['kind'].startswith('tray_'):
                    selected=[];color=item['kind'].split('_')[1]
                    for camera,v in clouds.items():
                        hsv=cv2.cvtColor(v['colors'][None],cv2.COLOR_BGR2HSV)[0] if len(v['colors']) else np.empty((0,3))
                        hue,sat,val=hsv.T
                        colored=(((hue>90)&(hue<140)) if color=='blue' else
                                 ((hue<14)|(hue>170)) if color=='red' else ((hue>45)&(hue<85)))&(sat>40)&(val>50)
                        selected.append(v['points'][colored])
                        if tray_support(item,selected[-1])['confirmed']:seen.append(camera)
                    support=tray_support(item,np.concatenate(selected))
                if support and support['confirmed']:
                    retained=dict(item,status='geometry_confirmed',seen_by=seen,camera_observations={},
                                  stamp=stamp,current_depth_support=support,
                                  identity_source='previous_color_with_current_colored_wall_depth')
                    self.tracks[key]=retained;result.append(retained)
                else:result.append(dict(item,status='remembered',seen_by=[],camera_observations={}))
        for item in result:
            item['age_sim_s']=round(stamp-item['stamp'],3)
        tubes=None
        if self.task.get('task')=='tubes':
            from .tube_vision import detect
            tubes=detect(clouds,plane,self.task,stamp)
            if self.task.get('variant')!='lying':result.extend(tubes['caps'])
        return dict(stamp=stamp,frame='waist_yaw_link',objects=result,plane=plane.tolist(),
                    tubes=tubes,expected_count=len(self.task.get('tubes') or {}) if tubes else None,
                    independent=independent,processing_ms=1000*(time.perf_counter()-begin),
                    position_meaning='Центр видимого объёма; для мыши — центр корпуса. Это не точка захвата.',
                    expected=([k for k,(_,name) in CATALOG.items() if name in (self.task.get('objects') or {})]
                              if self.task.get('objects') else ['tube_candidate' if self.task.get('variant')=='lying' else 'tube_standing'] if self.task.get('task')=='tubes' else list(CATALOG)[:6]),grid_resolution_mm=GRID*1000,
                    catalog={kind:label for kind,(label,_) in CATALOG.items()})


def overlay(image,scene,camera,pose,k):
    """Only mark independently supported camera detections, not an invisible projection."""
    canvas=image.copy()
    for item in scene['objects']:
        found=item.get('camera_observations',{}).get(camera)
        if not found:
            continue
        u,v=np.round(found['pixel']).astype(int)
        if not (0<=u<image.shape[1] and 0<=v<image.shape[0]):
            continue
        color=(90,230,90)
        cv2.drawMarker(canvas,(u,v),color,cv2.MARKER_CROSS,16,2)
        label=item['kind']
        cv2.putText(canvas,label,(max(0,u-20),max(15,v-12)),cv2.FONT_HERSHEY_SIMPLEX,.42,(10,10,10),3,cv2.LINE_AA)
        cv2.putText(canvas,label,(max(0,u-20),max(15,v-12)),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1,cv2.LINE_AA)
    return canvas

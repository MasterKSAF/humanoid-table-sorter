"""Separate coloured tube caps from racks using RGB-D; no hidden scene positions."""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def observations(view,plane,allowed_colors=None):
    p=view['points'];pixels=view['pixels'];colors=view['colors']
    if not len(p):return []
    hsv=cv2.cvtColor(colors[None],cv2.COLOR_BGR2HSV)[0]
    h=p[:,2]-p[:,:2]@plane[:2]-plane[2]
    detections=[]
    for color,hue in (('blue',(hsv[:,0]>=85)&(hsv[:,0]<=140)),
                      ('red',(hsv[:,0]<12)|(hsv[:,0]>168))):
        if allowed_colors and color not in allowed_colors:continue
        selected=hue&(hsv[:,1]>55)&(hsv[:,2]>70)&(h>.035)&(h<.20)
        indices=np.flatnonzero(selected)
        if not len(indices):continue
        # Two caps overlap in the torso image. Separate their actual 3-D
        # surfaces instead of merging neighbouring blue image pixels.
        pairs=cKDTree(p[indices]).query_pairs(.007,output_type='ndarray')
        graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(indices),len(indices)))
        count,labels=connected_components(graph,directed=False)
        for label in range(count):
            take=indices[labels==label]
            group=p[take]
            if len(group)<4:continue
            lo,hi=np.percentile(group,[2,98],axis=0);extent=hi-lo
            width=max(extent[:2])
            if not .008<width<.042 or extent[2]>.026:continue
            center=(lo+hi)/2
            detections.append(dict(color=color,xyz=center.tolist(),bounds=[lo.tolist(),hi.tolist()],
                pixel=np.mean(pixels[take],axis=0).tolist(),points=len(group),diameter_visible_mm=width*1000,
                geometry=dict(dimensions_mm=(extent*1000).round(1).tolist()),identity_source='cap_color_and_depth'))
    return detections


def detect(clouds,plane,task,stamp):
    allowed={v['cap'] for v in (task.get('tubes') or {}).values() if v.get('cap')}
    per_camera={camera:observations(view,np.asarray(plane),allowed) for camera,view in clouds.items()}
    fused=[]
    for camera,items in per_camera.items():
        for item in items:
            matches=[f for f in fused if f['color']==item['color'] and camera not in f['camera_observations']
                     and np.linalg.norm(np.asarray(f['xyz'])-item['xyz'])<.020]
            if len(matches)==1:
                found=matches[0];found['camera_observations'][camera]=item
                found['seen_by'].append(camera)
                found['xyz']=np.mean([o['xyz'] for o in found['camera_observations'].values()],axis=0).tolist()
            else:fused.append(dict(item,camera_observations={camera:item},seen_by=[camera]))
    types={x['type'] for x in (task.get('tubes') or {}).values()}
    for i,item in enumerate(fused):
        views=item['camera_observations'];xyz=np.asarray(item['xyz'])
        bounds=np.asarray([v['bounds'] for v in views.values()])
        item['bounds']=[bounds[:,0].min(axis=0).tolist(),bounds[:,1].max(axis=0).tolist()]
        diameter=max(v['diameter_visible_mm'] for v in views.values())
        tube_type=next(iter(types)) if len(types)==1 else ('falcon50' if diameter>23 else 'falcon15' if diameter<20 else None)
        dimensions=[diameter,diameter,float(np.ptp(np.asarray(item['bounds']),axis=0)[2]*1000)]
        disagreement=float(np.linalg.norm(np.diff(np.array([v['xyz'] for v in views.values()]),axis=0))*1000) if len(views)==2 else None
        item.update(id=f'tube-cap-{i}',kind='tube_standing',label=f'Пробирка: {item["color"]}, {tube_type or "размер неясен"}',
            status='observed',stamp=stamp,age_sim_s=0.,xyz_mm=(xyz*1000).tolist(),score=.65,
            height_mm=float((xyz[2]-xyz[:2]@np.asarray(plane[:2])-plane[2])*1000),
            camera_disagreement_mm=disagreement,tube_type=tube_type,
            geometry=dict(shape='visible_cap',dimensions_mm=dimensions,complete_shape=False,
                per_camera_mm={k:v['geometry']['dimensions_mm'] for k,v in views.items()},
                view_dimension_spread_mm=None,cross_view_status='two_visible_surfaces' if len(views)==2 else 'single_or_partial_view'),
            position_meaning='Центр видимой крышки, не центр пробирки и не готовая точка захвата',
            grasp_candidates=[dict(method='cap_side_pinch',hands=1,validated=False,
                reason='Боковой захват под крышкой большим и указательным пальцами',
                checks=['Измерить зазор до штатива','Проверить длину выступающей части','Найти пустое целевое гнездо'])])
    slots={camera:empty_slots(view,np.asarray(plane),per_camera[camera]) for camera,view in clouds.items()}
    return dict(caps=fused,independent=per_camera,slots_by_camera=slots,expected_count=len(task.get('tubes') or {}),
                slots_verified=False,rule=task.get('rule'),warning='Крышки и гнёзда-кандидаты: геометрия установки и безопасный перенос ещё не подтверждены')


def empty_slots(view,plane,caps):
    """Square dark openings inside an observed raised white rack, checked with depth."""
    p,pixels,colors=view['points'],view['pixels'],view['colors']
    if not len(p):return []
    hsv=cv2.cvtColor(colors[None],cv2.COLOR_BGR2HSV)[0]
    height=p[:,2]-p[:,:2]@plane[:2]-plane[2]
    white=(hsv[:,1]<50)&(hsv[:,2]>190)&(height>.020)&(height<.090)
    uv=(pixels//2).astype(int);shape=((view['shape'][0]+1)//2,(view['shape'][1]+1)//2)
    mask=np.zeros(shape,np.uint8);mask[uv[white,1],uv[white,0]]=1
    closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(closed,8);slots=[]
    for label in range(1,count):
        take=white&(labels[uv[:,1],uv[:,0]]==label)
        if take.sum()<25:continue
        group=p[take];extent=np.ptp(group,axis=0)
        if not .06<max(extent[:2])<.45:continue
        hull=cv2.convexHull(uv[take].astype(np.int32));inside=np.zeros(shape,np.uint8)
        cv2.fillConvexPoly(inside,hull,1);inside=cv2.erode(inside,np.ones((3,3),np.uint8))
        threshold=float(np.percentile(hsv[take,2],70))-12
        dark=(hsv[:,2]<threshold)&(hsv[:,1]<55)&(inside[uv[:,1],uv[:,0]]>0)
        holes=np.zeros(shape,np.uint8);holes[uv[dark,1],uv[dark,0]]=1
        n,lab,st,cent=cv2.connectedComponentsWithStats(holes,8)
        z=float(np.percentile(group[:,2],88));pose,k=view['pose'],view['k']
        for j in range(1,n):
            if st[j,cv2.CC_STAT_AREA]<4:continue
            selected=dark&(lab[uv[:,1],uv[:,0]]==j)
            contour=uv[selected].astype(float)*2
            if any(np.linalg.norm(np.asarray(c['pixel'])-cent[j]*2)<12 for c in caps):continue
            rays=np.column_stack(((contour[:,0]-k[2])/k[0],(contour[:,1]-k[5])/k[4],np.ones(len(contour))))@pose[:3,:3].T
            valid=abs(rays[:,2])>.05
            if not valid.all():continue
            points=pose[:3,3]+rays*((z-pose[2,3])/rays[:,2])[:,None]
            rectangle=cv2.minAreaRect(points[:,:2].astype(np.float32));dimensions=np.sort(rectangle[1])*1000
            size=('falcon50' if np.all(abs(dimensions-36)<7) else
                  'falcon15' if np.all(abs(dimensions-22.5)<4.5) else None)
            if size is None:continue
            recess=float(z-np.median(p[selected,2]))
            if recess<.008:continue
            slots.append(dict(xyz=[*map(float,rectangle[0]),z],pixel=(cent[j]*2).tolist(),
                visible_aperture_mm=dimensions.tolist(),recess_mm=recess*1000,
                suggested_type=size,verified=False))
    return slots

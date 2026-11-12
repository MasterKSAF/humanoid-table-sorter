"""Visible-surface measurements and grasp hypotheses, not hidden object geometry."""
import cv2
import numpy as np
from .grasp_priors import contact_strategy


def measure(points, plane, kind):
    p=np.asarray(points,float);plane=np.asarray(plane,float)
    rect=cv2.minAreaRect(p[:,:2].astype(np.float32))
    corners=cv2.boxPoints(rect).astype(float)
    edges=np.roll(corners,-1,axis=0)-corners
    lengths=np.linalg.norm(edges,axis=1);axis=edges[int(np.argmax(lengths))]
    axis/=max(np.linalg.norm(axis),1e-9)
    if axis[0]<0:axis=-axis
    across=np.array([-axis[1],axis[0]])
    projection=p[:,:2]@np.column_stack((axis,across))
    lo,hi=np.quantile(projection,[.01,.99],axis=0)
    dims=hi-lo
    height=p[:,2]-p[:,:2]@plane[:2]-plane[2]
    zlo,zhi=np.quantile(height,[.01,.99])
    shape=('container' if kind.startswith('tray_') else
           'upright_cylinder' if kind in ('gold_jar','blue_jar') else
           'upright_bottle' if kind=='lotion' else
           'box' if kind in ('drive','package_box','gelatin_box') else 'cable_coil' if kind=='cable' else
           'mouse_body' if kind in ('mouse','white_mouse') else 'unclassified')
    if shape=='unclassified' and dims[0]>.07 and dims[1]<.04 and zhi<.055:
        shape='slender_lying_candidate'
    return dict(shape=shape,dimensions_mm=(1000*np.r_[dims,zhi]).round(1).tolist(),
        visible_vertical_span_mm=round(1000*(zhi-zlo),1),long_axis_xy=axis.round(5).tolist(),
        corners_xy_m=corners.tolist(),points=len(p),
        meaning='Размер видимого объёма; высота от измеренной плоскости стола. Скрытая часть не восстановлена.',
        minimum_margin_mm=6.,complete_shape=False)


def grasps(item):
    g=item['geometry'];long,short,height=g['dimensions_mm'];kind=item['kind']
    if kind.startswith('tray_'):return []
    choices=[]
    def add(mode,hands,why,checks):
        choices.append(dict(mode=mode,hands=hands,reason=why,checks=checks,validated=False))
    if kind=='cable':
        add('supported_hook_pinch',2,'Гибкий моток: поддержка с одной стороны и подхват петли с другой',
            ['Выделить отдельную петлю и просвет снизу','Проверить палец под петлёй по камере','Не натянуть соседний провод'])
    elif kind in ('gold_jar','blue_jar','lotion'):
        add('side_wrap',1,'Стоящий сосуд: ладонь сбоку, большой палец напротив основных',
            ['Проверить диаметр и раскрытие','Учесть нижние пальцы и стол','Контакт двух сторон и подъём по камере'])
    elif kind in ('mouse','white_mouse'):
        add('body_pinch',1,'Брать широкий корпус; провод не является точкой захвата',
            ['Отделить провод от корпуса','Проверить просвет и положение большого пальца'])
    elif kind.startswith('figure_'):
        add('body_support',1,'Фигурка: поддержать массивную часть корпуса',
            ['Отделить корпус от выступающих лап, рога или крыльев','Проверить форму с другого ракурса'])
    elif kind in ('drive','green_speaker') or (height>75 and (long>150 or short>85)):
        add('two_palm_support',2,'Крупный объём: противоположные ладони с двух сторон',
            ['Обе стороны видны','Проверка взаимных столкновений рук','Согласованное движение и усилия обеих кистей','Предмет следует за обеими руками'])
        add('side_pinch',1,'Альтернатива, если измеренная узкая сторона помещается между пальцами',
            ['Проверить реальное раскрытие и центр масс','Проверить удержание при подъёме'])
    elif g['shape']=='slender_lying_candidate':
        add('precision_pinch',1,'Тонкий лежащий объект: захват поперёк оси',
            ['Подтвердить класс предмета','Отделить объект от соседей','Проверить просвет до стола'])
    elif g['shape']=='box':
        add('narrow_side_pinch',1,'Коробка: противоположные контакты поперёк узкой стороны',
            ['Измерить узкое сечение с двух камер','Проверить раскрытие, просвет и центр удержания'])
    else:
        add('inspect_more',0,'Формы недостаточно для выбора захвата',
            ['Другой ракурс головы','Сопоставить с камерой торса','Разделить слипшиеся объекты'])
    example = contact_strategy(item)
    if example:
        for choice in choices:
            if choice['hands'] == example['hands']:
                choice['human_examples'] = example['human_examples']
                choice['checks'] = list(dict.fromkeys(choice['checks']+example['checks']))
    return choices


def geometry_match(item,candidates):
    """Associate a partly unclassified view only with unique overlapping 3D support."""
    lo,hi=np.asarray(item['bounds']);matches=[]
    for c in candidates:
        if c['kind'] not in (item['kind'],'unknown'):continue
        clo,chi=np.asarray(c['bounds'])
        overlap=np.maximum(0,np.minimum(hi,chi)-np.maximum(lo,clo))
        fraction=np.prod(overlap)/max(1e-9,min(np.prod(hi-lo),np.prod(chi-clo)))
        distance=np.linalg.norm(np.asarray(c['xyz'])-item['xyz'])
        if fraction>.45 and distance<.065:matches.append((distance,c))
    matches.sort(key=lambda x:x[0])
    if not matches or (len(matches)>1 and matches[1][0]-matches[0][0]<.025):return None
    return matches[0][1]


def finish_measurement(item):
    g=item['geometry'];views=item.get('camera_observations',{})
    g['per_camera_mm']={camera:c['geometry']['dimensions_mm'] for camera,c in views.items()}
    if len(views)==2:
        dimensions=np.array(list(g['per_camera_mm'].values()))
        g['view_dimension_spread_mm']=np.ptp(dimensions,axis=0).round(1).tolist()
    else:g['view_dimension_spread_mm']=None
    g['cross_view_status']='two_visible_surfaces' if len(views)==2 else 'single_or_partial_view'
    item['grasp_candidates']=grasps(item)

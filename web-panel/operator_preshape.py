"""Operator-demonstrated preparation; never evidence of a successful grasp."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from team22_bringup.hand_geometry import HandGeometry


@lru_cache(maxsize=1)
def cylinder_ready():
    import team22_bringup.hand_geometry as geometry
    path = Path(geometry.__file__).with_name('operator_cylinder_ready.json')
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    for key in ('command', 'close_command'):
        values = np.asarray(data[key], dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError('Invalid operator cylinder hand command')
    if data['side'] != 'right':
        raise ValueError('Only the demonstrated right hand is calibrated')
    return data


def palm_entry(current, model, angles, observation):
    """Measured FK plus fresh tracked object centre; no simulator ground truth."""
    hand = model.wrist([angles[k] for k in model.names]) @ model.hand
    geometry=HandGeometry(model.side)
    vertices = geometry.vertices(angles)
    thumb_links={joint['child'] for joint in geometry.joints if 'thumb' in joint['name']}
    thumb = np.concatenate([v for name, v in vertices.items() if name in thumb_links])
    thumb_world = thumb @ hand[:3, :3].T + hand[:3, 3]
    original = current['original']
    center = np.asarray(observation['xyz'])
    bounds = np.asarray(original['bounds']) + center - np.asarray(original['xyz'])
    pocket = hand[:3, 3] + hand[:3, :3] @ model.pocket
    target = np.asarray(current['target']) + center - np.asarray(original['xyz'])
    error = float(np.linalg.norm(pocket - target))
    above = float(bounds[1, 2] - thumb_world[:, 2].max())
    inside = bool(np.all(pocket[:2] >= bounds[0, :2]) and np.all(pocket[:2] <= bounds[1, :2]))
    detail = dict(pocket_error_mm=1000*error, top_above_thumb_mm=1000*above,
                  pocket_inside_object_xy=inside, thumb_top_m=float(thumb_world[:, 2].max()))
    return bool(error <= .002 and above >= .005 and inside), detail


def table_entry(model, angles, plane, edge, hand_position):
    """Measured whole-hand staging gate; millimetres are model estimates.

    The staging hand is behind the observed near edge. Subsequent insertion
    must follow the measured table plane and check the swept hand separately.
    This gate alone is not a collision-checked approach or a successful grasp.
    """
    plane=np.asarray(plane,dtype=float);position=np.asarray(hand_position,dtype=float)
    if plane.shape!=(3,) or position.shape!=(6,) or not np.all(np.isfinite(np.r_[plane,edge,position])):
        raise ValueError('Missing finite table or finger measurements')
    geometry=HandGeometry(model.side)
    required={j['name'] for j in geometry.joints}|set(model.names)
    if not required.issubset(angles) or not all(np.isfinite(angles[k]) for k in required):
        raise ValueError('Missing measured arm or finger angles')
    hand=model.wrist([angles[k] for k in model.names])@model.hand
    meshes={name:v@hand[:3,:3].T+hand[:3,3] for name,v in geometry.vertices(angles).items()}
    gaps={name:float(np.min(v[:,2]-v[:,:2]@plane[:2]-plane[2])) for name,v in meshes.items()}
    lowest=min(gaps,key=gaps.get);gap=gaps[lowest]
    front=max(float(v[:,0].max()) for v in meshes.values());edge_gap=float(edge-front)
    ready=cylinder_ready();opened=bool(np.min(position[:4])>=.90)
    thumb=bool(np.max(abs(position[4:]-np.asarray(ready['command'][4:])))<=.06)
    reasons=[]
    if edge_gap<.001:reasons.append('Пальцы ещё заходят за ближний край стола')
    if gap<.001:reasons.append('Кисть ниже требуемой высоты')
    if gap>.008:reasons.append('Кисть ещё слишком высоко над столом')
    if not opened:reasons.append('Четыре пальца раскрыты недостаточно')
    if not thumb:reasons.append('Большой палец не в сохранённой готовности')
    return dict(ready=not reasons,reason='; '.join(reasons),lowest_link=lowest,
                table_gap_mm=gap*1000,behind_edge_mm=edge_gap*1000,
                four_fingers_open=opened,thumb_ready=thumb,frame='waist_yaw_link',
                estimate_source='measured_joint_fk_and_camera_table_plane')


@lru_cache(maxsize=1)
def last_trial():
    path=Path(__file__).with_name('operator-last-result.json')
    if not path.exists():return None
    return json.loads(path.read_text(encoding='utf-8'))

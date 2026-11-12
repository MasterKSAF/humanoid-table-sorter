"""Contact strategies from human examples; no human angles or forces are copied."""
import math
import json
from pathlib import Path
import numpy as np


def failed_pinch_prior(item,side,candidate):
    path=Path(__file__).with_name('grasp_failures.json')
    if not path.exists():return None
    measured=np.asarray(item.get('geometry',{}).get('dimensions_mm',[]))
    for old in json.loads(path.read_text(encoding='utf-8')):
        if old['kind']!=item['kind'] or old['side']!=side:continue
        dims=np.asarray(old.get('observed_dimensions_mm') or [])
        if measured.shape!=(3,) or dims.shape!=(3,) or np.any(abs(measured-dims)>np.maximum(10,dims*.25)):continue
        yaw=candidate['pinch_yaws'][0]-old['yaw']
        if (abs(math.atan2(math.sin(yaw),math.cos(yaw)))<.08 and
            abs(candidate['finger_down_tilt']-old['finger_down_tilt'])<.05 and
            np.max(abs(np.array(candidate['pocket'])-old['pocket']))<.006):return old['case']
    return None


def verified_prior(item, side):
    """A measured success ranks candidates; it never bypasses current checks."""
    path=Path(__file__).with_name('grasp_experience.json')
    if not path.exists():return None
    prior=json.loads(path.read_text(encoding='utf-8')).get(item['kind'])
    if not prior or prior['side']!=side:return None
    measured=np.asarray(item.get('geometry',{}).get('dimensions_mm',[]),float)
    reference=np.asarray(prior.get('observed_dimensions_mm',[]),float)
    if measured.shape!=(3,) or reference.shape!=(3,) or np.any(reference<=0):return None
    if np.any(abs(measured-reference)>np.maximum(10.,reference*.25)):return None
    return prior


def narrow_side_yaws(item, side):
    """For a palm-down pinch, align the finger axis along the object long axis.

    Opposed contacts then span its narrower dimension. Sizes are visible extents,
    not guaranteed object bounds; collision/IK and measured aperture still decide.
    """
    geometry = item.get('geometry', {})
    axis = np.asarray(geometry.get('long_axis_xy', []), float)
    dimensions = np.asarray(geometry.get('dimensions_mm', []), float)
    generic = [0., -.6, .6, -1.2, 1.2, -1.8, 1.8, 2.4, -2.4, math.pi]
    if (axis.shape != (2,) or dimensions.shape != (3,) or not np.isfinite(axis).all()
            or not np.isfinite(dimensions).all() or np.linalg.norm(axis) < .5
            or dimensions[1] <= 0 or dimensions[0]/dimensions[1] < 1.25):
        return generic
    sign = 1 if side == 'left' else -1
    base = math.atan2(sign*axis[1], axis[0])
    # Prefer contact across the short side; each is only a hypothesis.
    preferred = [base, base+math.pi, base-.20, base+.20, base+math.pi-.20, base+math.pi+.20]
    result = []
    for angle in preferred+generic:
        angle = (angle+math.pi) % (2*math.pi)-math.pi
        if all(abs(math.atan2(math.sin(angle-v), math.cos(angle-v))) > .03 for v in result):
            result.append(angle)
    return result


def contact_strategy(item):
    kind = item['kind']
    g = item.get('geometry', {})
    common = ['Проверить реальный просвет кисти с запасом на ошибку зрения',
              'Проверить весь подвод раскрытой кисти и движение пальцев',
              'Подтвердить противоположные контакты, затем пробный подъём',
              'Подтвердить перемещение предмета вместе с кистью по камере']
    if kind in ('gold_jar', 'blue_jar', 'lotion'):
        return dict(mode='side_wrap', hands=1, reason='Обхват корпуса: большой палец напротив согнутых пальцев',
                    checks=common+['Крышка не является точкой захвата без отдельной проверки её крепления'],
                    validated=False, human_examples=[1, 2, 3, 4])
    if g.get('shape') in ('box', 'mouse_body', 'slender_lying_candidate'):
        return dict(mode='narrow_side_pinch', hands=1, reason='Пальцы по разные стороны узкого сечения, кисть ориентирована вдоль предмета',
                    checks=common+['Удерживать ближе к центру массы; по фото масса неизвестна',
                                   'При недостаточном раскрытии перейти к двум рукам'],
                    validated=False, human_examples=list(range(6, 14)))
    if kind == 'cable':
        return dict(mode='local_loop_pinch', hands=1, reason='Кандидат: захват доступного участка петли большим и указательным пальцами',
                    checks=common+['Нужны различимая петля и просвет; фото браслета не доказывает подхват кабеля',
                                   'Если пальцу не подойти снизу, нужна поддержка второй рукой'],
                    validated=False, human_examples=[5], executable=False)
    return None

"""Operator working pose at the near edge of the table.

Angles are in radians, in the order used by Arm.names. Finger positions are
public hand command ratios. The targets are IK offsets from the measured
2026-11-08 working pose: each tool is 20 mm higher, the elbows are about
13.6 mm farther back, and hand orientation changes by less than 0.05 degrees.
Every arm route still needs a current scene map and clearance check.
"""
import numpy as np


POSE_ID = 'working_start_both_raised20_elbows_back_20261109'
ANGLES = {
    'left': (
        0.9350571468139854, 0.05153123093365478, 0.07071255026995746,
        -1.5085906773197426, -0.02085480874259896, -0.6816781818502817,
        0.18558901998676122,
    ),
    'right': (
        0.9350308937392112, -0.05154555992145724, -0.07073812721977793,
        -1.508623808919579, 0.02088363363162284, -0.6816648068444049,
        -0.18554581621694063,
    ),
}
HAND_POSITIONS = {
    'left': (
        0.9914684256752541, 0.9901671311905397, 0.9898633837912285,
        0.9900472351117234, 0.5821432831386725, 0.05925578741741766,
    ),
    'right': (
        0.9911812751009691, 0.9899540965662625, 0.9896213007058224,
        0.9902503556569805, 0.585368266950051, 0.059232541814017825,
    ),
}


def ready_angles(side):
    return np.array(ANGLES[side])


def working_hand_command(side):
    return HAND_POSITIONS[side]


def preparation_routes(side,current,repeat=False):
    """Direct candidate only; the caller checks it or searches the scene map.

    Rechecking the pose never makes an artificial outward-and-back excursion.
    ``repeat`` is retained for the existing planner/controller interface.
    """
    label = 'Вернуть руку в рабочую позу' if repeat else 'Установить руку в рабочую позу'
    return [[(label, ready_angles(side))]]

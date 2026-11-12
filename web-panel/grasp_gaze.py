"""Aim the calibrated head optical axis at a measured waist-frame target."""
import numpy as np
from scipy.optimize import least_squares
from team22_bringup.survey import Survey


def gaze_angles(target, current, rig=None):
    rig=rig or Survey();target=np.asarray(target,float)
    def residual(a):
        pose=rig.head_pose([a[0],a[1],0.])
        optical=pose[:3,:3].T@(target-pose[:3,3])
        return optical[:2]/max(.1,optical[2])
    # Stay inside the official head limits (pitch +/-25 degrees).
    low=np.array([-1.35,-.40]);high=np.array([1.35,.42])
    fit=least_squares(residual,np.clip(current,low+1e-6,high-1e-6),
                      bounds=(low,high),max_nfev=15)
    return fit.x

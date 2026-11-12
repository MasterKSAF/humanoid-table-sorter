"""Local RGB-D evidence for the gold jar, including while an arm occludes it."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from team22_bringup.survey import Survey,HEAD


class JarTracker:
    def __init__(self,original,plane):
        self.original=np.array(original['xyz']);self.plane=np.array(plane)
        self.kind=original.get('kind','gold_jar')
        bounds=np.asarray(original.get('bounds',[self.original-[.022,.022,.06],self.original+[.022,.022,.06]]))
        self.radius=float(np.clip(np.max(np.ptp(bounds,axis=0)[:2])/2,.018,.045))
        self.rig=Survey();self.baseline=None;self.latest=None

    def observe(self,frames,angles,stamp,expected):
        clouds=[];views=[];expected=np.array(expected)
        for camera,(bgr,depth,k) in frames.items():
            pose=self.rig.body if camera=='torso' else self.rig.head_pose([angles[n] for n in HEAD])
            hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
            color=((hsv[:,:,0]>=90)&(hsv[:,:,0]<=140) if self.kind=='blue_jar' else
                   (hsv[:,:,0]>=8)&(hsv[:,:,0]<=43))
            mask=color&(hsv[:,:,1]>50)&(hsv[:,:,2]>45)&(depth>=150)&(depth<2500)
            v,u=np.nonzero(mask);d=depth[v,u]/1000.
            p=np.column_stack(((u-k[2])*d/k[0],(v-k[5])*d/k[4],d))@pose[:3,:3].T+pose[:3,3]
            keep=(np.linalg.norm(p[:,:2]-expected[:2],axis=1)<.065)&(abs(p[:,2]-expected[2])<.11)
            keep &= p[:,2]-p[:,:2]@self.plane[:2]-self.plane[2]>.014
            p=p[keep]
            if len(p)>25:clouds.append(p);views.append(camera)
        if not clouds:
            self.latest=dict(stamp=stamp,valid=False,reason='Банка закрыта рукой или не видна');return self.latest
        p=np.concatenate(clouds);lo,hi=np.percentile(p,[5,95],axis=0)
        _,_,v=np.linalg.svd(p-p.mean(0),full_matrices=False)
        tilt=float(np.degrees(np.arccos(np.clip(abs(v[0,2]),0,1))))
        def residual(x):return np.linalg.norm(p[:,:2]-x[:2],axis=1)-x[2]
        fit=least_squares(residual,[*expected[:2],self.radius],bounds=([*(expected[:2]-.04),.014],[*(expected[:2]+.04),max(.035,self.radius+.012)]),loss='soft_l1',f_scale=.003,max_nfev=35)
        xyz=np.r_[fit.x[:2],(lo[2]+hi[2])/2]
        # Blue identifies a short printed band, not the full cylinder height.
        # Track both edges of that same band for subsequent lift confirmation.
        minimum_span=.025 if self.kind=='blue_jar' else .045
        valid=len(p)>60 and hi[2]-lo[2]>minimum_span and tilt<35 and np.median(abs(residual(fit.x)))<.009
        partial=(len(p)>70 and hi[2]-lo[2]>.007 and np.median(abs(residual(fit.x)))<.006
                 and np.linalg.norm(fit.x[:2]-expected[:2])<.03)
        surface_support=None
        if self.baseline is not None and not valid:
            # A short exposed arc cannot determine a free cylinder centre and
            # radius: the fit can jump to the wrong side of that arc. Use the
            # already measured radius and encoder-predicted centre to check
            # real RGB-D surface support. This never confirms the initial lift.
            radius=self.baseline['radius_m']
            radial=np.abs(np.linalg.norm(p[:,:2]-expected[:2],axis=1)-radius)
            supported=radial<.009
            surface_support=dict(points=int(supported.sum()),fraction=float(supported.mean()),
                median_radial_error_mm=float(np.median(radial)*1000),expected=expected.tolist())
            partial=partial or bool(supported.sum()>90 and supported.mean()>.65 and hi[2]-lo[2]>.007)
        result=dict(stamp=stamp,valid=bool(valid),xyz=xyz.tolist(),bounds=[lo.tolist(),hi.tolist()],
                    partial_visible=bool(partial),
                    tilt_deg=tilt,points=len(p),seen_by=views,radius_m=float(fit.x[2]),
                    fit_residual_mm=float(np.median(abs(residual(fit.x)))*1000))
        if surface_support is not None:result['predicted_surface_support']=surface_support
        if self.baseline is None and valid:self.baseline=result
        if self.baseline:
            result['delta']=(xyz-np.array(self.baseline['xyz'])).tolist()
            result['vertical_edges_delta']=(np.array([lo[2],hi[2]])-np.array(self.baseline['bounds'])[:,2]).tolist()
        self.latest=result
        return result


def lift_visible(observation,now,allow_partial=False):
    if not observation or not 0<=now-observation['stamp']<.4:return False
    d=np.array(observation.get('delta',[0.,0.,0.]));edges=np.array(observation.get('vertical_edges_delta',[0.,0.]))
    if observation.get('valid') and (.022<d[2]<.075 and np.linalg.norm(d[:2])<.04
            and np.all(edges>.018) and np.ptp(edges)<.025 and observation['tilt_deg']<25):
        return True
    if allow_partial and observation.get('partial_visible'):
        return bool(.022<d[2]<.075 and np.linalg.norm(d[:2])<.04
                    and observation.get('points',0)>150
                    and observation.get('prediction_error_mm',1000.)<=30.)
    return False

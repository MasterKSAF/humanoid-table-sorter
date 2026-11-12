"""RGB-D surface evidence for a dark object during a short diagnostic lift.

The encoder prediction locates the search window; only measured non-robot
points can establish motion. This tracker does not infer grasp from arm motion.
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from team22_bringup.grasp import Arm
from team22_bringup.hand_geometry import HandGeometry
from team22_bringup.robot_surface import RobotSurfaceMask
from team22_bringup.survey import Survey, HEAD


class ObjectTracker:
    def __init__(self, original, plane):
        self.original = np.array(original['xyz'])
        self.kind = original['kind']
        self.bounds = np.asarray(original['bounds'])
        self.plane = np.asarray(plane)
        self.rig = Survey()
        self.robot = RobotSurfaceMask()
        self.arms = {s: Arm(s) for s in ('left', 'right')}
        self.hands = {s: HandGeometry(s) for s in self.arms}
        self.baseline = self.latest = self.reference = None
        self.carrying=False
        self.expected_rotation=np.eye(3)
        self.cap_color=original.get('color')

    def remove_hands(self, points, angles):
        keep = np.ones(len(points), dtype=bool)
        for side, arm in self.arms.items():
            hand = arm.wrist([angles[n] for n in arm.names]) @ arm.hand
            local = (points-hand[:3, 3]) @ hand[:3, :3]
            for center, axes, extent in self.hands[side].boxes(angles):
                keep &= ~np.all(np.abs((local-center) @ axes) <= extent+.003, axis=1)
        return points[keep]

    def observe(self, frames, angles, stamp, expected):
        expected = np.asarray(expected)
        clouds, views = [], []
        extent = abs(self.expected_rotation)@(np.ptp(self.bounds, axis=0)/2) + [.025, .025, .045]
        if self.kind=='lotion' and self.carrying:
            # Follow the encoder-predicted in-hand rotation, with a bounded
            # margin. Resume reacquisition narrows it around the measured grip
            # offset so a nearby yellow jar cannot merge with the bottle cloud.
            margin=([.025,.025,.050] if getattr(self,'resume_reacquire',False)
                    else [.045,.045,.065])
            extent=abs(self.expected_rotation)@(np.ptp(self.bounds, axis=0)/2) + margin
        if self.kind=='tube_cap':extent=np.array([.026,.026,.055])
        for camera, (bgr, depth, k) in frames.items():
            pose = self.rig.body if camera == 'torso' else self.rig.head_pose([angles[n] for n in HEAD])
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            color=((hsv[:,:,0]>=8)&(hsv[:,:,0]<=43)&(hsv[:,:,1]>20)&(hsv[:,:,2]>45)
            if self.kind=='lotion' else hsv[:,:,2]<110)
            if self.kind=='drive':color=(hsv[:,:,1]<85)&(hsv[:,:,2]>90)
            if self.kind in ('mouse','white_mouse'):color=hsv[:,:,2]<220
            if self.kind.startswith('figure_'):color=np.ones(depth.shape,dtype=bool)
            if self.kind=='tube_cap':
                hue=hsv[:,:,0]
                color=(((hue>=85)&(hue<=140)) if self.cap_color=='blue' else ((hue<12)|(hue>168)))&(hsv[:,:,1]>55)&(hsv[:,:,2]>70)
            mask = color & (depth > 150) & (depth < 2500)
            v, u = np.nonzero(mask)
            d = depth[v, u]/1000.
            p = np.column_stack(((u-k[2])*d/k[0], (v-k[5])*d/k[4], d)) @ pose[:3, :3].T + pose[:3, 3]
            p = p[np.all(np.abs(p-expected) < extent, axis=1)]
            p = p[p[:, 2]-p[:, :2]@self.plane[:2]-self.plane[2] > .008]
            if len(p):
                p, _ = self.robot.filter(p, angles)
                p = self.remove_hands(p, angles)
            if len(p) > 15:
                clouds.append(p)
                views.append(camera)
        if not clouds:
            self.latest = dict(stamp=stamp, valid=False, reason='Нет видимой поверхности предмета вне кистей')
            return self.latest
        p = np.concatenate(clouds)
        lo, hi = np.percentile(p, [5, 95], axis=0)
        xyz = (lo+hi)/2
        result = dict(stamp=stamp, valid=False, xyz=xyz.tolist(), bounds=[lo.tolist(), hi.tolist()],
                      points=len(p), seen_by=views, tilt_deg=0., partial_visible=False,
                      method='dark_rgbd_surface_without_robot')
        if self.kind=='lotion':
            # Use both measured ends of the yellow body. Translation-only ICP
            # cannot align its two different faces after an in-hand rotation.
            # Saturation 20 also retains the illuminated face (old threshold 50
            # discarded nearly all of it). White robot meshes are excluded.
            core=p[np.all((p>=lo)&(p<=hi),axis=1)]
            if len(core)>60:
                values,vectors=np.linalg.eigh(np.cov(core.T))
                axis=vectors[:,-1];axis*=np.sign(axis[2])
                tilt=float(np.degrees(np.arccos(np.clip(axis[2],-1,1))))
                height=hi[2]-lo[2]
                width=float(np.max(hi[:2]-lo[:2]))
                axis_ratio=float(values[-1]/max(values[-2],1e-12))
                resume_full_candidate=(self.carrying and getattr(self,'resume_reacquire',False)
                    and len(p)>300 and .075<height<.28 and .035<width<.18
                    and tilt<55 and axis_ratio>1.20 and np.linalg.norm(xyz-expected)<.030)
                resume_partial_candidate=(self.carrying and getattr(self,'resume_reacquire',False)
                    and len(p)>350 and .050<height<.25 and .030<width<.18
                    and tilt<55 and axis_ratio>1.20 and np.linalg.norm(xyz-expected)<.030)
                resume_candidate=resume_full_candidate or resume_partial_candidate
                carried_full_candidate=(self.carrying and not getattr(self,'resume_reacquire',False)
                    and self.baseline is not None and len(p)>250 and .075<height<.28
                    and .035<width<.18 and tilt<55 and axis_ratio>1.20
                    and np.linalg.norm(xyz-expected)<.030)
                carried_partial_candidate=(self.carrying and not getattr(self,'resume_reacquire',False)
                    and self.baseline is not None and len(p)>350 and .050<height<.25
                    and .030<width<.18 and tilt<55 and axis_ratio>1.20
                    and np.linalg.norm(xyz-expected)<.030)
                shape_ok=((len(p)>150 and .11<height<.24 and .015<width<.105
                           and tilt<30 and axis_ratio>3) or resume_candidate
                          or carried_full_candidate or carried_partial_candidate)
                result.update(method='colored_bottle_rgbd_extents',tilt_deg=tilt,
                              principal_axis=axis.tolist(),shape_axis_ratio=axis_ratio,
                              shape_ok=bool(shape_ok),resume_reacquire_candidate=bool(resume_candidate),
                              carried_motion_candidate=bool(resume_candidate or carried_full_candidate
                                                            or carried_partial_candidate),
                              valid=bool(shape_ok))
                if self.baseline is None and shape_ok:
                    result.update(delta=[0.,0.,0.],vertical_edges_delta=[0.,0.])
                    self.baseline=result.copy();self.reference=p.copy()
                elif self.baseline is not None:
                    base=np.asarray(self.baseline['xyz']);bb=np.asarray(self.baseline['bounds'])
                    shift=xyz-base;span=height/np.ptp(bb[:,2])
                    error=float(np.linalg.norm(xyz-expected))
                    resume_occlusion_candidate=(self.carrying and getattr(self,'resume_reacquire',False)
                        and len(p)>250 and .07<height<.30 and .025<width<.20
                        and tilt<55 and axis_ratio>1.15 and .35<span<1.60 and error<.030)
                    carried_occlusion_candidate=(self.carrying and not getattr(self,'resume_reacquire',False)
                        and len(p)>250 and .07<height<.30 and .025<width<.20
                        and tilt<55 and axis_ratio>1.15 and .35<span<1.60 and error<.030)
                    if resume_occlusion_candidate or carried_occlusion_candidate:shape_ok=True
                    relaxed_carry=(resume_candidate or carried_full_candidate or carried_partial_candidate
                                   or resume_occlusion_candidate or carried_occlusion_candidate)
                    position_tolerance=.030 if self.carrying else .04
                    minimum_span=.55 if relaxed_carry else .75
                    maximum_span=1.50 if (resume_occlusion_candidate or carried_occlusion_candidate) else 1.25
                    valid=shape_ok and minimum_span<span<maximum_span and (error<position_tolerance if self.carrying else np.linalg.norm(shift[:2])<.04)
                    resume_partial=(self.carrying and getattr(self,'resume_reacquire',False)
                        and len(p)>350 and .050<height<.25 and .030<width<.18
                        and tilt<55 and axis_ratio>1.20 and error<.030)
                    carried_partial=(self.carrying and not getattr(self,'resume_reacquire',False)
                        and len(p)>350 and .050<height<.25 and .030<width<.18
                        and tilt<55 and axis_ratio>1.20 and error<.030)
                    valid=valid or resume_partial or carried_partial
                    partial=(self.carrying and len(p)>150 and .07<height<.24 and .015<width<.105
                             and tilt<55 and axis_ratio>1.15 and .015<shift[2]<.12
                             and np.linalg.norm(shift[:2])<.04 and error<.030)
                    result.update(valid=bool(valid),shape_ok=bool(shape_ok),delta=shift.tolist(),
                        vertical_edges_delta=(np.array([lo[2],hi[2]])-bb[:,2]).tolist(),
                        visible_height_fraction=float(span),prediction_error_mm=error*1000,
                        partial_visible=bool(partial),carried_shape_ok=bool(partial or resume_partial or carried_partial or resume_occlusion_candidate or carried_occlusion_candidate),
                        carried_motion_candidate=bool(resume_candidate or carried_full_candidate or carried_partial_candidate or resume_occlusion_candidate or carried_occlusion_candidate),
                        resume_reacquire_partial=bool(resume_partial),carried_motion_partial=bool(carried_partial))
                if self.carrying and self.baseline is not None and self.reference is not None and len(self.reference)>60:
                    # A held bottle may expose only one face. Fit the measured
                    # RGB-D surface to the original bottle cloud at the encoder-
                    # predicted hand rotation instead of treating the visible
                    # fragment's bounding-box centre as the bottle centre.
                    base=np.asarray(self.baseline['xyz'],float)
                    template=(np.asarray(self.reference,float)-base)@self.expected_rotation.T
                    tree=cKDTree(template);estimate=expected.copy()
                    for _ in range(6):
                        distances,ids=tree.query(p-estimate)
                        keep=distances<.022
                        if keep.sum()<45:break
                        candidate=np.median(p[keep]-template[ids[keep]],axis=0)
                        if np.linalg.norm(candidate-estimate)<.001:
                            estimate=candidate;break
                        estimate=candidate
                    distances,ids=tree.query(p-estimate)
                    support=distances<.014
                    fit_error=float(np.median(distances[support])) if np.count_nonzero(support)>=35 else float('inf')
                    prediction_error=float(np.linalg.norm(estimate-expected))
                    if (np.count_nonzero(support)>=45 and support.mean()>.18 and fit_error<.012
                            and prediction_error<=.030):
                        result.update(xyz=estimate.tolist(),valid=True,partial_visible=True,
                            shape_ok=True,carried_shape_ok=True,carried_motion_candidate=True,
                            prediction_error_mm=prediction_error*1000,fit_residual_mm=fit_error*1000,
                            method='colored_bottle_carried_rgbd_template_fit')
                self.latest=result
            return result
        if self.baseline is None:
            valid = (len(p)>60 and hi[2]-lo[2]>.005 and .010<np.max(hi[:2]-lo[:2])<.040 and len(views)==2
                     if self.kind=='tube_cap' else len(p)>90 and hi[2]-lo[2]>.012 and np.max(hi[:2]-lo[:2])>.025)
            if valid:
                result.update(valid=True, delta=[0., 0., 0.], vertical_edges_delta=[0., 0.])
                self.reference = p.copy()
                self.baseline = result.copy()
        else:
            base = np.asarray(self.baseline['xyz'])
            shift = expected-base
            reference=(self.reference-base)@self.expected_rotation.T+base
            tree = cKDTree(reference)
            # Encoders predict only template orientation and search position;
            # all translation evidence still comes from measured RGB-D points.
            for _ in range(5):
                distances, ids = tree.query(p-shift)
                keep = distances < .025
                if keep.sum() < 35:
                    break
                shift = np.median(p[keep]-reference[ids[keep]], axis=0)
            distances, ids = tree.query(p-shift)
            support = distances < .007
            visible = p[support]
            if len(visible) > 35:
                lo, hi = np.percentile(visible, [5, 95], axis=0)
                base_bounds = np.asarray(self.baseline['bounds'])
                span_fraction = (hi[2]-lo[2])/max(.01, np.ptp(base_bounds[:, 2]))
                prediction_error=np.linalg.norm(base+shift-expected)
                position_ok=(prediction_error<.030 if self.carrying else np.linalg.norm(shift[:2])<.04)
                valid = len(visible)>60 and support.mean()>.70 and span_fraction>.55 and position_ok
                partial=(self.carrying and len(visible)>35 and support.mean()>.35
                         and span_fraction>.20 and prediction_error<.030)
                carried_rgbd=(self.carrying and self.kind in ('drive','green_speaker')
                    and len(visible)>35 and span_fraction>.20 and support.mean()>.35
                    and prediction_error<.030 and np.median(distances)<.012)
                result.update(valid=bool(valid or carried_rgbd), partial_visible=bool(partial or carried_rgbd),
                              xyz=(base+shift).tolist(), delta=shift.tolist(),
                              bounds=[lo.tolist(), hi.tolist()],
                              vertical_edges_delta=(np.array([lo[2], hi[2]])-base_bounds[:, 2]).tolist(),
                              support_fraction=float(support.mean()), visible_height_fraction=float(span_fraction),
                              prediction_error_mm=float(prediction_error*1000),
                              fit_residual_mm=float(np.median(distances)*1000))
        self.latest = result
        return result

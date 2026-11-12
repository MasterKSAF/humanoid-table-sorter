"""Arm kinematics in waist_yaw_link. Distances in metres, joint positions in radians."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .posture import preparation_routes


CLEARANCE_M = 0.002


def origin(joint):
    pose = joint.find("origin")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", np.fromstring(pose.get("rpy"), sep=" ")).as_matrix()
    matrix[:3, 3] = np.fromstring(pose.get("xyz"), sep=" ")
    return matrix


def point(transform, xyz):
    return transform[:3, :3] @ xyz + transform[:3, 3]


class Arm:
    def __init__(self, side="left"):
        model = ET.parse(Path(__file__).with_name("upper_body.urdf"))
        self.side = side
        fixed = {j.get("name"): origin(j) for j in model.findall("joint[@type='fixed']")}
        self.camera = fixed["camera_body_front_joint"]
        joints = [j for j in model.findall("joint[@type='revolute']")
                  if j.get("name").endswith(f"_{side[0]}_joint")]
        self.names = [j.get("name") for j in joints]
        self.origins = [origin(j) for j in joints]
        self.axes = [np.fromstring(j.find("axis").get("xyz"), sep=" ") for j in joints]
        self.limits = np.array([[float(j.find("limit").get(k)) for j in joints] for k in ("lower", "upper")])
        self.hand = fixed[f"{side}_tcp_joint"] @ fixed[f"{side}_hand_to_tcp"]
        # Trial grasp pocket in L_base_link, between opposed thumb and curled fingers.
        # Set from finger roots in the URDF. Contact and grip strength need physics.
        self.pocket = np.array([0.035, 0., 0.135])

    def frames(self, q):
        pose = np.eye(4)
        frames = []
        for offset, axis, angle in zip(self.origins, self.axes, q):
            pose = pose @ offset
            pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(axis * angle).as_matrix()
            frames.append(pose)
        return frames

    def wrist(self, q):
        return self.frames(q)[-1]

    def tool(self, q):
        pose = self.wrist(q) @ self.hand
        pose[:3, 3] = point(pose, self.pocket)
        return pose

    def samples(self, q):
        frames = self.frames(q)
        hand = frames[-1] @ self.hand
        joints = [f[:3, 3] for f in frames] + [hand[:3, 3]]
        ends = [point(hand, p) for p in ((0., -.045, .215), (0., .045, .215), (.033, -.05 if self.side == "left" else .05, .10))]
        segments = [(a, b, .038) for a, b in zip(joints, joints[1:])] + [(joints[-1], b, .014) for b in ends]
        segments += [(joints[-1],point(hand,p),.010) for p in getattr(self,'extra_hand_points',())]
        samples = np.concatenate([np.linspace(a, b, 16) for a, b, _ in segments])
        radii = np.repeat([r for _, _, r in segments], 16)
        return samples, radii

    def clearances(self, q, table, other=None):
        plane, edge, obstacles = table
        samples, radii = self.samples(q)
        actual_hand=getattr(self,'hand_geometry',None)
        if actual_hand is not None:samples,radii=samples[:112],radii[:112]
        over = samples[:, 0] >= edge-radii
        vertical = samples[over, 2]-samples[over, :2] @ plane[:2]-plane[2]-radii[over]
        nearest = obstacles.query(samples)[0]-radii
        if other is not None:
            # Include both arm radii when checking the parked arm.
            nearest = np.minimum(nearest, other.query(samples)[0]-.038-radii)
        table_gap,object_gap=np.min(vertical, initial=np.inf),nearest.min()
        if actual_hand is not None:
            hand_gap,hand_objects=actual_hand.clearances(self.wrist(q)@self.hand,self.finger_angles,table,other)
            table_gap=min(table_gap,hand_gap);object_gap=min(object_gap,hand_objects)
        return table_gap,object_gap

    def clearance(self, q, table, other=None):
        return min(self.clearances(q, table, other))

    def path_clear(self, start, goal, table, other, minimum_clearance=CLEARANCE_M):
        if not np.all((goal >= self.limits[0]+.03) & (goal <= self.limits[1]-.03)):
            return False
        count = int(np.ceil(np.max(abs(goal-start))/.04))+1
        return all(self.clearance(s, table, other) > minimum_clearance for s in np.linspace(start, goal, count))

    def ready_route(self, q, table, other):
        for candidate in preparation_routes(self.side, q):
            route=[target for _,target in candidate]
            if all(self.path_clear(a, b, table, other, CLEARANCE_M) for a, b in zip([q, *route], route)):
                return route
        raise ValueError(f"no verified entry to {self.side} ready pose")

    def plan(self, xyz, q, table, other=None, contact_height=0.):
        target = point(self.camera, xyz)
        target[2] += contact_height
        if np.shape(q) != (7,) or not np.all(np.isfinite(target)) or not np.all(
                (q >= self.limits[0]) & (q <= self.limits[1])):
            raise ValueError("invalid target or joint state")
        sign = 1 if self.side == "left" else -1
        if sign*target[1] < 0 or np.linalg.norm(target - self.origins[0][:3, 3]) > 0.78:
            raise ValueError("outside arm workspace")
        # Palm faces down (+X of L_base_link), fingers extend horizontally (+Z).
        # Try pointing the left fingers diagonally inward first, then the alternate headings.
        for yaw in (-1.25, -.65, 0., .65, 1.25, -1.65, 1.65, -2., 2., -2.45, 2.45, np.pi):
            fingers = np.array([np.cos(yaw), sign*np.sin(yaw), 0.])
            palm = np.array([0., 0., -1.])
            rotation = np.column_stack((palm, np.cross(fingers, palm), fingers))
            proposed = []
            seed = np.array(q)
            heights = (0.18, 0.13, 0.08, 0.06, 0.04, 0.02, 0.)
            for height in heights:
                goal = target + [0., 0., height]

                def residual(angles):
                    pose = self.tool(angles)
                    delta = Rotation.from_matrix(rotation.T @ pose[:3, :3]).as_rotvec()
                    return np.r_[pose[:3, 3]-goal, 0.15*delta, 0.002*(angles-seed)]

                fit = least_squares(residual, seed, bounds=(self.limits[0]+.03, self.limits[1]-.03), max_nfev=90)
                error = residual(fit.x)
                if (np.linalg.norm(error[:3]) > 0.004 or np.linalg.norm(error[3:6]) > 0.15*0.05
                        or (proposed and np.max(np.abs(fit.x-seed)) > 0.35)):
                    break
                proposed.append(fit.x)
                seed = fit.x
            if len(proposed) == len(heights):
                if all(self.path_clear(a, b, table, other) for a, b in zip([q, *proposed], proposed)):
                    return target, rotation, proposed
        raise ValueError("no continuous collision-checked approach within joint limits")

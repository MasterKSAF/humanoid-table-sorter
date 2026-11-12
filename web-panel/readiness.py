"""Feedback-controlled, unloaded upper-body preparation. Angles use radians."""
from collections import deque
import os
import time

import numpy as np
from scipy.spatial import cKDTree
from team22_bringup.grasp import Arm
from team22_bringup.posture import ready_angles, preparation_routes, working_hand_command, POSE_ID


SIDES = ("left", "right")
LABELS = {"left": "Левая", "right": "Правая"}
AXES = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch",
        "elbow_yaw", "wrist_pitch", "wrist_roll")
HAND_TOLERANCE = np.array((.025, .025, .025, .025, .04, .025))
CLEARANCE_M = .002
NO_PROGRESS_TIMEOUT = min(15., max(4., float(os.environ.get('TEAM22_NO_PROGRESS_TIMEOUT', '15'))))


def working_hand_matches(side, position):
    measured = np.asarray(position, dtype=float)
    target = np.asarray(working_hand_command(side))
    return bool(measured.shape == target.shape and np.all(np.isfinite(measured))
                and np.all(np.abs(measured-target) <= HAND_TOLERANCE))


def observation_is_fresh(observation, now, processed_wall, wall):
    return (observation is not None and -.03 <= now-observation["stamp"] <= .30
            and wall-processed_wall <= 1.5)


class Timeline:
    """Keep simulator timestamps separate from receipt times on this host."""
    def __init__(self):
        self.data = {}
        self.bytes = {}
        self.rejected = 0

    def clear(self):
        self.data.clear()
        self.bytes.clear()

    def put(self, key, stamp, value, size=0, wall=None):
        wall = time.monotonic() if wall is None else wall
        if not np.isfinite(stamp) or stamp < 0:
            self.rejected += 1
            return False
        history = self.data.setdefault(key, deque(maxlen=700))
        if history and stamp <= history[-1][0]:
            self.rejected += 1
            return False
        history.append((stamp, wall, value))
        self.bytes[key] = size
        return True

    def latest(self, key):
        return self.data[key][-1] if self.data.get(key) else None

    def at(self, key, stamp, interpolate=False, tolerance=.025):
        history = self.data.get(key, ())
        if not history:
            return None
        before = next((s for s in reversed(history) if s[0] <= stamp), None)
        after = next((s for s in history if s[0] >= stamp), None)
        if interpolate and before and after:
            if after[0]-before[0] > 2*tolerance:
                return None
            if after[0] == before[0]:
                return before[2], 0.
            weight = (stamp-before[0])/(after[0]-before[0])
            value = {k: before[2][k]*(1-weight)+after[2][k]*weight for k in before[2] if k in after[2]}
            return value, max(stamp-before[0], after[0]-stamp)
        nearest = min(history, key=lambda s: abs(s[0]-stamp))
        return (nearest[2], abs(nearest[0]-stamp)) if abs(nearest[0]-stamp) <= tolerance else None

    def metrics(self, now, wall=None):
        wall = time.monotonic() if wall is None else wall
        result = {}
        for key, history in self.data.items():
            recent = [s for s in history if wall-s[1] <= 8.]
            hz_wall = hz_sim = None
            if len(recent) > 1:
                count = len(recent)-1
                hz_wall = count/max(recent[-1][1]-recent[0][1], 1e-6)
                hz_sim = count/max(recent[-1][0]-recent[0][0], 1e-6)
            last = history[-1]
            result[key] = dict(stamp=last[0], age_ms=1000*(now-last[0]),
                               receipt_age_ms=1000*(wall-last[1]), hz_wall=hz_wall, hz_sim=hz_sim,
                               payload_kbs=None if hz_wall is None else hz_wall*self.bytes.get(key, 0)/1000)
        return result


class ReadyController:
    """All movement phases share load, freshness and tracking checks."""
    def __init__(self, timeline):
        self.timeline = timeline
        self.models = {s: Arm(s) for s in SIDES}
        self.status = "idle"
        self.reason = "Ожидаю команду на исходную позу."
        self.stage = "Ожидание"
        self.route = deque()
        self.table = None
        self.side = None
        self.goal = {}
        self.reference = {}
        self.integral = {}
        self.command = {}
        self.last_tick = None
        self.confirmed = None
        self.started = None
        self.events = deque(maxlen=80)
        self.completed = 0
        self.progress_timeout = NO_PROGRESS_TIMEOUT
        self.targets = {s: ready_angles(s) for s in SIDES}
        self.pending_hold = False
        self.plan_preview = None
        self.planned_route = False
        self.prepare_working_pose = False
        self.finger_stage = False
        self.finger_sent = set()
        self.passive_hold = False
        self.startup_recovery = {}

    @property
    def active(self):
        return self.status in ("moving", "holding")

    def note(self, now, message):
        self.events.append(dict(stamp=now, message=message))

    def stop(self, reason, now):
        was_stopped = self.status == "stopped"
        self.status, self.reason = "stopped", reason
        self.stage = "Остановлен"
        if not was_stopped:
            self.pending_hold = True
            self.note(now, reason)
        self.route.clear()
        self.prepare_working_pose = False
        self.finger_stage = False
        self.passive_hold = False
        self.startup_recovery.clear()

    def hands_match(self):
        return all((sample := self.timeline.latest('hand_'+side)) is not None
                   and working_hand_matches(side, sample[2]['position']) for side in SIDES)

    def arm_matches(self, side, q, tolerance=.025):
        return all(key in q and np.isfinite(q[key]) and abs(q[key]-target) <= tolerance
                   for key, target in zip(self.models[side].names, self.targets[side]))

    def health(self, now, wall, passive_hold=False):
        # Stale force samples do not invalidate a stationary arm pose. A new
        # motion still requires fresh samples on the controller's first tick.
        passive_hold = passive_hold or self.passive_hold or self.status != "moving"
        joint_sample = self.timeline.latest("joints")
        q_for_freshness = joint_sample[2] if joint_sample else {}
        moving_sides = []
        if self.status == "moving":
            for side in SIDES:
                names = self.models[side].names
                if all(key in self.goal and key in q_for_freshness for key in names):
                    if max(abs(float(self.goal[key])-float(q_for_freshness[key]))
                           for key in names) > .01:
                        moving_sides.append(side)
        inactive_side = ({"left": "right", "right": "left"}.get(moving_sides[0])
                         if len(moving_sides) == 1 else None)
        inactive_stationary = bool(inactive_side and all(
            key in self.goal and key in q_for_freshness
            and abs(float(self.goal[key])-float(q_for_freshness[key])) < .01
            for key in self.models[inactive_side].names))
        for key in ("joints", "hand_left", "hand_right", "wrench_left", "wrench_right"):
            sample = self.timeline.latest(key)
            # Force messages can trail one simulator tick while the simulator
            # runs below real time. The wall-clock cap below still stops a real
            # sensor outage promptly.
            sim_lag_limit = .35
            sim_lag = now-sample[0] > sim_lag_limit if sample else True
            passive_wrench = passive_hold and key.startswith('wrench_')
            inactive_sensor = bool(inactive_stationary and key.endswith('_'+str(inactive_side)))
            if inactive_sensor and key.startswith('hand_') and sample:
                position = np.asarray(sample[2].get('position', ()), dtype=float)
                inactive_sensor = bool(position.shape == (6,) and np.all(np.isfinite(position))
                                       and np.min(position) > .9)
            wall_age_limit = 5. if passive_wrench else 1.5
            # Under slow simulation /clock can lag freshly received sensor
            # stamps by several physics frames. The wall-clock freshness cap
            # remains authoritative when a packet is actually delayed.
            future_limit = .4
            # A stale opposite-side stream is safe to reuse only while its arm
            # target is unchanged and its last hand sample confirms an open palm.
            # The active side, joints, and force threshold checks remain live.
            if (not sample or (sim_lag and not (passive_wrench or inactive_sensor))
                    or sample[0]-now > future_limit
                    or (wall-sample[1] > wall_age_limit and not inactive_sensor)):
                sim_age = None if not sample else max(0., now-sample[0])
                wall_age = None if not sample else max(0., wall-sample[1])
                future_age = None if not sample else max(0., sample[0]-now)
                return (f"Нет свежих показаний — {key}; sim={sim_age}, wall={wall_age}, "
                        f"future={future_age}, "
                        f"status={self.status}, moving={moving_sides}, inactive={inactive_side}, "
                        f"stationary={inactive_stationary}, allowed={inactive_sensor}")
        q = joint_sample[2]
        body_tolerance=.06 if getattr(self,'body_turn_active',False) else .025
        if abs(q.get("body_yaw_joint", 10.)-getattr(self,'body_reference',0.)) > body_tolerance:
            return "Корпус отклонился от проверенной опорной ориентации."
        for side in SIDES:
            value = self.timeline.latest("wrench_"+side)[2]
            force = float(np.linalg.norm(value["force"]))
            if force > 20.:
                return f"{LABELS[side]} кисть — нагрузка {force:.1f} Н превысила 20 Н."
            model = self.models[side]
            angles = np.array([q.get(k, np.nan) for k in model.names])
            if not np.all(np.isfinite(angles)) or np.any(angles < model.limits[0]-.025) or np.any(angles > model.limits[1]+.025):
                return f"{LABELS[side]} рука — недостоверные углы или выход за пределы."
        return None

    def palms(self, q, goals=None):
        result = {}
        goals = self.goal if goals is None else goals
        for side, model in self.models.items():
            if any(k not in q for k in model.names):
                continue
            angles = np.array([q[k] for k in model.names])
            pose = model.wrist(angles)@model.hand
            target = None
            error = None
            joint_error = None
            if all(k in goals for k in model.names):
                desired = np.array([goals[k] for k in model.names])
                target = (model.wrist(desired)@model.hand)[:3, 3]
                error = float(1000*np.linalg.norm(target-pose[:3, 3]))
                joint_error = float(np.degrees(np.max(abs(desired-angles))))
            height = None if self.table is None else 1000*(pose[2, 3]-pose[:2, 3]@self.table[0][:2]-self.table[0][2])
            result[side] = dict(xyz_mm=(pose[:3, 3]*1000).tolist(),
                                rotation=pose[:3, :3].tolist(), frame='waist_yaw_link',
                                target_mm=None if target is None else (target*1000).tolist(),
                                error_mm=error, joint_error_deg=joint_error, table_height_mm=height)
        return result

    def make_synchronized_route(self, q, repeat):
        """Check one direct paired move, including a short startup egress."""
        from bimanual import interarm_gap
        from transfer_trial import body_clear
        targets={side:ready_angles(side) for side in SIDES}
        starts={side:np.array([q[k] for k in self.models[side].names]) for side in SIDES}
        largest=max(float(np.max(abs(targets[side]-starts[side]))) for side in SIDES)
        steps=max(1,int(np.ceil(largest/.025)))
        recovery={}
        active_recovery=set()
        for fraction in np.linspace(0.,1.,steps+1):
            pair={side:starts[side]+fraction*(targets[side]-starts[side]) for side in SIDES}
            for side in SIDES:
                model=self.models[side]
                other_side='right' if side=='left' else 'left'
                other=cKDTree(self.models[other_side].samples(pair[other_side])[0])
                if not body_clear(model,pair[side]):
                    return None
                table_gap,obstacle_gap=model.clearances(pair[side],self.table,other)
                if fraction==0. and table_gap<CLEARANCE_M:
                    # The simulator's factory posture starts with a measured
                    # 7 mm table-envelope overlap. Permit only a verified,
                    # monotone escape over the first 25% of this direct path.
                    if table_gap<-.010 or obstacle_gap<.010:
                        return None
                    recovery[side]=dict(start=starts[side].copy(),goal=targets[side].copy(),
                                        start_gap=table_gap,last_gap=table_gap,last_fraction=0.,
                                        start_stamp=None)
                    active_recovery.add(side)
                if side in active_recovery:
                    if table_gap<CLEARANCE_M:
                        if (table_gap<-.010 or obstacle_gap<.010 or fraction>.25
                                or table_gap<recovery[side]['last_gap']-.0005):
                            return None
                        recovery[side]['last_gap']=table_gap
                    else:
                        active_recovery.remove(side)
                elif table_gap<CLEARANCE_M or obstacle_gap<CLEARANCE_M:
                    return None
            if interarm_gap(self.models,pair)<CLEARANCE_M:
                return None
        self.startup_recovery={side:dict(value,start=value['start'].copy(),goal=value['goal'].copy(),
                                         last_gap=value['start_gap'],last_fraction=0.,start_stamp=None)
                               for side,value in recovery.items()}
        label='Вернуть руки в рабочую позу' if repeat else 'Установить руки в рабочую позу'
        return [('both',label,targets)]

    def make_route(self, q, repeat):
        from transfer_trial import body_clear
        initial = {s: np.array([q[k] for k in m.names]) for s,m in self.models.items()}
        valid_routes = []
        failures = []

        # The first arm changes the clearance available to the second. Check both
        # serial orders against the live obstacle map, then choose the shorter
        # fully safe route.
        for order in (SIDES, tuple(reversed(SIDES))):
            route = []
            future = {s: value.copy() for s, value in initial.items()}
            cost = 0.
            failed = None
            for side in order:
                model = self.models[side]
                other_side = "right" if side == "left" else "left"
                other = cKDTree(self.models[other_side].samples(future[other_side])[0])
                accepted = None
                rejection = ''
                for candidate in preparation_routes(side, future[side], repeat):
                    current = future[side].copy()
                    candidate_cost = 0.
                    for label, target in candidate:
                        if np.any(target < model.limits[0]+.035) or np.any(target > model.limits[1]-.035):
                            rejection = f'{label}: недостаточный запас до предела сустава'
                            break
                        samples = np.linspace(current, target, int(np.ceil(np.max(abs(target-current))/.035))+1)
                        blocked = next((sample for sample in samples if not body_clear(model, sample)
                                        or model.clearance(sample, self.table, other) < CLEARANCE_M), None)
                        if blocked is not None:
                            if not body_clear(model, blocked):
                                rejection = f'{label}: пересечение корпуса'
                            else:
                                table_gap, obstacle_gap = model.clearances(blocked, self.table, other)
                                reason = (f'зазор до стола {table_gap*1000:.1f} мм' if table_gap < obstacle_gap
                                          else f'зазор до препятствия или второй руки {obstacle_gap*1000:.1f} мм')
                                rejection = f'{label}: {reason}, требуется 2 мм'
                            break
                        candidate_cost += float(np.max(abs(target-current)))
                        current = target
                    else:
                        accepted = candidate
                        cost += candidate_cost
                        break
                if accepted is None:
                    failed = f"{LABELS[side]} рука — нет маршрута в рабочую позу. {rejection}."
                    break
                for label, target in accepted:
                    route.append((side, label, target))
                    future[side] = target
            if failed is None:
                valid_routes.append((cost, route))
            else:
                failures.append(failed)

        if valid_routes:
            return min(valid_routes, key=lambda item: item[0])[1]
        raise ValueError('Ни один порядок рук не проходит проверку: ' + ' '.join(failures))

    def start(self, now, table, repeat=False, wall=None, plan=None):
        wall = time.monotonic() if wall is None else wall
        if self.status == "moving":
            raise ValueError("Переход уже выполняется.")
        problem = self.health(now, wall)
        if problem:
            self.stop(problem, now)
            raise ValueError(problem)
        q = self.timeline.latest("joints")[2]
        if all(self.arm_matches(side, q) for side in SIDES) and self.hands_match():
            self.table = table
            self.route.clear()
            self.plan_preview = None
            self.planned_route = self.synchronized = self.interarm_guard = False
            self.arm_speed_override = None
            self.prepare_working_pose = self.finger_stage = False
            self.finger_sent.clear()
            self.passive_hold = True
            self.pending_hold = False
            self.status, self.stage, self.reason = "holding", "Рабочая поза подтверждена", None
            self.goal = {key: target for side in SIDES for key, target in zip(self.models[side].names, self.targets[side])}
            self.reference = {key: q[key] for key in self.goal}
            self.command = dict(self.reference)
            self.integral = {key: 0. for key in self.goal}
            self.completed += 1
            self.note(now, self.stage)
            return
        if table is None:
            raise ValueError("Нужна свежая карта стола из обеих камер.")
        self.table = table
        synchronized_route=False
        if plan is not None:
            from motion_planner import start_matches
            if plan.get('error') or not plan.get('route') or not start_matches(plan, q):
                raise ValueError('План отклонён: ошибка расчёта или робот сдвинулся после снимка карты.')
            self.route = deque((s, name, np.asarray(target)) for s, name, target in plan['route'])
            self.plan_preview = plan['preview']
        else:
            self.plan_preview=None
            route=self.make_synchronized_route(q,repeat)
            if route is None:
                route=self.make_route(q,repeat)
            else:
                synchronized_route=True
            self.route = deque(route)
        speed = float(os.environ.get('TEAM22_ARM_SPEED', '.8'))
        if not np.isfinite(speed) or speed < .05:
            raise ValueError('TEAM22_ARM_SPEED должен быть не меньше 0,05 рад/с.')
        self.planned_route = plan is not None
        self.synchronized = self.planned_route or synchronized_route
        self.interarm_guard = self.planned_route or synchronized_route
        self.arm_speed_override = min(speed, .8)
        self.passive_hold = False
        self.status, self.reason = "moving", None
        self.pending_hold = False
        self.prepare_working_pose = True
        self.finger_stage = False
        self.finger_sent.clear()
        self.goal = {k: q[k] for m in self.models.values() for k in m.names}
        self.reference = dict(self.goal)
        self.command = dict(self.goal)
        self.integral = {k: 0. for k in self.goal}
        self.last_tick = now
        self.next_stage(now)

    def next_stage(self, now):
        self.confirmed = None
        if not self.route:
            if self.prepare_working_pose and not self.finger_stage:
                self.finger_stage = True
                self.stage = "Кисти — раскрыть пальцы рабочей позы"
                self.started = now
                self.segment_timeout = 22.
                self.progress_stamp = now
            else:
                self.status = "holding"
                self.passive_hold = True
                self.stage = ("Рабочая поза подтверждена" if self.prepare_working_pose
                              else "Маршрут подтверждён")
                self.completed += 1
                self.prepare_working_pose = False
                self.finger_stage = False
            self.note(now, self.stage)
            return
        self.finger_stage = False
        self.passive_hold = False
        self.side, name, target = self.route.popleft()
        self.segment_start = dict(self.goal)
        sides=SIDES if self.side=='both' else (self.side,)
        if getattr(self, 'synchronized', False) and not self.planned_route:
            measured = self.timeline.latest('joints')[2]
            for side in sides:
                for key in self.models[side].names:self.reference[key] = measured[key]
        self.stage = ('Обе руки' if self.side=='both' else LABELS[self.side])+" — "+name.lower()
        for side in sides:self.goal.update(zip(self.models[side].names, target[side] if self.side=='both' else target))
        requested_speed = getattr(self, 'arm_speed_override', None)
        if requested_speed is not None:
            self.motion_speed = float(np.clip(requested_speed, .02, .8))
        if self.planned_route:
            near_table=False
            if self.table is not None:
                for side in sides:
                    model=self.models[side]
                    for angles in (self.reference,self.goal):
                        near_table |= model.clearances(np.array([angles[k] for k in model.names]),self.table)[0]<.035
            self.tracking_joint_tolerance=.004 if near_table else .006
            if requested_speed is None:
                self.motion_speed = .12 if near_table else .18
            distance=max(abs(self.goal[k]-self.reference[k]) for k in self.goal)
            # Reference pauses for loaded joints. Budget travel plus settling,
            # rather than declaring a correctly tracking slow move a failure.
            # AG measured about .015 rad/s while supporting the outstretched
            # arm, despite a .18 rad/s reference. A separate progress watchdog
            # below still rejects a stalled joint within 15 simulation seconds.
            self.segment_timeout=min(180.,max(60.,12.*distance/self.motion_speed+12.))
            measured=self.timeline.latest('joints')[2]
            self.progress_error=max(abs(self.goal[k]-measured[k]) for k in self.goal)
            self.progress_stamp=now
        elif requested_speed is not None and getattr(self, 'synchronized', False):
            distance=max(abs(self.goal[k]-self.reference[k]) for k in self.goal)
            self.segment_timeout=min(180.,max(22.,12.*distance/self.motion_speed+12.))
        self.started = now
        self.note(now, self.stage)

    def hand_commands(self):
        """Return each needed hand command once, after the arms reach the target."""
        if self.status != 'moving' or not self.finger_stage:
            return {}
        from team22_bringup.hand_geometry import HandGeometry
        joints = self.timeline.latest('joints')[2]
        commands = {}
        for side, model in self.models.items():
            measured = np.asarray(self.timeline.latest('hand_'+side)[2]['position'], dtype=float)
            if working_hand_matches(side, measured) or side in self.finger_sent:
                continue
            target = np.asarray(working_hand_command(side))
            geometry = HandGeometry(side)
            other_side = 'right' if side == 'left' else 'left'
            other = self.models[other_side]
            other_q = np.array([joints[k] for k in other.names])
            other_points = cKDTree(other.samples(other_q)[0])
            for arm_q in (np.array([joints[k] for k in model.names]), self.targets[side]):
                hand = model.wrist(arm_q)@model.hand
                for fraction in np.linspace(0., 1., 11):
                    fingers = measured+(target-measured)*fraction
                    table_gap, object_gap = geometry.clearances(
                        hand, geometry.nominal_angles(fingers), self.table, other_points)
                    if min(table_gap, object_gap) < CLEARANCE_M:
                        raise ValueError(f'{LABELS[side]} кисть — для пальцев нет зазора 2 мм '
                                         f'(стол {table_gap*1000:.1f}, препятствие {object_gap*1000:.1f} мм).')
            commands[side] = tuple(target)
            self.finger_sent.add(side)
        return commands

    def tick(self, now, wall=None):
        wall = time.monotonic() if wall is None else wall
        if not self.active:
            return None
        problem = self.health(now, wall, passive_hold=self.passive_hold)
        if problem:
            self.stop(problem, now)
            return None
        if self.passive_hold:
            return None
        sample = self.timeline.latest("joints")
        stamp, _, q = sample
        if self.last_tick is None or stamp <= self.last_tick:
            return None
        dt = min(stamp-self.last_tick, .05)
        self.last_tick = stamp
        window = [s for s in self.timeline.data["joints"] if stamp-s[0] <= .32]
        settled = len(window)>2 and window[-1][0]-window[0][0]>=.25
        close = True
        reference_scale = 1.
        reference_speed = min(.8, getattr(self, 'arm_speed_override', None) or .25)
        if getattr(self, 'synchronized', False):
            distance = max((abs(self.goal[k]-self.reference[k]) for k in self.goal), default=0.)
            reference_scale = min(1., getattr(self,'motion_speed',.25)*dt/max(distance, 1e-12))
            if self.planned_route and max(abs(self.reference[k]-q[k]) for k in self.goal) > getattr(self,'tracking_joint_tolerance',.03):
                reference_scale = 0.  # Wait for the slowest loaded joint.
        for side, model in self.models.items():
            if hasattr(model,'hand_geometry'):model.finger_angles=q
            for i, key in enumerate(model.names):
                delta = self.goal[key]-self.reference[key]
                self.reference[key] += (delta*reference_scale if getattr(self, 'synchronized', False)
                                        else float(np.clip(delta, -reference_speed*dt, reference_speed*dt)))
                error = self.reference[key]-q[key]
                # Do not integrate ordinary servo lag while the planned reference
                # advances: it creates an overshoot at each waypoint. Accumulate
                # load compensation only while waiting or holding a reference.
                reference_moving = self.planned_route and abs(delta)>1e-8 and reference_scale>0.
                if not reference_moving:
                    self.integral[key] = float(np.clip(self.integral[key]+.65*error*dt, -.30, .30))
                # Preserve load compensation across waypoints and tiny encoder
                # overshoots. Clearing it here drops the gravity-support command
                # by up to 0.1 rad just as the next segment starts.
                # The simulator's position drives sag under gravity. A bounded
                # proportional correction reacts immediately while the integral
                # learns the static load; pausing the reference alone is too slow.
                correction=float(np.clip(1.5*error,-.045,.045)) if self.planned_route else 0.
                target = float(np.clip(self.reference[key]+self.integral[key]+correction, model.limits[0, i]+.035, model.limits[1, i]-.035))
                command_speed = max(.40, getattr(self, 'motion_speed', .40))
                self.command[key] += float(np.clip(target-self.command[key], -command_speed*dt, command_speed*dt))
                # PhysX velocity reports oscillate even with stationary encoder positions.
                # Require the complete measured position range to remain small for 0.25 s.
                close &= abs(self.goal[key]-q[key]) < (.003 if self.planned_route else .009) and settled
                if settled:
                    close &= np.ptp([s[2][key] for s in window]) < .003
            measured = np.array([q[k] for k in model.names])
            if self.planned_route and self.status == 'moving':
                a = np.array([self.segment_start[k] for k in model.names])
                b = np.array([self.goal[k] for k in model.names]); d = b-a
                fraction = np.clip(np.dot(measured-a, d)/max(np.dot(d, d), 1e-12), 0., 1.)
                expected = a+fraction*d
                deviation = np.linalg.norm(model.samples(measured)[0]-model.samples(expected)[0], axis=1).max()
                limit=.012
                if deviation>.012 and self.table is not None:
                    # A free-space servo deviation is not a table-side miss.
                    # Keep the 12 mm envelope near objects; expand only when
                    # both actual and checked poses have measured spare room.
                    free=min(*model.clearances(measured,self.table),*model.clearances(expected,self.table))
                    from bimanual import interarm_gap
                    pair={s:np.array([q[k] for k in self.models[s].names]) for s in SIDES}
                    free=min(free,interarm_gap(self.models,pair))
                    from transfer_trial import body_clear
                    if not body_clear(model,measured):free=0.
                    limit=min(.025,.012+max(0.,free-.035)*.5)
                if deviation > limit:
                    self.stop(f'{LABELS[side]} рука — отклонение от проверенной траектории {deviation*1000:.1f} мм (предел {limit*1000:.1f}).', now)
                    return None
            if self.table is not None:
                table_gap,obstacle_gap=model.clearances(measured,self.table)
                threshold=getattr(model,'minimum_table_gap',CLEARANCE_M)
                obstacle_threshold=getattr(model,'minimum_object_gap',CLEARANCE_M)
                recovery=self.startup_recovery.get(side)
                if recovery and table_gap>=threshold:
                    self.startup_recovery.pop(side,None)
                    recovery=None
                if table_gap<threshold and recovery and getattr(self,'synchronized',False) and self.side=='both':
                    from transfer_trial import body_clear
                    if recovery['start_stamp'] is None:
                        recovery['start_stamp']=stamp
                    direction=recovery['goal']-recovery['start']
                    fraction=float(np.clip(np.dot(measured-recovery['start'],direction)
                                           /max(np.dot(direction,direction),1e-12),0.,1.))
                    expected=recovery['start']+fraction*direction
                    deviation=float(np.max(np.linalg.norm(model.samples(measured)[0]
                                                         -model.samples(expected)[0],axis=1)))
                    valid=(table_gap>=-.010 and obstacle_gap>=obstacle_threshold and body_clear(model,measured)
                           and fraction<=.25 and fraction>=recovery['last_fraction']-.02
                           and deviation<=.012 and table_gap>=recovery['last_gap']-.001
                           and not (fraction<.03 and stamp-recovery['start_stamp']>3.))
                    if valid:
                        recovery['last_gap']=max(recovery['last_gap'],table_gap)
                        recovery['last_fraction']=max(recovery['last_fraction'],fraction)
                    else:
                        self.stop(f'{LABELS[side]} рука — стартовый выход из-за стола отклонён контролем безопасности.',now)
                        return None
                elif table_gap<threshold or obstacle_gap<obstacle_threshold:
                    self.stop(f"{LABELS[side]} рука — недостаточный зазор "
                              f"(стол {1000*table_gap:.1f} < {threshold*1000:.0f}, "
                              f"предметы {1000*obstacle_gap:.1f} < {obstacle_threshold*1000:.0f} мм).", now)
                    return None
        if getattr(self,'interarm_guard',False):
            from bimanual import interarm_gap
            gap=interarm_gap(self.models,{s:np.array([q[k] for k in self.models[s].names]) for s in SIDES})
            if gap<CLEARANCE_M:
                self.stop(f'Расстояние между руками меньше 2 мм: {gap*1000:.1f} мм',now)
                return None
        close &= all(p["error_mm"] is not None and p["error_mm"] < (3. if self.planned_route else 9.) for p in self.palms(q).values())
        if self.finger_stage:
            close &= self.hands_match()
        if self.status == "moving":
            if close:
                if self.confirmed is None:
                    self.confirmed = stamp
                elif stamp-self.confirmed >= .55:
                    self.next_stage(stamp)
            else:
                self.confirmed = None
            timeout=(getattr(self,'segment_timeout',22.) if self.planned_route
                     or getattr(self,'arm_speed_override',None) is not None else 22.)
            if self.planned_route and self.status=='moving' and not self.finger_stage:
                remaining=max(abs(self.goal[k]-q[k]) for k in self.goal)
                if remaining<self.progress_error-.01:
                    self.progress_error=remaining;self.progress_stamp=stamp
                progress_timeout=getattr(self,'progress_timeout',NO_PROGRESS_TIMEOUT)
                if not close and stamp-self.progress_stamp>progress_timeout:
                    self.stop(f'Нет измеримого приближения к цели за {progress_timeout:g} секунд симуляции.',now)
                    return None
            if self.status == "moving" and stamp-self.started > timeout:
                self.stop(f"Поза не достигнута за {timeout:.0f} секунд симуляции.", now)
                return None
        return dict(self.command)

    def snapshot(self, now):
        joints = self.timeline.latest("joints")
        return dict(status=self.status, stage=self.stage, reason=self.reason, completed=self.completed,
                    ready_pose=POSE_ID,
                    finger_targets={side:working_hand_command(side) for side in SIDES},
                    plan_preview=self.plan_preview,
                    stamp=None if joints is None else joints[0],
                    palms={} if joints is None else self.palms(joints[2]),
                    target_angles=self.goal, reference_angles=self.reference,
                    command_angles=self.command, compensation_angles=self.integral,
                    events=list(self.events))

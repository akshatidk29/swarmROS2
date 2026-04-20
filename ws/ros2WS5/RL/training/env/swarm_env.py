"""
Multi-agent swarm collector — Gymnasium environment with Pygame renderer.

CTDE: Shared policy (IPPO), 4 agents. The env cycles through agents each step.
From SB3's perspective this is a single-agent env.

Observation : 43-dim  (18 LiDAR + 25 fixed — no SLAM, no occ grid)
Action      : 3-dim continuous [vx, vy, wz] in [-1, 1]

Key design:
  - No oracle knowledge: agents discover objects only via camera FOV
  - Cross-pattern spawn: random centre + 4 agents at ±1m offsets
  - Safety hysteresis: prevents override oscillation
  - Action EMA smoothing: prevents velocity jerk
  - Collaboration: object broadcast + retroactive collab reward
  - Random-walk exploration: simple random targets when no objects visible
"""

import math
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    ARENA_HALF, ARENA_SIZE, N_ROBOTS, ROBOT_RADIUS, MAX_SPEED, MAX_WZ,
    OBJECT_DEFS, N_OBJECTS, CUBE_BASKET, SPHERE_BASKET,
    OBSTACLES, PICK_DIST, BASKET_DIST,
    SAFETY_DIST_ENTER, SAFETY_DIST_EXIT, ACTION_EMA,
    LIDAR_RAYS, LIDAR_MAX_DIST, LIDAR_MIN_DIST,
    CAMERA_FOV, CAMERA_RANGE,
    N_DETECTIONS, CARRY_TIMEOUT_STEPS,
    OBS_DIM, ACT_DIM,
    R_STEP, R_APPROACH, R_PICK,
    R_BROADCAST_BONUS, R_COLLAB_BONUS, R_DROP_CORRECT, R_DROP_WRONG,
    R_CARRY_OVERTIME, R_WALL_HIT, R_ROBOT_COLLISION,
    R_PROXIMITY_HARD, R_PROXIMITY_SOFT, R_OBSTACLE_NEAR,
    R_DANGLE, R_JERK, R_ALL_DONE,
    MAX_STEPS, DT, SPAWN_CLUSTER_SPREAD, EXPLORE_RETARGET_STEPS,
)
from env.arena import (
    norm_angle, check_wall_collision,
    check_obstacle_collision, point_near_any_obstacle,
    raycast, detect_in_fov,
    compute_repulsion_from_lidar,
)

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


# ═══════════════ SHARED STATE ═══════════════

class _SharedState:
    """Coordination bus — only contains information explicitly shared."""

    def __init__(self):
        self.collected = set()            # object names permanently removed
        self.broadcasts = []              # [(agent_idx, obj_type, x, y, step)]
        self.agent_poses = [(0.0, 0.0, 0.0)] * N_ROBOTS
        self.agent_carry = [0] * N_ROBOTS

    def reset(self):
        self.collected.clear()
        self.broadcasts.clear()
        self.agent_poses = [(0.0, 0.0, 0.0)] * N_ROBOTS
        self.agent_carry = [0] * N_ROBOTS


# ═══════════════ AGENT STATE ═══════════════

class AgentState:
    def __init__(self, idx, x, y):
        self.idx = idx
        self.x, self.y = x, y
        self.yaw = 0.0
        self.vx, self.vy = 0.0, 0.0
        self.carrying = 0           # 0=none, 1=cube, 2=sphere
        self.carrying_name = None
        self.carry_steps = 0
        self.target = None
        self.prev_target_dist = None
        self.prev_action = np.zeros(ACT_DIM)
        self.prev_prev_action = np.zeros(ACT_DIM)  # for jerk calculation

        # Per-agent detection memory (objects THIS agent has seen via camera)
        self.detected_objects = {}  # obj_name → (x, y, type, step)

        # Safety hysteresis state
        self.safety_active = False

        # Anti-dangle tracking
        self.near_obstacle_steps = 0
        self.prev_pos = (x, y)

        # Random-walk exploration
        self._explore_target = None
        self._explore_steps = 0

    def reset(self, x, y):
        self.x, self.y = x, y
        self.yaw = random.uniform(-math.pi, math.pi)
        self.vx = self.vy = 0.0
        self.carrying = 0
        self.carrying_name = None
        self.carry_steps = 0
        self.target = None
        self.prev_target_dist = None
        self.prev_action = np.zeros(ACT_DIM)
        self.prev_prev_action = np.zeros(ACT_DIM)
        self.detected_objects.clear()
        self.safety_active = False
        self.near_obstacle_steps = 0
        self.prev_pos = (x, y)
        self._explore_target = None
        self._explore_steps = 0


# ═══════════════ ENVIRONMENT ═══════════════

class SwarmCollectorEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 20}

    def __init__(self, render_mode=None, max_steps=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(-1, 1, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(ACT_DIM,), dtype=np.float32)

        self.agents = [AgentState(i, 0.0, 0.0) for i in range(N_ROBOTS)]
        self.shared = _SharedState()
        self.current_agent = 0
        self.step_count = 0
        self.max_steps = max_steps if max_steps is not None else MAX_STEPS
        self.objects = []

        # Episode stats
        self._ep_picks = 0
        self._ep_deliveries = 0
        self._ep_collisions = 0
        self._ep_broadcasts = 0
        self._ep_collabs = 0

        # Pygame
        self._screen = None
        self._clock = None
        self._px_scale = 60
        self._screen_size = int(ARENA_SIZE * self._px_scale)

    # ── RESET ──

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.current_agent = 0
        self.shared.reset()

        self._ep_picks = 0
        self._ep_deliveries = 0
        self._ep_collisions = 0
        self._ep_broadcasts = 0
        self._ep_collabs = 0

        # Cross-pattern spawn: random centre + ±1m offsets
        self._spawn_robots()

        # Fully randomized object placement
        self.objects = self._randomize_objects()

        # Initial target assignment
        self._assign_targets()
        return self._get_obs(self.current_agent), {}

    def _spawn_robots(self):
        """Spawn 4 robots in a cross pattern around a random centre."""
        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        margin = SPAWN_CLUSTER_SPREAD + ROBOT_RADIUS + 0.5

        for _ in range(200):
            cx = random.uniform(-ARENA_HALF + margin + 0.15,
                                ARENA_HALF - margin - 0.15)
            cy = random.uniform(-ARENA_HALF + margin + 0.15, 0.0)

            # Check all 4 positions are valid
            positions = []
            all_ok = True
            for dx, dy in offsets:
                px = cx + dx * SPAWN_CLUSTER_SPREAD
                py = cy + dy * SPAWN_CLUSTER_SPREAD
                if (abs(px) > ARENA_HALF - 0.5 or abs(py) > ARENA_HALF - 0.5):
                    all_ok = False
                    break
                if point_near_any_obstacle(px, py, ROBOT_RADIUS + 0.3):
                    all_ok = False
                    break
                positions.append((px, py))

            if all_ok and len(positions) == 4:
                for i, agent in enumerate(self.agents):
                    agent.reset(positions[i][0], positions[i][1])
                    self.shared.agent_poses[i] = (agent.x, agent.y, agent.yaw)
                return

        # Fallback: deterministic positions
        fallback = [(0.0, -2.0), (2.0, -2.0), (-2.0, -2.0), (0.0, -4.0)]
        for i, agent in enumerate(self.agents):
            agent.reset(fallback[i][0], fallback[i][1])
            self.shared.agent_poses[i] = (agent.x, agent.y, agent.yaw)

    def _randomize_objects(self):
        """Place all objects at fully random positions. No hardcoded nominal coords."""
        objects = []
        placed = []
        wall_limit = ARENA_HALF - 0.5 - 0.15  # inside wall face

        for name, otype in OBJECT_DEFS:
            for _ in range(300):
                ox = random.uniform(-wall_limit, wall_limit)
                oy = random.uniform(-wall_limit, wall_limit)

                # Not on baskets
                if math.sqrt((ox - CUBE_BASKET[0])**2 +
                             (oy - CUBE_BASKET[1])**2) < 1.2:
                    continue
                if math.sqrt((ox - SPHERE_BASKET[0])**2 +
                             (oy - SPHERE_BASKET[1])**2) < 1.2:
                    continue

                # Not on obstacles
                if point_near_any_obstacle(ox, oy, 0.4):
                    continue

                # Not on other placed objects
                too_close = False
                for px, py in placed:
                    if math.sqrt((ox - px)**2 + (oy - py)**2) < 0.6:
                        too_close = True
                        break
                if too_close:
                    continue

                # Not on robot spawn positions
                spawn_ok = True
                for agent in self.agents:
                    if math.sqrt((ox - agent.x)**2 + (oy - agent.y)**2) < 1.0:
                        spawn_ok = False
                        break
                if not spawn_ok:
                    continue

                break
            else:
                # Fallback: place somewhere safe
                ox = random.uniform(-3.0, 3.0)
                oy = random.uniform(-3.0, 3.0)

            placed.append((ox, oy))
            objects.append({
                'name': name, 'type': otype,
                'x': ox, 'y': oy, 'alive': True, 'placed': False,
            })
        return objects

    # ── STEP ──

    def step(self, action):
        action = np.clip(action, -1, 1)
        agent = self.agents[self.current_agent]
        reward = R_STEP

        # ── Action EMA smoothing ──
        smoothed = (1.0 - ACTION_EMA) * action + ACTION_EMA * agent.prev_action

        # ── Scale action to real units ──
        cmd_vx = smoothed[0] * MAX_SPEED
        cmd_vy = smoothed[1] * MAX_SPEED
        cmd_wz = smoothed[2] * MAX_WZ

        # ── Safety layer with HYSTERESIS ──
        lidar_obs = self._get_lidar(agent)
        lidar_min = min(lidar_obs) * LIDAR_MAX_DIST

        if not agent.safety_active and lidar_min < SAFETY_DIST_ENTER:
            agent.safety_active = True
        elif agent.safety_active and lidar_min > SAFETY_DIST_EXIT:
            agent.safety_active = False

        if agent.safety_active:
            rep_vx, rep_vy = compute_repulsion_from_lidar(
                lidar_obs, LIDAR_RAYS, LIDAR_MAX_DIST)
            cmd_vx, cmd_vy, cmd_wz = rep_vx, rep_vy, 0.0

        # ── Robot-frame → world-frame ──
        cos_y, sin_y = math.cos(agent.yaw), math.sin(agent.yaw)
        world_vx = cmd_vx * cos_y - cmd_vy * sin_y
        world_vy = cmd_vx * sin_y + cmd_vy * cos_y

        agent.vx, agent.vy = cmd_vx, cmd_vy
        new_x = agent.x + world_vx * DT
        new_y = agent.y + world_vy * DT
        agent.yaw = norm_angle(agent.yaw + cmd_wz * DT)

        # ── Wall collision ──
        new_x, new_y, wall_hit = check_wall_collision(new_x, new_y)
        if wall_hit:
            reward += R_WALL_HIT
            self._ep_collisions += 1

        # ── Obstacle collision ──
        new_x, new_y, obs_hit = check_obstacle_collision(
            new_x, new_y, agent.x, agent.y)
        if obs_hit:
            reward += R_WALL_HIT
            self._ep_collisions += 1

        agent.x, agent.y = new_x, new_y
        self.shared.agent_poses[agent.idx] = (agent.x, agent.y, agent.yaw)
        self.shared.agent_carry[agent.idx] = agent.carrying

        # ── Obstacle proximity penalty (continuous shaping) ──
        if point_near_any_obstacle(agent.x, agent.y, 0.5):
            reward += R_OBSTACLE_NEAR

        # ── Robot-robot proximity ──
        for other in self.agents:
            if other.idx == agent.idx:
                continue
            d = math.sqrt((agent.x - other.x) ** 2 + (agent.y - other.y) ** 2)
            if d < 2 * ROBOT_RADIUS:
                reward += R_ROBOT_COLLISION
                self._ep_collisions += 1
            elif d < 0.5:
                closeness = max(0, 1.0 - d / 0.5)
                reward += R_PROXIMITY_HARD * closeness
            elif d < 1.5:
                closeness = max(0, 1.0 - d / 1.5)
                reward += R_PROXIMITY_SOFT * closeness

        # ── Anti-dangle: stuck near obstacle ──
        move_d = math.sqrt(
            (agent.x - agent.prev_pos[0]) ** 2 +
            (agent.y - agent.prev_pos[1]) ** 2)
        if lidar_min < 0.7:
            agent.near_obstacle_steps += 1
        else:
            agent.near_obstacle_steps = 0
        if agent.near_obstacle_steps > 3 and move_d < 0.05:
            reward += R_DANGLE
        if move_d > 0.15:
            agent.near_obstacle_steps = 0
        agent.prev_pos = (agent.x, agent.y)

        # ── Camera object detection (no oracle — just FOV-gated) ──
        dets = detect_in_fov(agent.x, agent.y, agent.yaw, self.objects)
        for det_name, det_type, det_bearing, det_dist in dets:
            # Use exact world position (training env is ground truth)
            world_angle = agent.yaw + det_bearing
            det_wx = agent.x + det_dist * math.cos(world_angle)
            det_wy = agent.y + det_dist * math.sin(world_angle)
            if det_name not in agent.detected_objects and det_name not in self.shared.collected:
                agent.detected_objects[det_name] = (
                    det_wx, det_wy, det_type, self.step_count)
                self.shared.broadcasts.append(
                    (agent.idx, det_type, det_wx, det_wy, self.step_count))
                reward += R_BROADCAST_BONUS
                self._ep_broadcasts += 1

        # ── Merge broadcasts from other agents ──
        for src_idx, btype, bx, by, btime in self.shared.broadcasts:
            if src_idx == agent.idx:
                continue
            matched_name = self._match_object(btype, bx, by)
            if matched_name and matched_name not in agent.detected_objects:
                agent.detected_objects[matched_name] = (bx, by, btype, btime)

        # ── Target approach shaping ──
        if agent.target is not None:
            tx, ty = agent.target
            d = math.sqrt((agent.x - tx) ** 2 + (agent.y - ty) ** 2)
            if agent.prev_target_dist is not None:
                delta = agent.prev_target_dist - d
                reward += R_APPROACH * delta
            agent.prev_target_dist = d

        # ── Auto-pick (only from objects this agent has detected) ──
        if agent.carrying == 0:
            for obj in self.objects:
                if not obj['alive'] or obj['name'] in self.shared.collected:
                    continue
                d = math.sqrt((agent.x - obj['x'])**2 + (agent.y - obj['y'])**2)
                if d < PICK_DIST:
                    agent.carrying = 1 if obj['type'] == 'cube' else 2
                    agent.carrying_name = obj['name']
                    agent.carry_steps = 0
                    obj['alive'] = False
                    self.shared.collected.add(obj['name'])
                    reward += R_PICK
                    self._ep_picks += 1
                    # Collab bonus
                    for other in self.agents:
                        if other.idx != agent.idx:
                            if obj['name'] in other.detected_objects:
                                reward += R_COLLAB_BONUS
                                self._ep_collabs += 1
                    break

        # ── Auto-drop ──
        if agent.carrying > 0:
            agent.carry_steps += 1
            basket = CUBE_BASKET if agent.carrying == 1 else SPHERE_BASKET
            d = math.sqrt((agent.x - basket[0])**2 + (agent.y - basket[1])**2)
            if d < BASKET_DIST:
                obj_type = 'cube' if agent.carrying == 1 else 'sphere'
                correct = CUBE_BASKET if obj_type == 'cube' else SPHERE_BASKET
                if np.allclose(basket, correct):
                    reward += R_DROP_CORRECT
                    self._ep_deliveries += 1
                    for obj in self.objects:
                        if obj['name'] == agent.carrying_name:
                            obj['placed'] = True
                else:
                    reward += R_DROP_WRONG
                agent.carrying = 0
                agent.carrying_name = None
                agent.carry_steps = 0

            # Carry overtime penalty
            if agent.carry_steps > CARRY_TIMEOUT_STEPS:
                overtime = agent.carry_steps - CARRY_TIMEOUT_STEPS
                reward += R_CARRY_OVERTIME * (overtime / CARRY_TIMEOUT_STEPS)

        # ── Jerk penalty ──
        jerk = np.abs(action - 2 * agent.prev_action + agent.prev_prev_action).mean()
        reward += R_JERK * jerk
        agent.prev_prev_action = agent.prev_action.copy()
        agent.prev_action = action.copy()

        # ── Reassign targets ──
        self._assign_targets()

        # ── Check done ──
        all_placed = all(obj['placed'] for obj in self.objects)
        if all_placed:
            reward += R_ALL_DONE

        self.step_count += 1
        terminated = all_placed
        truncated = self.step_count >= self.max_steps * N_ROBOTS

        # Cycle agent
        self.current_agent = (self.current_agent + 1) % N_ROBOTS
        obs = self._get_obs(self.current_agent)

        if self.render_mode == 'human':
            self._render_pygame()

        info = {}
        if terminated or truncated:
            info['is_success'] = bool(all_placed)
            info['picks'] = self._ep_picks
            info['deliveries'] = self._ep_deliveries
            info['collisions'] = self._ep_collisions
            info['broadcasts'] = self._ep_broadcasts
            info['collabs'] = self._ep_collabs

        return obs, float(reward), terminated, truncated, info

    # ── LIDAR ──

    def _get_lidar(self, agent):
        """Return normalized LiDAR array. Ray 0 = forward (heading), CCW."""
        lidar = np.zeros(LIDAR_RAYS, dtype=np.float32)
        for r in range(LIDAR_RAYS):
            angle = agent.yaw + (r / LIDAR_RAYS) * 2 * math.pi
            dist = raycast(
                agent.x, agent.y, angle, LIDAR_MAX_DIST,
                self.agents, self.objects, exclude_idx=agent.idx)
            lidar[r] = dist / LIDAR_MAX_DIST
        return lidar

    # ── OBSERVATION (43-dim, lean) ──

    def _get_obs(self, agent_idx):
        agent = self.agents[agent_idx]
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        idx = 0

        # [0:18] LiDAR
        lidar = self._get_lidar(agent)
        obs[idx:idx + LIDAR_RAYS] = lidar
        idx += LIDAR_RAYS

        # [18:27] Other robots: 3 × (rel_x, rel_y, dist)
        others = [a for a in self.agents if a.idx != agent_idx]
        for i in range(3):
            if i < len(others):
                other = others[i]
                dx = other.x - agent.x
                dy = other.y - agent.y
                d = math.sqrt(dx * dx + dy * dy)
                cos_y = math.cos(-agent.yaw)
                sin_y = math.sin(-agent.yaw)
                rel_x = dx * cos_y - dy * sin_y
                rel_y = dx * sin_y + dy * cos_y
                obs[idx]     = np.clip(rel_x / ARENA_SIZE, -1, 1)
                obs[idx + 1] = np.clip(rel_y / ARENA_SIZE, -1, 1)
                obs[idx + 2] = np.clip(d / ARENA_SIZE, 0, 1)
            idx += 3

        # [27:30] Target: (sin_angle, cos_angle, dist_norm)
        if agent.target is not None:
            tx, ty = agent.target
            dx, dy = tx - agent.x, ty - agent.y
            d = math.sqrt(dx * dx + dy * dy)
            angle = math.atan2(dy, dx) - agent.yaw
            obs[idx]     = math.sin(angle)
            obs[idx + 1] = math.cos(angle)
            obs[idx + 2] = np.clip(d / ARENA_SIZE, 0, 1)
        idx += 3

        # [30:39] Camera detections: 3 × (type_sign, bearing_norm, dist_norm)
        dets = detect_in_fov(agent.x, agent.y, agent.yaw, self.objects)
        for i in range(N_DETECTIONS):
            if i < len(dets):
                _, otype, bearing, d = dets[i]
                obs[idx]     = 1.0 if otype == 'cube' else -1.0
                obs[idx + 1] = bearing / math.pi
                obs[idx + 2] = np.clip(d / CAMERA_RANGE, 0, 1)
            idx += 3

        # [39:42] Basket: (sin_angle, cos_angle, dist_norm)
        if agent.carrying > 0:
            basket = CUBE_BASKET if agent.carrying == 1 else SPHERE_BASKET
        else:
            basket = CUBE_BASKET
        dx, dy = basket[0] - agent.x, basket[1] - agent.y
        d = math.sqrt(dx * dx + dy * dy)
        angle = math.atan2(dy, dx) - agent.yaw
        obs[idx]     = math.sin(angle)
        obs[idx + 1] = math.cos(angle)
        obs[idx + 2] = np.clip(d / ARENA_SIZE, 0, 1)
        idx += 3

        # [42] Carry overtime fraction
        if agent.carrying > 0:
            obs[idx] = min(agent.carry_steps / CARRY_TIMEOUT_STEPS, 1.0)
        else:
            obs[idx] = 0.0
        idx += 1

        return obs

    # ── TARGET ASSIGNMENT (no SLAM, random-walk exploration) ──

    def _assign_targets(self):
        """Assign navigation hints. Only uses objects the agent has detected."""
        for agent in self.agents:
            if agent.carrying > 0:
                basket = CUBE_BASKET if agent.carrying == 1 else SPHERE_BASKET
                agent.target = (float(basket[0]), float(basket[1]))
            else:
                # Only objects THIS agent knows about
                best_d, best_pos = float('inf'), None
                for name, (kx, ky, ktype, _) in agent.detected_objects.items():
                    if name in self.shared.collected:
                        continue
                    d = math.sqrt((agent.x - kx)**2 + (agent.y - ky)**2)
                    if d < best_d:
                        best_d = d
                        best_pos = (kx, ky)

                if best_pos is not None:
                    agent.target = best_pos
                else:
                    # Random-walk exploration
                    agent._explore_steps += 1
                    if (agent._explore_target is None or
                            agent._explore_steps >= EXPLORE_RETARGET_STEPS):
                        agent._explore_target = (
                            random.uniform(-ARENA_HALF + 1.0, ARENA_HALF - 1.0),
                            random.uniform(-ARENA_HALF + 1.0, ARENA_HALF - 1.0),
                        )
                        agent._explore_steps = 0
                    # Check if arrived
                    if agent._explore_target is not None:
                        d = math.sqrt(
                            (agent.x - agent._explore_target[0])**2 +
                            (agent.y - agent._explore_target[1])**2)
                        if d < 1.0:
                            agent._explore_target = (
                                random.uniform(-ARENA_HALF + 1.0, ARENA_HALF - 1.0),
                                random.uniform(-ARENA_HALF + 1.0, ARENA_HALF - 1.0),
                            )
                            agent._explore_steps = 0
                    agent.target = agent._explore_target

    # ── HELPERS ──

    def _match_object(self, det_type, wx, wy, tolerance=1.5):
        """Match a detection to the nearest world object by type and proximity."""
        best_name, best_d = None, float('inf')
        for obj in self.objects:
            if obj['type'] != det_type or not obj['alive']:
                continue
            if obj['name'] in self.shared.collected:
                continue
            d = math.sqrt((wx - obj['x'])**2 + (wy - obj['y'])**2)
            if d < best_d:
                best_name, best_d = obj['name'], d
        return best_name if best_d < tolerance else None

    # ── RENDERING ──

    def _render_pygame(self):
        if not HAS_PYGAME:
            return
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode(
                (self._screen_size, self._screen_size))
            pygame.display.set_caption('Swarm Collector Training')
            self._clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._screen = None
                import sys as _sys
                _sys.exit(0)

        self._screen.fill((20, 20, 40))
        S = self._px_scale
        H = ARENA_HALF

        def w2p(wx, wy):
            return int((wx + H) * S), int((H - wy) * S)

        # Obstacles
        for ox, oy, hw, hh in OBSTACLES:
            px, py = w2p(ox - hw, oy + hh)
            pygame.draw.rect(self._screen, (120, 120, 140),
                             (px, py, int(2 * hw * S), int(2 * hh * S)))

        # Baskets
        px, py = w2p(*CUBE_BASKET)
        pygame.draw.rect(self._screen, (0, 200, 0),
                         (px - 15, py - 15, 30, 30), 3)
        px, py = w2p(SPHERE_BASKET[0], SPHERE_BASKET[1])
        pygame.draw.rect(self._screen, (200, 200, 0),
                         (px - 15, py - 15, 30, 30), 3)

        # Objects
        for obj in self.objects:
            if not obj['alive']:
                continue
            px, py = w2p(obj['x'], obj['y'])
            color = (220, 50, 50) if obj['type'] == 'cube' else (50, 50, 220)
            pygame.draw.circle(self._screen, color, (px, py), 6)

        # Robots
        bot_colors = [
            (230, 70, 70), (70, 150, 230), (70, 200, 100), (240, 160, 50)]
        for agent in self.agents:
            px, py = w2p(agent.x, agent.y)
            c = bot_colors[agent.idx]
            pygame.draw.circle(self._screen, c, (px, py), int(ROBOT_RADIUS * S))
            hx = px + int(ROBOT_RADIUS * S * 1.3 * math.cos(
                -agent.yaw + math.pi / 2))
            hy = py + int(ROBOT_RADIUS * S * 1.3 * math.sin(
                -agent.yaw + math.pi / 2))
            pygame.draw.line(self._screen, (255, 255, 255), (px, py), (hx, hy), 2)
            if agent.target:
                tx, ty = w2p(*agent.target)
                pygame.draw.line(self._screen, c, (px, py), (tx, ty), 1)
            if agent.carrying > 0:
                carry_c = (255, 100, 100) if agent.carrying == 1 else (100, 100, 255)
                pygame.draw.circle(
                    self._screen, carry_c,
                    (px, py - int(ROBOT_RADIUS * S) - 5), 4)

        # HUD
        font = pygame.font.SysFont('monospace', 14)
        collected = sum(1 for obj in self.objects
                        if obj['name'] in self.shared.collected)
        placed = sum(1 for obj in self.objects if obj['placed'])
        text = (f'Step:{self.step_count}  '
                f'Picked:{collected}/{N_OBJECTS}  '
                f'Placed:{placed}/{N_OBJECTS}  '
                f'Coll:{self._ep_collisions}')
        surf = font.render(text, True, (200, 200, 200))
        self._screen.blit(surf, (10, 10))

        pygame.display.flip()
        self._clock.tick(self.metadata['render_fps'])

    def render(self):
        if self.render_mode == 'human':
            self._render_pygame()

    def close(self):
        if self._screen is not None:
            import pygame as _pg
            _pg.quit()
            self._screen = None

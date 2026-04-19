"""
Multi-agent swarm collector — Gymnasium environment with Pygame renderer.

CTDE: Shared policy, 4 agents.  The env cycles through agents each step.
From SB3's perspective this is a single-agent env.

Observation : 65-dim (lidar + other bots + target + self + detections + basket)
Action      : 3-dim continuous [vx, vy, wz] in [-1, 1]
"""

import math
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import *

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


# ═══════════════ HELPERS ═══════════════

def _norm_angle(a):
    while a > math.pi:  a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def _rect_contains(ox, oy, hw, hh, px, py, margin=0.0):
    return (abs(px - ox) < hw + margin) and (abs(py - oy) < hh + margin)


def _circles_collide(x1, y1, r1, x2, y2, r2):
    dx, dy = x2 - x1, y2 - y1
    return dx*dx + dy*dy < (r1 + r2) ** 2


# ═══════════════ AGENT STATE ═══════════════

class AgentState:
    def __init__(self, idx, x, y):
        self.idx = idx
        self.x, self.y = x, y
        self.yaw = 0.0
        self.vx, self.vy = 0.0, 0.0
        self.carrying = 0       # 0=none, 1=cube, 2=sphere
        self.carrying_name = None
        self.target = None      # (tx, ty) or None
        self.prev_target_dist = None
        self.prev_action = np.zeros(ACT_DIM)

    def reset(self, x, y):
        self.x, self.y = x, y
        self.yaw = random.uniform(-math.pi, math.pi)
        self.vx = self.vy = 0.0
        self.carrying = 0
        self.carrying_name = None
        self.target = None
        self.prev_target_dist = None
        self.prev_action = np.zeros(ACT_DIM)


# ═══════════════ ENVIRONMENT ═══════════════

class SwarmCollectorEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 20}

    def __init__(self, render_mode=None, max_steps=None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(-1, 1, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(ACT_DIM,), dtype=np.float32)

        self.agents = [AgentState(i, *ROBOT_STARTS[i]) for i in range(N_ROBOTS)]
        self.current_agent = 0
        self.step_count = 0
        self.max_steps = max_steps if max_steps is not None else MAX_STEPS

        # Objects: list of dicts
        self.objects = []
        self.collected_set = set()
        self.visited = set()

        # Pygame
        self._screen = None
        self._clock = None
        self._px_scale = 60  # pixels per metre
        self._screen_size = int(ARENA_SIZE * self._px_scale)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.current_agent = 0
        self.collected_set = set()
        self.visited = set()

        for i, agent in enumerate(self.agents):
            agent.reset(*ROBOT_STARTS[i])

        self.objects = []
        for name, otype, ox, oy in OBJECTS:
            self.objects.append({
                'name': name, 'type': otype,
                'x': ox, 'y': oy, 'alive': True, 'placed': False,
            })

        self._assign_targets()
        obs = self._get_obs(self.current_agent)
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        agent = self.agents[self.current_agent]

        # Apply action (scale to real units)
        cmd_vx = action[0] * MAX_SPEED
        cmd_vy = action[1] * MAX_SPEED
        cmd_wz = action[2] * MAX_WZ

        # Robot-frame → world-frame
        cos_y, sin_y = math.cos(agent.yaw), math.sin(agent.yaw)
        world_vx = cmd_vx * cos_y - cmd_vy * sin_y
        world_vy = cmd_vx * sin_y + cmd_vy * cos_y

        agent.vx, agent.vy = cmd_vx, cmd_vy
        new_x = agent.x + world_vx * DT
        new_y = agent.y + world_vy * DT
        agent.yaw = _norm_angle(agent.yaw + cmd_wz * DT)

        # ── Collision checks ──
        reward = R_IDLE
        wall_hit = False
        robot_hit = False

        # Walls
        margin = ROBOT_RADIUS
        if new_x < -ARENA_HALF + margin:
            new_x = -ARENA_HALF + margin; wall_hit = True
        if new_x >  ARENA_HALF - margin:
            new_x =  ARENA_HALF - margin; wall_hit = True
        if new_y < -ARENA_HALF + margin:
            new_y = -ARENA_HALF + margin; wall_hit = True
        if new_y >  ARENA_HALF - margin:
            new_y =  ARENA_HALF - margin; wall_hit = True

        # Obstacles
        for ox, oy, hw, hh in OBSTACLES:
            if _rect_contains(ox, oy, hw, hh, new_x, new_y, margin):
                wall_hit = True
                new_x = agent.x   # bounce back
                new_y = agent.y
                break

        if wall_hit:
            reward += R_WALL_COLLISION

        agent.x, agent.y = new_x, new_y

        # Robot-robot
        for other in self.agents:
            if other.idx == agent.idx:
                continue
            d = math.sqrt((agent.x - other.x)**2 + (agent.y - other.y)**2)
            if d < 2 * ROBOT_RADIUS:
                robot_hit = True
                reward += R_ROBOT_COLLISION
            elif d < 1.5:
                # Proximity penalty
                closeness = max(0, 1.0 - d / 1.5)
                reward += R_ROBOT_PROXIMITY * closeness

        # ── Zone exploration ──
        ci = int((agent.x + ARENA_HALF) / GRID_RES)
        cj = int((agent.y + ARENA_HALF) / GRID_RES)
        ci = max(0, min(GRID_NX - 1, ci))
        cj = max(0, min(GRID_NY - 1, cj))
        cell = (ci, cj)
        if cell not in self.visited:
            self.visited.add(cell)
            reward += R_NEW_ZONE

        # ── Target approach shaping ──
        if agent.target is not None:
            tx, ty = agent.target
            d = math.sqrt((agent.x - tx)**2 + (agent.y - ty)**2)
            if agent.prev_target_dist is not None:
                delta = agent.prev_target_dist - d
                reward += R_TARGET_APPROACH * delta
            agent.prev_target_dist = d

            if d < 0.5:
                reward += R_TARGET_REACHED

        # ── Auto-pick ──
        if agent.carrying == 0:
            for obj in self.objects:
                if not obj['alive'] or obj['name'] in self.collected_set:
                    continue
                d = math.sqrt((agent.x - obj['x'])**2 + (agent.y - obj['y'])**2)
                if d < PICK_DIST:
                    agent.carrying = 1 if obj['type'] == 'cube' else 2
                    agent.carrying_name = obj['name']
                    obj['alive'] = False
                    self.collected_set.add(obj['name'])
                    reward += R_PICK
                    break

        # ── Auto-drop ──
        if agent.carrying > 0:
            basket = CUBE_BASKET if agent.carrying == 1 else SPHERE_BASKET
            d = math.sqrt((agent.x - basket[0])**2 + (agent.y - basket[1])**2)
            if d < BASKET_DIST:
                # Check if correct basket
                obj_type = 'cube' if agent.carrying == 1 else 'sphere'
                correct_basket = CUBE_BASKET if obj_type == 'cube' else SPHERE_BASKET
                if np.allclose(basket, correct_basket):
                    reward += R_DROP_CORRECT
                    for obj in self.objects:
                        if obj['name'] == agent.carrying_name:
                            obj['placed'] = True
                else:
                    reward += R_DROP_WRONG
                agent.carrying = 0
                agent.carrying_name = None

        # ── Smooth action bonus ──
        action_diff = np.abs(action - agent.prev_action).mean()
        reward += R_SMOOTH * (1.0 - action_diff)
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

        # Cycle to next agent
        self.current_agent = (self.current_agent + 1) % N_ROBOTS

        # Physics update for ALL agents happens every N_ROBOTS steps
        obs = self._get_obs(self.current_agent)

        if self.render_mode == 'human':
            self._render_pygame()

        info = {}
        if terminated or truncated:
            info['is_success'] = bool(all_placed)

        return obs, float(reward), terminated, truncated, info

    # ── OBSERVATION ──

    def _get_obs(self, agent_idx):
        agent = self.agents[agent_idx]
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        idx = 0

        # 1) LiDAR (36 rays)
        for r in range(LIDAR_RAYS):
            angle = agent.yaw + (r / LIDAR_RAYS) * 2 * math.pi
            dist = self._raycast(agent.x, agent.y, angle, LIDAR_MAX_DIST,
                                 exclude_idx=agent_idx)
            obs[idx] = dist / LIDAR_MAX_DIST
            idx += 1

        # 2) Other robots relative (3 × 3 = 9)
        others = [a for a in self.agents if a.idx != agent_idx]
        for other in others[:3]:
            dx = other.x - agent.x
            dy = other.y - agent.y
            d = math.sqrt(dx*dx + dy*dy)
            # Rotate to agent frame
            cos_y, sin_y = math.cos(-agent.yaw), math.sin(-agent.yaw)
            rel_x = dx * cos_y - dy * sin_y
            rel_y = dx * sin_y + dy * cos_y
            obs[idx]     = np.clip(rel_x / ARENA_SIZE, -1, 1)
            obs[idx + 1] = np.clip(rel_y / ARENA_SIZE, -1, 1)
            obs[idx + 2] = np.clip(d / ARENA_SIZE, 0, 1)
            idx += 3

        # 3) Target relative (3)
        if agent.target is not None:
            tx, ty = agent.target
            dx, dy = tx - agent.x, ty - agent.y
            d = math.sqrt(dx*dx + dy*dy)
            angle = math.atan2(dy, dx) - agent.yaw
            obs[idx]     = math.sin(angle)
            obs[idx + 1] = math.cos(angle)
            obs[idx + 2] = np.clip(d / ARENA_SIZE, 0, 1)
        idx += 3

        # 4) Self state (5)
        obs[idx]     = agent.carrying / 2.0
        obs[idx + 1] = np.clip(agent.vx / MAX_SPEED, -1, 1)
        obs[idx + 2] = np.clip(agent.vy / MAX_SPEED, -1, 1)
        obs[idx + 3] = math.sin(agent.yaw)
        obs[idx + 4] = math.cos(agent.yaw)
        idx += 5

        # 5) Detections in FOV (3 × 3 = 9)
        dets = self._detect_objects(agent)
        for i in range(N_DETECTIONS):
            if i < len(dets):
                otype, bearing, dist = dets[i]
                obs[idx]     = 1.0 if otype == 'cube' else -1.0
                obs[idx + 1] = bearing / math.pi
                obs[idx + 2] = np.clip(dist / CAMERA_RANGE, 0, 1)
            idx += 3

        # 6) Basket relative (3)
        if agent.carrying > 0:
            basket = CUBE_BASKET if agent.carrying == 1 else SPHERE_BASKET
        else:
            basket = CUBE_BASKET  # default
        dx, dy = basket[0] - agent.x, basket[1] - agent.y
        d = math.sqrt(dx*dx + dy*dy)
        angle = math.atan2(dy, dx) - agent.yaw
        obs[idx]     = math.sin(angle)
        obs[idx + 1] = math.cos(angle)
        obs[idx + 2] = np.clip(d / ARENA_SIZE, 0, 1)
        idx += 3

        return obs

    def _raycast(self, ox, oy, angle, max_dist, exclude_idx=-1, step=0.1):
        """Simple raycasting for LiDAR simulation."""
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for s in range(1, int(max_dist / step) + 1):
            d = s * step
            px = ox + d * cos_a
            py = oy + d * sin_a
            # Walls
            if abs(px) > ARENA_HALF or abs(py) > ARENA_HALF:
                return d
            # Obstacles
            for bx, by, hw, hh in OBSTACLES:
                if _rect_contains(bx, by, hw, hh, px, py):
                    return d
            # Other robots
            for agent in self.agents:
                if agent.idx == exclude_idx:
                    continue
                if (px - agent.x)**2 + (py - agent.y)**2 < ROBOT_RADIUS**2:
                    return d
            # Objects
            for obj in self.objects:
                if not obj['alive']:
                    continue
                if (px - obj['x'])**2 + (py - obj['y'])**2 < 0.1**2:
                    return d
        return max_dist

    def _detect_objects(self, agent):
        """Detect objects within camera FOV. Returns [(type, bearing, dist)]."""
        dets = []
        half_fov = CAMERA_FOV / 2
        for obj in self.objects:
            if not obj['alive']:
                continue
            dx = obj['x'] - agent.x
            dy = obj['y'] - agent.y
            d = math.sqrt(dx*dx + dy*dy)
            if d > CAMERA_RANGE or d < 0.1:
                continue
            angle = _norm_angle(math.atan2(dy, dx) - agent.yaw)
            if abs(angle) < half_fov:
                dets.append((obj['type'], angle, d))
        dets.sort(key=lambda x: x[2])
        return dets[:N_DETECTIONS]

    # ── TARGET ASSIGNMENT ──

    def _assign_targets(self):
        """Deterministic target assignment for each agent."""
        for agent in self.agents:
            if agent.carrying > 0:
                # Deliver to basket
                basket = CUBE_BASKET if agent.carrying == 1 else SPHERE_BASKET
                agent.target = (float(basket[0]), float(basket[1]))
            else:
                # Find nearest alive object
                best_d, best_pos = float('inf'), None
                for obj in self.objects:
                    if not obj['alive'] or obj['name'] in self.collected_set:
                        continue
                    d = math.sqrt((agent.x - obj['x'])**2 +
                                  (agent.y - obj['y'])**2)
                    if d < best_d:
                        best_d = d
                        best_pos = (obj['x'], obj['y'])
                if best_pos is not None:
                    agent.target = best_pos
                else:
                    # Explore: nearest unvisited cell
                    best_d2 = float('inf')
                    for i in range(GRID_NX):
                        for j in range(GRID_NY):
                            if (i, j) in self.visited:
                                continue
                            cx = (i + 0.5) * GRID_RES - ARENA_HALF
                            cy = (j + 0.5) * GRID_RES - ARENA_HALF
                            d = math.sqrt((agent.x - cx)**2 +
                                          (agent.y - cy)**2)
                            if d < best_d2:
                                best_d2 = d
                                agent.target = (cx, cy)

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
                import sys; sys.exit(0)

        self._screen.fill((20, 20, 40))
        S = self._px_scale
        H = ARENA_HALF

        def w2p(wx, wy):
            return int((wx + H) * S), int((H - wy) * S)

        # Grid
        for i in range(GRID_NX + 1):
            x = int(i * GRID_RES * S)
            pygame.draw.line(self._screen, (40, 40, 60),
                             (x, 0), (x, self._screen_size))
        for j in range(GRID_NY + 1):
            y = int(j * GRID_RES * S)
            pygame.draw.line(self._screen, (40, 40, 60),
                             (0, y), (self._screen_size, y))

        # Visited cells
        for i, j in self.visited:
            rx = int(i * GRID_RES * S)
            ry = int((GRID_NY - 1 - j) * GRID_RES * S)
            surf = pygame.Surface((int(GRID_RES * S), int(GRID_RES * S)))
            surf.set_alpha(30)
            surf.fill((0, 200, 0))
            self._screen.blit(surf, (rx, ry))

        # Obstacles
        for ox, oy, hw, hh in OBSTACLES:
            px, py = w2p(ox - hw, oy + hh)
            pygame.draw.rect(self._screen, (120, 120, 140),
                             (px, py, int(2*hw*S), int(2*hh*S)))

        # Baskets
        px, py = w2p(*CUBE_BASKET)
        pygame.draw.rect(self._screen, (0, 200, 0),
                         (px-15, py-15, 30, 30), 3)
        px, py = w2p(SPHERE_BASKET[0], SPHERE_BASKET[1])
        pygame.draw.rect(self._screen, (200, 200, 0),
                         (px-15, py-15, 30, 30), 3)

        # Objects
        for obj in self.objects:
            if not obj['alive']:
                continue
            px, py = w2p(obj['x'], obj['y'])
            color = (220, 50, 50) if obj['type'] == 'cube' else (50, 50, 220)
            pygame.draw.circle(self._screen, color, (px, py), 6)

        # Robots
        bot_colors = [(230, 70, 70), (70, 150, 230), (70, 200, 100), (240, 160, 50)]
        for agent in self.agents:
            px, py = w2p(agent.x, agent.y)
            c = bot_colors[agent.idx]
            pygame.draw.circle(self._screen, c, (px, py), int(ROBOT_RADIUS * S))
            # heading indicator
            hx = px + int(ROBOT_RADIUS * S * 1.3 * math.cos(-agent.yaw + math.pi/2))
            hy = py + int(ROBOT_RADIUS * S * 1.3 * math.sin(-agent.yaw + math.pi/2))
            pygame.draw.line(self._screen, (255, 255, 255), (px, py), (hx, hy), 2)
            # target line
            if agent.target:
                tx, ty = w2p(*agent.target)
                pygame.draw.line(self._screen, (*c, 100), (px, py), (tx, ty), 1)

        # HUD
        font = pygame.font.SysFont('monospace', 14)
        text = f'Step: {self.step_count}  Collected: {len(self.collected_set)}/{N_OBJECTS}'
        surf = font.render(text, True, (200, 200, 200))
        self._screen.blit(surf, (10, 10))

        pygame.display.flip()
        self._clock.tick(self.metadata['render_fps'])

    def render(self):
        if self.render_mode == 'human':
            self._render_pygame()

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None

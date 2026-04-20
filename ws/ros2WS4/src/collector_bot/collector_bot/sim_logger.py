"""
Simulation logger — records all swarm data, generates HTML report on shutdown.

Launch alongside robots.  On SIGINT it saves:
  simulate/logN/  →  trajectory_map.png, collisions.png, objects_timeline.png,
                      semantic_map.png, camera_snapshots/, metrics.json, report.html
"""

import os
import json
import time
import signal
import math
from datetime import datetime

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import cv2
import numpy as np

# ── Arena knowledge (must match config.py / arena.world) ──
ARENA_HALF = 6.0
ARENA_SIZE = 12.0
GRID_RES = 2.0
GRID_NX = int(ARENA_SIZE / GRID_RES)
GRID_NY = int(ARENA_SIZE / GRID_RES)
TOTAL_CELLS = GRID_NX * GRID_NY
OBJECTS = [
    ('cube_1',   'cube',    1.5,   1.5),
    ('cube_2',   'cube',   -3.0,   0.5),
    ('cube_3',   'cube',    4.0,  -3.0),
    ('cube_4',   'cube',   -1.0,  -3.5),
    ('cube_5',   'cube',    2.0,   3.5),
    ('sphere_1', 'sphere', -2.0,   3.0),
    ('sphere_2', 'sphere',  1.0,  -1.5),
    ('sphere_3', 'sphere', -4.0,  -1.0),
    ('sphere_4', 'sphere',  3.5,   1.0),
    ('sphere_5', 'sphere', -0.5,   4.0),
]
CUBE_BASKET   = (5.0,  5.0)
SPHERE_BASKET = (-5.0, 5.0)
OBSTACLES = [
    ( 3.0,  0.0, 0.25, 1.0),
    (-2.0, -2.0, 0.25, 1.0),
    ( 0.0,  3.0, 1.0,  0.25),
    (-4.0,  2.0, 0.25, 1.0),
]
ROBOT_NAMES = ['robot_1', 'robot_2', 'robot_3', 'robot_4']
COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']


class SimLogger(Node):
    def __init__(self):
        super().__init__('sim_logger')
        self.bridge = CvBridge()
        self.t0 = time.time()
        self._reported = False  # guard: only generate report once

        # ── data stores ──
        self.trails = {n: [] for n in ROBOT_NAMES}      # [(t, x, y)]
        self.collisions = {n: 0 for n in ROBOT_NAMES}
        self.events = []                                  # (t, event_type, data)
        self.visited_cells = set()
        self.snapshots = {n: [] for n in ROBOT_NAMES}     # [(t, cv_img)]
        self._snap_interval = 30.0
        self._last_snap = {n: 0.0 for n in ROBOT_NAMES}

        # ── subscribers ──
        for rn in ROBOT_NAMES:
            self.create_subscription(
                Odometry, f'/{rn}/odom',
                lambda msg, r=rn: self._odom_cb(r, msg), 10)
            self.create_subscription(
                Image, f'/{rn}/camera/image_raw',
                lambda msg, r=rn: self._image_cb(r, msg), 5)

        self.create_subscription(String, '/collision_stats', self._collision_cb, 10)
        self.create_subscription(String, '/claimed', self._claimed_cb, 10)
        self.create_subscription(String, '/collected', self._collected_cb, 10)
        self.create_subscription(String, '/dropped', self._dropped_cb, 10)
        self.create_subscription(String, '/visited_zones', self._visited_cb, 10)

        self.get_logger().info('SimLogger ready — will save report on shutdown')

    # ── callbacks ──

    def _odom_cb(self, rname, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.trails[rname].append((time.time() - self.t0, x, y))

    def _image_cb(self, rname, msg):
        now = time.time()
        if now - self._last_snap[rname] < self._snap_interval:
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.snapshots[rname].append((now - self.t0, img))
            self._last_snap[rname] = now
        except Exception:
            pass

    def _collision_cb(self, msg):
        parts = msg.data.split(':')
        if len(parts) == 2 and parts[0] in self.collisions:
            try:
                self.collisions[parts[0]] = int(parts[1])
            except ValueError:
                pass

    def _claimed_cb(self, msg):
        self.events.append((time.time() - self.t0, 'claimed', msg.data))

    def _collected_cb(self, msg):
        self.events.append((time.time() - self.t0, 'collected', msg.data))

    def _dropped_cb(self, msg):
        self.events.append((time.time() - self.t0, 'dropped', msg.data))

    def _visited_cb(self, msg):
        for part in msg.data.split(','):
            if ':' in part:
                a, b = part.split(':')
                try:
                    self.visited_cells.add((int(a), int(b)))
                except ValueError:
                    pass

    # ── report generation ──

    def generate_report(self):
        if self._reported:
            return
        self._reported = True
        log_dir = self._next_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        snap_dir = os.path.join(log_dir, 'camera_snapshots')
        os.makedirs(snap_dir, exist_ok=True)

        duration = time.time() - self.t0
        self.get_logger().info(f'Generating report in {log_dir}')

        # save camera snapshots
        for rn, snaps in self.snapshots.items():
            for idx, (t, img) in enumerate(snaps):
                path = os.path.join(snap_dir, f'{rn}_{idx:03d}_t{t:.0f}s.jpg')
                cv2.imwrite(path, img)

        # metrics
        collected = [e[2] for e in self.events if e[1] == 'collected']
        dropped = [e[2] for e in self.events if e[1] == 'dropped']
        
        coverage = (len(self.visited_cells) / TOTAL_CELLS * 100
                if TOTAL_CELLS > 0 else 0.0)

        metrics = {
            'duration_s': round(duration, 1),
            'objects_picked_up': len(collected),
            'objects_delivered': len(dropped),
            'collisions': self.collisions,
            'coverage_pct': round(coverage, 1),
            'timestamp': datetime.now().isoformat(),
        }
        with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        if HAS_MPL:
            self._plot_trajectories(log_dir)
            self._plot_collisions(log_dir)
            self._plot_timeline(log_dir)
            self._plot_semantic(log_dir)

        self._write_html(log_dir, metrics)
        self.get_logger().info(f'✓ Report saved to {log_dir}')

    def _next_log_dir(self):
        base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))),
            '..', '..', 'simulate')
        # Try to find the workspace root
        ws = os.environ.get('COLCON_PREFIX_PATH', '')
        if ws:
            base = os.path.join(os.path.dirname(ws), 'simulate')
        else:
            base = os.path.expanduser('~/Desktop/ros2WS4/simulate')
        n = 1
        while os.path.exists(os.path.join(base, f'log{n}')):
            n += 1
        return os.path.join(base, f'log{n}')

    def _plot_trajectories(self, log_dir):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_xlim(-ARENA_HALF, ARENA_HALF)
        ax.set_ylim(-ARENA_HALF, ARENA_HALF)
        ax.set_aspect('equal')
        ax.set_title('Robot Trajectories')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        # arena walls
        ax.add_patch(plt.Rectangle((-ARENA_HALF, -ARENA_HALF),
                                    2*ARENA_HALF, 2*ARENA_HALF,
                                    fill=False, edgecolor='gray', lw=2))
        # obstacles
        for ox, oy, hw, hh in OBSTACLES:
            ax.add_patch(plt.Rectangle((ox-hw, oy-hh), 2*hw, 2*hh,
                                        color='#7f8c8d', alpha=0.6))
        # baskets
        ax.plot(*CUBE_BASKET, 's', color='green', ms=12, label='Cube basket')
        ax.plot(*SPHERE_BASKET, 's', color='gold', ms=12, label='Sphere basket')

        # objects
        for name, otype, ox, oy in OBJECTS:
            c = 'red' if otype == 'cube' else 'blue'
            ax.plot(ox, oy, 'o', color=c, ms=6, alpha=0.5)

        # trails
        for i, rn in enumerate(ROBOT_NAMES):
            pts = self.trails[rn]
            if not pts:
                continue
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            ax.plot(xs, ys, '-', color=COLORS[i], alpha=0.7, lw=1, label=rn)
            ax.plot(xs[-1], ys[-1], 'D', color=COLORS[i], ms=8)

        ax.legend(loc='upper left', fontsize=8)
        fig.savefig(os.path.join(log_dir, 'trajectory_map.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    def _plot_collisions(self, log_dir):
        fig, ax = plt.subplots(figsize=(6, 4))
        names = list(self.collisions.keys())
        vals = [self.collisions[n] for n in names]
        bars = ax.bar(names, vals, color=COLORS[:len(names)])
        ax.set_title('Collisions per Robot')
        ax.set_ylabel('Count')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(v), ha='center', fontsize=10)
        fig.savefig(os.path.join(log_dir, 'collisions.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    def _plot_timeline(self, log_dir):
        fig, ax = plt.subplots(figsize=(10, 4))
        obj_names = [o[0] for o in OBJECTS]
        for i, on in enumerate(obj_names):
            claimed_t = None
            collected_t = None
            for t, etype, data in self.events:
                if data == on:
                    if etype == 'claimed' and claimed_t is None:
                        claimed_t = t
                    elif etype == 'collected' and collected_t is None:
                        collected_t = t
            if claimed_t is not None:
                end = collected_t if collected_t else claimed_t + 5
                ax.barh(i, end - claimed_t, left=claimed_t, height=0.5,
                        color='#3498db', alpha=0.7)
            if collected_t is not None:
                ax.plot(collected_t, i, 'y*', ms=8, label='Picked Up' if 'Picked Up' not in ax.get_legend_handles_labels()[1] else '')
                
            # Plot dropped event
            dropped_t = None
            for t, etype, data in self.events:
                if data == on and etype == 'dropped':
                    dropped_t = t
                    break
            if dropped_t is not None:
                ax.plot(dropped_t, i, 'g*', ms=12, label='Delivered' if 'Delivered' not in ax.get_legend_handles_labels()[1] else '')

        ax.set_yticks(range(len(obj_names)))
        ax.set_yticklabels(obj_names, fontsize=8)
        ax.set_xlabel('Time (s)')
        ax.set_title('Object Lifecycle Timeline')
        fig.savefig(os.path.join(log_dir, 'objects_timeline.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    def _plot_semantic(self, log_dir):
        fig, ax = plt.subplots(figsize=(8, 8))
        grid = np.zeros((GRID_NY, GRID_NX))
        for i, j in self.visited_cells:
            if 0 <= i < GRID_NX and 0 <= j < GRID_NY:
                grid[j, i] = 1.0
        ax.imshow(grid, extent=(-ARENA_HALF, ARENA_HALF, -ARENA_HALF, ARENA_HALF),
                  origin='lower', cmap='Greens', alpha=0.4, vmin=0, vmax=1)
        # objects
        for name, otype, ox, oy in OBJECTS:
            c = 'red' if otype == 'cube' else 'blue'
            ax.plot(ox, oy, 'o', color=c, ms=10)
            ax.annotate(name, (ox, oy), fontsize=6, ha='center', va='bottom')
        ax.set_xlim(-ARENA_HALF, ARENA_HALF)
        ax.set_ylim(-ARENA_HALF, ARENA_HALF)
        ax.set_aspect('equal')
        cov_pct = (len(self.visited_cells) / TOTAL_CELLS * 100
               if TOTAL_CELLS > 0 else 0.0)
        ax.set_title(f'Semantic Map (coverage: {cov_pct:.0f}%)')
        fig.savefig(os.path.join(log_dir, 'semantic_map.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    def _write_html(self, log_dir, metrics):
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Simulation Report — {metrics['timestamp']}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; max-width: 1000px; margin: auto;
         padding: 20px; background: #1a1a2e; color: #e0e0e0; }}
  h1 {{ color: #00d2ff; border-bottom: 2px solid #00d2ff; padding-bottom: 10px; }}
  h2 {{ color: #a8e6cf; }}
  .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;
              margin: 20px 0; }}
  .metric {{ background: #16213e; border-radius: 12px; padding: 20px;
             text-align: center; border: 1px solid #0f3460; }}
  .metric .value {{ font-size: 2em; color: #00d2ff; font-weight: bold; }}
  .metric .label {{ font-size: 0.9em; color: #a0a0a0; }}
  img {{ max-width: 100%; border-radius: 8px; margin: 10px 0;
         border: 1px solid #333; }}
  .gallery {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
</style></head><body>
<h1>🤖 Swarm Simulation Report</h1>
<p>Generated: {metrics['timestamp']} | Duration: {metrics['duration_s']}s</p>

<div class="metrics">
  <div class="metric"><div class="value">{metrics['objects_picked_up']}</div>
    <div class="label">Objects Picked Up</div></div>
  <div class="metric"><div class="value">{metrics['objects_delivered']}</div>
    <div class="label">Objects Delivered</div></div>
  <div class="metric"><div class="value">{metrics['coverage_pct']}%</div>
    <div class="label">Area Coverage</div></div>
  <div class="metric"><div class="value">{sum(metrics['collisions'].values())}</div>
    <div class="label">Total Collisions</div></div>
  <div class="metric"><div class="value">{metrics['duration_s']:.0f}s</div>
    <div class="label">Duration</div></div>
  <div class="metric"><div class="value">4</div>
    <div class="label">Robots</div></div>
</div>

<h2>📍 Trajectory Map</h2>
<img src="trajectory_map.png" alt="Trajectories">

<h2>💥 Collisions</h2>
<img src="collisions.png" alt="Collisions">

<h2>📦 Object Timeline</h2>
<img src="objects_timeline.png" alt="Timeline">

<h2>🗺️ Semantic Map</h2>
<img src="semantic_map.png" alt="Semantic">

<h2>📸 Camera Snapshots</h2>
<div class="gallery">"""
        snap_dir = os.path.join(log_dir, 'camera_snapshots')
        if os.path.exists(snap_dir):
            for fn in sorted(os.listdir(snap_dir))[:12]:
                html += f'\n  <img src="camera_snapshots/{fn}" alt="{fn}">'
        html += """
</div>
</body></html>"""
        with open(os.path.join(log_dir, 'report.html'), 'w') as f:
            f.write(html)


def main(args=None):
    rclpy.init(args=args)
    node = SimLogger()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.generate_report()
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.try_shutdown()

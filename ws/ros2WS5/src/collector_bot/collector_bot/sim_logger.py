"""
Simulation logger — records all swarm data, generates HTML report on shutdown.

Launch alongside robots.  On SIGINT it saves:
  simulate/logN/  →  trajectory_map.png, collisions.png, objects_timeline.png,
                      semantic_map.png, velocity_profiles.png, coverage_timeline.png,
                      camera_snapshots/, metrics.json, events.jsonl, report.html
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
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
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

from collector_bot.constants import (
    ARENA_HALF, ARENA_SIZE,
    OBJECT_DEFS, N_OBJECTS, CUBE_BASKET, SPHERE_BASKET,
    OBSTACLES, N_ROBOTS,
)
from collector_bot.paths import SIMULATE_DIR

ROBOT_NAMES = [f'robot_{i+1}' for i in range(N_ROBOTS)]
COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']


class SimLogger(Node):
    def __init__(self):
        super().__init__('sim_logger')

        self.declare_parameter('log_dir', '')
        self.bridge = CvBridge()
        self.t0 = time.time()
        self._reported = False  # guard: only generate report once

        # ── data stores ──
        self.trails  = {n: [] for n in ROBOT_NAMES}      # [(t, x, y)]
        self.velocities = {n: [] for n in ROBOT_NAMES}   # [(t, vx, vy, wz)]
        self.lidar_mins = {n: [] for n in ROBOT_NAMES}   # [(t, min_range)]
        self.collisions = {n: 0 for n in ROBOT_NAMES}
        self.events  = []                                  # (t, event_type, data)
        self.snapshots = {n: [] for n in ROBOT_NAMES}     # [(t, cv_img)]
        self.known_objects_log = []                        # [(t, msg)]
        self._snap_interval = 30.0
        self._last_snap = {n: 0.0 for n in ROBOT_NAMES}

        # Per-robot event counters
        self.pick_events  = {n: [] for n in ROBOT_NAMES}   # [(t, obj_name)]
        self.drop_events  = {n: [] for n in ROBOT_NAMES}   # [(t, obj_name)]

        # ── subscribers ──
        for rn in ROBOT_NAMES:
            self.create_subscription(
                Odometry, f'/{rn}/odom',
                lambda msg, r=rn: self._odom_cb(r, msg), 10)
            self.create_subscription(
                Image, f'/{rn}/camera/image_raw',
                lambda msg, r=rn: self._image_cb(r, msg), 5)
            self.create_subscription(
                Twist, f'/{rn}/cmd_vel',
                lambda msg, r=rn: self._cmd_vel_cb(r, msg), 10)
            self.create_subscription(
                LaserScan, f'/{rn}/scan',
                lambda msg, r=rn: self._scan_cb(r, msg), 10)

        self.create_subscription(String, '/collision_stats', self._collision_cb, 10)
        self.create_subscription(String, '/claimed', self._claimed_cb, 10)
        self.create_subscription(String, '/collected', self._collected_cb, 10)
        self.create_subscription(String, '/dropped', self._dropped_cb, 10)
        self.create_subscription(String, '/known_objects', self._known_objects_cb, 10)

        self.get_logger().info('SimLogger ready — will save report on shutdown')

    # ── callbacks ──

    def _odom_cb(self, rname, msg):
        t = time.time() - self.t0
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.trails[rname].append((t, x, y))

    def _cmd_vel_cb(self, rname, msg):
        t = time.time() - self.t0
        self.velocities[rname].append(
            (t, msg.linear.x, msg.linear.y, msg.angular.z))

    def _scan_cb(self, rname, msg):
        t = time.time() - self.t0
        ranges = [r for r in msg.ranges if not math.isinf(r) and not math.isnan(r)]
        if ranges:
            self.lidar_mins[rname].append((t, min(ranges)))

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

    def _known_objects_cb(self, msg):
        t = time.time() - self.t0
        self.known_objects_log.append((t, msg.data))
        self.events.append((t, 'broadcast', msg.data))

    # ── report generation ──

    def generate_report(self):
        if self._reported:
            return
        self._reported = True

        log_dir = self._next_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        snap_dir = os.path.join(log_dir, 'camera_snapshots')
        os.makedirs(snap_dir, exist_ok=True)
        self.get_logger().info(f'Generating report in {log_dir}')

        duration = time.time() - self.t0
        metrics = self._compute_metrics(duration)

        # Save metrics
        with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save events (JSONL)
        with open(os.path.join(log_dir, 'events.jsonl'), 'w') as f:
            for t, etype, data in self.events:
                f.write(json.dumps({
                    't': round(t, 2), 'type': etype, 'data': data}) + '\n')

        # Save camera snapshots
        for rn in ROBOT_NAMES:
            for i, (t, img) in enumerate(self.snapshots[rn]):
                fname = f'{rn}_t{int(t):04d}_{i:02d}.jpg'
                cv2.imwrite(os.path.join(snap_dir, fname), img)

        # Plots
        if HAS_MPL:
            self._plot_trajectories(log_dir, metrics)
            self._plot_collisions(log_dir)
            self._plot_timeline(log_dir)
            self._plot_semantic(log_dir)
            self._plot_velocities(log_dir)

        # HTML
        self._write_html(log_dir, metrics)
        self.get_logger().info(f'✓ Report written to {log_dir}')

    def _next_log_dir(self):
        base = self.get_parameter('log_dir').value or SIMULATE_DIR
        os.makedirs(base, exist_ok=True)
        n = 1
        while os.path.exists(os.path.join(base, f'log{n}')):
            n += 1
        return os.path.join(base, f'log{n}')

    def _compute_metrics(self, duration):
        picked = set(d for t, e, d in self.events if e == 'collected')
        dropped = set(d for t, e, d in self.events if e == 'dropped')
        broadcast_count = sum(1 for t, e, d in self.events if e == 'broadcast')

        # Per-robot distance
        per_robot = {}
        for rn in ROBOT_NAMES:
            trail = self.trails[rn]
            dist = 0.0
            for i in range(1, len(trail)):
                _, x0, y0 = trail[i - 1]
                _, x1, y1 = trail[i]
                dist += math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            robot_picks = sum(
                1 for t, e, d in self.events
                if e == 'collected' and d.startswith(rn.replace('robot_', '')))
            per_robot[rn] = {
                'distance_m': round(dist, 1),
                'collisions': self.collisions.get(rn, 0),
            }

        return {
            'duration_s': round(duration, 1),
            'objects_picked_up': len(picked),
            'objects_delivered': len(dropped),
            'completion_ratio': round(len(dropped) / max(N_OBJECTS, 1), 2),
            'collisions': dict(self.collisions),
            'total_collisions': sum(self.collisions.values()),
            'broadcast_events': broadcast_count,
            'per_robot': per_robot,
            'timestamp': datetime.now().isoformat(),
        }

    # ── plots ──

    def _arena_fig(self, title=''):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-ARENA_HALF - 0.5, ARENA_HALF + 0.5)
        ax.set_ylim(-ARENA_HALF - 0.5, ARENA_HALF + 0.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14)
        ax.add_patch(mpatches.Rectangle(
            (-ARENA_HALF, -ARENA_HALF), ARENA_SIZE, ARENA_SIZE,
            edgecolor='black', facecolor='#f5f5f5', linewidth=2))
        for ox, oy, hw, hh in OBSTACLES:
            ax.add_patch(mpatches.Rectangle(
                (ox - hw, oy - hh), 2 * hw, 2 * hh,
                facecolor='#bdc3c7', edgecolor='#7f8c8d'))
        ax.plot(*CUBE_BASKET, 's', color='green', markersize=12,
                label='Cube basket')
        ax.plot(*SPHERE_BASKET, 's', color='goldenrod', markersize=12,
                label='Sphere basket')
        return fig, ax

    def _plot_trajectories(self, log_dir, metrics):
        fig, ax = self._arena_fig('Robot Trajectories')
        for i, rn in enumerate(ROBOT_NAMES):
            trail = self.trails[rn]
            if not trail:
                continue
            xs = [p[1] for p in trail]
            ys = [p[2] for p in trail]
            ax.plot(xs, ys, alpha=0.6, linewidth=1.0, color=COLORS[i], label=rn)
            ax.plot(xs[0], ys[0], 'o', color=COLORS[i], markersize=8)
        for name, otype in OBJECT_DEFS:
            c = '#e74c3c' if otype == 'cube' else '#3498db'
            # Object positions are randomized, can't plot nominal positions
        ax.legend(loc='upper right', fontsize=9)
        fig.savefig(os.path.join(log_dir, 'trajectory_map.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_collisions(self, log_dir):
        fig, ax = plt.subplots(figsize=(8, 4))
        names = list(self.collisions.keys())
        vals = [self.collisions[n] for n in names]
        bars = ax.bar(names, vals, color=COLORS[:len(names)])
        ax.set_ylabel('Collisions')
        ax.set_title('Collisions per Robot')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    str(v), ha='center')
        fig.savefig(os.path.join(log_dir, 'collisions.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_timeline(self, log_dir):
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, (t, etype, data) in enumerate(self.events):
            color = {'claimed': '#f39c12', 'collected': '#27ae60',
                     'dropped': '#2980b9', 'broadcast': '#8e44ad'}.get(
                         etype, '#bdc3c7')
            ax.barh(0, 0.5, left=t, height=0.3, color=color, alpha=0.7)
            ax.text(t, 0.2, f'{etype[:4]}', fontsize=6, rotation=45)
        ax.set_xlabel('Time (s)')
        ax.set_title('Object Events Timeline')
        ax.set_yticks([])
        fig.savefig(os.path.join(log_dir, 'objects_timeline.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_semantic(self, log_dir):
        fig, ax = self._arena_fig('Exploration Coverage')
        for (i, j) in self.visited_cells:
            x0 = i * GRID_RES - ARENA_HALF
            y0 = j * GRID_RES - ARENA_HALF
            ax.add_patch(mpatches.Rectangle(
                (x0, y0), GRID_RES, GRID_RES,
                facecolor='#27ae60', alpha=0.25))
        fig.savefig(os.path.join(log_dir, 'semantic_map.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_velocities(self, log_dir):
        """Plot velocity profiles (vx, vy) over time per robot."""
        fig, axes = plt.subplots(len(ROBOT_NAMES), 1, figsize=(12, 3 * len(ROBOT_NAMES)),
                                 sharex=True)
        if len(ROBOT_NAMES) == 1:
            axes = [axes]
        for i, rn in enumerate(ROBOT_NAMES):
            ax = axes[i]
            data = self.velocities[rn]
            if not data:
                ax.set_title(f'{rn} — no velocity data')
                continue
            ts = [d[0] for d in data]
            vxs = [d[1] for d in data]
            vys = [d[2] for d in data]
            ax.plot(ts, vxs, alpha=0.7, label='vx', color=COLORS[i])
            ax.plot(ts, vys, alpha=0.5, label='vy', linestyle='--',
                    color=COLORS[i])
            ax.set_ylabel('m/s')
            ax.set_title(f'{rn} Velocity')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time (s)')
        fig.tight_layout()
        fig.savefig(os.path.join(log_dir, 'velocity_profiles.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)



    def _write_html(self, log_dir, metrics):
        imgs = [f for f in os.listdir(log_dir) if f.endswith('.png')]
        dur = metrics['duration_s']
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Simulation Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 2em; background: #fafafa; }}
  h1 {{ color: #2c3e50; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1em; }}
  .grid img {{ width: 100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
  .metrics {{ background: #ecf0f1; padding: 1em; border-radius: 6px; margin-bottom: 1.5em; }}
  .metrics span {{ display: inline-block; min-width: 160px; font-weight: bold; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1em; }}
  td, th {{ border: 1px solid #bdc3c7; padding: 6px 12px; text-align: left; }}
  th {{ background: #34495e; color: white; }}
</style></head><body>
<h1>🤖 Swarm Simulation Report</h1>
<div class="metrics">
  <p><span>Duration:</span> {dur}s</p>
  <p><span>Objects picked:</span> {metrics['objects_picked_up']} / {N_OBJECTS}</p>
  <p><span>Objects delivered:</span> {metrics['objects_delivered']} / {N_OBJECTS}</p>
  <p><span>Completion:</span> {metrics['completion_ratio'] * 100:.0f}%</p>
  <p><span>Total collisions:</span> {metrics['total_collisions']}</p>
  <p><span>Broadcasts:</span> {metrics['broadcast_events']}</p>
</div>

<h2>Per-Robot Stats</h2>
<table>
  <tr><th>Robot</th><th>Distance (m)</th><th>Collisions</th></tr>
"""
        for rn in ROBOT_NAMES:
            pr = metrics['per_robot'].get(rn, {})
            html += f"  <tr><td>{rn}</td><td>{pr.get('distance_m', 0)}</td>"
            html += f"<td>{pr.get('collisions', 0)}</td></tr>\n"

        html += """</table>

<h2>Visualizations</h2>
<div class="grid">
"""
        for img in sorted(imgs):
            html += f'  <img src="{img}" alt="{img}">\n'
        html += """</div>
</body></html>"""

        with open(os.path.join(log_dir, 'report.html'), 'w') as f:
            f.write(html)


def main(args=None):
    rclpy.init(args=args)
    node = SimLogger()

    def _shutdown_handler(signum, frame):
        node.generate_report()
        node.destroy_node()
        rclpy.try_shutdown()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        if not node._reported:
            node.generate_report()
        node.destroy_node()
        rclpy.try_shutdown()

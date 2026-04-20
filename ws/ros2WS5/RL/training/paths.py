"""Centralized path config for RL training. All file paths defined here."""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))          # RL/training/
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))    # ros2WS5/

MODEL_DIR    = os.path.join(_ROOT, 'RL', 'model')
LOG_BASE_DIR = os.path.join(_HERE, 'logs')
SIMULATE_DIR = os.path.join(_ROOT, 'simulate')


def model_path(name='policy'):
    """Full path to a model zip (without .zip extension)."""
    return os.path.join(MODEL_DIR, name)


def next_run_dir():
    """Create and return a timestamped run directory under logs/."""
    import datetime
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    d = os.path.join(LOG_BASE_DIR, f'run_{ts}')
    os.makedirs(d, exist_ok=True)
    return d

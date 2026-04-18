import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/silver/Desktop/silver_quick/cs671_2026_hack/swarm_ws/install/swarm_nav'

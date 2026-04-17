import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/akshatidk29/Desktop/ros2WS2/install/collector_bot'

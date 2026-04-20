import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'collector_bot'

setup(
    name=package_name,
    version='0.3.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='akshatidk29',
    maintainer_email='akshatidk29@todo.todo',
    description='RL-augmented omni-drive collector swarm with simulation reporting',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'swarm_brain = collector_bot.swarm_brain:main',
            'sim_logger = collector_bot.sim_logger:main',
            'safety_coordinator = collector_bot.safety_coordinator:main',
        ],
    },
)

import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'bot'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Ament index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # Package manifest
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # URDF / xacro models
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*')),
        # World files
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='akshatidk29',
    maintainer_email='akshatidk29@todo.todo',
    description='Multi-robot swarm with LiDAR obstacle avoidance',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'avoid = bot.avoid:main',
        ],
    },
)

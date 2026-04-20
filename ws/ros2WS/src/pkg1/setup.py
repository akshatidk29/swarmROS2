from setuptools import find_packages, setup

package_name = 'pkg1'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='akshatidk29',
    maintainer_email='akshatidk29@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hello = pkg1.hello:main',
            'talker = pkg1.talker:main',
            'listener = pkg1.listener:main',
        ],
    },
)

from setuptools import setup, find_packages

package_name = 'pioneer3at_gz'

setup(
    name=package_name,
    version='0.0.1',

    
    packages=find_packages(where='.'),

    package_dir={'': '.'},

    data_files=[
        ('share/' + package_name + '/launch', ['launch/pioneer3at.launch.py']),
        ('share/' + package_name, ['package.xml']),
    ],

    install_requires=['setuptools'],
    zip_safe=True,

    entry_points={
        'console_scripts': [
            'pioneer_controller = pioneer_controller:main',
            'fire_detection = fire_node.fire_detection_node:main',
        ],
    },
)


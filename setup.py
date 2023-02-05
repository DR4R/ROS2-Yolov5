from setuptools import setup
import glob
import os

package_name = 'yolov5'
share_dir = 'share/' + package_name

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (share_dir, ['package.xml']),
        (share_dir + '/launch' , glob.glob(os.path.join('launch', '*.launch.py'))),
        (share_dir + '/param' , glob.glob(os.path.join('param', '*.yaml')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='michael',
    maintainer_email='kimh060612@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'img_publisher = yolov5.camera:main',
            'image_subscriber = yolov5.imageview:main',
            'yolov5 = yolov5.yolov5:main'
        ],
    },
)

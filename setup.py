from setuptools import setup
import filament_annotation_3d

setup(
    name='filament_annotator_3d',
    version=filament_annotation_3d.__version__,
    author="Anna Medyukhina",
    author_email='anna.medyukhina@gmail.com',
    packages=['filament_annotation_3d',
              'filament_annotation_3d.utils'
              ],
    license='Apache License 2.0',
    include_package_data=True,

    install_requires=[
        'numpy',
        'scipy',
        'ddt',
        'pytest',
        'sklearn',
        'pandas',
        'Geometry3D',
        'jupyter',
        'jupyterlab<4.0',
    ],
)

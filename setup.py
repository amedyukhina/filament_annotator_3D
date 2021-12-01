from setuptools import setup

setup(
    name='filament_annotator_3d',
    version='0.0',
    author="Anna Medyukhina",
    author_email='anna.medyukhina@gmail.com',
    packages=['lib'
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
        'networkx',
        'Geometry3D',
        'jupyter',
        'jupyterlab<4.0',
    ],
)

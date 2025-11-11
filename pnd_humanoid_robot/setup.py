from setuptools import find_packages
from distutils.core import setup

setup(
    name='pnd_humanoid_robot',
    version='0.0.1',
    author='Jony Zhang',
    license="BSD-3-Clause",
    packages=find_packages(),
    description='Isaac Gym environments for pnd_humanoid_robot Humanoid Robot',
    install_requires=['isaacgym',
                      'rsl-rl',
                      'matplotlib']
)
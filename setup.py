from setuptools import setup, find_packages
from codecs import open
import os.path as osp


working_dir = osp.abspath(osp.dirname(__file__))
ROOT = osp.abspath(osp.dirname(__file__))

# READ the README
with open(osp.join(ROOT, 'README.md'), encoding='utf-8') as f:
    README = f.read()

with open(osp.join(ROOT, 'requirements.txt'), encoding='utf-8') as f:
    REQ = f.read().splitlines()

setup(name='fedsimul',
      version='0.1',
      description='Simulation of Asynchronous Federated Learning',
      long_descript=README,
      packages=find_packages(exclude=['out*']),
      install_requires=REQ,
      )

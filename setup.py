import os
from setuptools import setup, find_packages

version = '0.2.1'

install_requires = [i.strip() for i in open("requirements.txt").readlines()]
tests_require = ['nose']
docs_require = ['Sphinx']

setup(name='nems',
      version=version,
      description='Neural encoding models',
      author='Niru Maheshwaranathan',
      author_email='nirum@stanford.edu',
      url='https://github.com/ganguli-lab/nems',
      requires=install_requires,
      license='MIT',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      packages=find_packages(),
)

import os
from distutils.core import setup

version = '0.1.0'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.md')).read()
except IOError:
    README = ''

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
      long_description=README,
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      packages=['nems'],
)
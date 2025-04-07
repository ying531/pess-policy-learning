from os.path import abspath, dirname, join
from setuptools import setup

here = abspath(dirname(__file__))

with open(join(here, 'README.md')) as f:
    readme = f.read()

with open(join(here, 'LICENSE')) as f:
    lic = f.read()

setup(name='pess_policy_learning', 
      py_modules=['utils'],
      install_requires=[
          "numpy>=1.17.0",
          "pandas>=0.25.0",
          "scipy>=1.3.0",
          "scikit-learn>=0.21.3",
          "autograd>=1.2",
          "statsmodels>=0.10.1",
          "ipykernel>=5.1.2",
          "autograd",
          "dill",
          "jupyterlab>=1.1.4",
          "matplotlib>=3.1.1",
          "seaborn",
          "sklearn",
          "openml",
          "zipp",
          "matplotlib",
      ],
      classifiers=[
          'Development Status :: 1 - Planning',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3'
      ],
      license=lic)
from setuptools import setup
import os
import sys

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'RDME.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'MVPR',
  packages = ['MVPR'],
  version = '1.31',
  license='MIT',
  description = 'Multi-variable polynomial regression for curve fitting.',
  long_description_content_type='text/markdown',
  long_description = long_description,
  author = 'Joel Hampton',
  author_email = 'joelelihampton@outlook.com',
  url = 'https://github.com/Joel-H-dot/MVPR',
  download_url = 'https://github.com/Joel-H-dot/MVPR/archive/refs/tags/1.1.tar.gz',
  keywords = ['Machine Learning', 'Regression', 'polynomial'],
  install_requires=[
          'numpy',
          'sklearn',
          'scipy',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research ',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
)
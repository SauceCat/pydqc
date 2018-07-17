"""
pydqc
-------------------------------
 - Modified version by Eugenio Marinetto
 - nenetto@gmail.com
-------------------------------
"""

from setuptools import setup, find_packages
from codecs import open
from os import path
import sys


here = path.abspath(path.dirname(__file__))

# PRE INSTALL COMMANDS COMES HERE
sys.path.append(here)


# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
        name='pydqc',
        version='0.1.0',
        description='',
        long_description=long_description,
        url='https://github.com/nenetto/pymake',
        author='SauceCat - modified by E. Marinetto',
        author_email='jiangchun.lee@gmail.com',
        license='MIT',
        packages=find_packages(exclude=("tests",)),
        install_requires=['openpyxl>=2.5.0',
                          'xlsxwriter>=1.0.2',
                          'seaborn>=0.8.0',
                          'matplotlib_venn>=0.11.5',
                          'scikit-learn>=0.19.1',
                          'openpyxl>=2.5.4',
                          'Pillow>=5.2.0'],
        include_package_data=True,
        package_data={'pydqc': ['templates/*.txt']}
        )

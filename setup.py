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
        name='pymake',
        version='1.0',
        description='',
        long_description=long_description,
        url='https://github.com/nenetto/pymake',
        author='Eugenio Marinetto',
        author_email='nenetto@gmail.com',
        packages=find_packages(exclude=("tests",)),
        install_requires=['pexpect>=4.3.0',
                          'setuptools>=38.4.0',
                          'botocore>=1.10.16',
                          'boto3>=1.7.16',
                          'pipreqs>=0.4.9',
                          'psycopg2>=2.7.1',
                          'tabulate>=0.8.2',
                          'pandas>=0.22.0',
                          'pyodbc>=4.0.23',
                          'pyathena>=1.2.3',
                          'unidecode>=1.0.22',
                          'openpyxl>=2.5.4'],
        include_package_data=True,
        package_data={'pydqc': ['templates/*.txt']}
        )

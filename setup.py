from setuptools import setup, find_packages

setup(name='pydqc',
	packages=find_packages(exclude=("tests",)),
	package_data={'pydqc': ['templates/*.txt']},
	version='0.1.0',
	description='python automatic data quality check',
	author='SauceCat',
	author_email='jiangchun.lee@gmail.com',
	url='https://github.com/SauceCat/pydqc',
	download_url = '',
	license='MIT',
	classifiers = [],
	install_requires=[
		'openpyxl>=2.5.0',
		'xlsxwriter>=1.0.2',
		'seaborn>=0.8.0',
		'matplotlib_venn>=0.11.5',
		'scikit-learn',
        'openpyxl',
        'Pillow'
    ],
	zip_safe=False)

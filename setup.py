from setuptools import setup

setup(
    name='cimem',
    version='0.0.0',
    packages=['cimem'],
    url='https://github.com/sdeslauriers/cimem',
    license='GPL-3.0',
    author='Samuel Deslauriers-Gauthier',
    author_email='sam.deslauriers@gmail.com',
    description='A Python package that implements Connectivity Informed '
                'Maximum Entropy on the Mean (CIMEM)',
    install_requires=['numpy', 'bayesnet', 'scipy']
)

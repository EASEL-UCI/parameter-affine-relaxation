from setuptools import setup, find_packages

setup(
    name='dimpc',
    version='0.0.0',
    packages=find_packages(include=[
        'dimpc',
        'dimpc.*'
    ]),
    install_requires=[
        'numpy==1.26.4',
        'scipy==1.12.0',
        'matplotlib==3.8.3',
        'casadi==3.6.3',
    ]
)

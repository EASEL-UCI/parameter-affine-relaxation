from setuptools import setup, find_packages

setup(
    name='par',
    version='0.0.0',
    packages=find_packages(include=[
        'par',
        'par.*'
    ]),
    install_requires=[
        'numpy==1.26.4',
        'scipy==1.14.1',
        'matplotlib==3.8.3',
        'casadi==3.6.3',
    ]
)

from setuptools import setup, find_packages

setup(
    name="par",
    version="0.0.0",
    packages=find_packages(include=[
        "par",
        "par.*"
    ]),
    install_requires=[
        "casadi==3.6.7",
        "numpy==2.1.2",
        "scipy==1.14.1",
        "matplotlib==3.9.2",
        "PyQt6==6.7.1"
    ]
)

from setuptools import setup, find_packages

packages = find_packages(".", exclude=["tests"])
setup(
    name='hospital_in_a_box',
    version='0.1.0',
    packages=packages
)


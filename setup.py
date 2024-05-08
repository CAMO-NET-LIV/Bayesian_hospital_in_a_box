from setuptools import setup, find_packages

#packages = find_packages(".", exclude=["tests"])
packages = find_packages(".")
setup(
    name='bayesian_hospital_in_a_box',
    version='0.1.0',
    author='Peter L Green, Alessandro Gerada, Conor Rosato',
    author_email='p.l.green@liverpool.ac.uk',
    packages=packages,
    install_requires=["numpy", "scipy", "salabim", "pandas", "tqdm"]
)


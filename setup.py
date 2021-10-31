from setuptools import setup, find_packages


# parse_requirements() returns generator of pip.req.InstallRequirement objects
# requirements = list[map(str.strip, open("requirements.txt").readlines())]

setup(
    packages=find_packages(),
    # install_requires=requirements,
)

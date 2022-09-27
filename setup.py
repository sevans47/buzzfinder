from setuptools import setup
from setuptools import find_packages

# list dependencies from file
with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(name='buzzfinder',
      description='DL audio classification that can identify a note played on guitar as buzzy or clean',
      packages=find_packages(),
      install_requires=requirements)

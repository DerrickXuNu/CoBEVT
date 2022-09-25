from os.path import dirname, realpath
from setuptools import setup, find_packages, Distribution


def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='OpenCOOD',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/DerrickXuNu/CoBEVT.git',
    license='MIT',
    author='Runsheng Xu',
    author_email='rxx3386@ucla.edu',
    description='CoBEVT',
    long_description=open("README.md").read(),
    install_requires=_read_requirements_file(),
)

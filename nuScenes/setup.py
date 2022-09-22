from setuptools import setup, find_packages


__version__ = '0.0.1'

setup(
    name='cross_view_transformer',
    version=__version__,
    author='Runsheng Xu, Brady Zhou (Thanks for creating this awesome repo)',
    author_email='rxx3386@ucla.edu',
    license='MIT',
    packages=find_packages(include=['cross_view_transformer', 'cross_view_transformer.*']),
    zip_safe=False,
)

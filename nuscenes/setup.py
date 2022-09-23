from setuptools import setup, find_packages


__version__ = '0.0.1'

setup(
    name='sinbevt',
    version=__version__,
    author='Runsheng Xu(Providing SinBEVT model), Brady Zhou(Providing the pipeline)',
    author_email='rxx3386@ucla.edu',
    license='MIT',
    packages=find_packages(include=['cross_view_transformer', 'cross_view_transformer.*']),
    zip_safe=False,
)

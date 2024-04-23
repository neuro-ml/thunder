import runpy
from pathlib import Path

from setuptools import find_packages, setup


classifiers = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3 :: Only',
]

name = 'thunder'
root = Path(__file__).resolve().parent
with open(root / 'requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()
with open(root / 'README.md', encoding='utf-8') as file:
    long_description = file.read()
version = runpy.run_path(root / name / '__version__.py')['__version__']

setup(
    name=name,
    packages=find_packages(include=(name,)),
    author='NeuroML Group',
    author_email='max@ira-labs.com',
    include_package_data=True,
    version=version,
    description='A small experiments runner for Pytorch Lightning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/neuro-ml/thunder',
    download_url='https://github.com/neuro-ml/thunder/archive/v%s.tar.gz' % version,
    keywords=['deep learning', 'experiments'],
    entry_points={
        'console_scripts': ['thunder = thunder.cli:main'],
    },
    classifiers=classifiers,
    install_requires=requirements,
    python_requires='>=3.8',
)

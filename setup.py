#!/usr/bin/env python

"""The setup script."""

import os
import liverctsegmentationpackage as module
from setuptools import setup, find_packages


def walker(base, *paths):
    file_list = set([])
    cur_dir = os.path.abspath(os.curdir)

    os.chdir(base)
    try:
        for path in paths:
            for dname, dirs, files in os.walk(path):
                for f in files:
                    file_list.add(os.path.join(dname, f))
    finally:
        os.chdir(cur_dir)

    return list(file_list)


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup_requirements = 'pytest-runner'

test_requirements = 'pytest'

setup(
    author="Luis Kuhn Cuellar",
    author_email='luis.kuhn@gmail.com',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="Prediction package for U-Net models traine
d on the LiTS dataset.",
    entry_points={
        'console_scripts': [
            'liverctsegmentationpackage=liverctsegmentationpackage.cli_pytorch:main',
        ],
    },
    install_requires=requirements,
    license="MIT",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='liverctsegmentationpackage',
    name='liverctsegmentationpackage',
    packages=find_packages(include=['liverctsegmentationpackage', 'liverctsegmentationpackage.*']),
    package_data={
        module.__name__: walker(
            os.path.dirname(module.__file__),
            'models', 'data'
        ),
    },
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    url='https://github.com/luiskuhn/liverctsegmentationpackage',
    version='0.1.0-SNAPSHOT',
    zip_safe=False,
)
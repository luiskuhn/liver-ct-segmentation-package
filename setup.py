#!/usr/bin/env python

"""The setup script."""

import os
import liver_ct_segmentation_package as module
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
    description="Prediction package for U-Net models trained on the LiTS dataset.",
    entry_points={
        'console_scripts': [
            'liver-ct-seg-pred=liver_ct_segmentation_package.cli_pred:main',
            'liver-ct-seg-uncert=liver_ct_segmentation_package.cli_uncert_pred:main',
            'liver-ct-seg-feat-ggcam=liver_ct_segmentation_package.cli_feat_imp_ggcam:main',
            'liver-ct-seg-model-dl=liver_ct_segmentation_package.cli_model_dl:main',
        ],
    },
    install_requires=requirements,
    license="MIT",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='liver-ct-segmentation-package',
    name='liver-ct-segmentation-package',
    packages=find_packages(include=['liver_ct_segmentation_package', 'liver_ct_segmentation_package.*']),
    package_data={
        module.__name__: walker(
            os.path.dirname(module.__file__),
            'model', 'data'
        ),
    },
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    url='https://github.com/luiskuhn/liver-ct-segmentation-package',
    version='1.6.0',
    zip_safe=False,
)

from pathlib import Path

from setuptools import setup, find_packages

ROOT = Path(__file__).resolve().parent

with (ROOT / 'README.md').open('r', encoding='utf-8') as f:
    long_description = f.read()


def load_requirements(filename):
    with (ROOT / filename).open('r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


common_requirements = load_requirements('requirements.txt')

setup(
    name='motionscorehrpqct',
    version='2.2.3',
    author='Matthias Walle',
    author_email='matthias.walle@ucalgary.ca',
    description='MotionScoreHRpQCT core CLI for dataset-first HR-pQCT motion grading',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/wallematthias/MotionScoreHRpQCT',
    packages=find_packages(),
    install_requires=common_requirements,
    extras_require={
        'torch': [
            'torch>=2.2',
        ],
        'preview': [
            'matplotlib',
        ],
        'explain': [
            'SimpleITK',
            'matplotlib',
        ],
        'test': [
            'pytest>=7',
            'pytest-cov>=4',
            'matplotlib',
        ],
    },
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'motionscore=motionscore.cli:main',
            'motionscorehrpqct=motionscore.cli:main',
        ],
    },
)

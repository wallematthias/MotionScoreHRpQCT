from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


def load_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


common_requirements = load_requirements('requirements.txt')
mac_requirements = load_requirements('requirements-mac.txt')
unix_requirements = load_requirements('requirements-unix.txt')
tf_requirements = sorted(set(mac_requirements + unix_requirements)) or ['tensorflow']

setup(
    name='motionscorehrpqct',
    version='2.1.0',
    author='Matthias Walle',
    author_email='matthias.walle@ucalgary.ca',
    description='MotionScoreHRpQCT core CLI for dataset-first HR-pQCT motion grading',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/wallematthias/MotionScoreHRpQCT',
    packages=find_packages(),
    install_requires=common_requirements,
    extras_require={
        'tensorflow': tf_requirements,
        'mac': tf_requirements,
        'unix': tf_requirements,
        'torch': [
            'torch>=2.2',
        ],
        'h5': [
            'h5py',
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

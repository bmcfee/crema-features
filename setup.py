from setuptools import setup, find_packages

import imp

version = imp.load_source('crema.version', 'crema/version.py')

setup(
    name='crema',
    version=version.version,
    description='Convolutional REpresentations for Music Analysis',
    long_description='Convolutional REpresentations for Music Analysis',
    author='CREMA developers',
    author_email='brian.mcfee@nyu.edu',
    url='http://github.com/bmcfee/crema',
    download_url='http://github.com/bmcfee/crema/releases',
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
    ],
    keywords='machine learning, audio, music',
    license='ISC',
    install_requires=[
        'joblib>=0.8',
        'six',
        'numpy>=1.8',
        'scipy',
        'scikit-learn',
        'pandas',
        'pescador>=0.1',
        'librosa>=0.4.1',
        'jams>=0.2',
        'lasagne>=0.2',
        'presets>=0.1',
    ],
    extras_require={
        'docs': ['numpydoc']
    }
)

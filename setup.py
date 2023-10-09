from setuptools import setup, find_packages

VERSION = '0.27.12'
DESCRIPTION = 'Machine Learning project startup utilities'
LONG_DESCRIPTION = 'My commonly used utilities for machine learning projects'

setup(
    name='stefutils',
    version=VERSION,
    license='MIT',
    author='Yuzhao Stefan Heng',
    author_email='stefan.hg@outlook.com',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url='https://github.com/StefanHeng/stef-util',
    download_url='https://github.com/StefanHeng/stef-util/archive/refs/tags/v0.27.12.tar.gz',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'sty', 'colorama', 'pygments',
        'numpy', 'pandas', 'scikit-learn', 'torch', 'transformers>=4.26.2',
        'matplotlib', 'seaborn', 'tensorboard',
        'tqdm', 'icecream'
    ],
    keywords=['python', 'nlp', 'machine-learning', 'deep-learning'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: MacOS X',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Utilities'
    ]
)

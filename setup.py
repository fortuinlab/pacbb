from setuptools import setup

setup(
    name='pacbb',
    version='0.0.1',
    description='PAC Bayes Bound toolset',
    url='https://github.com/Yauhenii/pacbb',
    author='Yauheni Mardan, Maksym Tretiakov',
    author_email='yauhenmardan@gmail.com',
    license='MIT',
    packages=['core'],
    install_requires=[
        'torch>=2.2.1',
        'numpy>=1.26.4',
        'torchvision>=0.17.1',
        'tqdm>=4.66.2',
        'lightning>=2.2.1',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
    zip_safe=False,
)

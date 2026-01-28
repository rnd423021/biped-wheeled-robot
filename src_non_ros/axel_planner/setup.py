import setuptools

NAME = 'axel_planner'

setuptools.setup(
    name=NAME,
    packages=[NAME],
    package_dir={NAME: '.'},

    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
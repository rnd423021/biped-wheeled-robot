import setuptools

NAME = 'axel_planner'

setuptools.setup(
    name=NAME,
    packages=setuptools.find_packages(where='axel_planner'),
    package_dir={'': 'axel_planner'},

    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
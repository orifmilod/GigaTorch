from setuptools import setup, find_packages
import setuptools_black

setup(
    name="gigatorch",
    version="0.1.0",
    author="Orif Milod",
    packages=find_packages(exclude=["test"]),
    test_suite="tests",
    install_requires=["graphviz", "torch", "setuptools-black", "Pillow", "numpy"],
    cmdclass={
        "build": setuptools_black.BuildCommand,
    },
    extras_require={
        "testing": [
            "torch",
            "pytest",
        ],
    },
)

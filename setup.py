from setuptools import setup, find_packages
import setuptools_black

setup(
    name="mytorch",
    version="0.1.0",
    author="Orif Milod",
    packages=find_packages(exclude=["test"]),
    test_suite="tests",
    install_requires=["graphviz", "torch", "setuptools-black"],
    cmdclass={
        "build": setuptools_black.BuildCommand,
    },
)

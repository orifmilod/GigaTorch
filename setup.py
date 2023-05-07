from setuptools import setup, find_packages

setup(
    name = 'mytorch',
    version='0.1.0',
    author="Orif Milod",
    packages = find_packages(),
    test_suite="tests",
    packages = find_packages(exclude=["test"]),
    install_requires=[
       'graphviz',
       'torch',
    ]
)

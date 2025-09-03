from setuptools import setup, find_packages

setup(
    name="xaipatimg",
    version="0.0.1",
    python_requires=">=3.6",
    packages=find_packages(include=["xaipatimg", "xaipatimg.*"]),
)

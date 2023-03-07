from pathlib import Path

from setuptools import find_packages, setup

PACKAGE_NANE = "nmrcryspy"


with open("nmrcryspy/__init__.py") as f:
    for line in f.readlines():
        if "__version__" in line:
            before_keyword, keyword, after_keyword = line.partition("=")
            version = after_keyword.strip()[1:-1]
            print("nmrcryspy version ", version)
            break


def get_readme():
    filename = Path(__file__).parent.joinpath("README.md")
    with open(filename) as f:
        readme = f.read()
    return readme


setup(
    name=PACKAGE_NANE,
    version=version,
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "pymatgen",
        # "jsonargparse==3.19.2",
    ],
    extras_require={
        "test": ["pytest"],
    },
    author="Maxwell Venetos",
    author_email="mvenetos@berkeley.edu",
    url="https://xxx.yyy.zzz",
    description="An NMR crystallography toolkit.",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    license="BSD-3-Clause",
    zip_safe=False,
)

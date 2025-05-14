from setuptools import find_packages  # type: ignore
from setuptools import setup  # type: ignore

setup(
    name="pescoid_modelling",
    version="0.1.dev0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "pescoid = pescoid_modelling.cli:main",
        ],
    },
)

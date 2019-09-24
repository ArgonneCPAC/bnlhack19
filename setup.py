from setuptools import setup, find_packages


PACKAGENAME = "chopperhack19"
VERSION = "0.0.dev"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author=["Matt Becker", "Andrew Hearin", "Antonio Villarreal"],
    author_email=["ahearin@anl.gov"],
    description="Source code for the exachoppers at BNL GPU Hackathon 2019",
    long_description="Source code for the exachoppers at BNL GPU Hackathon 2019",
    install_requires=["numpy", "numba", "scipy"],
    packages=find_packages(),
    url="https://github.com/ArgonneCPAC/chopperhack19"
)

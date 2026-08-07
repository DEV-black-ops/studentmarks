from setuptools import setup, find_packages


def get_requirements(file_path:str) -> list:
    with open(file_path) as f:
        requirements = f.read().splitlines()
    return requirements 




setup(
    name="MLOPS",
    version="0.1.0",
    packages=find_packages(),
    install_requires= get_requirements("requirements.txt")   

)


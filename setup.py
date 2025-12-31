from setuptools import find_packages, setup

def get_requirements(file_path: str):
    """
    This function gets all the requirements as per the requirements.txt file
    """
    HYPEN_E_DOT = "-e ."

    with open(file_path, "r") as f:
        requirements = []
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


setup(
    name= "mlproject-1",
    author="Divyanshu",
    version="0.0.1",
    author_email="divyanshuasadeveloper@gmail.com",
    packages=find_packages(),
    requires=get_requirements("requirements.txt")
)
import setuptools

#with open("README.md", "r", encoding="utf-8") as f:
    #long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "Alzheimer-s-Disease-Detection"
AUTHOR_USER_NAME = "gaurang"
SRC_REPO = "alzheimer_disease"
AUTHOR_EMAIL = "gauranggirimeghanathi@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="App to detect Alzheimer disease",
    #long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
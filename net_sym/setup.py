import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="net-sym",
    version="1.0.0",
    author="Kumar Harsha",
    author_email="kumar.harsha@tum.de",
    description="Network symmetrization toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hkumar6/thesis-notebooks",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

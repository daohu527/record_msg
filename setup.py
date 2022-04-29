import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="record_msg",
    version="0.0.7",
    author="daohu527",
    author_email="daohu527@gmail.com",
    description="Record message parse helper function",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daohu527/record_msg",
    project_urls={
        "Bug Tracker": "https://github.com/daohu527/record_msg/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    install_requires=[
        'opencv-python',
        'lzf',
    ],
    python_requires=">=3.6",
)

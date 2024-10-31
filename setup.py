from setuptools import find_packages, setup


def _fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


setup(
    name="nanorl",
    version='0.0.1',
    packages=find_packages(
        exclude=("examples")
    ),
    description='An nano rlhf framework',
    long_description=open('README.md').read(),
    install_requires=_fetch_requirements("requirements.txt"),
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.10',
)

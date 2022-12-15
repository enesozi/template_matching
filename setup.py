#!python
import setuptools
import glob

setuptools.setup(
    name="feature_matcher",
    version="0.0.0",
    description="Feature Matcher",
    scripts=glob.glob("scripts/*.py"),
    packages=setuptools.find_packages(),
    package_dir={"feature_matcher": "feature_matcher"},
)

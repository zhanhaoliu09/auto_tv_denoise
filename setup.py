from setuptools import setup, find_packages

REQUIRED_PACKAGES = []

setup(
    name="auto_tv_denoise",
    version="0.1.0",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Total variation denoising with automatic parameter estimation"
)

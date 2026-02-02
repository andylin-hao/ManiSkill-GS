from setuptools import setup, find_packages

setup(
    name="mani_skill_gs",
    version="0.0.1",
    packages=["mani_skill_gs"],
    include_package_data=True,
    install_requires=[
        "gsplat==1.4.0",
        "mani-skill==3.0.0b21",
        "gymnasium==0.29.1",
        "nerfstudio==1.1.5",
        "networkx==3.4.2",
        "e3nn==0.5.9"
        ],
)
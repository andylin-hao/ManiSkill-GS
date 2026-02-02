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
        "urdfpy @ git+https://github.com/mmatl/urdfpy.git",  # use git: unpins networkx (PyPI 0.0.22 pins networkx==2.2, conflicts with mani-skill)
        "networkx==3.4.1",  # compatible with both urdfpy (git) and mani-skill
        "e3nn==0.5.9"
        ],
)
import re
from setuptools import setup, find_packages

# Read version from video_viewer/__init__.py
with open("video_viewer/__init__.py") as f:
    version = re.search(r'__version__\s*=\s*"(.+?)"', f.read()).group(1)

setup(
    name="video_viewer",
    version=version,
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "PySide6",
        "pyqtgraph",
        "scikit-image",
    ],
    entry_points={
        'console_scripts': [
            'video_viewer=video_viewer.main:main',
        ],
    },
)

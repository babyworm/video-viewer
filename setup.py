from setuptools import setup, find_packages

setup(
    name="video_viewer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "PyQt6",
    ],
    entry_points={
        'console_scripts': [
            'video_viewer=video_viewer.main:main',
        ],
    },
)

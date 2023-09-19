# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="semantic_segment_anything",
    version="1.0",
    install_requires=[
        "groundingdino @ git+https://github.com/IDEA-Research/GroundingDINO@beeb4c29cbfa915b7aafce9bf0a23f2208498a43",
        "scikit-image",
        "open_clip_torch"
    ],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        "all": ["matplotlib", "pycocotools", "opencv-python"],
        "dev": ["flake8", "isort", "black", "mypy"],
    },
)

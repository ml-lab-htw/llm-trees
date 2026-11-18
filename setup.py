from pathlib import Path
from setuptools import find_packages, setup

setup(
    name='llm-trees',
    version='0.3.0',
    description='Zero-shot decision tree induction and embedding using large language models',
    author='Ricardo Knauer, Mario Koddenbrock',
    author_email='ricardo.knauer@htw-berlin.de',
    url='https://github.com/ml-lab-htw/llm-trees',
    packages=find_packages(),
    install_requires=[
        'pandas~=2.2.3',
        'scikit-learn>=1.7,<2',
        'numpy>=1.26,<3',
        'openai~=1.57.2',
        'python-dotenv~=1.2.2',
    ],
    extras_require={
        'test': ['pytest'],
    },
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)

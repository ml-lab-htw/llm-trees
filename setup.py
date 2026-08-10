from setuptools import setup, find_packages

setup(
    name='llm_trees',
    version='0.2.0',
    description='A project for generating and evaluating decision trees using LLMs',
    author='Mario Koddenbrock, Ricardo Knauer',
    author_email='mario.koddenbrock@htw-berlin.de',
    url='https://github.com/ml-lab-htw/llm-trees',
    packages=find_packages(),
    install_requires=[
        'pandas~=2.2.3',
        'scikit-learn~=1.6.0',
        'numpy<2.0.0',
        'matplotlib~=3.9.3',
        'seaborn~=0.13.2',
        'prettytable~=3.12.0',
        'tqdm~=4.67.1',
        'cachetools~=5.5.0',
        'anthropic~=0.40.0',
        'google-cloud-aiplatform~=1.133.0',
        'openai~=1.57.2',
        'python-dotenv~=1.2.2',
        'xgboost~=2.1.3',
    ],
    entry_points={
        'console_scripts': [
            'llm_trees=llm_trees.cli:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)

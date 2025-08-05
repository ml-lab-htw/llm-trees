# LLM Trees: Decision Trees with Language Models
Official repo: [“Oh LLM, I’m Asking Thee, Please Give Me a Decision Tree”: Zero-Shot Decision Tree Induction and Embedding with Large Language Models](https://dl.acm.org/doi/10.1145/3711896.3736818)

Large language models (LLMs) provide powerful means to leverage prior knowledge for predictive modeling when data is limited. 
In this work, we demonstrate how LLMs can use their compressed world knowledge to generate intrinsically interpretable machine learning models, i.e., decision trees, without any training data. We find that these zero-shot decision trees can even surpass data-driven trees on some small-sized tabular datasets and that embeddings derived from these trees perform better than data-driven tree-based embeddings on average. Our knowledge-driven decision tree induction and embedding approaches therefore serve as strong new baselines for data-driven machine learning methods in the low-data regime. Furthermore, they offer ways to harness the rich world knowledge within LLMs for tabular machine learning tasks.

## Results Reproduction
The results presented in the paper can be reproduced by running the `calculations.py` script. 
However, these results have already been generated and are available in the `result` folder. 
You can visualize these results using the `evaluations.ipynb` notebook.

# The Python Package

## Overview

`llm_trees` is a Python package that allows you to generate and evaluate decision trees using various language models (LLMs) such as GPT (4o and o1), Gemini, and Claude. 
This package provides a Command Line Interface (CLI) to facilitate these operations. 
The trees can be evaluated directly (induction) or as embeddings followed by a [multi-layer perceptron classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) as a probe. 


## Installation

To install the package, use the following command:

```sh
pip install llm_trees
```


## Configuration

To configure the `.env` file for accessing the LLMs, you need to set the respective API keys based on the model you intend to use. The `.env` file should include the following keys:

- `OPENAI_API_KEY` for GPT models
- `GOOGLE_CLOUD_PROJECT` for Gemini
- `ANTHROPIC_API_KEY` for Claude

You only need to provide the key for the model you plan to use. 
Place the `.env` file in the root directory of your project.

## Usage

The CLI provides three main commands: `generate`, `eval_induction`, and `eval_embedding`.

### Generate Decision Trees

To generate decision trees, use the `generate` command. Below are the available options:

```sh
python -m llm_trees.cli generate [OPTIONS]
```

**Options:**

- `--root`: Root directory for the project (default: `.`)
- `--dataset`: Dataset name (default: `penguins`)
- `--method`: The LLM method to use (default: `gpt-4o`, choices: `gpt-4o`, `gpt-o1`, `gemini`, `claude`)
- `--temperature`: Temperature for the LLM (default: `1`)
- `--iter`: Iteration counter (default: `0`)
- `--force_decision_tree`: Force the generation of a decision tree or let the LLM decide (only for induction) (default: `True`)
- `--include_description`: Include dataset descriptions in the prompt (default: `False`)
- `--llm_dialogue`: Enable LLM dialogue mode as described in the paper or directly prompting the python code (default: `True`)
- `--max_tree_depth`: Maximum depth of the decision tree (default: `2`, no maximum depth by selecting `0`)
- `--num_examples`: Number of examples to provide in the prompt (default: `1`)
- `--num_retry_llm`: Number of retries for generating a valid tree (default: `10`)
- `--use_role_prompt`: Use role-based prompts for the LLM (default: `False`)
- `--seed`: Random seed (default: `42`)
- `--generate_tree_if_missing`: Generate tree if missing (default: `True`)
- `--regenerating_invalid_trees`: Regenerate invalid trees (default: `True`)

### Evaluate Induction

To evaluate the induction process, use the `eval_induction` command. Below are the available options:

```sh
python -m llm_trees.cli eval_induction [OPTIONS]
```

**Options:**

- `--root`: Root directory for the project (default: `.`)
- `--dataset`: Dataset name (default: `penguins`)
- `--method`: The LLM method to use (default: `gpt-4o`, choices: `gpt-4o`, `gpt-o1`, `gemini`, `claude`)
- `--temperature`: Temperature for the LLM (default: `1`)
- `--iter`: Iteration counter (default: `0`)
- `--num_iters`: Number of iterations (default: `5`)
- `--train_split`: Train/test split ratio (default: `0.67`)
- `--force_decision_tree`: Force the generation of a decision tree or let the LLM decide (only for induction) (default: `True`)
- `--include_description`: Include dataset descriptions in the prompt (default: `False`)
- `--llm_dialogue`: Enable LLM dialogue mode as described in the paper or directly prompting the python code (default: `True`)
- `--max_tree_depth`: Maximum depth of the decision tree (default: `2`, no maximum depth by selecting `0`)
- `--num_examples`: Number of examples to provide in the prompt (default: `1`)
- `--num_retry_llm`: Number of retries for generating a valid tree (default: `10`)
- `--use_role_prompt`: Use role-based prompts for the LLM (default: `False`)
- `--num_trees`: Number of trees (default: `5`)
- `--seed`: Random seed (default: `42`)
- `--generate_tree_if_missing`: Generate tree if missing (default: `True`)
- `--regenerating_invalid_trees`: Regenerate invalid trees (default: `True`)
- `--skip_existing`: Skip existing results and load them from the csv file (default: `True`)

### Evaluate Embedding

To evaluate the embedding process, use the `eval_embedding` command. Below are the available options:

```sh
python -m llm_trees.cli eval_embedding [OPTIONS]
```

**Options:**

- `--root`: Root directory for the project (default: `.`)
- `--dataset`: Dataset name (default: `penguins`)
- `--method`: The LLM method to use (default: `gpt-4o`, choices: `gpt-4o`, `gpt-o1`, `gemini`, `claude`)
- `--temperature`: Temperature for the LLM (default: `1`)
- `--iter`: Iteration counter (default: `0`)
- `--num_iters`: Number of iterations (default: `5`)
- `--train_split`: Train/test split ratio (default: `0.67`)
- `--append_raw_features`: Append raw features to the embeddings (default: `True`)
- `--classifier`: The downstream classifier to use (default: `mlp`, choices: `mlp`, `hgbdt`, `lr`)
- `--include_description`: Include feature descriptions of the dataset in the prompt (default: `False`)
- `--llm_dialogue`: Enable LLM dialogue mode as described in the paper or directly prompting the python code (default: `True`)
- `--max_tree_depth`: Maximum depth of the decision tree (default: `2`, no maximum depth by selecting `0`)
- `--num_examples`: Number of examples to provide in the prompt (default: `1`)
- `--num_retry_llm`: Number of retries for generating a valid tree (default: `10`)
- `--use_role_prompt`: Use role-based prompts for the LLM (default: `False`)
- `--num_trees`: Number of trees (default: `5`)
- `--seed`: Random seed (default: `42`)
- `--generate_tree_if_missing`: Generate tree if missing (default: `True`)
- `--regenerating_invalid_trees`: Regenerate invalid trees (default: `True`)
- `--skip_existing`: Skip existing results and load them from the csv file (default: `True`)

## Examples

### Generate Decision Trees

```sh
python -m llm_trees.cli generate --method gpt-4o --dataset penguins --temperature 1.0
```

### Evaluate Induction

```sh
python -m llm_trees.cli eval_induction --method gpt-4o --dataset penguins --temperature 1.0
```

### Evaluate Embedding

```sh
python -m llm_trees.cli eval_embedding --method gpt-4o --dataset penguins --temperature 1.0
```


## CLI Help

se the `--help` flag for more information.

### Main Command Help

```sh
python -m llm_trees.cli --help
```

### Subcommand Help

#### Generate Command

```sh
python -m llm_trees.cli generate --help
```

#### Eval Induction Command

```sh
python -m llm_trees.cli eval_induction --help
```

#### Eval Embedding Command

```sh
python -m llm_trees.cli eval_embedding --help
```


## Run with your own Data
To integrate your own dataset into the `llm_trees` project, follow these steps:

1. **Create a New Folder**: Create a new folder under `data_sets` with the name of your dataset.

2. **Add Data Files**: Inside this folder, add two CSV files: `X.csv` for the features and `y.csv` for the target variable.

3. **Create `prompt.txt`**: Add a `prompt.txt` file with the prompt for the LLM.

4. **Create `feature_description.txt`**: Add a `feature_description.txt` file with detailed descriptions of the features.

5. **Create `description.txt` (Optional)**: Add a `description.txt` file with additional information about the dataset.

### Example Structure

Assume your dataset is named `my_dataset`.

```
data_sets/
└── my_dataset/
    ├── X.csv
    ├── y.csv
    ├── prompt.txt
    ├── feature_description.txt
    └── description.txt (optional)
```

### Example Files

#### `X.csv`
```csv
island,culmen_length_mm,culmen_depth_mm,flipper_length_mm,body_mass_g,sex
2,39.1,18.7,181.0,3750.0,2.0
2,39.5,17.4,186.0,3800.0,1.0
2,40.3,18.0,195.0,3250.0,1.0
```

#### `y.csv`
```csv
species
0
1
2
```

#### `prompt.txt`
```unknown
I want you to induce a decision tree classifier based on features. I first give an example below. 
Then, I provide you with Features and want you to build a decision tree with a maximum depth of 2 using the most important features. 
The tree should classify the species of penguins (Adelie / Chinstrap / Gentoo).

Features: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)

Decision tree:
|--- petal width (cm) <= 0.80
||--- class: setosa
|--- petal width (cm) > 0.80
||--- petal width (cm) <= 1.75
|||--- class: versicolor
||--- petal width (cm) > 1.75
|||--- class: virginica

Features: island (Biscoe / Dream / Torgersen), culmen length (mm), culmen depth (mm), flipper length (mm), body mass (g), sex (male / female)

Decision Tree:
```

#### `feature_description.txt`
```unknown
Features:
island: 3 islands in the Palmer Archipelago, Antarctica (0 = Biscoe / 1 = Dream / 2 = Torgersen)
culmen_length_mm: The culmen is the upper ridge of a bird’s bill. This feature is the length of the culmen in mm.
culmen_depth_mm: The culmen is the upper ridge of a bird’s bill. This feature is the depth of the culmen in mm.
flipper_length_mm: Flipper length in mm.
body_mass_g: Body Mass Index
sex: (0 = nan / 1 = female / 2 = male)

Target variable:
species: penguin species (0 = Adelie / 1 = Chinstrap / 2 = Gentoo)
```

#### `description.txt` (Optional)
```unknown
This dataset contains measurements of penguins from three different islands in the Palmer Archipelago, Antarctica. The features include physical measurements such as culmen length, culmen depth, flipper length, and body mass, as well as the sex of the penguins. The target variable is the species of the penguins, which can be Adelie, Chinstrap, or Gentoo.
```

By following these steps, you can integrate your own dataset into the `llm_trees` project and use it with the provided CLI commands.



## License

This project is licensed under the [MIT License](https://github.com/ml-lab-htw/llm-trees/blob/main/LICENSE). See the `LICENSE` file for details.


## Authors

- Mario Koddenbrock (HTW Berlin)
- Ricardo Knauer (HTW Berlin)

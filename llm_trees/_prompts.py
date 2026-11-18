"""Two-step LLM prompt construction from estimator parameters.

The first prompt asks the LLM to induce a human-readable decision tree for
the given target (rendered by its description) and features (each rendered
as ``name: description``), with optional complexity constraints. The second
prompt asks for the induced tree to be converted into a Python function; it
carries the exact feature keys, the target, the feature descriptions, and
reference examples of the expected function shape.
"""

from ._prompt_examples import get_example

FIRST_PROMPT_TEMPLATE = """I want you to induce a decision tree based on features. I first give {example_phrase}. Then, I provide you with new features and want you to build a decision tree{complexity} using the most important features. The tree should predict the {target_description}.

Features: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)

Decision tree:
|--- petal width (cm) <= 0.80
||--- class: setosa
|--- petal width (cm) > 0.80
||--- petal width (cm) <= 1.75
|||--- class: versicolor
||--- petal width (cm) > 1.75
|||--- class: virginica

Features: {features}

Decision Tree:
"""


def get_role_prompt():
    return """
You are a domain expert with years of experience in building the best-performing decision trees. 
You have an astounding ability to identify the best features for the task at hand, and you know how to combine them to get the best predictions. 
Impressively, your profound world knowledge allows you to do that without looking at any real-world data.

"""


def complexity_clause(max_tree_depth: int | None, num_features: int | None) -> str:
    """Prompt fragment for the active complexity constraints."""
    if max_tree_depth is not None and num_features is not None:
        return f" with a maximum depth of {max_tree_depth} and at most {num_features} distinct features"
    if max_tree_depth is not None:
        return f" with a maximum depth of {max_tree_depth}"
    if num_features is not None:
        return f" with at most {num_features} distinct features"
    return ""


def target_phrase(target: dict) -> str:
    """Render the single target entry as ``name: description``."""
    (name, description), = target.items()
    return f"{name}: {description}"


def build_first_prompt(
    *,
    target: dict,
    features: dict,
    max_tree_depth: int | None = None,
    num_features: int | None = None,
    num_examples: int = 1,
    use_role_prompt: bool = False,
) -> str:
    """First-step prompt: induce a human-readable decision tree.

    The tree is asked to predict the target's description (the target name
    itself is not needed for induction), and every feature is rendered as
    ``name: description`` so the LLM sees each column key and its meaning
    (value encoding, units) before inducing the tree: with cryptic column
    keys alone the model has nothing to reason about. Features with empty
    descriptions fall back to name-only rendering.
    """
    if num_examples == 1:
        example_phrase = "an example below"
    elif num_examples > 1:
        example_phrase = f"{num_examples} examples below"
    else:
        example_phrase = "no examples below"

    features_str = ", ".join(
        f"{name}: {description}" if description else str(name) for name, description in features.items()
    )

    (_name, target_description), = target.items()
    prompt = FIRST_PROMPT_TEMPLATE.format(
        example_phrase=example_phrase,
        complexity=complexity_clause(max_tree_depth, num_features),
        target_description=target_description,
        features=features_str,
    )

    if use_role_prompt:
        prompt = get_role_prompt() + prompt

    return prompt


def build_second_prompt(
    *,
    target: dict,
    features: dict,
    num_examples: int,
    max_tree_depth: int | None,
) -> str:
    """Second-step prompt: convert the (step-1) tree into a Python function."""
    second_prompt = """
Now, format this decision tree as a Python function that returns a single prediction as well as a list representing the truth values of the inner nodes. 
The entries of this list should be 1 if the condition of the corresponding inner node is satisfied, and 0 otherwise. 

If your decision tree has placeholders for thresholds, try to fill them with meaningful values. 

"""

    feature_description = "".join(f"{name}: {description}\n" for name, description in features.items())

    feature_string = f"""
        
Here are the exact keys of the features. Pay close attention to their spelling (uppercase, lowercase, spaces, etc.):
{list(features)}

Here is the target. Pay close attention to the target value encoding:
{target_phrase(target)}

Here is a description of the features with optional feature encoding or ranges:
{feature_description}
"""
    second_prompt += feature_string

    if num_examples == 1:
        second_prompt += "Here is an example of what the function should look like:"
    else:
        second_prompt += f"Here are {num_examples} examples of what the function should look like:"

    for i in range(num_examples):
        second_prompt += get_example(i, max_tree_depth)

    return second_prompt

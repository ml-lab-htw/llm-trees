"""Deterministic test fixtures for llm_trees.

PROMPT_TEMPLATE is the pinned first-prompt template; the suite asserts that
the built first prompt equals this template rendered with concrete
target/features/depth values.
"""

PROMPT_TEMPLATE = '''I want you to induce a decision tree based on features. I first give an example below. Then, I provide you with new features and want you to build a decision tree with a maximum depth of {max_tree_depth} using the most important features. The tree should predict the {target_description}.

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
'''

# Dialogue separator (``25 * "#"``) produced between the two step responses.
# Pinned independently of ``llm_trees._tree.DIALOGUE_SEPARATOR`` on purpose:
# tests/test_llm.py asserts the two agree, so drift in the production
# constant is caught.
DIALOGUE_SEPARATOR_FIXTURE = "#" * 25

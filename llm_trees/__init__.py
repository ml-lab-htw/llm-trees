"""LLM Trees: Decision Trees with Language Models"""

from ._llm_tree_classifier import LLMTreeClassifier
from ._llm_tree_embedding_classifier import LLMTreeEmbeddingClassifier
from ._llm_tree_embedding_regressor import LLMTreeEmbeddingRegressor
from ._llm_tree_regressor import LLMTreeRegressor

__all__ = [
    "LLMTreeClassifier",
    "LLMTreeRegressor",
    "LLMTreeEmbeddingClassifier",
    "LLMTreeEmbeddingRegressor",
]

# Define the version of the package
__version__ = "0.3.0"

# Package metadata
__author__ = "Ricardo Knauer and Mario Koddenbrock"
__license__ = "MIT"
__maintainer__ = "Ricardo Knauer and Mario Koddenbrock"
__email__ = "ricardo.knauer@htw-berlin.de"
__status__ = "Development"
__url__ = "https://github.com/ml-lab-htw/llm-trees"
__description__ = "LLM Trees: Decision Trees with Language Models"
__doc__ = __description__

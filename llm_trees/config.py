
# Configuration for a single run
class Config:

    root = "." # Root directory for the project
    dataset = "penguins"  # Set the dataset name here
    method = "openrouter"  # LLM method

    temperature = 0  # Set the temperature of the LLM

    # Settings for train/test splits
    iter = 0 # iteration counter
    num_iters = 5 # Number of iterations
    train_split = 0.67 # Train/test split ratio

    # embedding settings
    classifier = "mlp" # Downstream classifier
    append_raw_features = True # Append raw features to the embeddings

    # LLM prompting settings
    force_decision_tree = True  # If False, generate free-form model
    include_description = False  # Include feature descriptions of the dataset in the prompt
    llm_dialogue = True  # If True, generate a two-step dialogue with the LLM
    max_tree_depth = 2  # Maximum depth of the decision tree
    num_examples = 1  # Number of examples to provide in the prompt
    num_retry_llm = 10  # Number of retries for generating a valid tree
    use_role_prompt = False  # Use role-based prompts for the LLM

    # Additional settings
    num_trees = 5
    seed = 42
    generate_tree_if_missing = True
    regenerating_invalid_trees = True
    skip_existing = True

    def __init__(self, **kwargs):
        # Override default properties with any provided keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        if not self.regenerating_invalid_trees:
            self.num_retry_llm = 1

    def split_str(self):
        return f"{100 * self.train_split:.0f}/{100 * (1 - self.train_split):.0f}"

    def __str__(self):
        config_items = [
            f"{key}: {value}"
            for key, value in self.__dict__.items()
            if key != "root"  # Exclude the root property
        ]
        config_items.append(f"Computed Temperature: {self.get_temperature()}")
        return "Current Configuration:\n" + "\n".join(f"  {item}" for item in config_items)

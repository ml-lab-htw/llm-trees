import argparse
import os

from .config import Config
from .induction_utils import eval_induction
from .embeddings import eval_embedding
from .utils import generate_tree, get_tree_path


def main():
    parser = argparse.ArgumentParser(description="LLM Trees: Decision Trees with Language Models")
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Subparser for the generate command
    parser_generate = subparsers.add_parser('generate', help='Generate decision trees')
    parser_generate.add_argument('--root', type=str, default=".", help="Root directory for the project")
    parser_generate.add_argument('--dataset', type=str, default="penguins", help="Dataset name")
    parser_generate.add_argument('--temperature', type=float, default=0, help="Temperature for the LLM")
    parser_generate.add_argument('--iter', type=int, default=0, help="Iteration counter")
    parser_generate.add_argument('--num_iters', type=int, default=5, help="Number of iterations")
    parser_generate.add_argument('--force_decision_tree', type=bool, default=True, help="Force the generation of a decision tree")
    parser_generate.add_argument('--include_description', type=bool, default=False, help="Include feature descriptions of the dataset in the prompt")
    parser_generate.add_argument('--llm_dialogue', type=bool, default=True, help="Enable LLM dialogue mode")
    parser_generate.add_argument('--max_tree_depth', type=int, default=2, help="Maximum depth of the decision tree")
    parser_generate.add_argument('--num_examples', type=int, default=3, help="Number of examples to provide in the prompt")
    parser_generate.add_argument('--num_retry_llm', type=int, default=10, help="Number of retries for generating a valid tree")
    parser_generate.add_argument('--use_role_prompt', type=bool, default=False, help="Use role-based prompts for the LLM")
    parser_generate.add_argument('--num_trees', type=int, default=5, help="Number of trees")
    parser_generate.add_argument('--seed', type=int, default=42, help="Random seed")
    parser_generate.add_argument('--generate_tree_if_missing', type=bool, default=True, help="Generate tree if missing")
    parser_generate.add_argument('--regenerating_invalid_trees', type=bool, default=True, help="Regenerate invalid trees")

    # Subparser for the eval_induction command
    parser_eval_induction = subparsers.add_parser('eval_induction', help='Evaluate induction')
    parser_eval_induction.add_argument('--root', type=str, default=".", help="Root directory for the project")
    parser_eval_induction.add_argument('--dataset', type=str, default="penguins", help="Dataset name")
    parser_eval_induction.add_argument('--temperature', type=float, default=0, help="Temperature for the LLM")
    parser_eval_induction.add_argument('--iter', type=int, default=0, help="Iteration counter")
    parser_eval_induction.add_argument('--num_iters', type=int, default=5, help="Number of iterations")
    parser_eval_induction.add_argument('--train_split', type=float, default=0.67, help="Train/test split ratio")
    parser_eval_induction.add_argument('--force_decision_tree', type=bool, default=True, help="Force the generation of a decision tree")
    parser_eval_induction.add_argument('--include_description', type=bool, default=False, help="Include feature descriptions of the dataset in the prompt")
    parser_eval_induction.add_argument('--llm_dialogue', type=bool, default=True, help="Enable LLM dialogue mode")
    parser_eval_induction.add_argument('--max_tree_depth', type=int, default=2, help="Maximum depth of the decision tree")
    parser_eval_induction.add_argument('--num_examples', type=int, default=3, help="Number of examples to provide in the prompt")
    parser_eval_induction.add_argument('--num_retry_llm', type=int, default=10, help="Number of retries for generating a valid tree")
    parser_eval_induction.add_argument('--use_role_prompt', type=bool, default=False, help="Use role-based prompts for the LLM")
    parser_eval_induction.add_argument('--num_trees', type=int, default=5, help="Number of trees")
    parser_eval_induction.add_argument('--seed', type=int, default=42, help="Random seed")
    parser_eval_induction.add_argument('--generate_tree_if_missing', type=bool, default=True, help="Generate tree if missing")
    parser_eval_induction.add_argument('--regenerating_invalid_trees', type=bool, default=True, help="Regenerate invalid trees")
    parser_eval_induction.add_argument('--skip_existing', type=bool, default=True, help="Skip existing trees")

    # Subparser for the eval_embedding command
    parser_eval_embedding = subparsers.add_parser('eval_embedding', help='Evaluate embedding')
    parser_eval_embedding.add_argument('--root', type=str, default=".", help="Root directory for the project")
    parser_eval_embedding.add_argument('--dataset', type=str, default="penguins", help="Dataset name")
    parser_eval_embedding.add_argument('--temperature', type=float, default=1, help="Temperature for the LLM")
    parser_eval_embedding.add_argument('--iter', type=int, default=0, help="Iteration counter")
    parser_eval_embedding.add_argument('--num_iters', type=int, default=5, help="Number of iterations")
    parser_eval_embedding.add_argument('--train_split', type=float, default=0.67, help="Train/test split ratio")
    parser_eval_embedding.add_argument('--append_raw_features', type=bool, default=True, help="Append raw features to the embeddings")
    parser_eval_embedding.add_argument('--classifier', type=str, default="mlp", choices=['mlp','hgbdt', 'lr'], help="The downstream classifier to use (multi-layer perceptron, histogram-based gradient-boosted decision tree, logistic regression)")
    parser_eval_embedding.add_argument('--include_description', type=bool, default=False, help="Include feature descriptions of the dataset in the prompt")
    parser_eval_embedding.add_argument('--llm_dialogue', type=bool, default=True, help="Enable LLM dialogue mode")
    parser_eval_embedding.add_argument('--max_tree_depth', type=int, default=2, help="Maximum depth of the decision tree")
    parser_eval_embedding.add_argument('--num_examples', type=int, default=3, help="Number of examples to provide in the prompt")
    parser_eval_embedding.add_argument('--num_retry_llm', type=int, default=10, help="Number of retries for generating a valid tree")
    parser_eval_embedding.add_argument('--use_role_prompt', type=bool, default=False, help="Use role-based prompts for the LLM")
    parser_eval_embedding.add_argument('--num_trees', type=int, default=5, help="Number of trees")
    parser_eval_embedding.add_argument('--seed', type=int, default=42, help="Random seed")
    parser_eval_embedding.add_argument('--generate_tree_if_missing', type=bool, default=True, help="Generate tree if missing")
    parser_eval_embedding.add_argument('--regenerating_invalid_trees', type=bool, default=True, help="Regenerate invalid trees")
    parser_eval_embedding.add_argument('--skip_existing', type=bool, default=True, help="Skip existing trees")

    args = parser.parse_args()

    skip_existing = getattr(args, 'skip_existing', True) if args.command != 'generate' else True
    force_decision_tree = getattr(args, 'force_decision_tree', True) if args.command != 'eval_embedding' else True
    append_raw_features = getattr(args, 'append_raw_features', True) if args.command == 'eval_embedding' else True
    train_split = getattr(args, 'train_split', 0.67) if args.command != 'generate' else 0.67
    classifier = getattr(args, 'classifier', 'mlp') if args.command == 'eval_embedding' else 'mlp'

    config = Config(
        root=args.root,
        dataset=args.dataset,
        method="openrouter",
        temperature=args.temperature,
        iter=args.iter,
        num_iters=args.num_iters,
        train_split=train_split,
        append_raw_features=append_raw_features,
        classifier=classifier,
        force_decision_tree=force_decision_tree,
        include_description=args.include_description,
        llm_dialogue=args.llm_dialogue,
        max_tree_depth=args.max_tree_depth,
        num_examples=args.num_examples,
        num_retry_llm=args.num_retry_llm,
        use_role_prompt=args.use_role_prompt,
        num_trees=args.num_trees,
        seed=args.seed,
        generate_tree_if_missing=args.generate_tree_if_missing,
        regenerating_invalid_trees=args.regenerating_invalid_trees,
        skip_existing=skip_existing
    )

    if args.command == 'generate':
        tree_path = get_tree_path(config)
        tree_exists = os.path.exists(tree_path)

        generate_tree(config)

        if tree_exists:
            print(f"Tree already exists: {tree_path}")
        else:
            print(f"Tree generated: {tree_path}")

    elif args.command == 'eval_induction':
        accuracy, f1_score = eval_induction(config)
        print(f"Induction Evaluation - Accuracy: {100 * accuracy:.0f}%, F1 Score: {100 * f1_score:.0f}%")
    elif args.command == 'eval_embedding':
        accuracy, f1_score = eval_embedding(config)
        print(f"Embedding Evaluation - Accuracy: {100 * accuracy:.0f}%, F1 Score: {100 * f1_score:.0f}%")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

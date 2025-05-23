from utils import experiments

run_config = {
    "induction": True,
    "embedding": True,
    "setting_1_tree_depth_induction": True,
    "setting_2_tree_depth_embeddings": True,
    "setting_3_number_of_trees_embeddings": True,
    "setting_4_temperature_induction": True,
    "setting_5_temperature_embeddings": True,
    "setting_6_number_of_examples_induction": True,
    "setting_7_number_of_examples_embeddings": True,
    "setting_8_append_raw_features": True,
    "setting_9_decision_tree_vs_free_form_induction": True,
    "setting_10_description_induction": True,
    "setting_11_description_embeddings": True,
    "setting_12_classifier_embeddings": True,
}

### Induction Experiments
if run_config["induction"]:
    result_path_induction = "./results/induction_results.csv"
    experiments.induction(
        is_ablation=False,
        result_path=result_path_induction,
        force_decision_tree_list=[True],
        include_description_list=[False],
        max_tree_depth_list=[2],
        num_examples_list=[1],
        temperature_list=[1],
    )

### Embedding Experiments
if run_config["embedding"]:
    result_path_embeddings = "./results/embedding_results.csv"
    experiments.embeddings(
        is_ablation=False,
        result_path=result_path_embeddings,
        append_raw_features_list=[True],
        include_description_list=[False],
        max_tree_depth_list=[0],
        num_examples_list=[1],
        num_trees_list=[5],
        temperature_list=[1],
        classifier_list=["mlp"],
    )

    result_path_dimensions = "./results/embedding_dimensions"
    experiments.embedding_dimensions(
        result_path=result_path_dimensions,
        model_list=["claude", "gemini", "gpt-4o", "gpt-o1"],
    )

### Ablation Setting 1: Tree Depth for Induction
if run_config["setting_1_tree_depth_induction"]:
    result_path_tree_depth_induction = "results/ablations/setting_1_optimal_tree_depth_induction.csv"
    experiments.induction(
        result_path=result_path_tree_depth_induction,
        num_examples_list=[1],
        max_tree_depth_list=[1, 2, 3, 4, 5],
        temperature_list=[1],
        force_decision_tree_list=[True],
        include_description_list=[False],
    )

### Ablation Setting 2: Tree Depth for Embeddings
if run_config["setting_2_tree_depth_embeddings"]:
    result_path_tree_depth_embeddings = "./results/ablations/setting_2_optimal_tree_depth_embeddings.csv"
    experiments.embeddings(
        result_path=result_path_tree_depth_embeddings,
        num_trees_list=[5],
        num_examples_list=[1],
        max_tree_depth_list=[1, 2, 3, 4, 5],
        temperature_list=[1],
        append_raw_features_list=[True],
        include_description_list=[False],
        model_list=["gpt-4o", "gpt-o1", "gemini", "claude"],
        classifier_list=["mlp"],
    )

### Ablation Setting 3: Number of Trees
if run_config["setting_3_number_of_trees_embeddings"]:
    result_path_number_of_trees_embeddings = "./results/ablations/setting_3_optimal_number_of_trees_embeddings.csv"
    experiments.embeddings(
        result_path=result_path_number_of_trees_embeddings,
        num_trees_list=[1, 2, 3, 4, 5],
        num_examples_list=[1],
        max_tree_depth_list=[0],
        temperature_list=[1],
        append_raw_features_list=[True],
        include_description_list=[False],
        model_list=["gpt-4o", "gpt-o1", "gemini", "claude"],
        classifier_list=["mlp"],
    )

### Ablation Setting 4: Temperature Induction
if run_config["setting_4_temperature_induction"]:
    result_path_temperature_induction = "./results/ablations/setting_4_optimal_temperature_induction.csv"
    temperatures = [
        {"gpt-4o": 0, "gemini": 0, "claude": 0},
        {"gpt-4o": 0.5, "gemini": 0.5, "claude": 0.5},
        {"gpt-4o": 1, "gemini": 1, "claude": 1},
    ]
    experiments.induction(
        result_path=result_path_temperature_induction,
        num_examples_list=[1],
        max_tree_depth_list=[2],
        temperature_list=temperatures,
        force_decision_tree_list=[True],
        include_description_list=[False],
        model_list=["gpt-4o", "gemini", "claude"],
    )

### Ablation Setting 5: Temperature Embeddings
if run_config["setting_5_temperature_embeddings"]:
    result_path_temperature_embeddings = "./results/ablations/setting_5_optimal_temperature_embeddings.csv"
    temperatures = [
        {"gpt-4o": 0, "gemini": 0, "claude": 0},
        {"gpt-4o": 0.5, "gemini": 0.5, "claude": 0.5},
        {"gpt-4o": 1, "gemini": 1, "claude": 1},
    ]
    experiments.embeddings(
        result_path=result_path_temperature_embeddings,
        num_trees_list=[5],
        num_examples_list=[1],
        max_tree_depth_list=[0],
        temperature_list=temperatures,
        append_raw_features_list=[True],
        include_description_list=[False],
        model_list=["gpt-4o", "gemini", "claude"],
        classifier_list=["mlp"],
    )

### Ablation Setting 6: Number of In-Context Examples for Induction
if run_config["setting_6_number_of_examples_induction"]:
    result_path_number_of_examples_induction = "./results/ablations/setting_6_optimal_number_of_examples_induction.csv"
    experiments.induction(
        result_path=result_path_number_of_examples_induction,
        num_examples_list=[1, 2, 3],
        max_tree_depth_list=[2],
        temperature_list=[1],
        force_decision_tree_list=[True],
        include_description_list=[False],
        model_list=["gpt-4o", "gpt-o1", "gemini", "claude"],
    )

### Ablation Setting 7: Number of In-Context Examples for Embeddings
if run_config["setting_7_number_of_examples_embeddings"]:
    result_path_number_of_examples_embeddings = "./results/ablations/setting_7_optimal_number_of_examples_embeddings.csv"
    experiments.embeddings(
        result_path=result_path_number_of_examples_embeddings,
        num_trees_list=[5],
        num_examples_list=[1, 2, 3],
        max_tree_depth_list=[0],
        temperature_list=[1],
        append_raw_features_list=[True],
        include_description_list=[False],
        model_list=["gpt-4o", "gpt-o1", "gemini", "claude"],
        classifier_list=["mlp"],
    )

### Ablation Setting 8: Two Embedding Variants
if run_config["setting_8_append_raw_features"]:
    result_path_append_raw_features = "results/ablations/setting_8_optimal_append_raw_features_embeddings.csv"
    experiments.embeddings(
        result_path=result_path_append_raw_features,
        num_trees_list=[5],
        num_examples_list=[1],
        max_tree_depth_list=[0],
        temperature_list=[1],
        append_raw_features_list=[False, True],
        include_description_list=[False],
        model_list=["gpt-4o", "gpt-o1", "gemini", "claude"],
        classifier_list=["mlp"],
    )

### Ablation Setting 9: Decision Tree vs. Free Form
if run_config["setting_9_decision_tree_vs_free_form_induction"]:
    result_path_decision_tree_vs_free_form_induction = "./results/ablations/setting_9_optimal_decision_tree_vs_free_form_induction.csv"
    experiments.induction(
        result_path=result_path_decision_tree_vs_free_form_induction,
        num_examples_list=[1],
        max_tree_depth_list=[2],
        temperature_list=[0],
        force_decision_tree_list=[True, False],
        include_description_list=[False],
        model_list=["gpt-4o", "gpt-o1", "gemini", "claude"],
    )

### Ablation Setting 10: Description for Induction
if run_config["setting_10_description_induction"]:
    result_path_description = "./results/ablations/setting_10_optimal_description_induction.csv"
    experiments.induction(
        result_path=result_path_description,
        num_examples_list=[1],
        max_tree_depth_list=[2],
        temperature_list=[0],
        force_decision_tree_list=[True],
        include_description_list=[True, False],
        dataset_list=["bankruptcy"],
        model_list=["gpt-4o", "gpt-o1", "gemini", "claude"],
    )

### Ablation Setting 11: Description for Embeddings
if run_config["setting_11_description_embeddings"]:
    result_path_description = "./results/ablations/setting_11_optimal_description_embeddings.csv"
    experiments.embeddings(
        result_path=result_path_description,
        num_trees_list=[5],
        num_examples_list=[1],
        max_tree_depth_list=[2],
        temperature_list=[1],
        append_raw_features_list=[False, True],
        include_description_list=[True, False],
        dataset_list=["bankruptcy"],
        model_list=["gpt-4o", "gpt-o1", "gemini", "claude"],
        classifier_list=["mlp"],
    )

### Ablation Setting 12: Three Downstream Classifier Variants
if run_config["setting_12_classifier_embeddings"]:
    result_path_classifier = "results/ablations/setting_12_optimal_classifier_embeddings.csv"
    experiments.embeddings(
        result_path=result_path_classifier,
        num_trees_list=[5],
        num_examples_list=[1],
        max_tree_depth_list=[0],
        temperature_list=[1],
        append_raw_features_list=[True],
        include_description_list=[False],
        model_list=["gpt-4o", "gpt-o1", "gemini", "claude"],
        classifier_list=["mlp", "hgbdt", "lr"],
    )
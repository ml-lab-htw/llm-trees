"""Tests for llm_trees._tree_ast: structural analysis of generated tree functions."""

from llm_trees._tree import compile_tree_function
from llm_trees._tree_ast import leaves, leaf_index, parse_tree_function


def test_off_path_nodes_none_do_not_break_leaf_walk():
    # nodes[1] is assigned but never consulted by the if-chain, so its
    # value (including None) must not affect the walk
    code = (
        "def predict(X: dict):\n"
        "    nodes = 3 * [None]\n"
        '    nodes[0] = X["a"] <= 1.0\n'
        '    nodes[1] = X["b"] <= 2.0\n'
        '    nodes[2] = X["a"] <= 0.5\n'
        "    if nodes[0]:\n"
        "        prediction = 0\n"
        "    else:\n"
        "        if nodes[2]:\n"
        "            prediction = 1\n"
        "        else:\n"
        "            prediction = 2\n"
        "    return prediction, nodes\n"
    )
    pt = parse_tree_function(code)
    # on-path values booleans, off-path entries None
    assert leaf_index([True, None, None], pt.structure) == 0
    assert leaf_index([False, None, True], pt.structure) == 1
    assert leaf_index([False, None, False], pt.structure) == 2
    # the unused node carries an arbitrary value: same leaves
    assert leaf_index([True, False, None], pt.structure) == 0
    assert leaf_index([False, True, True], pt.structure) == 1


def test_parse_non_conforming_function_returns_none():
    assert parse_tree_function("def predict(X):\n    return X['a'] * 2\n") is None
    assert parse_tree_function("def other(X):\n    pass\n") is None


def test_constant_tree_parses_as_depth_zero():
    # a single-leaf (constant) tree is a valid member of the generation
    # format: depth 0, no features, one leaf
    code = (
        "def predict(X: dict):\n"
        "    nodes = 0 * [None]\n"
        "    prediction = 0\n"
        "    return prediction, nodes\n"
    )
    pt = parse_tree_function(code)
    assert pt is not None
    assert pt.depth == 0
    assert pt.features == ()
    assert [leaf["prediction"] for leaf in leaves(pt)] == [0]
    # the leaf walk is the identity: every node vector (incl. empty) maps
    # to the single leaf
    assert leaf_index([], pt.structure) == 0
    assert leaf_index([True, False], pt.structure) == 0
    # non-numeric leaf values are accepted too
    str_code = code.replace("prediction = 0", 'prediction = "abnormal"')
    pt2 = parse_tree_function(str_code)
    assert pt2 is not None
    assert [leaf["prediction"] for leaf in leaves(pt2)] == ["abnormal"]


def test_parse_negative_leaf_values():
    # -1.0 is parsed by Python as UnaryOp(USub, Constant(1.0)); the leaf parser
    # must recognize signed numeric literals without executing expressions
    code = (
        "def predict(X: dict):\n"
        "    nodes = 1 * [None]\n"
        '    nodes[0] = X["a"] <= 1.0\n'
        "    if nodes[0]:\n"
        "        prediction = -1.0\n"
        "    else:\n"
        "        prediction = 2.0\n"
        "    return prediction, nodes\n"
    )
    pt = parse_tree_function(code)
    assert pt is not None
    assert pt.depth == 1
    assert pt.features == ("a",)
    assert pt.structure["children"][0]["prediction"] == -1.0
    assert pt.structure["children"][1]["prediction"] == 2.0

    # signed int literals and + prefixed literals
    code2 = (
        "def predict(X: dict):\n"
        "    nodes = 1 * [None]\n"
        '    nodes[0] = X["a"] <= 1.0\n'
        "    if nodes[0]:\n"
        "        prediction = -3\n"
        "    else:\n"
        "        prediction = +5\n"
        "    return prediction, nodes\n"
    )
    pt2 = parse_tree_function(code2)
    assert pt2 is not None
    assert pt2.structure["children"][0]["prediction"] == -3
    assert pt2.structure["children"][1]["prediction"] == 5

    # non-numeric signed expressions must NOT be evaluated
    code3 = (
        "def predict(X: dict):\n"
        "    nodes = 1 * [None]\n"
        '    nodes[0] = X["a"] <= 1.0\n'
        "    if nodes[0]:\n"
        "        prediction = 1 - 2\n"
        "    else:\n"
        "        prediction = 0\n"
        "    return prediction, nodes\n"
    )
    assert parse_tree_function(code3) is None


X_GET_FEATURE_CODE = (
    "def predict(X: dict):\n"
    "    nodes = 2 * [None]\n"
    '    nodes[0] = X.get("a") <= 1.0\n'
    '    nodes[1] = X["b"] <= 2.0\n'
    "    if nodes[0]:\n"
    "        prediction = 0\n"
    "    else:\n"
    "        if nodes[1]:\n"
    "            prediction = 1\n"
    "        else:\n"
    "            prediction = 2\n"
    "    return prediction, nodes\n"
)


def test_x_get_access_counts_as_feature():
    pt = parse_tree_function(X_GET_FEATURE_CODE)
    assert pt is not None
    assert pt.features == ("a", "b")


def test_unverifiable_feature_usage_rejected():
    # X referenced in a form we cannot analyze -> structure is unrecognized
    code = (
        "def predict(X: dict):\n"
        "    nodes = 1 * [None]\n"
        '    nodes[0] = X["a"] <= 1.0\n'
        "    if nodes[0]:\n"
        "        prediction = 0\n"
        "    else:\n"
        "        prediction = 1\n"
        "    return prediction, list(X.shape)\n"
    )
    assert parse_tree_function(code) is None


TRY_HIDDEN_DEPTH_CODE = (
    "def predict(X: dict):\n"
    "    nodes = 2 * [None]\n"
    '    nodes[0] = X["a"] <= 1.0\n'
    '    nodes[1] = X["b"] <= 2.0\n'
    "    if nodes[0]:\n"
    "        prediction = 0\n"
    "        try:\n"
    "            if nodes[1]:\n"
    "                prediction = 1\n"
    "        except Exception:\n"
    "            prediction = 2\n"
    "    else:\n"
    "        prediction = 3\n"
    "    return prediction, nodes\n"
)


def test_try_block_with_hidden_branch_makes_tree_unparsable():
    # the try block hides an extra ``if nodes[1]`` branch (runtime depth 2);
    # the parsed structure must not report the shallower valid-looking tree
    pt = parse_tree_function(TRY_HIDDEN_DEPTH_CODE)
    assert pt is None


def test_try_block_overwriting_leaf_makes_tree_unparsable():
    # a try branch that reassigns ``prediction`` would make the parsed leaf
    # value wrong, so the structure must be unrecognized
    code = (
        "def predict(X: dict):\n"
        "    nodes = 1 * [None]\n"
        '    nodes[0] = X["a"] <= 1.0\n'
        "    if nodes[0]:\n"
        '        prediction = "normal"\n'
        "    else:\n"
        '        prediction = "abnormal"\n'
        "        try:\n"
        '            prediction = "hidden"\n'
        "        except Exception:\n"
        "            pass\n"
        "    return prediction, nodes\n"
    )
    assert parse_tree_function(code) is None


def test_non_canonical_return_makes_tree_unparsable():
    # a branch returning a literal is a leaf assignment that cannot be
    # verified against the ``prediction`` assignment
    code = (
        "def predict(X: dict):\n"
        "    nodes = 1 * [None]\n"
        '    nodes[0] = X["a"] <= 1.0\n'
        "    if nodes[0]:\n"
        '        prediction = "normal"\n'
        "    else:\n"
        '        prediction = "abnormal"\n'
        '        return "hidden", nodes\n'
        "    return prediction, nodes\n"
    )
    assert parse_tree_function(code) is None


def test_post_decision_node_overwrite_makes_tree_unparsable():
    # nodes[0] is re-assigned after the if read it: the returned vector no
    # longer reproduces the decision, so both leaves would walk to the same
    # parsed leaf (mixed leaf probabilities)
    code = (
        "def predict(X: dict):\n"
        "    nodes = 1 * [None]\n"
        '    nodes[0] = X["a"] <= 1.0\n'
        "    if nodes[0]:\n"
        "        prediction = 0\n"
        "    else:\n"
        "        prediction = 1\n"
        "    nodes[0] = True\n"
        "    return prediction, nodes\n"
    )
    assert parse_tree_function(code) is None


def test_whole_container_reassignment_makes_tree_unparsable():
    # replacing the whole nodes container after decisions invalidates every
    # recorded decision path
    code = (
        "def predict(X: dict):\n"
        "    nodes = 1 * [None]\n"
        '    nodes[0] = X["a"] <= 1.0\n'
        "    if nodes[0]:\n"
        "        prediction = 0\n"
        "    else:\n"
        "        prediction = 1\n"
        "    nodes = 1 * [None]\n"
        "    return prediction, nodes\n"
    )
    assert parse_tree_function(code) is None


def test_write_to_undecided_slot_in_other_branch_still_parses():
    # nodes[1] is only READ inside the if branch; assigning it in the else
    # branch never overwrites a decided slot on any path
    code = (
        "def predict(X: dict):\n"
        "    nodes = 2 * [None]\n"
        '    nodes[0] = X["a"] <= 1.0\n'
        "    if nodes[0]:\n"
        '        nodes[1] = X["b"] <= 2.0\n'
        "        if nodes[1]:\n"
        "            prediction = 0\n"
        "        else:\n"
        "            prediction = 1\n"
        "    else:\n"
        '        nodes[1] = X["b"] <= 3.0\n'
        "        prediction = 2\n"
        "    return prediction, nodes\n"
    )
    pt = parse_tree_function(code)
    assert pt is not None
    assert pt.depth == 2
    assert pt.features == ("a", "b")


def test_prediction_overwrite_after_split_makes_tree_unparsable():
    # the executed prediction is overwritten after the split: the parsed
    # leaf values would be stale, and the true split on b is invisible to
    # the structure (and to complexity limits)
    code = (
        "def predict(X):\n"
        '    nodes = [X["a"] > 0]\n'
        "    if nodes[0]:\n"
        "        prediction = 0\n"
        "    else:\n"
        "        prediction = 1\n"
        '    prediction = 2 if X["b"] > 0 else 3\n'
        "    return prediction, nodes\n"
    )
    assert parse_tree_function(code) is None


def test_prediction_constant_overwrite_after_split_makes_tree_unparsable():
    # a constant overwrite cannot silently retain the obsolete leaf values
    code = (
        "def predict(X):\n"
        '    nodes = [X["a"] > 0]\n'
        "    if nodes[0]:\n"
        "        prediction = 0\n"
        "    else:\n"
        "        prediction = 1\n"
        "    prediction = 7\n"
        "    return prediction, nodes\n"
    )
    assert parse_tree_function(code) is None


def test_prediction_overwrite_after_nested_split_makes_tree_unparsable():
    # the same pattern one level deeper: an overwrite in a branch that
    # itself contains a node-test if
    code = (
        "def predict(X: dict):\n"
        "    nodes = 2 * [None]\n"
        '    nodes[0] = X["a"] > 0\n'
        "    if nodes[0]:\n"
        '        nodes[1] = X["b"] > 0\n'
        "        if nodes[1]:\n"
        "            prediction = 0\n"
        "        else:\n"
        "            prediction = 1\n"
        '        prediction = 2 if X["a"] > 0 else 3\n'
        "    else:\n"
        "        prediction = 4\n"
        "    return prediction, nodes\n"
    )
    assert parse_tree_function(code) is None


def test_prediction_default_before_split_makes_tree_unparsable():
    # a pre-split default is dead code outside the verifiable generation
    # format; the rule is "prediction assignments live in terminal branches"
    code = (
        "def predict(X):\n"
        '    nodes = [X["a"] > 0]\n'
        "    prediction = 5\n"
        "    if nodes[0]:\n"
        "        prediction = 0\n"
        "    else:\n"
        "        prediction = 1\n"
        "    return prediction, nodes\n"
    )
    assert parse_tree_function(code) is None


def test_terminal_branch_prediction_double_assignment_uses_last():
    # terminal branches keep execution-order semantics: the last
    # assignment holds at runtime, so the parsed leaf value is the last one
    code = (
        "def predict(X: dict):\n"
        "    nodes = 1 * [None]\n"
        '    nodes[0] = X["a"] > 0\n'
        "    if nodes[0]:\n"
        "        prediction = 0\n"
        "        prediction = 9\n"
        "    else:\n"
        "        prediction = 1\n"
        "    return prediction, nodes\n"
    )
    pt = parse_tree_function(code)
    assert pt is not None
    assert pt.structure["children"][0]["prediction"] == 9
    assert pt.structure["children"][1]["prediction"] == 1


def _value_side_mutation_code(mutation: str) -> str:
    """Tree whose else branch mutates the node vector through the assignment
    value (target slot 1 is not read on that path, so target-based checks pass)."""
    return (
        "def predict(X: dict):\n"
        "    nodes = 2 * [None]\n"
        '    nodes[0] = X["a"] <= 1.0\n'
        '    nodes[1] = X["b"] <= 2.0\n'
        "    if nodes[0]:\n"
        "        if nodes[1]:\n"
        "            prediction = 0\n"
        "        else:\n"
        "            prediction = 1\n"
        "    else:\n"
        f"        nodes[1] = {mutation}\n"
        "        prediction = 2\n"
        "    return prediction, nodes\n"
    )


def test_value_side_node_mutation_makes_tree_unparsable():
    # the reported case: nodes.pop(0) in the assignment VALUE shifts the
    # container after the nodes[0] decision, so the returned vector no longer
    # reproduces the decision and two runtime leaves walk to the same parsed
    # leaf (predict stays correct, predict_proba silently wrong)
    code = _value_side_mutation_code("nodes.pop(0)")
    assert parse_tree_function(code) is None


def test_value_side_node_mutations_make_tree_unparsable():
    for mutation in [
        "nodes.pop()",
        "nodes.insert(0, True)",
        "nodes.append(True)",
        "nodes.extend([True])",
        "nodes.remove(True)",
        "nodes.clear()",
        "nodes.reverse()",
        "nodes.sort()",
        "nodes.__setitem__(0, True)",
        "nodes.__delitem__(0)",
    ]:
        assert parse_tree_function(_value_side_mutation_code(mutation)) is None, mutation


def test_read_only_nodes_rhs_after_decision_still_parses():
    # copying from the container is read-only: the decided slots are intact
    # in the returned vector, so the leaf walk still matches the execution
    code = (
        "def predict(X: dict):\n"
        "    nodes = 2 * [None]\n"
        '    nodes[0] = X["a"] <= 1.0\n'
        '    nodes[1] = X["b"] <= 2.0\n'
        "    if nodes[0]:\n"
        "        if nodes[1]:\n"
        "            prediction = 0\n"
        "        else:\n"
        "            prediction = 1\n"
        "    else:\n"
        "        prediction = 2\n"
        "        nodes[1] = nodes[0]\n"
        "    return prediction, nodes\n"
    )
    pt = parse_tree_function(code)
    assert pt is not None
    assert pt.depth == 2
    predict = compile_tree_function(code)
    for a, b in [(0.5, 1.0), (0.5, 3.0), (2.0, 1.0), (2.0, 3.0)]:
        pred, nodes = predict({"a": a, "b": b})
        assert [leaf["prediction"] for leaf in leaves(pt)][leaf_index(nodes, pt.structure)] == pred


def test_pre_decision_value_side_mutation_still_parses():
    # a mutation before any decision only changes the container before
    # decisions are read from it: the returned vector is still consistent
    code = (
        "def predict(X: dict):\n"
        "    nodes = 3 * [None]\n"
        "    nodes[1] = nodes.pop(0)\n"
        '    nodes[0] = X["a"] <= 1.0\n'
        "    if nodes[0]:\n"
        "        prediction = 0\n"
        "    else:\n"
        "        prediction = 1\n"
        "    return prediction, nodes\n"
    )
    pt = parse_tree_function(code)
    assert pt is not None
    assert pt.depth == 1

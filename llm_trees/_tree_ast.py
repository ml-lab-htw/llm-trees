"""AST-based structural analysis of generated tree functions.

Parses the ``predict(X)`` function produced by the two-step LLM workflow into an
explicit tree structure. This powers (1) complexity-constraint validation,
(2) the ``support_``/``sparsity_`` fitted attributes, and (3) leaf-based
``predict_proba`` (leaf class fractions).

The parser recognizes the generation format: ``nodes[i] = <condition>``
assignments (anywhere in the function) and an ``if``/``elif``/``else`` chain
whose tests are ``nodes[i]`` (or ``not nodes[i]``), with each terminal branch
ending in ``prediction = <constant>``. A function without any ``if`` is a
single-leaf (constant) tree: depth 0, no features, one leaf. Off-path
``nodes`` entries may be ``None``; only on-path values are read during the
leaf walk. Returns ``None`` when the function does not follow the expected
form.

Every statement in every branch must be verifiable control flow — a
``nodes``/``nodes[i]``/``prediction`` assignment, an ``if`` (with a
``nodes[i]`` test), or ``return prediction, nodes``. Anything else
(``try``/``except`` blocks, loops, ``with``, ``raise``, bare expressions,
other assignments, non-canonical returns) can hide branches or leaf
assignments the structure cannot see, so the whole tree is unrecognized.

The returned node vector must also reproduce the decisions actually taken:
once an index has been read by an ``if`` on an execution path, no later
statement on that path may write to that ``nodes[i]`` slot or reassign the
whole ``nodes`` container. Post-decision overwrites (including
whole-container replacement) make the tree unrecognized, so the leaf walk
can never map two different runtime leaves to the same parsed leaf. The
same applies to mutations of the container through the assignment value
(e.g. ``nodes[1] = nodes.pop(0)``): a method call on ``nodes`` in the RHS
of a later assignment on that path can shift, reorder, or clear the slots
(including decided ones), so it also makes the tree unrecognized. A
mutation before any decision on the path is fine: only the post-mutation
container is ever read by decisions, so the returned vector still
reproduces them.

Likewise, the returned prediction must reproduce the leaf values: a
``prediction`` assignment lives in a terminal branch (execution order
there is verifiable — the last assignment wins). A ``prediction = ...``
in the same statement list as a node-test ``if`` is either a dead default
(before the if) or an overwrite of the subtree's result (after the if,
which would leave the parsed leaf values stale and hide any split the
overwrite expression introduces), so it makes the tree unrecognized.
"""

import ast
from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["ParsedTree", "parse_tree_function", "leaf_index", "leaves"]


@dataclass(frozen=True)
class ParsedTree:
    """Parsed tree structure, referenced features, and depth.

    ``structure``: nested dict. Internal node:
    ``{"node": int, "negated": bool, "children": (left, right)}``.
    Leaf: ``{"prediction": value, "index": int}`` (DFS leaf index).
    """

    structure: dict
    features: tuple
    depth: int


def _node_test(test_ast) -> tuple[int, bool] | None:
    """Return ``(node_idx, negated)`` for ``nodes[i]`` / ``not nodes[i]``, else ``None``."""
    t = test_ast
    negated = False
    if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not):
        negated = True
        t = t.operand
    if (
        isinstance(t, ast.Subscript)
        and isinstance(t.value, ast.Name)
        and t.value.id == "nodes"
        and isinstance(t.slice, ast.Constant)
        and isinstance(t.slice.value, int)
    ):
        return t.slice.value, negated
    return None


def _leaf_value(value_ast):
    """The constant value of a leaf assignment's RHS, or None if unsupported.

    Accepts plain constants (strings, numerics, booleans) and signed numeric
    literals: Python's AST represents ``-1.0`` as ``UnaryOp(USub,
    Constant(1.0))``, so signed numerics are folded without evaluating any
    arbitrary expression.
    """
    if isinstance(value_ast, ast.UnaryOp) and isinstance(value_ast.operand, ast.Constant):
        operand = value_ast.operand.value
        if isinstance(operand, (int, float)) and not isinstance(operand, bool):
            if isinstance(value_ast.op, ast.USub):
                return -operand
            if isinstance(value_ast.op, ast.UAdd):
                return operand
        return None
    if isinstance(value_ast, ast.Constant):
        return value_ast.value
    return None


def _terminal(stats) -> dict | None:
    """Parse a terminal branch (no nested ``if``) ending in ``prediction = <const>``."""
    prediction = None
    found = False
    for st in stats:
        if (
            isinstance(st, ast.Assign)
            and len(st.targets) == 1
            and isinstance(st.targets[0], ast.Name)
            and st.targets[0].id == "prediction"
        ):
            prediction = _leaf_value(st.value)
            found = True
    # an unsupported RHS (arbitrary expression, ``None``, ...) cannot be
    # verified, so the branch is unrecognized
    if not found or prediction is None:
        return None
    return {"prediction": prediction}


def _known_assignment_target(target) -> bool:
    """True for ``nodes = ...``, ``nodes[i] = ...``, or ``prediction = ...``."""
    if isinstance(target, ast.Name):
        return target.id in ("nodes", "prediction")
    return (
        isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id == "nodes"
        and isinstance(target.slice, ast.Constant)
        and isinstance(target.slice.value, int)
    )


def _canonical_return(st) -> bool:
    """True only for ``return prediction, nodes`` (the verified exit form)."""
    v = st.value
    return (
        isinstance(v, ast.Tuple)
        and len(v.elts) == 2
        and isinstance(v.elts[0], ast.Name)
        and v.elts[0].id == "prediction"
        and isinstance(v.elts[1], ast.Name)
        and v.elts[1].id == "nodes"
    )


def _parse_branch(stats, leaf_counter: list[int]) -> dict | None:
    for st in stats:
        if isinstance(st, ast.If):
            continue
        if isinstance(st, ast.Return):
            if not _canonical_return(st):
                return None
            continue
        if (
            isinstance(st, ast.Assign)
            and len(st.targets) == 1
            and _known_assignment_target(st.targets[0])
        ):
            continue
        # any other statement (try/except, loops, with, raise, side effects,
        # other assignments) is control flow that cannot be verified
        return None
    ifs = [st for st in stats if isinstance(st, ast.If)]
    if ifs:
        if len(ifs) != 1:
            return None
        if_node = ifs[0]
        test = _node_test(if_node.test)
        if test is None:
            return None
        # a ``prediction`` assignment in the same statement list as a
        # node-test ``if`` is outside the verifiable generation format:
        # before the if it is a default the subtree unconditionally
        # overwrites (dead code), after the if it overwrites the subtree's
        # result, leaving the parsed leaf values stale (and any split the
        # overwrite expression introduces invisible to the structure and
        # to complexity limits). Terminal branches keep execution-order
        # semantics (last assignment wins) via ``_terminal``.
        if any(
            isinstance(st, ast.Assign)
            and len(st.targets) == 1
            and isinstance(st.targets[0], ast.Name)
            and st.targets[0].id == "prediction"
            for st in stats
        ):
            return None
        idx, negated = test
        left = _parse_branch(if_node.body, leaf_counter)
        right = _parse_branch(if_node.orelse, leaf_counter)
        if left is None or right is None:
            return None
        return {"node": idx, "negated": negated, "children": (left, right)}
    leaf = _terminal(stats)
    if leaf is None:
        return None
    leaf["index"] = leaf_counter[0]
    leaf_counter[0] += 1
    return leaf


def _calls_nodes_method(value_ast) -> bool:
    """True if the expression calls any method on the ``nodes`` container.

    A bound-method call on the container in an assignment value (``pop``,
    ``insert``, ``append``, ``sort``, ..., or dunder forms such as
    ``__setitem__``) can mutate or reorder it, shifting or clearing the
    slots the leaf walk reads. Even the read-only list methods (``copy``,
    ``count``, ``index``) are not part of the verifiable generation format,
    so any method call on ``nodes`` inside an RHS is treated as
    unverifiable.
    """
    for sub in ast.walk(value_ast):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == "nodes"
        ):
            return True
    return False


def _decisions_live_after(stats, decided: frozenset[int]) -> frozenset[int] | None:
    """Scan a branch for node writes that would invalidate decision replay.

    ``decided`` is the set of node indices already read by an ``if`` on the
    current execution path. A write to an already-decided slot — or a
    whole-container ``nodes = ...`` reassignment after any decision — would
    make the returned vector disagree with the value the decision actually
    used, so the leaf walk could not be reconstructed: the tree is
    unparseable. A method call on ``nodes`` in the value of a later
    assignment on the path is the same class of write (``nodes.pop(0)``
    shifts the decided slots even though the assignment target is a
    different slot), so it is rejected as well. Mutations before any
    decision on the path leave the post-mutation container, which is what
    the decisions read, so they stay accepted. Returns the live decided set
    after the branch, or ``None``. Branch rejoins conservatively union the
    decided sets of both paths.
    """
    for st in stats:
        if isinstance(st, ast.If):
            test = _node_test(st.test)
            if test is None:
                continue  # non-node test: the structure parse rejects it
            idx, _negated = test
            after = decided | frozenset((idx,))
            left = _decisions_live_after(st.body, after)
            if left is None:
                return None
            right = _decisions_live_after(st.orelse, after)
            if right is None:
                return None
            decided = left | right
            continue
        if isinstance(st, ast.Assign) and len(st.targets) == 1:
            t = st.targets[0]
            if isinstance(t, ast.Name) and t.id == "nodes":
                if decided:
                    return None
            elif (
                isinstance(t, ast.Subscript)
                and isinstance(t.value, ast.Name)
                and t.value.id == "nodes"
                and isinstance(t.slice, ast.Constant)
                and isinstance(t.slice.value, int)
            ):
                if t.slice.value in decided:
                    return None
            if decided and _calls_nodes_method(st.value):
                return None
    return decided


def _depth(node: dict) -> int:
    if "node" in node:
        return 1 + max(_depth(node["children"][0]), _depth(node["children"][1]))
    return 0


def _features(fn: ast.FunctionDef):
    """Referenced feature names, or ``None`` if feature usage cannot be verified.

    Recognized access forms: ``X["name"]`` and ``X.get("name")``. Any other
    reference to ``X`` (e.g. ``X.shape``) means the feature set cannot be
    verified, so the structure is reported as unrecognized.
    """
    feats: list[str] = []
    recognized: set[int] = set()
    for st in ast.walk(fn):
        if (
            isinstance(st, ast.Subscript)
            and isinstance(st.value, ast.Name)
            and st.value.id == "X"
            and isinstance(st.slice, ast.Constant)
            and isinstance(st.slice.value, str)
        ):
            recognized.add(id(st.value))
            if st.slice.value not in feats:
                feats.append(st.slice.value)
        elif (
            isinstance(st, ast.Call)
            and isinstance(st.func, ast.Attribute)
            and st.func.attr == "get"
            and isinstance(st.func.value, ast.Name)
            and st.func.value.id == "X"
            and len(st.args) == 1
            and not st.keywords
            and isinstance(st.args[0], ast.Constant)
            and isinstance(st.args[0].value, str)
        ):
            recognized.add(id(st.func.value))
            if st.args[0].value not in feats:
                feats.append(st.args[0].value)
    for st in ast.walk(fn):
        if isinstance(st, ast.Name) and st.id == "X" and id(st) not in recognized:
            return None
    return tuple(feats)


def parse_tree_function(tree_fcn_str: str) -> ParsedTree | None:
    """Parse the tree function code; returns ``None`` when the form is unexpected."""
    try:
        module = ast.parse(tree_fcn_str)
    except SyntaxError:
        return None
    fn = next(
        (
            n
            for n in module.body
            if isinstance(n, ast.FunctionDef) and n.name == "predict"
        ),
        None,
    )
    if fn is None:
        return None
    body = fn.body
    # a leading string docstring is a verified no-op (real model outputs
    # include them); any other leading expression is not
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    leaf_counter = [0]
    structure = _parse_branch(body, leaf_counter)
    if structure is None:
        return None
    if _decisions_live_after(body, frozenset()) is None:
        return None  # the returned nodes would not reproduce the decisions
    features = _features(fn)
    if features is None:
        return None  # feature usage cannot be verified
    return ParsedTree(structure=structure, features=features, depth=_depth(structure))


def leaf_index(nodes: Sequence[Any], structure: dict) -> int:
    """Walk the if-chain with a sample's on-path ``nodes`` values; returns the DFS leaf index."""
    node = structure
    while "node" in node:
        val = bool(nodes[node["node"]])
        if node["negated"]:
            val = not val
        node = node["children"][0] if val else node["children"][1]
    return node["index"]


def leaves(pt: ParsedTree) -> list[dict]:
    """All leaves of a parsed tree, ordered by DFS leaf index."""
    out: list[dict] = []

    def walk(node: dict) -> None:
        if "node" in node:
            walk(node["children"][0])
            walk(node["children"][1])
        else:
            out.append(node)

    walk(pt.structure)
    out.sort(key=lambda leaf: leaf["index"])
    return out

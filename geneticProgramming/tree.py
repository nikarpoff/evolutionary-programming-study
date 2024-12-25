import random

import numpy as np


class Node:
    def __init__(self, label: str):
        self.label = label

    def build(self) -> str:
        pass

    def count(self, x) -> np.array:
        pass

    def get_depth(self) -> int:
        pass


class TerminalNode(Node):
    def __init__(self, label: str):
        super().__init__(label)

    def build(self) -> str:
        return self.label

    def count(self, x) -> np.array:
        # We consider that all labels with x<id> are variables while other labels - constants.
        if 'x' in self.label:
            variable_id = int(self.label.replace('x', '')) - 1
            return x[:, variable_id]
        else:
            return int(self.label)

    def get_depth(self) -> int:
        return 0


class UnaryFunctionNode(Node):
    def __init__(self, label: str, function, child):
        super().__init__(label)
        self.child = child
        self.function = function

    def build(self) -> str:
        if self.child is None:
            return f'{self.label}()'

        return f'{self.label}({self.child.build()})'

    def count(self, x) -> np.array:
        return self.function(self.child.count(x))

    def get_depth(self) -> int:
        return 1 + self.child.get_depth()


class BinaryFunctionNode(Node):
    def __init__(self, label: str, function, left: Node, right: Node):
        super().__init__(label)
        self.function = function
        self.left = left
        self.right = right

    def build(self) -> str:
        return f'{self.left.build()} {self.label} {self.right.build()}'

    def count(self, x) -> np.array:
        return self.function(self.left.count(x), self.right.count(x))

    def get_depth(self) -> int:
        return 1 + max(self.left.get_depth(), self.right.get_depth())


class TreeGenerator:
    def __init__(self, terminals: list, unary_functions: dict, binary_functions: dict, functions: list):
        self.terminals = terminals
        self.listed_unary_functions = list(unary_functions.keys())
        self.listed_binary_functions = list(binary_functions.keys())
        self.unary_functions = unary_functions
        self.binary_functions = binary_functions
        self.functions = functions

    def generate(self, depth):
        # If depth is 0 -> generate terminal.
        if depth == 0:
            return TerminalNode(random.choice(self.terminals))
        else:
            if depth == 1:
                # With probability 1/2 we can generate terminal not in last layer.
                if random.choice([True, False]):
                    return TerminalNode(random.choice(self.terminals))

            function = random.choice(self.functions)

            if function in self.unary_functions:
                return UnaryFunctionNode(function, self.unary_functions[function], self.generate(depth - 1))
            else:
                return BinaryFunctionNode(function, self.binary_functions[function],
                                          self.generate(depth - 1), self.generate(depth - 1))


def collect_nodes(root: Node) -> list:
    """
    Collects all nodes to list.
    :param root: root of tree
    :return: list of nodes
    """
    nodes = []

    def traverse(node):
        nodes.append(node)

        if isinstance(node, UnaryFunctionNode):
            traverse(node.child)
        elif isinstance(node, BinaryFunctionNode):
            traverse(node.left)
            traverse(node.right)

    traverse(root)
    return nodes


def replace_node(root: Node, node: Node, replacement: Node):
    def equals_node(node1: Node, node2: Node) -> bool:
        return isinstance(node1, type(node2)) and node1.label == node2.label and node1.get_depth() == node2.get_depth()

    def search(current):
        if isinstance(current, UnaryFunctionNode):
            if equals_node(current.child, node):
                return current

            search(current.child)
        elif isinstance(current, BinaryFunctionNode):
            if equals_node(current.left, node) or equals_node(current.right, node):
                return current

            search(current.left)
            search(current.right)

    found_node = search(root)

    if found_node is not None:
        if isinstance(found_node, UnaryFunctionNode):
            found_node.child = replacement
        elif isinstance(node, BinaryFunctionNode):
            if equals_node(found_node.left, node):
                found_node.left = replacement
            else:
                found_node.right = replacement
    else:
        if isinstance(root, UnaryFunctionNode):
            root.child = replacement
        elif isinstance(node, BinaryFunctionNode):
            root.left = replacement

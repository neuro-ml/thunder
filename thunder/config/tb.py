from collections import defaultdict
from functools import lru_cache
from typing import Any

from lightning.pytorch.loggers import TensorBoardLogger

from .hyperparams import register


def log_connectome_layer(logger: TensorBoardLogger, name: str, layer):
    root = parse(name, layer._container.edges)
    # these 3 lines are copy-pasted from torch
    stats = RunMetadata(step_stats=StepStats(dev_stats=[DeviceStepStats(device="/device:CPU:0")]))
    gr = GraphDef(node=root, versions=VersionDef(producer=22))
    logger.experiment._get_file_writer().add_graph((gr, stats))


try:
    from connectome import CallableLayer
    from connectome.engine import IdentityEdge, TreeNode
    from tensorboard.compat.proto.config_pb2 import RunMetadata
    from tensorboard.compat.proto.graph_pb2 import GraphDef
    from tensorboard.compat.proto.node_def_pb2 import NodeDef
    from tensorboard.compat.proto.step_stats_pb2 import DeviceStepStats, StepStats
    from tensorboard.compat.proto.versions_pb2 import VersionDef

    register(TensorBoardLogger, CallableLayer)(log_connectome_layer)

except ImportError:
    pass


def node_proto(name, inputs, op):
    return NodeDef(name=name.encode(encoding="utf_8"), op=op, input=inputs)


def remove_identity(nodes):
    @lru_cache(None)
    def _remove(node: TreeNode):
        if node.is_leaf:
            return node

        if isinstance(node.edge, IdentityEdge) and node.parents[0].name == node.name:
            return _remove(node.parents[0])

        return TreeNode(node.name, (node.edge, list(map(_remove, node.parents))), node.details)

    return tuple(map(_remove, nodes))


def node_to_str(root_name, node, unique_names):
    def name_in_scope(details, name, sentinel: Any):
        local = unique_names[details, name]
        if sentinel in local:
            return local[sentinel]

        if not local:
            result = name
        else:
            result = f'{name}[{len(local)}]'

        local[sentinel] = result
        return result

    def scope_id(scope):
        if scope is None:
            return root_name
        return f'{root_name}/{scope_id(scope.parent)}/{name_in_scope(scope.parent, scope.layer, scope)}'.strip('/')

    return (scope_id(node.details) + '/' + name_in_scope(node.details, node.name, node)).strip('/')


def parse(root_name, edges):
    # optimize out unneeded edges
    edges = TreeNode.to_edges(remove_identity(TreeNode.from_edges(edges).values()))
    nodes = {}
    unique_names = defaultdict(dict)
    all_nodes = set()
    for edge, inputs, output in edges:
        all_nodes.update(inputs)

        nodes[output] = node_proto(
            node_to_str(root_name, output, unique_names), [node_to_str(root_name, x, unique_names) for x in inputs],
            type(edge).__name__,
        )

    for node in all_nodes - set(nodes):
        nodes[node] = node_proto(node_to_str(root_name, node, unique_names), [], 'Input')

    return list(nodes.values())

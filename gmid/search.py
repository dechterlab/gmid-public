from constants import *
from gmid.graph_algorithms import MessageGraph


def message_graph_as_heuristic(message_graph, elim_order, pseudo_tree, mini_buckets, num_var, num_fun):
    """
    This function exports a static mini-bucket heuristic table from a message_graph that stores
    messages of mini bucket elimination or weighted mini bucket elimination.

    call this function after executing CTE or WeightedMBE
    to create a dictionary of heuristic_info
    after this step, write a file using a function in fileio,
    ``write_mini_bucket_heuristic_from_info(file_name, heuristic_info)``

    Parameters
    ----------

    message_graph: MessageGraph.message_graph
        networkx graph that contains factors and messages as a result of running mbe like algorithms


    elim_order: List
        a list of ints (var ids) following the **elimination order**

    pseudo_tree: List
        a list of ints (var ids) encoding a pseudo tree;
        position is var id and pseudo_tree[var_id] is the id of its parent; -1 indicates it is the root

    mini_buckets: defaultdict(SortedSet)
        a defaultdict of SortedSet
            key is the variable id
            value is a SortedSet of mini_bucket id (var_id, mini_bucket count)

    num_var: int
        the total number of variables

    num_fun: int
        the total number of functions

    Returns
    -------
    heuristic_info: Dict
        heuristic information required to be exported
        num vars, num messages, the num id of the first message,
        dict of buckets, bucket[var_id] is the list of messages needed to evaluate heuristic
        dict of factors, factor[factor_id] is factor object or valuation object with scope, and table

    """
    def path_to_ancestors(var_from, var_to):
        """
        find a path from ``var_from`` (exclusive) to ``var_to`` (inclusive) in the pseudo tree and
        return a list of var_ids

        use this function to know what buckets a message pass through (a message is stored in an
        edge between two nodes)
        """
        path_through = []
        next_node = pseudo_tree[var_from]
        while next_node != var_to:
            path_through.append(next_node)
            next_node = pseudo_tree[next_node]
        if var_to != -1:
            path_through.append(var_to)
        return path_through

    heuristic_info = {}
    heuristic_info['num_var'] = num_var
    heuristic_info['msg_id_start'] = num_fun
    msg_id = num_fun
    bucket_msg = {i: [] for i in range(num_var)}
    msg_indexer = {}       # off set by num_fun; the message id is indexer ind + num_fun

    for var in elim_order[:-1]:     # no need to process the last layer it gives global upper bound
        for node in mini_buckets[var]:
            if message_graph.node[node]['msg'] is not None:             # node store 'msg' when it is a root (for mbe)
                msg_indexer[msg_id] = message_graph.node[node]['msg']
                buckets_pass_through = path_to_ancestors(var, -1)       # add constant msg to bucket_msg list
                for b in buckets_pass_through:
                    bucket_msg[b].append(msg_id)
            else:
                for node_from, node_to in message_graph.out_edges_iter([node]):
                    if elim_order.index(node_from[0]) < elim_order.index(node_to[0]): # msg passed down in the elim tree
                        msg_indexer[msg_id] = message_graph.edge[node_from][node_to]['msg']
                        break       # one message out going from the current node
                buckets_pass_through = path_to_ancestors(node_from[0], node_to[0])
                for b in buckets_pass_through:
                    bucket_msg[b].append(msg_id)
            msg_id += 1

    heuristic_info['bucket_msg'] = bucket_msg
    heuristic_info['msg_indexer'] = msg_indexer
    heuristic_info['num_msg'] = len(msg_indexer)

    return heuristic_info




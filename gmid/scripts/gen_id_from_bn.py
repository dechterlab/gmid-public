PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gmid.fileio import *
import random
import pprint
pp = pprint.PrettyPrinter(indent=4)


def generate_id_from_bn(name="BN_0.uai", dec_ratio=0.05):
    ### read a bn file
    bn_path = os.path.join(BETA_PATH, name)
    bn_dict = read_uai_bn(bn_path)
    num_vars = len(bn_dict)
    max_scope_size = max([len(bn_dict[el]['scope']) for el in bn_dict])
    max_domain_size = max([bn_dict[el]['domain_size'] for el in bn_dict])
    print("num_vars:{}\tmax_domain_size:{}\tmax_scope_size:{}".format(num_vars, max_scope_size, max_domain_size))
    pp.pprint(bn_dict)

    ### build a bn dag
    bn_dag = nx.DiGraph()
    for var_id in sorted(bn_dict):
        bn_dag.add_edges_from([(pa, var_id) for pa in bn_dict[var_id]['parents']])

    ### find a toplogical sort of variables from bn_dag
    vars_toposort = nx.topological_sort(bn_dag)
    print("vars_toposort:{}".format(vars_toposort))

    ### randomly pick up decision variables, at least dec_ratio
    num_dec_vars = int(num_vars*dec_ratio)+1
    decision_variables = sorted(random.sample(range(num_vars), num_dec_vars), key=lambda x: vars_toposort.index(x))
    print("dec_vars {}/{}:{}".format(num_dec_vars, num_vars, decision_variables))

    ### define partial variable ordering
    partial_variable_ordering = []
    history = set()
    for dec_var in decision_variables:
        current_block = [el for el in bn_dict[dec_var]['parents'] if el not in history]
        if len(current_block) > 0:
            partial_variable_ordering.append(current_block)     # add observation
        history.update(current_block)
        partial_variable_ordering.append([dec_var])
        history.update([dec_var])
    hidden = [el for el in range(num_vars) if el not in history]
    partial_variable_ordering.append(hidden)
    partial_variable_ordering.reverse()     # elimination order
    print("partial_variable_ordering:{}".format(partial_variable_ordering))

    ### create influence diagram without utility functions
    influence_diagram = nx.DiGraph()
    for n in range(num_vars):
        if n in decision_variables:
            influence_diagram.add_node(n, {'node_type':'D',
                                           'domain_size' : bn_dict[n]['domain_size'],
                                           'parents':bn_dict[n]['parents'], 'scope':bn_dict[n]['scope'],
                                           'table_length':bn_dict[n]['table_length'], 'table':bn_dict[n]['table']})
        else:
            influence_diagram.add_node(n, {'node_type':'C',
                                           'domain_size': bn_dict[n]['domain_size'],
                                           'parents':bn_dict[n]['parents'], 'scope':bn_dict[n]['scope'],
                                           'table_length':bn_dict[n]['table_length'], 'table':bn_dict[n]['table']})
    for var_id in range(num_vars):
        if var_id in decision_variables:
            for pa in bn_dict[var_id]['parents']:
                influence_diagram.add_edge(pa, var_id, {'edge_type':'informational_arc'})
        else:
            for pa in bn_dict[var_id]['parents']:
                influence_diagram.add_edge(pa, var_id, {'edge_type': 'probability_arc'})

    ### add utility nodes and functions
    ### 1 utility per decision, scope of the utility function taken from decision | hidden state | obs
    ### values are all int [0, 10] and scope size is randomly chosen less than max scope size
    num_util_nodes = num_dec_vars
    utility_dict = {}
    for ind, n in enumerate(range(num_vars, num_vars+num_util_nodes)):
        utility_dict[n] = {}
        ### random scope of utility function
        current_scope_size = random.randint(2, max(3,max_scope_size))
        current_dec_var = decision_variables[ind]
        current_obs = bn_dict[current_dec_var]['parents']
        var_pool = hidden + current_obs
        current_pa = random.sample(var_pool, current_scope_size-1)
        current_scope = current_pa + [current_dec_var]
        utility_dict[n]['scope'] = current_scope
        utility_dict[n]['parents'] = current_scope     # parents in influence diagram, same as scope
        ### create a random function, rows
        current_table_length = 1
        for var_id in current_scope:
            current_table_length *= bn_dict[var_id]['domain_size']
        # current_table = list(np.randi.uniform(0, 1, current_table_length))     # uniform random variable
        current_table = list(np.random.randint(0, 10, current_table_length))
        utility_dict[n]['table'] = current_table
        utility_dict[n]['table_length'] = current_table_length

    for n in range(num_vars, num_vars+num_util_nodes):
        influence_diagram.add_node(n, {'node_type': 'U',
                                       'parents': utility_dict[n]['parents'],
                                       'scope': utility_dict[n]['scope'],
                                       'table_length': utility_dict[n]['table_length'],
                                       'table': utility_dict[n]['table']})
    for n in range(num_vars, num_vars+num_util_nodes):
        for pa in utility_dict[n]['parents']:
            influence_diagram.add_edge(pa, n, {'edge_type':'utility_arc'})

    ### write pvo, id files
    problem_name = "ID_from_" + name.replace(".uai", '')
    write_pvo_from_partial_elim_order(os.path.join(BETA_PATH, problem_name + '.pvo'), partial_variable_ordering)
    var_types = ['D' if el in decision_variables else 'C' for el in range(num_vars)]
    func_types = ['P']*(num_vars-num_dec_vars) + ['U']*num_util_nodes       # enumerate probability and then utility
    write_id_from_types(os.path.join(BETA_PATH, problem_name + '.id'), var_types, func_types)

    ### write uai file
    write_uai_from_nx_graph(os.path.join(BETA_PATH, problem_name + ".uai"), influence_diagram)
    # temporal_ordering = []
    # for block in reversed(partial_variable_ordering):   # reverse elim blocks
    #     for v in block:
    #         temporal_ordering.append(v)
    # print("temporal_ordering:{}".format(temporal_ordering))
    # assert len(temporal_ordering) == num_vars, "the number of variables did not match"
    # write_erg_from_nx_graph(os.path.join(BETA_PATH, problem_name + ".erg"), influence_diagram, temporal_ordering)
    # write_limid_from_nx_graph(os.path.join(BETA_PATH, problem_name + ".limid.uai"), influence_diagram)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_id_from_bn(sys.argv[1])
    else:
        generate_id_from_bn()






PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
import random
from gmid.constants import *

def create_cpt(diagram, domain_size, parent_nodes):
    cpt = []
    configs = 1  # product of domain size of all parent nodes
    for n in parent_nodes:
        configs *= diagram.node[n]['domain_size']
    for _ in range(configs):
        conditional_cpt = np.random.rand(domain_size)  # generate 1 conditional cpt
        conditional_cpt = conditional_cpt/sum(conditional_cpt)  # normalize
        cpt.extend(list(conditional_cpt))
    return cpt  # return a list of random numbers


def set_parents(diagram, current_node, min_parents, max_parents, set_decision_parent=True):
    num_parents = random.randint(min_parents, max_parents)
    decision_node = None
    previous_states = []
    for n in diagram.nodes_iter():
        if diagram.node[n]['step'] == 0:
            previous_states.append(n)
        if diagram.node[n]['step'] == 1:
            decision_node = n

    if num_parents > len(previous_states) + 1:
        num_parents = len(previous_states) + 1

    if set_decision_parent:
        parents = list(np.random.choice(previous_states, num_parents-1, replace=False))
        parents.append(decision_node)
    else:
        parents = list(np.random.choice(previous_states+[decision_node], num_parents, replace=False))

    for p in parents:
        diagram.add_edge(p, current_node)
    return sorted(parents)


def create_utility(diagram, parents, min_u, max_u):
    configs = 1
    for n in parents:
        configs *= diagram.node[n]['domain_size']
    table = np.random.rand(configs)*max_u + min_u
    return list(table)


def make_mdp_diagram(n_sv=3, n_u=2, n_a=1, min_k=2, max_k=2, min_pa=1, max_pa=2, min_reward=0, max_reward=1):
    mdp_2t = nx.DiGraph()
    node_label = 0

    # init state vars
    for n in range(node_label, node_label + n_sv):
        mdp_2t.add_node(n)
        mdp_2t.node[n]['step'] = 0
        mdp_2t.node[n]['type'] = 'initial_state'
        mdp_2t.node[n]['domain_size'] = random.randint(min_k, max_k)
        mdp_2t.node[n]['parents'] = []  # init state has no parents
        mdp_2t.node[n]['table'] = create_cpt(mdp_2t, mdp_2t.node[n]['domain_size'], [])
    node_label += n_sv

    # add 1 decision var
    mdp_2t.add_node(node_label)
    mdp_2t.node[node_label]['step'] = 1
    mdp_2t.node[node_label]['type'] = 'decision'
    mdp_2t.node[node_label]['domain_size'] = n_a
    mdp_2t.node[node_label]['parents'] = list(range(n_sv))  # all initial_state vars
    mdp_2t.node[node_label]['table'] = []  # don't need to define a table
    node_label += 1

    # create utility nodes
    for n in range(node_label, node_label+n_u):
        mdp_2t.add_node(n)
        mdp_2t.node[n]['step'] = 2
        mdp_2t.node[n]['type'] = 'utility'
        mdp_2t.node[n]['domain_size'] = None  # this is a node for a function
        set_dec_parent = True if n == node_label else False
        # set_dec_parent = True
        mdp_2t.node[n]['parents'] = set_parents(mdp_2t, n, min_pa, max_pa, set_decision_parent=set_dec_parent)  # from prev states and dec
        mdp_2t.node[n]['table'] = create_utility(mdp_2t, mdp_2t.node[n]['parents'], min_reward, max_reward)
    node_label += n_u

    # create next state vars and state transitions
    for n in range(node_label, node_label+n_sv):
        mdp_2t.add_node(n)
        mdp_2t.node[n]['step'] = 3
        mdp_2t.node[n]['type'] = 'state'
        init_state_label = n - node_label  # match current node id to the init state node id
        mdp_2t.node[n]['domain_size'] = mdp_2t.node[init_state_label]['domain_size']
        set_dec_parent = True if n == node_label else False

        # if n==5:
        #     mdp_2t.node[n]['parents'] = [0, 1]
        #     for p in [0, 1]:
        #         mdp_2t.add_edge(p, node_label)
        #
        # if n==6:
        #     mdp_2t.node[n]['parents'] = [0, 2]
        #     for p in [0, 2]:
        #         mdp_2t.add_edge(p, node_label)

        mdp_2t.node[n]['parents'] = set_parents(mdp_2t, n, min_pa, max_pa, set_decision_parent=set_dec_parent)  # from prev states and dec
        mdp_2t.node[n]['table'] = create_cpt(mdp_2t, mdp_2t.node[n]['domain_size'], mdp_2t.node[n]['parents'])
    node_label += n_sv

    return mdp_2t


def make_fh_id(mdp_2t, time_horizon, file_name):
    initial_states = []
    decision_node = None
    utility_nodes = []
    states = []
    for n in mdp_2t.nodes_iter():
        node_type = mdp_2t.node[n]['type']
        if node_type == 'initial_state':
            initial_states.append(n)
        elif node_type == 'decision':
            decision_node = n
        elif node_type == 'utility':
            utility_nodes.append(n)
        elif node_type == 'state':
            states.append(n)
    initial_states = sorted(initial_states)
    utility_nodes = sorted(utility_nodes)
    states = sorted(states)

    one_step_nodes = len(states) + 1 + len(utility_nodes)
    # total_nodes = (one_step_nodes) * time_horizon
    partial_ordering = []
    domains = []
    # create influence diagram as nx digraph
    id_node_to_mdp = {}
    id_node_to_var_id = {}
    id_diagram = nx.DiGraph()
    node_label = 0          # node label inside nx graph
    var_id = 0              # var (node) id for uai file
    for th in range(time_horizon):  # from 0 to T-1
        # states
        current_state_nodes = []
        for m_ind, m_node in enumerate(states):
            id_diagram.add_node(node_label)
            id_node_to_var_id[node_label] = var_id
            current_state_nodes.append(node_label)
            if th == 0:
                id_node_to_mdp[node_label] = initial_states[m_ind]
                id_diagram.node[node_label]['parents'] = []
            else:
                id_node_to_mdp[node_label] = m_node
                # set parent child in influence diagram
                mdp_parents = mdp_2t.node[m_node]['parents']
                id_diagram.node[node_label]['parents'] = [el + one_step_nodes*(th-1) for el in mdp_parents]
                for pa in id_diagram.node[node_label]['parents']:
                    id_diagram.add_edge(pa, node_label)
            domains.append(mdp_2t.node[m_node]['domain_size'])
            node_label += 1
            var_id += 1
        partial_ordering.append(current_state_nodes)

        # decision
        id_diagram.add_node(node_label)
        id_node_to_var_id[node_label] = var_id
        id_node_to_mdp[node_label] = decision_node
        # set parent child in influence diagram
        mdp_parents = mdp_2t.node[decision_node]['parents']
        id_diagram.node[node_label]['parents'] = [el + one_step_nodes * th for el in mdp_parents]
        for pa in id_diagram.node[node_label]['parents']:
            id_diagram.add_edge(pa, node_label)
        domains.append(mdp_2t.node[decision_node]['domain_size'])
        partial_ordering.append([node_label])
        node_label += 1
        var_id += 1

        # utility
        for m_node in utility_nodes:
            id_diagram.add_node(node_label)
            id_node_to_mdp[node_label] = m_node
            # set parent child in influence diagram
            mdp_parents = mdp_2t.node[m_node]['parents']
            id_diagram.node[node_label]['parents'] = [el + one_step_nodes * th for el in mdp_parents]
            for pa in id_diagram.node[node_label]['parents']:
                id_diagram.add_edge(pa, node_label)
            node_label += 1

    write_id_for_fhmdp(file_name, time_horizon, domains, partial_ordering, len(states), 1, len(utility_nodes),
                       mdp_2t, id_diagram, id_node_to_mdp, id_node_to_var_id)
    return id_diagram


def make_fh_id_multi_dec(mdp_2t, time_horizon, file_name):
    initial_states = []
    decision_nodes = []
    utility_nodes = []
    states = []
    for n in mdp_2t.nodes_iter():
        node_type = mdp_2t.node[n]['type']
        if node_type == 'initial_state':
            initial_states.append(n)
        elif node_type == 'decision':
            decision_nodes.append(n)
        elif node_type == 'utility':
            utility_nodes.append(n)
        elif node_type == 'state':
            states.append(n)
    initial_states = sorted(initial_states)
    utility_nodes = sorted(utility_nodes)
    states = sorted(states)

    one_step_nodes = len(states) + len(decision_nodes) + len(utility_nodes)
    # total_nodes = (one_step_nodes) * time_horizon
    partial_ordering = []
    domains = []
    # create influence diagram as nx digraph
    id_node_to_mdp = {}
    id_node_to_var_id = {}
    id_diagram = nx.DiGraph()
    node_label = 0          # node label inside nx graph
    var_id = 0              # var (node) id for uai file
    for th in range(time_horizon):  # from 0 to T-1
        # states
        current_state_nodes = []
        for m_ind, m_node in enumerate(states):
            id_diagram.add_node(node_label)
            id_node_to_var_id[node_label] = var_id
            current_state_nodes.append(node_label)
            if th == 0:
                id_node_to_mdp[node_label] = initial_states[m_ind]
                id_diagram.node[node_label]['parents'] = []
            else:
                id_node_to_mdp[node_label] = m_node
                # set parent child in influence diagram
                mdp_parents = mdp_2t.node[m_node]['parents']
                id_diagram.node[node_label]['parents'] = [el + one_step_nodes*(th-1) for el in mdp_parents]
                for pa in id_diagram.node[node_label]['parents']:
                    id_diagram.add_edge(pa, node_label)
            domains.append(mdp_2t.node[m_node]['domain_size'])
            node_label += 1
            var_id += 1
        partial_ordering.append(current_state_nodes)

        # decision
        if th < time_horizon-1:     # don't add the last decision, no transition
            current_decision_nodes = []
            for m_ind, m_node in enumerate(decision_nodes):
                id_diagram.add_node(node_label)
                id_node_to_var_id[node_label] = var_id
                id_node_to_mdp[node_label] = m_node
                current_decision_nodes.append(node_label)

                mdp_parents = mdp_2t.node[m_node]['parents']
                id_diagram.node[node_label]['parents'] = [el + one_step_nodes*th for el in mdp_parents]
                for pa in id_diagram.node[node_label]['parents']:
                    id_diagram.add_edge(pa, node_label)
                domains.append(mdp_2t.node[m_node]['domain_size'])
                node_label += 1
                var_id += 1
            partial_ordering.append(current_decision_nodes)

        # utility
        for m_node in utility_nodes:
            id_diagram.add_node(node_label)
            id_node_to_mdp[node_label] = m_node
            # set parent child in influence diagram
            mdp_parents = mdp_2t.node[m_node]['parents']
            id_diagram.node[node_label]['parents'] = [el + one_step_nodes * th for el in mdp_parents]
            for pa in id_diagram.node[node_label]['parents']:
                id_diagram.add_edge(pa, node_label)
            node_label += 1

    write_id_for_fhmdp(file_name, time_horizon, domains, partial_ordering,
                       len(states), len(decision_nodes), len(utility_nodes),
                       mdp_2t, id_diagram, id_node_to_mdp, id_node_to_var_id)
    return id_diagram


def write_id_for_fhmdp(file_name, time_horizon, domains, partial_ordering,
                       num_state_nodes, num_decision_nodes, num_util_nodes,
                       mdp_2t, id_diagram, id_node_to_mdp, id_node_to_var_id):
    ####################################################################################################################
    # create uai file
    f = open(file_name + '.uai', 'w')
    f.write('ID\n')
    # vars
    # n_vars = (num_state_nodes + num_decision_nodes) * time_horizon
    n_vars = sum([len(bk) for bk in partial_ordering] )
    f.write('{}\n'.format(n_vars))
    for d in domains:
        f.write('{} '.format(d))
    f.write('\n')
    # factors
    n_factors = (num_state_nodes + num_util_nodes) * time_horizon
    f.write('{}\n'.format(n_factors))
    for n in sorted(id_diagram.nodes_iter()):
        if mdp_2t.node[id_node_to_mdp[n]]['type'] in ['initial_state', 'state', 'constraint']:
            scope = id_diagram.node[n]['parents'] + [n]
        elif mdp_2t.node[id_node_to_mdp[n]]['type'] == 'utility':
            scope = id_diagram.node[n]['parents']
        else:
            continue
        f.write('{} '.format(len(scope)))
        for s in scope:
            try:
                f.write('{} '.format(id_node_to_var_id[s]))
            except:
                print('err')
        f.write('\n')
    f.write('\n')
    # tables
    for n in sorted(id_diagram.nodes_iter()):
        if mdp_2t.node[id_node_to_mdp[n]]['type'] != 'decision':
            f.write('{}\n'.format(len(mdp_2t.node[id_node_to_mdp[n]]['table'])))
            for t in mdp_2t.node[id_node_to_mdp[n]]['table']:
                f.write('{}\n'.format(float(t)))
            f.write('\n')
    f.close()

    ####################################################################################################################
    # create id file (identify types of vars and factors)
    f = open(file_name + '.id', 'w')
    f.write('{}\n'.format(n_vars))
    # var types
    for n in sorted(id_diagram.nodes_iter()):
        if mdp_2t.node[id_node_to_mdp[n]]['type'] in ['initial_state', 'state', 'constraint']:
            f.write('C ')
        elif mdp_2t.node[id_node_to_mdp[n]]['type'] == 'decision':
            f.write('D ')
    f.write('\n')
    # factor types
    f.write('{}\n'.format(n_factors))
    for n in sorted(id_diagram.nodes_iter()):
        if mdp_2t.node[id_node_to_mdp[n]]['type'] in ['initial_state', 'state', 'constraint']:
            f.write('P ')
        elif mdp_2t.node[id_node_to_mdp[n]]['type'] == 'utility':
            f.write('U ')
    f.write('\n')
    f.close()

    ####################################################################################################################
    # create pvo file (reverse temporal ordering for elimination)
    f = open(file_name + '.pvo', 'w')
    f.write('{};\n'.format(n_vars))
    f.write('{};\n'.format(len(partial_ordering)))

    for block in reversed(partial_ordering):
        for b in block:
            f.write('{} '.format(id_node_to_var_id[b]))
        f.write(';\n')
    f.close()
    return id_diagram


########################################################################################################################
def run():
    n_sv = 40
    n_a = 3
    n_u = 4
    min_k = 2
    max_k = 2
    min_pa = 3
    max_pa = 5
    min_reward = 0
    max_reward = 10
    time_horizon = 5
    file_name = 'mdp15-' + '_'.join([str(n_sv), str(n_a), str(n_u), str(time_horizon)])


    mdp2t = make_mdp_diagram(n_sv, n_u, n_a, min_k, max_k, min_pa, max_pa, min_reward, max_reward)
    nx.write_gpickle(mdp2t, os.path.join(BETA_PATH, file_name + '.gpickle'))
    # nx.read_gpickle(os.path.join(file_path, file_name + '.gpickle'))
    id_diagram = make_fh_id(mdp2t, time_horizon, os.path.join(BETA_PATH, file_name))
    print('done')

if __name__ == '__main__':
    run()


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


def create_utility(diagram, parents, min_u, max_u):
    configs = 1
    for n in parents:
        configs *= diagram.node[n]['domain_size']
    table = np.random.rand(configs)*max_u + min_u
    return list(table)


def set_util_parents(diagram, current_node, min_parents, max_parents, set_decision_parent=True):
    num_parents = random.randint(min_parents, max_parents)
    decision_node = None
    previous_states = []
    for n in diagram.nodes_iter():
        if diagram.node[n]['step'] == 0:
            previous_states.append(n)
        if diagram.node[n]['step'] == 1:
            decision_node = n

    if num_parents > len(previous_states) + 1:  # 1 is for the decision var
        num_parents = len(previous_states) + 1

    if set_decision_parent:
        parents = list(np.random.choice(previous_states, num_parents-1, replace=False))
        parents.append(decision_node)
    else:
        parents = list(np.random.choice(previous_states+[decision_node], num_parents, replace=False))
    for p in parents:
        diagram.add_edge(p, current_node)
    return sorted(parents)


def set_state_parents(diagram, current_node, min_parents, max_parents, set_decision_parent=True):
    num_parents = random.randint(min_parents, max_parents)
    decision_node = None
    previous_states = []
    for n in diagram.nodes_iter():
        if diagram.node[n]['step'] == 0 and diagram.node[n]['type'] == 'initial_hidden':
            previous_states.append(n)
        if diagram.node[n]['step'] == 1:
            decision_node = n

    if num_parents > len(previous_states) + 1:  # 1 is for the decision var
        num_parents = len(previous_states) + 1

    if set_decision_parent:
        parents = list(np.random.choice(previous_states, num_parents-1, replace=False))
        parents.append(decision_node)
    else:
        parents = list(np.random.choice(previous_states+[decision_node], num_parents, replace=False))
    for p in parents:
        diagram.add_edge(p, current_node)
    return sorted(parents)


def set_initobs_parent(diagram, current_node, min_parents, max_parents):
    num_parents = random.randint(min_parents, max_parents)
    possible_parents = []
    for n in diagram.nodes_iter():
        if diagram.node[n]['step'] == 0 and diagram.node[n]['type'] == 'initial_hidden':
            possible_parents.append(n)
    if num_parents > len(possible_parents):
        num_parents = len(possible_parents)
    parents = list(np.random.choice(possible_parents, num_parents, replace=False))
    for p in parents:
        diagram.add_edge(p, current_node)
    return sorted(parents)


def make_pomdp_diagram(n_sh=2, n_so=1, n_u=2, n_a=1, min_k=2, max_k=2, min_pa=1, max_pa=2, min_reward=0, max_reward=1):
    pomdp_2t = nx.DiGraph()
    node_label = 0

    # init state vars
    for n in range(node_label, node_label + n_sh):
        pomdp_2t.add_node(n)
        pomdp_2t.node[n]['step'] = 0
        pomdp_2t.node[n]['type'] = 'initial_hidden'
        pomdp_2t.node[n]['domain_size'] = random.randint(min_k, max_k)
        pomdp_2t.node[n]['parents'] = []  # init state has no parents
        pomdp_2t.node[n]['table'] = create_cpt(pomdp_2t, pomdp_2t.node[n]['domain_size'], [])
    node_label += n_sh

    # init obs state vars
    for n in range(node_label, node_label + n_so):
        pomdp_2t.add_node(n)
        pomdp_2t.node[n]['step'] = 0
        pomdp_2t.node[n]['type'] = 'initial_obs'
        pomdp_2t.node[n]['domain_size'] = random.randint(min_k, max_k)
        pomdp_2t.node[n]['parents'] = set_initobs_parent(pomdp_2t, n, min_pa, max_pa)
        pomdp_2t.node[n]['table'] = create_cpt(pomdp_2t, pomdp_2t.node[n]['domain_size'], pomdp_2t.node[n]['parents'])
    node_label += n_so

    # add 1 decision var
    pomdp_2t.add_node(node_label)
    pomdp_2t.node[node_label]['step'] = 1
    pomdp_2t.node[node_label]['type'] = 'decision'
    pomdp_2t.node[node_label]['domain_size'] = n_a
    obs_nodes = sorted([n for n in pomdp_2t.nodes_iter() if pomdp_2t.node[n]['type'] == 'initial_obs'])
    pomdp_2t.node[node_label]['parents'] = obs_nodes
    pomdp_2t.node[node_label]['table'] = []  # don't need to define a table
    node_label += 1

    # create utility nodes
    for n in range(node_label, node_label+n_u):
        pomdp_2t.add_node(n)
        pomdp_2t.node[n]['step'] = 2
        pomdp_2t.node[n]['type'] = 'utility'
        pomdp_2t.node[n]['domain_size'] = None  # this is a node for a function
        set_dec_parent = True if n == node_label else False
        pomdp_2t.node[n]['parents'] = set_util_parents(pomdp_2t, n, min_pa, max_pa, set_decision_parent=set_dec_parent)
        pomdp_2t.node[n]['table'] = create_utility(pomdp_2t, pomdp_2t.node[n]['parents'], min_reward, max_reward)
    node_label += n_u

    # create next hidden state vars and state transitions
    for n in range(node_label, node_label+n_sh):
        pomdp_2t.add_node(n)
        pomdp_2t.node[n]['step'] = 3
        pomdp_2t.node[n]['type'] = 'hidden'
        init_state_label = n - (n_sh+n_so+1+n_u)  # match current node id to the init state node id
        pomdp_2t.node[n]['domain_size'] = pomdp_2t.node[init_state_label]['domain_size']
        set_dec_parent = True if n == node_label else False
        pomdp_2t.node[n]['parents'] = set_state_parents(pomdp_2t, n, min_pa, max_pa, set_decision_parent=set_dec_parent)
        pomdp_2t.node[n]['table'] = create_cpt(pomdp_2t, pomdp_2t.node[n]['domain_size'], pomdp_2t.node[n]['parents'])
    node_label += n_sh

    # create next obs state vars [ copy of the init stage]
    for n in range(node_label, node_label+n_so):
        pomdp_2t.add_node(n)
        pomdp_2t.node[n]['step'] = 3
        pomdp_2t.node[n]['type'] = 'obs'
        init_state_label = n - (n_sh+n_so+1+n_u)  # match current node id to the init state node id
        pomdp_2t.node[n]['domain_size'] = pomdp_2t.node[init_state_label]['domain_size']
        initobs_parents = pomdp_2t.node[init_state_label]['parents']
        obs_parents = sorted([el+(n_sh+n_so+1+n_u) for el in initobs_parents])
        pomdp_2t.node[n]['parents'] = obs_parents
        pomdp_2t.node[n]['table'] = pomdp_2t.node[init_state_label]['table']
    return pomdp_2t


def make_fh_id(pomdp_2t, time_horizon, file_name):
    initial_hidden = []
    initial_obs = []
    decision_node = None
    utility_nodes = []
    hidden = []
    obs = []
    for n in pomdp_2t.nodes_iter():
        node_type = pomdp_2t.node[n]['type']
        if node_type == 'initial_hidden':
            initial_hidden.append(n)
        elif node_type == 'initial_obs':
            initial_obs.append(n)
        elif node_type == 'decision':
            decision_node = n
        elif node_type == 'utility':
            utility_nodes.append(n)
        elif node_type == 'hidden':
            hidden.append(n)
        elif node_type == 'obs':
            obs.append(n)
    initial_hidden = sorted(initial_hidden)
    initial_obs = sorted(initial_obs)
    utility_nodes = sorted(utility_nodes)
    hidden = sorted(hidden)
    obs = sorted(obs)

    # create influence diagram as nx digraph
    id_node_to_pomdp = {}
    id_node_to_var_id = {}
    id_diagram = nx.DiGraph()
    one_step_nodes = len(hidden) + len(obs) + 1 + len(utility_nodes)
    one_step_vars = len(hidden) + len(obs)  + 1                     # chance, decision
    one_step_funcs = len(hidden) + len(obs) + len(utility_nodes)    # prob, utility
    node_label = 0
    var_id = 0
    domains = []
    partial_ordering = []
    hidden_vars = []

    for th in range(time_horizon):  # from 0 to T-1
        current_obs_nodes = []
        # state variables
        if th == 0:
            for m_ind, m_node in enumerate(initial_hidden):
                id_diagram.add_node(node_label)
                id_node_to_var_id[node_label] = var_id   # map influence diagram node and variable id
                id_node_to_pomdp[node_label] = m_node    # map influence diagram node to pomdp2t node
                domains.append(pomdp_2t.node[m_node]['domain_size'])
                id_diagram.node[node_label]['parents'] = []
                hidden_vars.append(node_label)
                node_label += 1
                var_id += 1
            for m_ind, m_node in enumerate(initial_obs):
                id_diagram.add_node(node_label)
                id_node_to_var_id[node_label] = var_id
                id_node_to_pomdp[node_label] = m_node  # map influence diagram node to pomdp2t node
                domains.append(pomdp_2t.node[m_node]['domain_size'])
                pomdp_parents = pomdp_2t.node[m_node]['parents']
                id_diagram.node[node_label]['parents'] = [el for el in pomdp_parents]   # th 0 same!
                for pa in id_diagram.node[node_label]['parents']:
                    id_diagram.add_edge(pa, node_label)
                current_obs_nodes.append(node_label)
                node_label += 1
                var_id += 1
        else:
            for m_ind, m_node in enumerate(hidden):
                id_diagram.add_node(node_label)
                id_node_to_var_id[node_label] = var_id
                id_node_to_pomdp[node_label] = m_node
                domains.append(pomdp_2t.node[m_node]['domain_size'])
                pomdp_parents = pomdp_2t.node[m_node]['parents']
                id_diagram.node[node_label]['parents'] = [el + one_step_nodes*(th-1) for el in pomdp_parents]
                for pa in id_diagram.node[node_label]['parents']:
                    id_diagram.add_edge(pa, node_label)
                hidden_vars.append(node_label)
                node_label += 1
                var_id += 1
            for m_ind, m_node in enumerate(obs):
                id_diagram.add_node(node_label)
                id_node_to_var_id[node_label] = var_id
                id_node_to_pomdp[node_label] = m_node
                domains.append(pomdp_2t.node[m_node]['domain_size'])
                pomdp_parents = pomdp_2t.node[m_node]['parents']
                id_diagram.node[node_label]['parents'] = [el + one_step_nodes * (th-1) for el in pomdp_parents]
                for pa in id_diagram.node[node_label]['parents']:
                    id_diagram.add_edge(pa, node_label)
                current_obs_nodes.append(node_label)
                node_label += 1
                var_id += 1

        partial_ordering.append(current_obs_nodes)

        # decision
        id_diagram.add_node(node_label)
        id_node_to_var_id[node_label] = var_id
        id_node_to_pomdp[node_label] = decision_node
        domains.append(pomdp_2t.node[decision_node]['domain_size'])
        pomdp_parents = pomdp_2t.node[decision_node]['parents']
        id_diagram.node[node_label]['parents'] = [el + one_step_nodes * th for el in pomdp_parents]
        for pa in id_diagram.node[node_label]['parents']:
            id_diagram.add_edge(pa, node_label)
        partial_ordering.append([node_label])
        node_label += 1
        var_id += 1

        # utility
        for m_node in utility_nodes:
            id_diagram.add_node(node_label)
            id_node_to_pomdp[node_label] = m_node
            pomdp_parents = pomdp_2t.node[m_node]['parents']
            id_diagram.node[node_label]['parents'] = [el + one_step_nodes * th for el in pomdp_parents]
            for pa in id_diagram.node[node_label]['parents']:
                id_diagram.add_edge(pa, node_label)
            node_label += 1

    partial_ordering.append(hidden_vars)

    ####################################################################################################################
    # create uai file
    f = open(file_name + '.uai', 'w')
    f.write('ID\n')
    # vars
    n_vars = one_step_vars * time_horizon
    f.write('{}\n'.format(n_vars))
    for d in domains:
        f.write('{} '.format(d))
    f.write('\n')
    # factors
    n_factors = one_step_funcs * time_horizon
    f.write('{}\n'.format(n_factors))
    for n in sorted(id_diagram.nodes_iter()):
        if pomdp_2t.node[id_node_to_pomdp[n]]['type'] in ['initial_hidden', 'initial_obs', 'hidden', 'obs']:
            scope = id_diagram.node[n]['parents'] + [n]
        elif pomdp_2t.node[id_node_to_pomdp[n]]['type'] == 'utility':
            scope = id_diagram.node[n]['parents']
        else:
            continue
        f.write('{} '.format(len(scope)))
        for s in scope:
            f.write('{} '.format(id_node_to_var_id[s]))
        f.write('\n')
    f.write('\n')
    # tables
    for n in sorted(id_diagram.nodes_iter()):
        if pomdp_2t.node[id_node_to_pomdp[n]]['type'] != 'decision':
            f.write('{}\n'.format(len(pomdp_2t.node[id_node_to_pomdp[n]]['table'])))
            for t in pomdp_2t.node[id_node_to_pomdp[n]]['table']:
                f.write('{}\n'.format(float(t)))
            f.write('\n')
    f.close()

    ####################################################################################################################
    # create id file (identify types of vars and factors)
    f = open(file_name + '.id', 'w')
    f.write('{}\n'.format(n_vars))
    # var types
    for n in sorted(id_diagram.nodes_iter()):
        if pomdp_2t.node[id_node_to_pomdp[n]]['type'] in ['initial_hidden', 'initial_obs', 'hidden', 'obs']:
            f.write('C ')
        elif pomdp_2t.node[id_node_to_pomdp[n]]['type'] == 'decision':
            f.write('D ')
    f.write('\n')
    # factor types
    f.write('{}\n'.format(n_factors))
    for n in sorted(id_diagram.nodes_iter()):
        if pomdp_2t.node[id_node_to_pomdp[n]]['type'] in ['initial_hidden', 'initial_obs', 'hidden', 'obs']:
            f.write('P ')
        elif pomdp_2t.node[id_node_to_pomdp[n]]['type'] == 'utility':
            f.write('U ')
    f.write('\n')
    f.close()

    ####################################################################################################################
    # create pvo file (reverse temporal ordering for elimination)
    f = open(file_name + '.pvo', 'w')
    f.write('{};\n'.format(n_vars))
    f.write('{};\n'.format(len(partial_ordering)))

    for block in reversed(partial_ordering):
        for b in reversed(block):
            f.write('{} '.format(id_node_to_var_id[b]))
        f.write(';\n')
    f.close()
    return id_diagram


########################################################################################################################
def run():
    n_sh = 20
    n_so = 12
    n_a = 2
    n_u = 4
    min_k = 2
    max_k = 3
    min_pa = 2
    max_pa = 5
    min_reward = 0
    max_reward = 10
    time_horizon = 6
    head = 'pomdp15-'
    file_name = head + '_'.join([str(n_sh), str(n_so), str(n_a), str(n_u), str(time_horizon)])

    pomdp2t = make_pomdp_diagram(n_sh, n_so, n_u, n_a, min_k, max_k, min_pa, max_pa, min_reward, max_reward)
    nx.write_gpickle(pomdp2t, os.path.join(BETA_PATH, file_name + '.gpickle'))
    # nx.read_gpickle(os.path.join(file_path, file_name + '.gpickle'))
    id_diagram = make_fh_id(pomdp2t, time_horizon, os.path.join(BETA_PATH, file_name))
    print('done {}'.format(file_name))

if __name__ == '__main__':
    run()


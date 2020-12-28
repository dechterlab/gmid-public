PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gmid.fileio import *
from gmid.graphical_models import *
from gmid.graph_algorithms import *
from gmid.gddid_hinge_shift import *
from gmid.cte import *
# mdpsmall-2_2_2_2
# mdp1-4_2_2_5
# pomdp1-4_2_2_2_3
def run(name="pomdp1-4_2_2_2_3", ibound=1, mbe_init=True):
    #### read files
    problem_path = os.path.join(PROBLEM_PATH_ID, name)
    if name.endswith(".erg"):
        print("read erg problem file:{}".format(problem_path))
        file_info = read_erg(problem_path)
        ordering_path = problem_path.replace(".erg", '') + '.vo'
    elif name.endswith(".limid"):
        print("read limid problem file:{}".format(problem_path))
        file_info = read_limid(problem_path)        # error in blocks??? todo fix limid generator?
        ordering_path = problem_path.replace(".limid", '') + '.vo'
    else:   # only name without extension -> uai/pvo/id
        print("read problem files:{}".format(problem_path))
        file_info = read_uai_id(problem_path)
        ordering_path = problem_path + '.vo'
    factors = file_info['factors']
    blocks = file_info['blocks']
    var_types = file_info['var_types']
    fun_types = file_info['factor_types']
    num_var = file_info['nvar']
    num_fun = len(file_info['factors'])

    #### create graphical model
    valuations = [factor_to_valuation(factor, factor_type, False) for factor, factor_type in zip(factors, fun_types)]
    weights = [1.0 if var_type == 'C' else 0.0 for var_type in var_types]
    is_log = False
    is_valuation = True
    gm = GraphicalModel(valuations, weights, is_log=is_log)

    ### primal graph, find elimination order
    pg = PrimalGraph(gm)
    try:
        ordering, iw = read_vo(ordering_path)
        print("read ordering_path file:{}".format(ordering_path))
        iw = get_induced_width_from_ordering(pg, ordering)
    except:
        print("generate ordering_path file:{}".format(ordering_path))
        ordering, iw = iterative_greedy_variable_order(5000, pg.nx_diagram, ps=8, pe=-1, ct=inf, pv=blocks)
        print("write ordering_path file:{}".format(ordering_path))
        write_vo_from_elim_order(problem_path + '.vo', ordering, iw)
    print('constrained induced width of the primal graph:{}'.format(iw))
    print("number of connected components in primal graph:{}".format(nx.number_connected_components(pg.nx_diagram)))

    #### join graph decomposition
    verbose_level = 0
    if mbe_init:
        mbtd, mini_buckets = mini_bucket_tree(gm, ordering, ibound, False, random_partition=False)
        add_mg_attr_to_nodes(mbtd)
        add_const_factors(mbtd, gm.variables, is_valuation, is_log)
        mbe_str = 'Mbe_i='+str(ibound)
        mbe_logfile = os.path.join(LOG_PATH_ID, name+mbe_str)
        tree_mp = CTE(verbose_level, mbtd, ordering, weights, is_log, is_valuation, log_file_name=mbe_logfile)
        tree_mp.schedule()
        tree_mp.init_propagate()
        mbe_bound = tree_mp.propagate(propagation_type='mbe_cost_shifting', max_sum=False)
        tree_mp.print_log('mbe_bound at i:{} is {}'.format(ibound, mbe_bound))
        tree_mp.mg.set_fs_from_mg_to_rg()
        jgd = join_graph(tree_mp.mg, mini_buckets, ordering, connect_mb_only=False, make_copy=False)
    else:
        mbtd, mini_buckets = mini_bucket_tree(gm, ordering, ibound, False, random_partition=False)
        jgd = join_graph(mbtd, mini_buckets, ordering, connect_mb_only=False, make_copy=False)
    add_mg_attr_to_nodes(jgd)
    add_mg_attr_to_edges(jgd, gm.variables, is_log)
    add_const_factors(jgd, gm.variables, is_valuation, is_log)

    #### GDD for id
    gdd_str ='_GddIdHingeShift_i=' + str(ibound) + '_init=' + str(mbe_init)
    log_file_name = os.path.join(LOG_PATH_ID, name+gdd_str)
    gd_options = {
        'gd_steps': 10, 'tol': TOL,
        'ls_steps' : 30, 'armijo_thr' : 1e-4, 'armijo_step_back' : 0.5, 'ls_tol': TOL
    }

    weight_options = {
        'gd_steps': 10, 'tol': TOL,
        'ls_steps': 30, 'armijo_thr': 1e-4, 'armijo_step_back': 0.5, 'ls_tol': TOL
    }

    util_options = {
        'gd_steps': 50, 'tol': TOL,
        'ls_steps': 30, 'armijo_thr': 1e-4, 'armijo_step_back': 0.5, 'ls_tol': TOL
    }
    gdd_mp = GddIdHingeShift(verbose_level, jgd, ordering, weights, is_log, epsilon=WEPS, log_file_name=log_file_name)
    gdd_mp.print_log("GDDIDHingeShift")
    gdd_mp.print_log("gd_options:{}".format(gd_options))
    gdd_mp.print_log("weight_options:{}".format(weight_options))
    gdd_mp.print_log("util_options:{}".format(util_options))
    gdd_mp.print_log("start solving:{}".format(name))
    gdd_mp.print_log("iw={}, elim_order={}, ibound={}".format(iw, ordering, ibound))
    gdd_mp.print_log("nv={}, nf={}".format(num_var, num_fun))
    gdd_mp.print_log("weights={}".format(weights))
    gdd_mp.print_log("number of clusters={}".format(len(jgd.region_graph.nodes())))

    #### Start GDD
    gdd_mp.schedule()
    gdd_mp.init_propagate()
    best_eu = gdd_mp.propagate(time_limit=10800, iter_limit=2000, cost_options=gd_options, weight_options=weight_options, util_options=util_options)
    gdd_mp.print_log("finished solving:{}".format(name))
    gdd_mp.print_log("best bound found={}".format(best_eu))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(sys.argv)
        run(str(sys.argv[1]), int(sys.argv[2]), True)
    else:
        run()

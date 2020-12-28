PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gmid.fileio import *
from gmid.graphical_models import *
from gmid.graph_algorithms import *
from gmid.gddid_hinge_shift_proj import *
from gmid.cte import *
import datetime

def run(name="pomdp1-4_2_2_2_3", ibound=20, time_limit=3600, iter_limit=5):
    print("Start JOB:{},{} at {}".format(name, ibound, datetime.datetime.now()))
    problem_path = os.path.join(TEST_PATH, name)
    if name.endswith(".erg"):
        print("read erg problem file:{}".format(problem_path))
        file_info = read_erg(problem_path)
        ordering_path = problem_path.replace(".erg", '') + '.vo'
    elif name.endswith(".limid"):
        print("read limid problem file:{}".format(problem_path))
        file_info = read_limid(problem_path)
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

    valuations = [factor_to_valuation(factor, factor_type, False) for factor, factor_type in zip(factors, fun_types)]
    weights = [1.0 if var_type == 'C' else 0.0 for var_type in var_types]
    is_log = False
    is_valuation = True
    gm = GraphicalModel(valuations, weights, is_log=is_log)
    pg = PrimalGraph(gm)
    ordering, iw = read_vo(ordering_path)
    print("read ordering_path file:{}".format(ordering_path))
    iw = get_induced_width_from_ordering(pg, ordering)
    print('constrained induced width of the primal graph:{}'.format(iw))
    print("number of connected components in primal graph:{}".format(nx.number_connected_components(pg.nx_diagram)))

    #### join graph decomposition
    verbose_level = 0
    mbtd, mini_buckets = mini_bucket_tree(gm, ordering, ibound, False, random_partition=True)
    add_mg_attr_to_nodes(mbtd)
    add_const_factors(mbtd, gm.variables, is_valuation, is_log)
    mbe_str = '_GddIdMbeInit_ibd='+str(ibound)
    mbe_logfile = os.path.join(LOG_PATH_ID, name+mbe_str)
    tree_mp = CTE(verbose_level, mbtd, ordering, weights, is_log, is_valuation, log_file_name=mbe_logfile)
    tree_mp.schedule()
    tree_mp.init_propagate()
    mbe_bound = tree_mp.propagate(propagation_type='mbe_cost_shifting', max_sum=True)
    tree_mp.print_log('mbe_bound at i:{} is {}'.format(ibound, mbe_bound))
    tree_mp.mg.set_fs_from_mg_to_rg()
    jgd = join_graph(tree_mp.mg, mini_buckets, ordering, False, False)
    add_mg_attr_to_nodes(jgd)
    # add_mg_attr_to_edges(jgd, gm.variables, is_log)
    add_const_factors(jgd, gm.variables, is_valuation, is_log)

    #### GDD for id
    gdd_str = "_".join( [name, "GddIdHingeShiftPrj", "iw="+str(iw), "ibd="+str(ibound),
                         "il="+str(iter_limit), "tl="+str(time_limit)])
    log_file_name = os.path.join(LOG_PATH_ID, gdd_str)
    cost_options = {
        'gd_steps': 10, 'tol': TOL,
        'ls_steps' : 30, 'armijo_thr' : 1e-4, 'armijo_step_back' : 0.5, 'ls_tol': TOL,
        'max_bcd_scope_size' : 15
    }

    weight_options = {
        'gd_steps': 10, 'tol': TOL,
        'ls_steps': 30, 'armijo_thr': 1e-4, 'armijo_step_back': 0.5, 'ls_tol': TOL
    }

    util_options = {
        'gd_steps': 50, 'tol': TOL,
        'ls_steps': 30, 'armijo_thr': 1e-4, 'armijo_step_back': 0.5, 'ls_tol': TOL,
        'trigger_util_update' : 0.09, 'util_update_period' : 1, 'no_util_update_iter' : 1
    }
    gdd_mp = GddIdHingeShiftProjected(verbose_level, jgd, ordering, weights, is_log, WEPS, log_file_name, ibound)
    gdd_mp.print_log("GddIdHingeShiftPrj")
    gdd_mp.print_log("START")
    gdd_mp.print_log("name:{}".format(name))
    gdd_mp.print_log("num vars:{}".format(file_info['nvar']))
    gdd_mp.print_log("num factors:{}".format(len(file_info['factors'])))
    gdd_mp.print_log("max domain:{}".format(max(file_info['domains'])))
    gdd_mp.print_log("max scope:{}".format(max([len(s) for s in file_info['scopes']])))
    gdd_mp.print_log('induced width:{}'.format(iw))
    gdd_mp.print_log("ordering:{}".format(ordering))
    gdd_mp.print_log("weights={}".format(weights))
    gdd_mp.print_log("connected components:{}".format(nx.number_connected_components(pg.G)))
    gdd_mp.print_log("number of clusters={}".format(len(jgd.region_graph.nodes())))
    gdd_mp.print_log("cost_options:{}".format(cost_options))
    gdd_mp.print_log("weight_options:{}".format(weight_options))
    gdd_mp.print_log("util_options:{}".format(util_options))

    t0 = time.time()
    gdd_mp.schedule()
    gdd_mp.init_propagate()
    best_eu = gdd_mp.propagate(time_limit=time_limit, iter_limit=iter_limit, cost_options=cost_options, weight_options=weight_options, util_options=util_options)
    t1 = time.time()
    gdd_mp.print_log("final MEU:{}".format(best_eu))
    gdd_mp.print_log("Finish JOB:{}, with i-bound {} at {}".format(name, ibound, datetime.datetime.now()))
    gdd_mp.print_log("total time (sec) for solving problem:{}".format(t1-t0))
    gdd_mp.print_log("END")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(sys.argv)
        name=str(sys.argv[1])
        name=name.split("/")[-1]
        print("read {}".format(name))
        # if not name.endswith(".uai"):
        #     assert False, name + " is wrong input file"
        name=name.replace(".uai", "")
        ibound = int(sys.argv[2])
        time_limit = int(sys.argv[3])
        iter_limit = int(sys.argv[4])
        run(name, ibound, time_limit, iter_limit)
    else:
        run()

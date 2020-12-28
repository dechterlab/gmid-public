PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gmid.fileio import *
from gmid.graphical_models import *
from gmid.graph_algorithms import *
from gmid.cte import *
from libs.pyGM.factor import Factor
import pprint
pp = pprint.PrettyPrinter(indent=4)


def run_policy(problem_home, problems):
    policy_group = []
    acc_Z = []
    acc_eu = []
    acc_eu_z = []
    for problem in problems:
        problem_path = os.path.join(problem_home, problem)
        file_info = read_limid(problem_path)
        name=problem_path.split("/")[-1]
        name=name.replace(".limid","")
        factors = file_info['factors']
        blocks = file_info['blocks']
        var_types = file_info['var_types']
        fun_types = file_info['factor_types']

        valuations = [factor_to_valuation(factor, factor_type, False) for factor, factor_type in zip(factors, fun_types)]
        weights = [1.0 if var_type == 'C' else 0.0 for var_type in var_types]
        is_log = False
        is_valuation = True
        gm = GraphicalModel(valuations, weights, is_log=is_log)
        pg = PrimalGraph(gm)
        ordering, iw = iterative_greedy_variable_order(100, pg.nx_diagram, ps=8, pe=-1, ct=inf, pv=blocks)
        elim_ordering = ordering[:-1]       # exclude the last elimination, the first action (it's blind action)

        mbtd, mini_buckets = mini_bucket_tree(graphical_model=gm, elim_order=elim_ordering, ibound=100, ignore_msg=False, random_partition=False)
        add_mg_attr_to_nodes(mbtd)
        add_const_factors(mbtd, gm.variables, is_valuation, is_log)
        verbose_level = 0
        log_file_name = os.path.join(LOG_PATH, name)
        tree_mp = CTE(verbose_level, mbtd, ordering, weights, is_log, is_valuation, log_file_name=log_file_name)
        tree_mp.schedule()
        tree_mp.init_propagate()
        EU_at_root, Z_at_root = tree_mp.propagate()

        action_var = Z_at_root.vars[0]  # sorted set, 1 variable
        action_var.label = 0        # replace label
        Z = Factor([action_var], Z_at_root.table)
        EU = Factor([action_var], EU_at_root.table)
        policy_group.append(Valuation(Z, EU))
        # tree_mp.print_log(("{}".format(problem)))
        # tree_mp.print_log("Z:\n{}".format(Z))
        # tree_mp.print_log("EU:\n{}".format(EU))
        # tree_mp.print_log("vars\nZ:{}\tEU:{}".format(Z.vars, EU.vars))

    # total_policy = policy_group[0].copy()
    total_policy = Valuation(1.0, 0.0)
    for policy in policy_group:
        total_policy = total_policy * policy        # combining (adding) valuations
        print("policyp:{}".format(policy.prob))
        print("policyu:{}".format(policy.util))
        print("progressp:{}".format(total_policy.prob))
        print("progressu:{}".format(total_policy.util))
    total_policy = total_policy * Valuation(1.0/len(problems),0.0)
    print(total_policy.util)
    return "batch:{}\taction:{}\tZ:{}\tEU:{}".format(len(problems), np.argmax(total_policy.util.table),
                                                     len(problems)*np.max(total_policy.prob.table),
                                                     np.max(total_policy.util.table))

def run_batch(problem_home, problems, batch_size):
    for it, i in enumerate(range(0,len(problems), batch_size)):
        if i+batch_size <= len(problems):
            print("iter:{}\t{}".format(it+1, run_policy(problem_home, problems[i:i+batch_size])))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        problem_home = str(sys.argv[1])
        ns = int(sys.argv[2])
        nc = int(sys.argv[3])
        na = int(sys.argv[4])
        batch_size = int(sys.argv[5])
        problems =  []
        for f in os.listdir(problem_home):
            params = [int(el.split("=")[-1]) for el in f.split("_")[1].split("-")]
            if params[0] == ns and params[1] == nc and params[2] == na:
                problems.append(f)
        random.shuffle(problems, random.random)
        print("START:{}-{}-{}-{}-{}".format(problem_home.split("/")[-1], batch_size, ns, nc, na))
        run_batch(problem_home, problems, batch_size)
        print(("END:{}-{}-{}-{}-{}".format(problem_home.split("/")[-1], batch_size, ns, nc, na)))

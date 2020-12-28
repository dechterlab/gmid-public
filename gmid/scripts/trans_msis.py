PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
from gmid.fileio import *
from gmid.graph_algorithms import *
import os
import pprint

pp = pprint.PrettyPrinter(indent=4)     # it's a class so make an object


def run_trans_msis(file_name, draw_diagram=False):
    file_info = read_uai_id(file_name, sort_scope=False)
    original_influence_diagram = NxInfluenceDiagram()
    nvar = file_info['nvar']
    scopes = file_info['scopes']                # list of Vars
    scopes = scopes_of_vars_to_int(scopes)      # list of ints (label)
    scope_types = file_info['scope_types']      # P or U    -> don't show decision scope
    partial_elim_order = file_info['blocks']                # list of blocks following elim order
    var_types = file_info['var_types']          # C or D

    original_influence_diagram.create_id_from_scopes(nvar, var_types, scopes, scope_types, partial_elim_order)
    relaxed_influence_diagram = get_relaxed_influence_diagram(original_influence_diagram)
    if draw_diagram:
        original_influence_diagram.draw_diagram(file_name)
        relaxed_influence_diagram.draw_diagram(file_name + ".relaxed")
    write_pvo_from_partial_elim_order(file_name + ".relaxed.pvo", relaxed_influence_diagram.partial_elim_order)
    if nvar != sum([len(bk) for bk in relaxed_influence_diagram.partial_elim_order]):
        fix_pvo(file_name + ".relaxed.pvo")

    print("\tdecision vars{}".format(original_influence_diagram.decision_nodes))
    print("\tpartial elim ordering")
    print("\tbefore relaxation:{}".format(original_influence_diagram.partial_elim_order))
    print("\tafter  relaxation:{}".format(relaxed_influence_diagram.partial_elim_order))


if __name__ == "__main__":
    files = [f for f in os.listdir(TRANS_PATH) if os.path.isfile(os.path.join(TRANS_PATH, f))
             and f.endswith(".uai")
             and not f.endswith(".mixed.uai")
             and not f.endswith(".mmap.uai")
             ]
    print("total {} files".format(len(files)))
    pp.pprint(sorted(files))
    for uai_name in sorted(files):
        # if uai_name.startswith("rand-"):
        print("start {}".format(uai_name))
        run_trans_msis(os.path.join(TRANS_PATH, uai_name.replace(".uai", "")),  True)
        print("processed {}".format(uai_name))

    # uai_name = "ID_from_BN_0_w32d11"
    # run_trans_msis(os.path.join(TRANS_PATH, uai_name.replace(".uai", "")), True)

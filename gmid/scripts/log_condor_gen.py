PRJ_PATH = "/home/junkyul/conda/gmid"
import sys
sys.path.append(PRJ_PATH)
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)


####################################################################################################################

MEM_HEADER = """
universe = vanilla
notification = never
should_transfer_files = yes
when_to_transfer_output = always
copy_to_spool = false
executable = mmap_2018

    """

MEM_JOB = """
requirements = regexp("slot({use_slots})@pedigree-({use_clusters}).ics.uci.edu", Name)
initialdir = {init_dir}
output = {uai_file}__{option_strings_name}__ibd={i_bd}.search
error  = err_{uai_file}__{option_strings_name}__ibd={i_bd}.search
log    = log_{uai_file}__{option_strings_name}__ibd={i_bd}.search
transfer_input_files = {program}, {uai_file}, {map_file}
arguments = -f {uai_file} -M {map_file} {option_strings_cmd} --time-limit {time_limit} --ibound {i_bd}
queue

    """

####################################################################################################################

program_name = 'mmap_2018'
haplo1_root = '/home/junkyul/mmap_trans'  # e.g. haplo1_root / park / promedas / or_chacin... /
clusters_available = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

local_root = "/home/junkyul/gmid_data/exec/mmap2018/mmap_trans"
problem_set = "/home/junkyul/gmid_data/exec/mmap2018/mmap_trans/problem_set"

command_per_solver = {}
command_per_solver['aaobf+wmb-mm'] = '-a any-aaobf -H wmb-mm --verbose --positive --seed 12345678 --cache-size 4g'
command_per_solver['aaobf+wmb-jg'] = '-a any-aaobf -H wmb-jg --jglp 2000 --verbose --positive --seed 12345678 --cache-size 4g'
# ibounds = [1, 3, 5, 10, 12, 16, 20, 24, 28, 32]
ibounds = [1, 5, 10, 15, 20, 25, 30]


def run_condor_gen(solver_name, time_limit=1, memory_limit=1):
    instance_names = [f for f in os.listdir(problem_set) if os.path.isfile(os.path.join(problem_set, f))
                      and f.endswith(".mmap.uai")]
    cluster_assigned = '|'.join([str(c) for c in clusters_available])
    condor_file = open(os.path.join(local_root, solver_name+"mmap_trans.condor"), 'w')
    condor_file.write(MEM_HEADER)

    for instance in sorted(instance_names):
        uai_file = instance
        map_file = instance.replace(".uai", ".map")
        vo_file = instance.replace(".uai", ".vo")

        for i_bd in ibounds:
            template_solver_filled = MEM_JOB.format(
                use_slots="[1-1]",
                use_clusters=cluster_assigned,
                init_dir=haplo1_root + '/problem_set',
                uai_file=uai_file,
                map_file=map_file,
                vo_file=vo_file,
                option_strings_name = solver_name,
                option_strings_cmd=command_per_solver[solver_name],
                program='../' + program_name,           # 1 level above init
                program_path='./' + program_name,       # next to *.condor
                # time_limit_shell=time_limit + 300,
                time_limit=time_limit,
                # memory_limit_kb=memory_limit * 1000,
                i_bd=i_bd
            )
            condor_file.write(template_solver_filled)
    condor_file.close()


if __name__ == "__main__":
    run_condor_gen("aaobf+wmb-mm")



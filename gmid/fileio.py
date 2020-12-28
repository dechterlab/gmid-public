from constants import *
from pyGM.filetypes import readUai, readEvidence14 #readFileByTokens
from pyGM.varset_py import Var, VarSet
from pyGM import Factor
from valuation import Valuation


########################################################################################################################
# helper functions
class FileReadError(Exception):
    """Error while reading an input problem file"""


class FileWriteError(Exception):
    """Error while writing an output problem file"""


def get_a_line(filename):
    for line in open(filename):
        line_strip = line.strip()
        if line_strip:
            yield line_strip


def get_a_token(filename):
    for each_line in get_a_line(filename):
        for each_token in each_line.split():
            if each_token:
                yield each_token


def make_valuations(factors, factor_types):
    valuations = []
    assert len(factors) == len(factor_types), 'length of both list should match'
    for f, t in zip(factors, factor_types):
        if t == 'P':
            valuations.append(Valuation(f, Factor([], 0.0) ))
        elif t == 'U':
            valuations.append(Valuation(Factor([], 1.0), f))
        else:
            raise FileReadError
    return valuations


def scope_of_vars_to_int(scope):
    return [v.label for v in scope]


def scopes_of_vars_to_int(scopes):
    return [scope_of_vars_to_int(sc) for sc in scopes]


def make_weights(var_types):
    return [1.0 if el == 'C' else 0.0 for el in var_types]


def remove_constant_factors(factors, is_log = False, factor_types=None):
    raise NotImplementedError


def remove_free_variables(factors, variables, var_types=None, weights=None):
    raise NotImplementedError


def remove_evidence():
    raise NotImplementedError


########################################################################################################################
# simple read and write functions
def read_uai(filename):
    # todo re-write and remove dependency
    """ read conventional uai file """
    return readUai(filename)        # returns list of factors


def read_evid(filename):
    # todo re-write and remove dependency
    return readEvidence14(filename)


def read_vo(filename):
    vars = []
    iw = None
    for each_line in get_a_line(filename):
        if not each_line.startswith('#'):
            vars.append(int(each_line))
        if iw is None and 'iw=' in each_line:
            iw = int(each_line.split('iw=')[-1])
    if vars[0] != len(vars)-1:
        raise FileReadError
    return vars[1:], iw         # elim_order, induced width


def read_map(filename):
    vars = []
    for each_line in get_a_line(filename):
        if not each_line.startswith('#'):
            for each_token in each_line.split():
                vars.append(int(each_token))
    if vars[0] != len(vars)-1:
        raise FileReadError
    return vars[1:]


def read_pvo(filename):
    fp = open(filename, 'r')
    blocks = []
    for each_block in fp.read().split(';'):
        if len(each_block.strip()):
            if each_block.startswith('#'):
                continue
            current_block = [int(el) for el in each_block.split()]
            blocks.append(current_block)
            # blocks.append([int(el.strip()) for el in each_block.split() if not el.strip().startswith('#') and el.strip()])
    nvars = blocks[0][0]        # first block total number of variables
    nblocks = blocks[1][0]      # second block total number of blocks
    assert len(blocks[2:]) == nblocks, "number of blocks wrong while reading pvo"
    # if len(blocks[2:]) != nblocks:
    #     raise FileReadError
    assert nvars == sum([len(el) for el in blocks[2:]]), "number of variables wrong while reading pvo"
    # if nvars != sum([len(el) for el in blocks[2:]]):
    #     raise FileReadError
    fp.close()
    return blocks[2:], nblocks, nvars


def fix_pvo(filename):
    fp = open(filename, 'r')
    blocks = []
    for each_block in fp.read().split(';'):
        if len(each_block.strip()):
            if each_block.startswith('#'):
                continue
            current_block = [int(el) for el in each_block.split()]
            blocks.append(current_block)
    nvars = blocks[0][0]        # first block total number of variables
    nblocks = blocks[1][0]      # second block total number of blocks
    assert len(blocks[2:]) == nblocks, "number of blocks wrong while reading pvo"
    fp.close()

    history=set()
    old_blocks = blocks[2:]
    new_blocks = []
    if nvars != sum([len(el) for el in old_blocks]):
        for old_block in reversed(old_blocks):  # temporal order
            current_bk = [b for b in old_block if b not in history]
            if current_bk:
                new_blocks.append(current_bk)
                history.update(current_bk)
    new_blocks = list(reversed(new_blocks))     # elim order
    assert sum([len(el) for el in new_blocks]) == nvars, "the number of variables should match"
    write_pvo_from_partial_elim_order(filename, new_blocks)


def read_mi(filename):
    gen = get_a_token(filename)
    num_vars = int(next(gen))
    var_types = [next(gen).upper() for _ in range(num_vars)]
    return num_vars, var_types           # num vars, type of vars in upper case


def read_id(filename):
    gen = get_a_token(filename)
    num_vars = int(next(gen))
    var_types = [next(gen).upper() for _ in range(num_vars)]
    num_funcs = int(next(gen))
    factor_types = [next(gen).upper() for _ in range(num_funcs)]
    return num_vars, var_types, num_funcs, factor_types


def read_pt(filename):
    try:
        gen = get_a_token(filename)
        num_vars = int(next(gen))
        return [int(next(gen)) for _ in range(num_vars)]
    except:
        return []


def read_standard_uai(file_name, sort_scope=True, skip_table=False):
    uai_file_name = file_name if file_name.endswith(".uai") else file_name + ".uai"
    file_info = {'domains': [], 'scopes': [], 'factors': []}
    gen = get_a_token(uai_file_name)
    type = next(gen)
    file_info['nvar'] = int(next(gen))
    file_info['domains'] = [int(next(gen)) for i in range(file_info['nvar'])]
    nfuncs = int(next(gen))
    for i in range(nfuncs):
        scope_size = int(next(gen))
        current_scope = []
        for s in range(scope_size):
            var_id = int(next(gen))
            current_scope.append(Var(var_id, file_info['domains'][var_id]))
        file_info['scopes'].append(current_scope)
    if skip_table:
        for i in range(nfuncs):
            file_info['factors'].append(Factor(file_info['scopes'][i]))
    else:
        for i in range(nfuncs):
            num_rows = int(next(gen))
            current_table = [float(next(gen)) for _ in range(num_rows)]
            if ZERO > 0:
                current_table = [el + ZERO for el in current_table if el == 0]
            factor_size = tuple(v.states for v in file_info['scopes'][i]) if len(file_info['scopes'][i]) else (1,)
            tab = np.array(current_table, dtype=float, order='C').reshape(factor_size)
            if sort_scope:
                tab = np.transpose(tab, tuple(np.argsort([v.label for v in file_info['scopes'][i]])))
            file_info['factors'].append(Factor(file_info['scopes'][i]))     # Factor takes list of Vars as a VarSet, sorted
            file_info['factors'][-1].table = np.array(tab, dtype=float, order=orderMethod)
    return file_info


def write_standard_uai(file_name, file_info, file_type):
    uai_file = open(file_name, 'w')
    uai_file.write("{}\n".format(file_type))
    uai_file.write("{}\n".format(file_info['nvar']))
    uai_file.write("{}\n".format(" ".join(str(el) for el in file_info['domains'])))
    uai_file.write("{}\n".format(len(file_info['scopes'])))
    for each_scope in file_info['scopes']:
        uai_file.write("{}\n".format(' '.join([str(len(each_scope))]+[str(el) for el in each_scope])))
    uai_file.write("\n")
    for each_factor in file_info['factors']:
        uai_file.write("{:d}\n".format(each_factor.numel()) + "\n".join(map(str, each_factor.t.ravel(order='C'))) + "\n\n")
    uai_file.close()


def write_vo_from_elim_order(file_name, elim_order, iw):
    vo_file_name = file_name if file_name.endswith(".vo") else file_name + ".vo"
    vo_file = open(vo_file_name, 'w')
    vo_file.write("# iw={}\n".format(iw))
    vo_file.write(("{}\n".format(len(elim_order))))
    for el in elim_order:
        vo_file.write("{}\n".format(el))
    vo_file.close()


def write_pvo_from_partial_elim_order(file_name, partial_variable_ordering):
    pvo_file = open(file_name, 'w')
    # partial variable ordering defines blocks of variables
    pvo_list = [el for el in partial_variable_ordering if len(el) > 0]      # exclude zero length sub-list
    num_blocks = len(pvo_list)
    num_var = max(max(el) for el in pvo_list) + 1                           # total var = largest var id + 1
    pvo_file.write("{};\n".format(num_var))
    pvo_file.write("{};\n".format(num_blocks))
    for block in pvo_list:
        pvo_file.write("{};\n".format(" ".join((str(v) for v in block))))
    pvo_file.close()


def write_id_from_types(file_name, var_types, func_types):
    id_file = open(file_name, 'w')
    id_file.write("{}\n".format(len(var_types)))
    id_file.write("{}\n".format(" ".join((el.upper() for el in var_types))))     # C, D
    id_file.write("{}\n".format(len(func_types)))
    id_file.write("{}\n".format(" ".join((el.upper() for el in func_types))))     # P, U
    id_file.close()


def write_mi_from_types(file_name, var_types):
    mi_file = open(file_name, 'w')
    mi_file.write("{}\n".format(len(var_types)))
    mi_file.write("{}\n".format(" ".join((el.upper() for el in var_types))))
    mi_file.close()


def write_map_from_types(file_name, var_types):
    map_file = open(file_name, 'w')
    dec_vars = [str(el) for el in range(len(var_types)) if var_types[el] == "D"]
    map_file.write("{}\n".format(len(dec_vars)))
    map_file.write("{}\n".format("\n".join(dec_vars)))
    map_file.close()


def write_map_from_list(file_name, map_vars):
    map_file = open(file_name, 'w')
    map_file.write("{}\n".format(len(map_vars)))
    map_file.write("{}\n".format("\n".join(str(el) for el in map_vars)))
    map_file.close()

########################################################################################################################
# read influence diagram or related
def read_uai_id(file_name, sort_scope=True, skip_table=False):
    uai_file_name = file_name if file_name.endswith(".uai") else file_name +".uai"
    pvo_file_name = file_name.replace(".uai", ".pvo") if file_name.endswith(".uai") else file_name + ".pvo"
    id_file_name = file_name.replace(".uai", ".id") if file_name.endswith(".uai") else file_name + ".id"
    pt_file_name = file_name.replace(".uai", ".pt") if file_name.endswith(".uai") else file_name + ".pt"

    file_info = {'nvar':0, 'domains':[], 'nprob':0, 'ndec':0, 'nutil':0,
                 'scopes':[], 'scope_types': [],
                 'factors':[], 'factor_types':[],
                 'blocks':[], 'var_types':[]}
    blocks, num_blocks, nvars = read_pvo(pvo_file_name)
    nvars, var_types, nfuncs, func_types = read_id(id_file_name)
    pseudo_tree = read_pt(pt_file_name)

    file_info['nvar'] = nvars
    file_info['blocks'] = blocks
    file_info['var_types'] = var_types
    file_info['factor_types'] = func_types
    file_info['scope_types'] = func_types       # scope and func types are the same because no decision appears
    nprob = len([el for el in var_types if el == 'C'])
    ndec = nvars - nprob
    nutil = nfuncs - nprob
    file_info['nprob'] = nprob
    file_info['ndec'] = ndec
    file_info['nutil'] = nutil
    file_info['pseudo_tree'] = pseudo_tree

    # gen = readFileByTokens(uai_file_name, '(),')  # split on white space, (, ), and ,
    gen = get_a_token(uai_file_name)
    type = next(gen)
    assert int(next(gen)) == nvars, "nvars error"
    file_info['domains'] = [int(next(gen)) for _ in range(nvars)]
    assert len(file_info['domains']) == nvars, "domains error"
    assert int(next(gen)) == nfuncs, "nfuncs error"
    for _ in range(nfuncs):
        scope_size = int(next(gen))
        current_scope = []
        for s in range(scope_size):
            var_id = int(next(gen))
            current_scope.append(Var(var_id, file_info['domains'][var_id]))
        file_info['scopes'].append(current_scope)

    if skip_table:
        for i in range(nfuncs):
            file_info['factors'].append(Factor(file_info['scopes'][i]))
    else:
        for i in range(nfuncs):
            num_rows = int(next(gen))
            current_table = [float(next(gen)) for _ in range(num_rows)]
            if ZERO > 0:
                current_table = [el + ZERO for el in current_table if el == 0]
            factor_size = tuple(v.states for v in file_info['scopes'][i]) if len(file_info['scopes'][i]) else (1,)
            tab = np.array(current_table, dtype=float, order='C').reshape(factor_size)
            if sort_scope:
                tab = np.transpose(tab, tuple(np.argsort([v.label for v in file_info['scopes'][i]])))
            file_info['factors'].append(Factor(file_info['scopes'][i]))     # Factor takes list of Vars as a VarSet, sorted
            file_info['factors'][-1].table = np.array(tab, dtype=float, order=orderMethod)
    return file_info


def read_erg(filename, sort_scope=True, skip_table=False):
    file_info = {'nvar':0, 'domains':[], 'nprob':0, 'ndec':0, 'nutil':0,
                 'scopes':[], 'scope_types': [],
                 'factors':[], 'factor_types':[],
                 'blocks':[], 'var_types':[]}
    # gen = readFileByTokens(filename, '(),')     # split on white space, (, ), and ,
    gen = get_a_token(filename)
    type = next(gen)
    nvar = int(next(gen))
    file_info['nvar'] = nvar
    domains = []
    for i in range(nvar):
        domains.append(int(next(gen)))
    file_info['domains'] = domains
    var_type_dict = {}
    dec_vars = []
    for i in range(nvar):
        var_type_dict[i] = next(gen)
        if var_type_dict[i] in ['d', 'D']:
            dec_vars.append(i)
        file_info['var_types'].append('D' if var_type_dict[i]=='d' else 'C')
    file_info['ndec'] = len(dec_vars)
    file_info['nprob'] = nvar - file_info['ndec']
    temporal_order = []
    for i in range(nvar):
        temporal_order.append(int(next(gen)))
    nfactor = int(next(gen))            # sum of prob, dec, util
    file_info['nutil'] = nfactor - file_info['ndec'] - file_info['nprob']
    func_type_dict = {}
    for i in range(nfactor):
        func_type_dict[i] = next(gen)   # p d u
        scope_size = int(next(gen))     # num vars in func
        current_scope = []
        for s in range(scope_size):
            var_id = int(next(gen))     # get a var id
            current_scope.append(Var(var_id, file_info['domains'][var_id]))
        file_info['scopes'].append(current_scope)
        file_info['scope_types'].append(func_type_dict[i].upper())
    for i in range(len(file_info['scope_types'])):            # read each table
        num_rows = int(next(gen))       # read num rows
        if num_rows == 0:
            continue
        else:
            # file_info['factors'].append(Factor(file_info['scopes'][i]))
            file_info['factor_types'].append(file_info['scope_types'][i])
            assert file_info['scope_types'][i] != 'D'
            current_table = [float(next(gen)) for _ in range(num_rows)]
            if ZERO > 0:
                current_table = [el + ZERO for el in current_table if el == 0]
            # current_table = []
            # for t in range(num_rows):
            #     val = float(next(gen))
            #     if val == 0:
            #         val += ZERO
            #     current_table.append(val)
            factor_size = tuple(v.states for v in file_info['scopes'][i]) if len(file_info['scopes'][i]) else (1,)
            tab = np.array(current_table, dtype=float, order='C').reshape(factor_size)
            if sort_scope:
                tab = np.transpose(tab, tuple(np.argsort([v.label for v in file_info['scopes'][i]])))
            file_info['factors'].append(Factor(file_info['scopes'][i]))  # Factor takes list of Vars as a VarSet, sorted
            file_info['factors'][-1].table = np.array(tab, dtype=float, order=orderMethod)
    # from temporal ordering recover blocks; chance | dec | chance | dec | ... | hidden
    blocks = []
    current_block = []
    for i in temporal_order:
        if var_type_dict[i] in ['c', 'C']:
            current_block.append(i)
        else:
            if len(current_block):                  # first add obs
                blocks.append(current_block)
                current_block = []
            blocks.append([i])                      # then add dec
    if len(current_block):
        blocks.append(current_block)                # hidden chance vars
    file_info['blocks'] = list(reversed(blocks))   # return elim order; hidden vars -> dec -> obs
    file_info['nblock'] = len(blocks)
    return file_info


def read_limid(filename, sort_scope=True):
    file_info = {'nvar':0, 'domains':[], 'nprob':0, 'ndec':0, 'nutil':0,
                 'scopes':[], 'scope_types' : [],
                 'factors':[], 'factor_types':[],
                 'blocks':[], 'var_types':[]}
    # gen = readFileByTokens(filename, '(),')     # split on white space, (, ), and ,
    gen = get_a_token(filename)
    type = next(gen)
    nvar = int(next(gen))
    file_info['nvar'] = nvar
    domains = []
    for i in range(nvar):
        domains.append(int(next(gen)))
    file_info['domains'] = domains
    file_info['nprob'] = int(next(gen))
    file_info['ndec'] = int(next(gen))
    file_info['nutil'] = int(next(gen))

    for i in range(file_info['nprob'] + file_info['ndec'] + file_info['nutil']):
        scope_size = int(next(gen))
        current_scope = []
        for s in range(scope_size):
            var_id = int(next(gen))     # get a var id
            current_scope.append(Var(var_id, file_info['domains'][var_id]))
        file_info['scopes'].append(current_scope)
        if i < file_info['nprob']:
            file_info['scope_types'].append('P')
        elif i < file_info['nprob'] + file_info['ndec']:
            file_info['scope_types'].append('D')
        else:
            file_info['scope_types'].append('U')

    for i in range(file_info['nprob'] + file_info['ndec'] + file_info['nutil']):
        if file_info['nprob'] <= i < file_info['nprob'] + file_info['ndec']:
            continue        # skip decision tables; not shown in the file
        else:
            num_rows = int(next(gen))
            current_scope = file_info['scopes'][i]
            file_info['factor_types'].append('P' if i < file_info['nprob'] else 'U')
            current_table = [float(next(gen)) for _ in range(num_rows)]
            if ZERO > 0:
                current_table = [el + ZERO for el in current_table if el == 0]
            factor_size = tuple(v.states for v in current_scope) if len(current_scope) else (1,)
            tab = np.array(current_table, dtype=float, order='C').reshape(factor_size)
            if sort_scope:
                tab = np.transpose(tab, tuple(np.argsort([v.label for v in current_scope])))
            file_info['factors'].append(Factor(file_info['scopes'][i]))  # Factor takes list of Vars as a VarSet, sorted
            file_info['factors'][-1].table = np.array(tab, dtype=float, order=orderMethod)

    blocks = []
    decision_vars = []
    observed_vars = set()
    for i in range(file_info['nprob'], file_info['nprob']+file_info['ndec']):
        current_scope = [v.label for v in file_info['scopes'][i]]
        blocks.append(current_scope[:-1])    # the last var label is for decision
        blocks.append([current_scope[-1]])   # put decision in a separate block
        decision_vars.append(current_scope[-1])
        observed_vars.update(current_scope[:-1])
    hidden_vars = []
    for el in range(file_info['nvar']):
        if el in decision_vars:
            file_info['var_types'].append('D')
        elif el in observed_vars:
            file_info['var_types'].append('C')
        else:
            file_info['var_types'].append('C')
            hidden_vars.append(el)
    if hidden_vars:
        blocks.append(hidden_vars)
    file_info['blocks'] = list(reversed(blocks))        # return elim order; hidden vars -> ..
    file_info['nblock'] = len(blocks)
    return file_info

def read_maua(filename, sort_scope=False):
    file_info = {'nvar':0, 'domains':[], 'nprob':0, 'ndec':0, 'nutil':0,
                 'scopes':[], 'scope_types' : [],
                 'factors':[], 'factor_types':[],
                 'blocks':[], 'var_types':[]}
    gen = get_a_token(filename)
    file_type = next(gen)
    if "/*" in file_type:
        while "*/" not in next(gen):
            pass
        file_type = next(gen)
    assert file_type == "LIMID"
    file_info['nprob'] = int(next(gen))
    file_info['ndec'] = int(next(gen))
    file_info['nutil'] =int(next(gen))
    file_info['nvar'] = file_info['nprob'] + file_info['ndec']
    # by conevnetion var from 0 to nprob-1 are chance vars and nprob to nvar-1 are decision vars
    file_info['domains'] = [int(next(gen)) for _ in range(file_info['nvar'])]
    # scopes are defined per each variable because ID is a DAG and 1 node can hold 1 function
    # maua's format only defines parents, different from UAI format https://github.com/denismaua/kpu-pp
    for i in range(file_info['nprob'] + file_info['ndec'] + file_info['nutil']):
        num_pa = int(next(gen))
        if i < file_info['nprob'] + file_info['ndec']:
            current_scope = list(reversed([int(next(gen)) for _ in range(num_pa)])) + [i]
        else:
            current_scope = list(reversed([int(next(gen)) for _ in range(num_pa)]))
        current_scope = [Var(v, file_info['domains'][v]) for v in current_scope]
        file_info['scopes'].append(current_scope)
        if i < file_info['nprob']:
            file_info['scope_types'].append('P')
        elif i < file_info['nprob'] + file_info['ndec']:
            file_info['scope_types'].append('D')
        else:
            file_info['scope_types'].append('U')
    # read tables
    for i in range(file_info['nprob'] + file_info['ndec'] + file_info['nutil']):
        if file_info['nprob'] <= i < file_info['nprob'] + file_info['ndec']:
            continue        # skip decision tables; not shown in the file
        else:
            num_rows = int(next(gen))
            current_scope = file_info['scopes'][i]
            file_info['factor_types'].append('P' if i < file_info['nprob'] else 'U')
            current_table = [float(next(gen)) for _ in range(num_rows)]
            if ZERO > 0:
                current_table = [el + ZERO for el in current_table if el == 0]
            factor_size = tuple(v.states for v in current_scope) if len(current_scope) else (1,)
            try:
                tab = np.array(current_table, dtype=float, order='C').reshape(factor_size)
            except:
                print("err")
            if sort_scope:
                tab = np.transpose(tab, tuple(np.argsort([v.label for v in current_scope])))
            file_info['factors'].append(Factor(file_info['scopes'][i]))  # Factor takes list of Vars as a VarSet, sorted
            file_info['factors'][-1].table = np.array(tab, dtype=float, order=orderMethod)
    # read partial variable ordering from decision scopes assuming decisions follow temporal order
    blocks = []
    decision_vars = []
    observed_vars = set()
    for i in range(file_info['nprob'], file_info['nprob']+file_info['ndec']):
        current_scope = [v.label for v in file_info['scopes'][i]]
        blocks.append(current_scope[:-1])    # the last var label is for decision
        blocks.append([current_scope[-1]])   # put decision in a separate block
        decision_vars.append(current_scope[-1])
        observed_vars.update(current_scope[:-1])
    hidden_vars = []
    for el in range(file_info['nvar']):
        if el in decision_vars:
            file_info['var_types'].append('D')
        elif el in observed_vars:
            file_info['var_types'].append('C')
        else:
            file_info['var_types'].append('C')
            hidden_vars.append(el)
    if hidden_vars:
        blocks.append(hidden_vars)
    file_info['blocks'] = list(reversed(blocks))        # return elim order, so starting from hidden vars -> ..
    file_info['nblock'] = len(blocks)
    return file_info


def read_uai_bn(filename):
    # read a BN and return a dict of dict
    #       { var_id:
    #               {domain_size, parent_vars, scope_vars, table_length, table}
    #       }
    # BN uai file assumes that functions are shown in the order of variable labels and
    # the last variable of a scope is the head
    bn_lines = get_a_line(filename)
    bn_dict = {}
    print("reading uai file type of {}".format(next(bn_lines)))
    num_vars = int(next(bn_lines))
    for ind, k in enumerate(next(bn_lines).split()):
        bn_dict[ind] = {}
        bn_dict[ind]['domain_size'] = int(k)
    assert len(bn_dict) == num_vars, "num vars error"
    num_funcs = int(next(bn_lines))
    ### encode scope
    for ind in range(num_funcs):
        scope = [int(v) for v in next(bn_lines).split()]    # the first element is the scope size
        var_id = scope[-1]
        assert ind == var_id, "bn assumes the n-th function is defined by the n-th variable"
        assert scope[0] == (len(scope)-1), "scope size error"
        bn_dict[var_id]['parents'] = scope[1:-1]    # parent variables
        bn_dict[var_id]['scope'] = scope[1:]        # include self, scope shown in the file
        assert len(bn_dict[var_id]['scope']) > 0, "scope must be greater than 0"
        bn_dict[var_id]['table_length'] = 1
        for v in bn_dict[var_id]['scope']:
            bn_dict[var_id]['table_length'] *= bn_dict[v]['domain_size']
    ### store tables as a list
    for var_id in range(num_funcs):
        table_length = int(next(bn_lines))
        assert table_length == bn_dict[var_id]['table_length'], "table length error"
        cpt_values = []
        while len(cpt_values) < table_length:
            cpt_values.extend([float(val) for val in next(bn_lines).split()])
        bn_dict[var_id]['table'] = cpt_values
    return bn_dict


########################################################################################################################
# read pure mmap or mixed mmap
def read_uai_mmap(file_name, sort_scope=True):
    uai_file_name = file_name if file_name.endswith(".uai") else file_name + ".uai"
    map_file_name = file_name if file_name.endswith(".map") else file_name + ".map"

    file_info = {'nvar': 0, 'domains': [], 'nprob': 0, 'ndec': 0, 'nutil': 0,
                 'scopes': [], 'scope_types': [],
                 'factors': [], 'factor_types': [],
                 'blocks': [], 'var_types': []}

    uai_info = read_standard_uai(uai_file_name, sort_scope)
    map_vars = read_map(map_file_name)

    file_info['nvar'] = len(uai_info['domains'])
    file_info['domains'] = uai_info['domains']
    file_info['ndec'] = len(map_vars)
    file_info['nprob'] = file_info['nvar'] - file_info['ndec']
    file_info['scopes'] = uai_info['scopes']        # list of Vars
    file_info['scope_types'] = ['P']* len(file_info['scopes'])
    file_info['factors'] = uai_info['factors']      # list of factors
    file_info['factor_types'] = file_info['scope_types']
    file_info['blocks'] = [[el for el in range(file_info['nvar']) if el not in map_vars], map_vars]
    file_info['var_types'] = ['D' if el in map_vars else 'C' for el in range(file_info['nvar'])]

    return file_info


def read_uai_mixed(file_name, sort_scope=True, skip_table=False):
    uai_file_name = file_name if file_name.endswith(".uai") else file_name + ".uai"
    pvo_file_name = file_name.replace(".uai", ".pvo") if file_name.endswith(".uai") else file_name + ".pvo"
    mi_file_name = file_name.replace(".uai", ".mi") if file_name.endswith(".uai") else file_name + ".mi"

    file_info = {'nvar': 0, 'domains': [], 'nprob': 0, 'ndec': 0, 'nutil': 0,
                 'scopes': [], 'scope_types': [], 'factors': [], 'factor_types': [], 'blocks': [], 'var_types': []}

    uai_info = read_standard_uai(uai_file_name, sort_scope, skip_table)
    blocks, nblocks, nvars = read_pvo(pvo_file_name)
    num_vars, var_types = read_mi(mi_file_name)
    dec_vars = [el for el in range(num_vars) if var_types[el] == 'D']

    file_info['nvar'] = nvars
    file_info['domains'] = uai_info['domains']
    file_info['ndec'] = len(dec_vars)
    file_info['nprob'] = nvars - file_info['ndec']
    file_info['scopes'] = uai_info['scopes']
    file_info['scope_types'] = ['P'] * len(file_info['scopes'])
    file_info['factor_types'] = file_info['scope_types']
    file_info['factors'] = uai_info['factors']      # list of factors
    file_info['blocks'] = blocks
    file_info['var_types'] = var_types

    return file_info


########################################################################################################################
# translate influence diagram to  pure mmap or mixed mmap
def translate_uai_id_to_mixed(id_file_info):
    mmap_file_info = {'nvar': 0, 'domains': [], 'nprob': 0, 'ndec': 0, 'nutil': 0, 'scopes': [], 'scope_types': [],
                      'factors': [], 'factor_types': [], 'blocks': [], 'var_types': []}

    mmap_file_info['nvar'] = id_file_info['nvar'] + 1
    mmap_file_info['domains'] = id_file_info['domains'] + [id_file_info['nutil']]       # append 1 latent var
    mmap_file_info['nprob'] = id_file_info['nprob'] + 1
    mmap_file_info['ndec'] = id_file_info['ndec']
    latent_var_id = mmap_file_info['nvar'] - 1
    latent_var = Var(latent_var_id, id_file_info['nutil'])
    for scope_ind, scope in enumerate(id_file_info['scopes']):
        if id_file_info['scope_types'][scope_ind] == 'U':
            mmap_file_info['scopes'].append([latent_var] + scope)           # prepend latent var
        else:
            mmap_file_info['scopes'].append(scope)
        mmap_file_info['scope_types'] = 'P'
        mmap_file_info['factor_types'] = 'P'
    mmap_file_info['var_types'] = id_file_info['var_types'] + ['C']         # the last latent variable is Chance/sum

    dec_vars = [i for i in range(id_file_info['nvar']) if id_file_info['var_types'][i] == 'D']
    if id_file_info['blocks'][0][0] in dec_vars:
        mmap_file_info['blocks'] = [[latent_var_id]] + [el for el in id_file_info['blocks']]
    else:
        mmap_file_info['blocks'] = [[latent_var_id]+id_file_info['blocks'][0]] + [el for el in id_file_info['blocks'][1:]]

    util_factor_count = 0
    for factor_ind, factor in enumerate(id_file_info['factors']):
        if id_file_info['factor_types'][factor_ind] == 'U':
            mmap_scope = mmap_file_info['scopes'][factor_ind]               # scopes and factors follow same index
            mmap_factor_dim = tuple(v.states for v in mmap_scope) if len(mmap_scope) else (1,)
            util_factor_table = factor.t
            assert util_factor_table.shape == mmap_factor_dim[1:]      # the first dim associated with the latent var
            assert mmap_factor_dim[0] == id_file_info['nutil']
            mmap_factor_table = np.ones(mmap_factor_dim, dtype=float, order='C')
            mmap_factor_table[util_factor_count] = util_factor_table
            util_factor_count += 1
            mmap_factor = Factor(mmap_scope)
            mmap_factor.t = mmap_factor_table
            mmap_file_info['factors'].append(mmap_factor)
        else:
            mmap_file_info['factors'].append(factor)    # append the same factor
    return mmap_file_info


# def translate_id_to_mmap(id_file_info):
#     mmap_file_info = {'nvar': 0, 'domains': [], 'nprob': 0, 'ndec': 0, 'nutil': 0, 'scopes': [], 'scope_types': [],
#                       'factors': [], 'factor_types': [], 'blocks': [], 'var_types': []}
#     raise NotImplementedError
#     return mmap_file_info
#
#
# def translate_mmap_to_id(mmap_file_info):
#     id_file_info = {'nvar': 0, 'domains': [], 'nprob': 0, 'ndec': 0, 'nutil': 0, 'scopes': [], 'scope_types': [],
#                       'factors': [], 'factor_types': [], 'blocks': [], 'var_types': []}
#     raise NotImplementedError
#     return id_file_info
#
#
# def translate_mixed_to_id(mmap_file_info):
#     id_file_info = {'nvar': 0, 'domains': [], 'nprob': 0, 'ndec': 0, 'nutil': 0, 'scopes': [], 'scope_types': [],
#                       'factors': [], 'factor_types': [], 'blocks': [], 'var_types': []}
#     raise NotImplementedError
#     return id_file_info


########################################################################################################################
# wrtie influence diagrams from file_info
def write_uai_id_from_info(file_name, file_info):
    uai_file_name = file_name if file_name.endswith(".uai") else file_name +".uai"
    pvo_file_name = file_name.replace(".uai", ".pvo") if file_name.endswith(".uai") else file_name + ".pvo"
    id_file_name = file_name.replace(".uai", ".id") if file_name.endswith(".uai") else file_name + ".id"
    uai_file = open(uai_file_name, 'w')
    uai_file.write("ID\n")
    uai_file.write("{}\n".format(file_info['nvar']))
    uai_file.write("{}\n".format(' '.join((str(el) for el in file_info['domains']))))
    uai_file.write("{}\n".format(file_info['nprob']+file_info['nutil']))
    for scope_ind, each_scope in enumerate(file_info['scopes']):      # exclude decisions scope;
        if file_info['scope_types'][scope_ind] != 'D':                # skip decision scopes in uai file
            uai_file.write("{}\n".format(' '.join([str(len(each_scope))]+[str(el) for el in each_scope])))
    uai_file.write("\n")
    for each_factor in file_info['factors']:        # scopes and factors follow the same order after removing decisions
        uai_file.write("{:d}\n".format(each_factor.numel()) + "\n".join(map(str, each_factor.t.ravel(order='C'))) + "\n\n")
    uai_file.close()
    write_pvo_from_partial_elim_order(pvo_file_name, file_info['blocks'])
    write_id_from_types(id_file_name, file_info['var_types'], file_info['factor_types'])


def write_limid_from_info(file_name, file_info):
    limid_file_name = file_name if file_name.endswith(".limid") else file_name + ".limid"
    limid_file = open(limid_file_name, 'w')
    limid_file.write("LIMID\n")
    limid_file.write("{}\n".format(file_info['nvar']))
    limid_file.write("{}\n".format(' '.join([str(el) for el in file_info['domains']])))
    limid_file.write("{}\n".format(file_info['nprob']))
    limid_file.write("{}\n".format(file_info['ndec']))
    limid_file.write("{}\n".format(file_info['nutil']))

    for scope_ind, each_scope in enumerate(file_info['scopes']):
        if file_info['scope_types'][scope_ind] == 'P':
            limid_file.write("{}\n".format(' '.join([str(len(each_scope))] + [str(el) for el in each_scope])))
    previous_block = []
    decision_temporal_order = []
    for each_block in reversed(file_info['blocks']):
        if len(each_block) == 1 and file_info['var_types'][each_block[0]] == 'D':
            decision_scope = previous_block + each_block         # [ parents ] + [ decision ]
            decision_temporal_order.append(each_block[0])
            limid_file.write("{}\n".format(' '.join([str(len(decision_scope))] + [str(el) for el in decision_scope])))
        previous_block = each_block
    for scope_ind, each_scope in enumerate(file_info['scopes']):
        if file_info['scope_types'][scope_ind] == 'U':
            limid_file.write("{}\n".format(' '.join([str(len(each_scope))] + [str(el) for el in each_scope])))
    limid_file.write("\n")

    for f_ind, each_factor in enumerate(file_info['factors']):
        if file_info['factor_types'][f_ind] == 'P':
            limid_file.write("{:d}\n".format(each_factor.numel()) + "\n".join(map(str, each_factor.t.ravel(order='C'))) + "\n\n")
    for f_ind, each_factor in enumerate(file_info['factors']):
        if file_info['factor_types'][f_ind] == 'U':
            limid_file.write("{:d}\n".format(each_factor.numel()) + "\n".join(map(str, each_factor.t.ravel(order='C'))) + "\n\n")
    limid_file.close()


def write_erg_from_info(file_name, vo_file, file_info):
    erg_file_name = file_name if file_name.endswith(".erg") else file_name + ".erg"
    erg_file = open(erg_file_name, 'w')
    erg_file.write("ID\n")
    erg_file.write("{}\n".format(file_info['nvar']))
    erg_file.write("{}\n".format(' '.join((str(el) for el in file_info['domains']))))
    erg_file.write("{}\n".format(' '.join((el.lower() for el in file_info['var_types']))))
    ### erg file shows a temporal ordering (reverse of the elimination ordering in vo_file)
    elim_order, iw = read_vo(vo_file)
    erg_file.write("{}\n".format(' '.join((str(el) for el in reversed(elim_order)))))
    erg_file.write("{}\n".format(file_info['nprob']+file_info['ndec']+file_info['nutil']))

    ### write probability, decision, utility scopes
    for f_ind, each_scope in enumerate(file_info['scopes']):
        if file_info['factor_types'][f_ind] == 'P':
            erg_file.write("p {}\n".format(' '.join([str(len(each_scope))] + [str(el) for el in each_scope])))
    ### decision scopes
    previous_block = []
    for each_block in reversed(file_info['blocks']):
        if len(each_block) == 1 and file_info['var_types'][each_block[0]] == 'D':
            decision_scope = previous_block + each_block         # [ parents ] + [ decision ]
            erg_file.write("d {}\n".format(' '.join([str(len(decision_scope))] + [str(el) for el in decision_scope])))
        previous_block = each_block
    for f_ind, each_scope in enumerate(file_info['scopes']):
        if file_info['factor_types'][f_ind] == 'U':
            erg_file.write("u {}\n".format(' '.join([str(len(each_scope))] + [str(el) for el in each_scope])))
    erg_file.write("\n")

    ### write probability, decision, utility tables
    for f_ind, each_factor in enumerate(file_info['factors']):
        if file_info['factor_types'][f_ind] == 'P':
            erg_file.write("{:d}\n".format(each_factor.numel()) + "\n".join(map(str, each_factor.t.ravel(order='C'))) + "\n\n")
    for _ in range(file_info['ndec']):
        erg_file.write("0\n")
    for f_ind, each_factor in enumerate(file_info['factors']):
        if file_info['factor_types'][f_ind] == 'U':
            erg_file.write("{:d}\n".format(each_factor.numel()) + "\n".join(map(str, each_factor.t.ravel(order='C'))) + "\n\n")
    erg_file.close()


########################################################################################################################
# write pure mmap or mixed mmap from file_info
def write_uai_mmap(file_name, file_info, uai_type="MARKOV"):
    uai_file_name = file_name if file_name.endswith(".uai") else file_name +".uai"
    map_file_name = file_name.replace(".uai", ".map") if file_name.endswith(".uai") else file_name + ".map"

    write_standard_uai(uai_file_name, file_info, uai_type)
    write_map_from_types(map_file_name, file_info['var_types'])


def write_uai_mixed(file_name, file_info, uai_type="MARKOV"):
    uai_file_name = file_name if file_name.endswith(".uai") else file_name + ".uai"
    pvo_file_name = file_name.replace(".uai", ".pvo") if file_name.endswith(".uai") else file_name + ".pvo"
    mi_file_name = file_name.replace(".uai", ".mi") if file_name.endswith(".uai") else file_name + ".mi"
    write_standard_uai(uai_file_name, file_info, uai_type)
    write_pvo_from_partial_elim_order(pvo_file_name, file_info['blocks'])
    write_mi_from_types(mi_file_name, file_info['var_types'])

########################################################################################################################
# convert formats
def convert_uai_id_to_mixed(uai_file_name, mixed_file_name):
    id_file_info = read_uai_id(uai_file_name, sort_scope=False)
    mmap_file_info = translate_uai_id_to_mixed(id_file_info)
    write_uai_mixed(mixed_file_name, mmap_file_info)


def convert_mmap_to_mixed(mmap_file_name, mixed_file_name):
    file_info = read_uai_mmap(mmap_file_name, sort_scope=False)
    write_uai_mixed(mixed_file_name, file_info)


def convert_uai_to_limid(uai_file_name, limid_file_name):
    file_info = read_uai_id(uai_file_name, sort_scope=False)
    write_limid_from_info(limid_file_name, file_info)


def convert_uai_to_erg(uai_file_name, vo_file, erg_file_name):
    file_info = read_uai_id(uai_file_name, sort_scope=False)
    write_erg_from_info(erg_file_name, vo_file, file_info)


def convert_erg_to_uai(erg_file_name, uai_file_name):
    file_info = read_erg(erg_file_name, sort_scope=False)
    write_uai_id_from_info(uai_file_name, file_info)


def convert_erg_to_limid(erg_file_name, limid_file_name):
    file_info = read_erg(erg_file_name, sort_scope=False)
    write_limid_from_info(limid_file_name, file_info)


def convert_limid_to_uai(limid_file_name, uai_file_name):
    file_info = read_limid(limid_file_name, sort_scope=False)
    write_uai_id_from_info(uai_file_name, file_info)


def convert_limid_to_erg(limid_file_name, vo_file, erg_file_name):
    raise NotImplementedError       # detour uai_id


def convert_maua_to_uai(maua_file_name, uai_file_name):
    file_info = read_maua(maua_file_name, sort_scope=False)
    write_uai_id_from_info(uai_file_name, file_info)


########################################################################################################################
# write influce diagrams from nx graph
def write_uai_from_nx_graph(file_name, influence_diagram):
    ### var_id and node_id are identical for chance, decision variables
    ### values nodes are shown up at the end of the nodes
    ### open a file
    uai_file = open(file_name, 'w')
    uai_file.write("ID\n")

    ### read variables
    chance_variables = []
    decision_variables = []
    value_nodes = []
    for n in sorted(influence_diagram.nodes_iter()):
        if influence_diagram.node[n]['node_type'] == 'C':
            chance_variables.append(n)
        elif influence_diagram.node[n]['node_type'] == 'D':
            decision_variables.append(n)
        elif influence_diagram.node[n]['node_type'] == 'U':
            value_nodes.append(n)
        else:
            assert False, "unknown node type in influenec diagram"

    ### write domains of variables
    num_vars = len(chance_variables) + len(decision_variables)
    num_funcs = len(chance_variables) + len(value_nodes)
    uai_file.write("{}\n".format(num_vars))
    domains = [influence_diagram.node[n]['domain_size'] for n in sorted(influence_diagram.nodes_iter())
                    if influence_diagram.node[n]['node_type'] in ['C', 'D']]
    uai_file.write("{}\n".format(" ".join([str(el) for el in domains])))

    # prob functions -> utility functions
    ### write scope of functions
    uai_file.write("{}\n".format(num_funcs))
    for var_id in chance_variables + value_nodes:
        if var_id in decision_variables:    # only write probability functions
            continue
        scope_line = [len(influence_diagram.node[var_id]['scope'])] + influence_diagram.node[var_id]['scope']
        uai_file.write("{}\n".format(" ".join(str(el) for el in scope_line)))

    uai_file.write("\n")
    ### write tables
    for var_id in chance_variables + value_nodes:
        table_length = influence_diagram.node[var_id]['table_length']
        uai_file.write("{}\n".format(table_length))
        table = influence_diagram.node[var_id]['table']
        uai_file.write("{}\n".format("\n".join(str(el) for el in table)))
        uai_file.write("\n")
    uai_file.close()


def write_erg_from_nx_graph(file_name, influence_diagram, temporal_ordering):
    ### var_id and node_id are identical for chance, decision variables
    ### values nodes are shown up at the end of the nodes
    ### open a file
    uai_file = open(file_name, 'w')
    uai_file.write("ID\n")

    ### read variables
    chance_variables = []
    decision_variables = []
    value_nodes = []
    for n in sorted(influence_diagram.nodes_iter()):
        if influence_diagram.node[n]['type'] == 'chance':
            chance_variables.append(n)
        elif influence_diagram.node[n]['type'] == 'decision':
            decision_variables.append(n)
        elif influence_diagram.node[n]['type'] == 'value':
            value_nodes.append(n)
        else:
            assert False, "unknown node type in influenec diagram"

    ### write num vars, domains, types
    num_vars = len(chance_variables) + len(decision_variables)
    uai_file.write("{}\n".format(num_vars))
    domains = [influence_diagram.node[n]['domain_size'] for n in sorted(influence_diagram.nodes_iter())
                    if influence_diagram.node[n]['type'] in ['chance', 'decision']]
    uai_file.write("{}\n".format(" ".join([str(el) for el in domains])))
    var_types = ['d' if el in decision_variables else 'c' for el in range(num_vars)]
    uai_file.write("{}\n".format(" ".join(var_types)))
    var_ordering = [temporal_ordering[el] for el in range(num_vars)]
    uai_file.write("{}\n".format(" ".join([str(el) for el in var_ordering])))

    ### write functions scopes
    num_funcs = len(chance_variables) + len(decision_variables) + len(value_nodes)
    uai_file.write("{}\n".format(num_funcs))
    for n in chance_variables + decision_variables + value_nodes:        # node id are strating from 0 to num_var-1 | util nodes
        scope_line = []
        if influence_diagram.node[n]['type'] == 'chance':
            scope_line.append('p')
        elif influence_diagram.node[n]['type'] == 'decision':
            scope_line.append('d')
        elif influence_diagram.node[n]['type'] == 'value':
            scope_line.append('u')
        else:
            assert False, "unknown node type in influence diagram"
        scope_line.append(len(influence_diagram.node[n]['scope']))
        scope_line.extend(influence_diagram.node[n]['scope'])
        uai_file.write("{}\n".format(" ".join([str(el) for el in scope_line])))

    uai_file.write("\n")
    ### write tables
    for n in chance_variables + decision_variables + value_nodes:  # node id are strating from 0 to num_var-1 | util nodes
        if n in decision_variables:
            uai_file.write("{}\n".format(0))
        else:
            table_length = influence_diagram.node[n]['table_length']
            uai_file.write("{}\n".format(table_length))
            table = influence_diagram.node[n]['table']
            uai_file.write("{}\n".format("\n".join(str(el) for el in table)))
        uai_file.write("\n")
    uai_file.close()


def write_limid_from_nx_graph(file_name, influence_diagram):
    ### var_id and node_id are identical for chance, decision variables
    ### values nodes are shown up at the end of the nodes
    ### open a file
    uai_file = open(file_name, 'w')
    uai_file.write("LIMID\n")

    ### read variables
    chance_variables = []
    decision_variables = []
    value_nodes = []
    for n in sorted(influence_diagram.nodes_iter()):
        if influence_diagram.node[n]['type'] == 'chance':
            chance_variables.append(n)
        elif influence_diagram.node[n]['type'] == 'decision':
            decision_variables.append(n)
        elif influence_diagram.node[n]['type'] == 'value':
            value_nodes.append(n)
        else:
            assert False, "unknown node type in influenec diagram"

    ### write num vars, domains, types
    num_vars = len(chance_variables) + len(decision_variables)
    uai_file.write("{}\n".format(num_vars))
    domains = [influence_diagram.node[n]['domain_size'] for n in sorted(influence_diagram.nodes_iter())
                    if influence_diagram.node[n]['type'] in ['chance', 'decision']]
    uai_file.write("{}\n".format(" ".join([str(el) for el in domains])))

    ### write function scopes, prob-dec-util
    uai_file.write("{}\n".format(len(chance_variables)))
    uai_file.write("{}\n".format(len(decision_variables)))
    uai_file.write("{}\n".format(len(value_nodes)))
    for n in chance_variables + decision_variables + value_nodes:
        scope_line = [len(influence_diagram.node[n]['scope'])]
        scope_line.extend(influence_diagram.node[n]['scope'])
        uai_file.write("{}\n".format(" ".join([str(el) for el in scope_line])))

    uai_file.write("\n")
    ### write tables
    for n in chance_variables + value_nodes:  # node id are strating from 0 to num_var-1 | util nodes
        table_length = influence_diagram.node[n]['table_length']
        uai_file.write("{}\n".format(table_length))
        table = influence_diagram.node[n]['table']
        uai_file.write("{}\n".format("\n".join(str(el) for el in table)))
        uai_file.write("\n")
    uai_file.close()


########################################################################################################################
# write mini bucket heuristic
def write_mini_bucket_heuristic_from_info(file_name, heur_info):
    file_name = file_name if file_name.endswith(".heur") else file_name + ".heur"
    heur_file = open(file_name, 'w')
    heur_file.write("{} {} {}\n".format(heur_info['num_var'], heur_info['num_msg'], heur_info['msg_id_start']))
    for var in range(heur_info['num_var']):
        if len(heur_info['bucket_msg'][var]) == 0:
            msg_str = ""
        else:
            msg_str = ' '.join([str(i) for i in heur_info['bucket_msg'][var]])
        heur_file.write("{} {} {}\n".format(var, len(heur_info['bucket_msg'][var]), msg_str ))
    heur_file.write("\n")

    for msg_id in range(heur_info['msg_id_start'], heur_info['msg_id_start']+heur_info['num_msg']):
        prob_msg = heur_info['msg_indexer'][msg_id].prob
        if type(prob_msg) != Factor:
            heur_file.write("0\n")
            heur_file.write("1\n")
            heur_file.write("{}\n".format(prob_msg))
        else:
            scope_str = " ".join([str(el) for el in prob_msg.vars] )
            heur_file.write( "{} {}\n".format(len(prob_msg.vars), scope_str) )
            heur_file.write("{}\n".format(prob_msg.numel()))
            table_str = "\n".join(map(str, prob_msg.table.ravel(order='C')))
            heur_file.write( "{}\n".format(table_str))

        value_msg = heur_info['msg_indexer'][msg_id].util
        if type(value_msg) != Factor:
            heur_file.write("0\n")
            heur_file.write("1\n")
            heur_file.write("{}\n".format(value_msg))
        else:
            scope_str = " ".join([str(el) for el in value_msg.vars])
            heur_file.write("{} {}\n".format(len(value_msg.vars), scope_str))
            heur_file.write("{}\n".format(value_msg.numel()))
            table_str = "\n".join(map(str, value_msg.table.ravel(order='C')))
            heur_file.write("{}\n".format(table_str))

        heur_file.write("\n")







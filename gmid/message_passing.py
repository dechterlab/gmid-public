import time
import pickle

class MessagePassingError(Exception):
    """ error related to MessagePassing class """


class MessagePassing(object):
    def __init__(self, verbose_level, message_graph, elim_order, weights, is_log, log_file_name):
        self.verbose = False if verbose_level <= 0 else True
        self.terse = False if verbose_level <= 1 else True
        self.mg = message_graph
        self.elim_order = elim_order
        self.weights = weights
        self.is_log = is_log
        self.log_file_name = log_file_name+'@' + str(int(time.time()))

    def store_message_graph(self):
        self.pickle_counter = 0
        pickel_file = self.log_file_name.split('@')[0] + '@' + str(self.pickle_counter) + '.pickle'
        pickle.dump(self.mg, open(pickel_file, 'wb') )
        self.pickle_counter += 1

    def load_message_graph(self, counter_from_the_last=0):
        load_counter = max(0, self.pickle_counter - counter_from_the_last)
        pickel_file = self.log_file_name.split('@')[0] + '@' + str(load_counter) + '.pickle'
        mg = pickle.load(open(pickel_file, 'rb'))
        return mg

    def print_log(self, string_message):
        print(string_message)
        self.file_out = open(self.log_file_name, 'a')
        with open(self.log_file_name, 'a') as file_out:
            file_out.write(string_message)
            file_out.write('\n')

    def schedule(self):
        raise NotImplemented

    def init_propagate(self):
        raise NotImplemented

    def propagate(self):
        raise NotImplemented

    def _propagate_one_pass(self):
        raise NotImplemented

    def bounds(self):
        raise NotImplemented
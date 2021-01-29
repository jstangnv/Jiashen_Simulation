#
# Define Functions Required for Managing & Executing Pulse Sequences
#

from nvsys import *

class pulse_seq():
    def __init__(self, inst_qutrit_sys, t_start = 0, montecarlo=False):

        self.arr_pulses = []
        self.tot_time = 0
        self.qutrit_sys = copy.deepcopy(inst_qutrit_sys)
        self.t_start = t_start
        self.monte = montecarlo


    def tail_pulse_edit(self, inst_pulse):
        #
        # If adding pulse to the end of current pulse sequence, this function
        # edits the start time and end time attributes of inst_pulse appropriately
        #

        if inst_pulse.starttime is None:
            if len(self.arr_pulses):
                # find largest end time for existing pulses and set to start time
                inst_pulse.starttime = max([elem.endtime for elem in self.arr_pulses])
                inst_pulse.endtime = inst_pulse.starttime + inst_pulse.duration
            else:
                inst_pulse.starttime = self.t_start
                inst_pulse.endtime = self.t_start + inst_pulse.duration
        else:
            inst_pulse.endtime = inst_pulse.starttime + inst_pulse.duration

    def add_pulse(self, inst_pulse):
        #
        # Add input pulse to pulse sequence
        #        
        self.arr_pulses.append(inst_pulse)
        
    def remove_pulse(self, idx_pulse):
        #
        # Removes pulse at given index
        #
        if idx_pulse >= len(self.arr_pulses):
            raise Exception('Invalid index of pulse sequence')
        else:
            del self.arr_pulses[idx_pulse]
            
    def edit_pulse(self, idx_pulse):
        #
        # Removes pulse at given index, and returns it for modification
        #
        if idx_pulse >= len(self.arr_pulses):
            raise Exception('Invalid index of pulse sequence')
        else:
            return self.arr_pulses.pop(idx_pulse)
        
    def print_pulses(self):
        for elem in self.arr_pulse:
            elem.print_info()

    def exec_pulse(self, init_state, t_list=[], c_ops=[], e_ops=[]): ## add opts=None and also edit in pulse_evolve
        if not len(t_list): 
            n_timesteps_test = int(self.endtime_seq()-self.starttime_seq()/self.qutrit_sys.max_timestep)
            if n_timesteps_test > 2:
                t_list = np.linspace(self.starttime_seq(), self.endtime_seq(), n_timesteps_test)
            else:
                t_list = np.linspace(self.starttime_seq(), self.endtime_seq(), 2)
        return self.qutrit_sys.pulse_evolve(init_state, t_list, c_ops=c_ops, e_ops=e_ops, arr_pulses=self.arr_pulses, montecarlo=self.monte)

    def exec_pulse_serial(self, init_state, pulsesupp=None, t_list=[], c_ops=[], e_ops=[]):
        if not len(t_list):
            t_list = np.linspace(self.starttime_seq(), self.endtime_seq(), 2)
        
        state0 = init_state
        for pulseobj in self.sort_arrpulses():
            if pulsesupp is not None:
                pulsesupp.duration = pulseobj.duration
                pulsesupp.endtime = pulsesupp.starttime + pulsesupp.duration
                pulseobj.Ham_args.update(pulsesupp.Ham_args)
                pulseobj.Ham_args.update({pulsesupp.pulsetype[:3] + '_t_offset1': pulseobj.starttime})
                #print(pulseobj.Ham_args, '\n')
                state0 = self.qutrit_sys.serial_evolve(state0, pulseobj, Hsupp = pulsesupp.Ham_entry, montecarlo=self.monte)
            else:
                state0 = self.qutrit_sys.serial_evolve(state0, pulseobj, Hsupp = None, montecarlo=self.monte)
            #print(state0)
        return state0
    
    # Return ending time of sequence
    def endtime_seq(self):
        return max([elem.endtime for elem in self.arr_pulses])

    # Return starting time of sequence
    def starttime_seq(self):
        return min([elem.starttime for elem in self.arr_pulses])
        
    # Sort pulses in ascending order of start time, return sorted arr_pulses
    def sort_arrpulses(self):
        ls_starttimes = [pulse_obj.starttime for pulse_obj in self.arr_pulses]
        idxsort = np.argsort(ls_starttimes)
        return [self.arr_pulses[idx] for idx in idxsort]


#
# Object to be fed to pulse_seq(), contains information about pulse
# Example info includes 
#  - pulse type e.g. microwave, AC field, shaped pulse, etc.
#  - timings: start time, duration, end time
#  - Hamiltonian matrix to be fed to mesolve, time dependent coefficient
#
class Pulse():
    def __init__(self, pulsetype, duration, starttime=None, time_offset = 0):
        self.pulsetype = pulsetype #check that we get a string?
        self.duration = duration
        #self.time_offset = time_offset
        self.starttime = starttime #+ self.time_offset
        
        if starttime is None:
            self.endtime = None
        else:
            self.endtime = self.starttime + self.duration
            
        self.Ham_entry = None
        self.Ham_coeff_str = None
        self.Ham_args = {}
        
        
    def pulse_parser(self, inst_qutrit_sys, *args):
        if self.starttime is None:
            raise Exception('Define starttime and endtime')
        if self.pulsetype[:3].upper() in inst_qutrit_sys.pulsefuncs:
            self.Ham_entry, Ham_args_update = inst_qutrit_sys.pulsefuncs[self.pulsetype[:3]](self, *args)
            self.Ham_args.update(Ham_args_update)
        else:
            raise Exception('Pulse type not available:', self.pulsetype)
            
    #
    # Pulse parser that accepts returns time-dependent coefficient as a string
    #            
    def pulse_parser_strargs(self, inst_qutrit_sys, *args):
        args += tuple([True]) # Set str_args=True so time-dep coefficient is returned as a string
        #print(args)
        if self.starttime is None:
            raise Exception('Define starttime and endtime')
        if self.pulsetype[:3].upper() in inst_qutrit_sys.pulsefuncs:
            self.Ham_entry, self.Ham_coeff_str = inst_qutrit_sys.pulsefuncs[self.pulsetype[:3]](self, *args)
            #print(self.Ham_coeff_str)
        else:
            raise Exception('Pulse type not available:', self.pulsetype)
            
    def print_info(self):
        print('Pulse Type:', self.pulsetype)
        print('Start Time:', self.starttime)# - self.time_offset)
        print('End Time:', self.endtime)# - self.time_offset)
        print('Duration:', self.duration)
        print('')


        
#
# For an input list of pulse_seq() objects, functions for executing 1 or all 
# of the pulse sequences in parallel (or series)
#
class parallel_pulseseq():
    def __init__(self, ls_pulseseqs, state_init, t_list=None, bool_ret_states=True):
        self.ls_pulsehandlers = ls_pulseseqs
        self.state0 = state_init
        self.bool_ret_states = bool_ret_states
        if t_list is not None:
            self.t_list = t_list
        else:
            self.t_list = None
    
    #
    # Loop through list of pulse_seq() objects, execute in parallel or in series
    #
    def exec_pulseseqs(self, bool_ret_states=True, bool_parallel=True):
        #print(pulse_handler_upd.exec_pulse_serial(state_init))
        if bool_parallel:
            return parallel_map(self.exec_1pulseseq, np.arange(len(self.ls_pulsehandlers))) #, task_args=(self.ls_pulsehandlers, self.state0, self.bool_ret_states))
        else:
            return self.ls_pulsehandlers[0].exec_pulse(self.state0, self.t_list).states

    #
    # Execute a single pulse sequence located at given index (idx) within the list of pulse_seq() objects
    #    
    def exec_1pulseseq(self, idx):
        if self.bool_ret_states:
            return self.ls_pulsehandlers[idx].exec_pulse(self.state0, t_list=self.t_list).states
        else:
            return self.ls_pulsehandlers[idx].exec_pulse(self.state0, t_list=self.t_list)
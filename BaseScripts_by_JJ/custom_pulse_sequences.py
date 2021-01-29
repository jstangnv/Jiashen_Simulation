#
# Design Specific Pulse Sequence Experiments
#
from pulsehandler import *

class Ramsey():
    def __init__(self):
        self.B0 = np.zeros(3)
        self.B_amp = None
        self.detuning = None
        self.frq_trans = None
        self.time_pi = None
        
        self.phase_90_1 = 0
        self.phase_90_2 = 0
        
        self.qutrit_sys = None
        self.state_init = None
        
        self.t_offset_global = 0
    
    def vary_freeprecession(self, times_fp, parallel = True, montecarlo = False, progressbar = True):
        #
        # Input: array of times to run free precession between pi/2 pulses
        #
        time_startsim = time.time()
        
        pulse_handler = pulse_seq(self.qutrit_sys, montecarlo=montecarlo)
            
        #
        # 1st pi/2 pulse
        #
        pulse_90_1 = Pulse('ACDrive', self.time_pi/2)
        pulse_handler.tail_pulse_edit(pulse_90_1)
        pulse_90_1.pulse_parser(self.qutrit_sys, self.B_amp, self.frq_trans, self.phase_90_1, self.detuning, 0)
        pulse_handler.add_pulse(pulse_90_1)

        #
        # Evolve state until the end of pi/2 pulse. Save state at end of pulse
        #
        output_prefp = pulse_handler.exec_pulse(self.state_init)
        state_prefp = output_prefp.states[-1]

        #
        # Allow state to undergo free precession (no AC field)
        # Save states at time intervals during free precession
        #
        qutrit_sys_fp = copy.deepcopy(self.qutrit_sys)
        evolver_post_fp = qutrit_sys_fp.evolve(state_prefp, times_fp)
        states_post_fp = evolver_post_fp.states
        
        #
        # Define useful parameters for 2nd pi/2 pulse 
        # (this makes parallelization faster if we don't have to redefine these during each loop)
        #
        self.ramsey_pt2_args = {}
        self.ramsey_pt2_args['pulse_handler'] = pulse_seq(self.qutrit_sys)
        self.ramsey_pt2_args['states_postfp'] = states_post_fp
        self.ramsey_pt2_args['t_prefp'] = pulse_handler.endtime_seq()
        #print("self.ramsey_pt2_args['t_prefp']", self.ramsey_pt2_args['t_prefp'])
        self.ramsey_pt2_args['times_fp'] = times_fp
        
        time_postfp = time.time()
        print('Time taken to find states after free precession:', round(time_postfp - time_startsim, 3), 's')
        if parallel:
            if progressbar:
                arr_ramsey = parallel_map(self.ramsey_pt2_evolve, np.arange(len(times_fp)), progress_bar=progressbar)
            else:
                arr_ramsey = parallel_map(self.ramsey_pt2_evolve, np.arange(len(times_fp)))
        else:
            arr_ramsey = []
            for idx_tfp in np.arange(len(times_fp)):
                arr_ramsey.append(self.ramsey_pt2_evolve(idx_tfp))
        
        del self.ramsey_pt2_args
        if not parallel:
            print('2nd pi/2 pulse applied. Time taken:', round(time.time() - time_postfp, 3))
        return arr_ramsey

    
    def ramsey_pt2_evolve(self, idx_tfp):
        #
        # Function used alongside vary_freeprecession, when sweeping free precession time 
        # between pi/2 pulses.
        #
        
        # Index correct state after free precession, set offset time for time-dependent coefficient
        state_post_fp = self.ramsey_pt2_args['states_postfp'][idx_tfp]
        t_freepre = self.ramsey_pt2_args['times_fp'][idx_tfp] #time allowed for free precession
        t_offset = t_freepre + self.ramsey_pt2_args['t_prefp'] #offset time from beginning of experiment
        #print('t_offset', t_offset)
        pulse_handler_loop = self.ramsey_pt2_args['pulse_handler'] # pulse sequence handler object

        #
        # Create pi/2 pulse
        #
        pulse_90 = Pulse('ACDrive', self.time_pi/2)
        pulse_handler_loop.tail_pulse_edit(pulse_90)
        #print('self.phase_90_2', self.phase_90_2)
        pulse_90.pulse_parser(self.qutrit_sys, self.B_amp, self.frq_trans, self.phase_90_2, self.detuning, t_offset)
        pulse_handler_loop.add_pulse(pulse_90)
        
        # Execute pulse, returns state after pulse
        output_ramsey = pulse_handler_loop.exec_pulse(state_post_fp)
        return output_ramsey.states[-1]

    
    def exec_single_seq(self, time_fp):
        #
        # Executes a single Ramsey sequence, returns final state after sequence is done.
        #
        #------
        
        # 1st pi/2 pulse
        pulse_90_1 = Pulse('ACDrive', self.time_pi/2)
        pulse_handler.tail_pulse_edit(pulse_90_1)
        pulse_90_1.pulse_parser(self.qutrit_sys, self.B_amp, self.frq_trans, self.phase_90_1, self.detuning, self.t_offset_global)
        pulse_handler.add_pulse(pulse_90_1)
        
        # Create free precession pulse
        pulse_fp = Pulse("NON", time_fp)
        pulse_handler.tail_pulse_edit(pulse_fp)
        pulse_fp.pulse_parser(self.qutrit_sys)
        pulse_handler.add_pulse(pulse_fp)

        # 2nd pi/2 pulse
        phase_90_2 = 0 #phase shift for 2nd pi/2 pulse
        pulse_90_2 = Pulse('ACDrive', self.time_pi/2)
        pulse_handler.tail_pulse_edit(pulse_90_2)
        pulse_90_2.pulse_parser(self.qutrit_sys, self.B_amp, self.frq_trans, self.phase_90_2, self.detuning, self.t_offset_global)
        pulse_handler.add_pulse(pulse_90_2)

        #t_start_seq = pulse_handler.starttime_seq()
        #t_end_seq = pulse_handler.endtime_seq()

        output_evolver = pulse_handler.exec_pulse(self.state_init)
        
        return output_evolver.states[-1]
    
class Ramsey_DQ():
    def __init__(self):
        self.B0 = np.zeros(3)
        
        self.B_amp1 = None
        self.B_amp2 = None
        self.detuning1 = None
        self.detuning2 = None
        self.frq_trans1 = None
        self.frq_trans2 = None
        self.time_DQhalfpi = None
        self.times_fp = None
        
        self.phase_90init_1 = 0
        self.phase_90init_2 = 0
        self.phase_90end_1 = 0
        self.phase_90end_2 = 0
        
        self.qutrit_sys = None
        self.state_init = None
        
        self.t_offset_global = 0
    
    def vary_freeprecession(self, times_fp, parallel = True, montecarlo = False, progressbar = True):
        #
        # Input: array of times to run free precession between pi/2 pulses
        # If n_allstates is defined as a positive number greater than 2, then break up time taken to do rotation pulses
        # into this number of intervals and return system state at these times
        #
        time_startsim = time.time()
        
        pulse_handler = pulse_seq(self.qutrit_sys, montecarlo=montecarlo)
        
        #
        # Initial pi/2 pulses
        #
        pulse_90_1 = Pulse('ACDrive', duration=self.time_DQhalfpi, starttime=0)
        pulse_90_1.pulse_parser(self.qutrit_sys, self.B_amp1, self.frq_trans1, self.phase_90init_1, self.detuning1, 0)
        pulse_handler.add_pulse(pulse_90_1)
        
        pulse_90_2 = Pulse('ACDrive', duration=self.time_DQhalfpi, starttime=0)
        pulse_90_2.pulse_parser(self.qutrit_sys, self.B_amp2, self.frq_trans2, self.phase_90init_2, self.detuning2, 0)
        pulse_handler.add_pulse(pulse_90_2)

                
        #
        # Evolve state until the end of pi/2 pulse. Save state(s)
        #
        output_prefp = pulse_handler.exec_pulse(self.state_init)
        state_prefp = output_prefp.states[-1]

        #
        # Allow state to undergo free precession (no AC field)
        # Save states at time intervals during free precession
        #
        qutrit_sys_fp = copy.deepcopy(self.qutrit_sys)
        evolver_post_fp = qutrit_sys_fp.evolve(state_prefp, times_fp)
        states_post_fp = evolver_post_fp.states

        
        #
        # Define useful parameters for 2nd pi/2 pulse 
        # (this makes parallelization faster if we don't have to redefine these during each loop)
        #
        self.ramsey_pt2_args = {}
        self.ramsey_pt2_args['pulse_handler'] = pulse_seq(self.qutrit_sys)
        self.ramsey_pt2_args['states_postfp'] = states_post_fp
        self.ramsey_pt2_args['t_prefp'] = pulse_handler.endtime_seq()
        self.ramsey_pt2_args['times_fp'] = times_fp
        
        time_postfp = time.time()
        print('Time taken to find states after free precession:', round(time_postfp - time_startsim, 3), 's')
        if parallel:
            if progressbar:
                arr_ramsey = parallel_map(self.ramsey_pt2_evolve, np.arange(len(times_fp)), progress_bar=progressbar)
            else:
                arr_ramsey = parallel_map(self.ramsey_pt2_evolve, np.arange(len(times_fp)))                
        else:
            arr_ramsey = []
            for idx_tfp in np.arange(len(times_fp)):
                arr_ramsey.append(self.ramsey_pt2_evolve(idx_tfp))
        
        del self.ramsey_pt2_args
        if not parallel:
            print('2nd pi/2 pulse applied. Time taken:', round(time.time() - time_postfp, 3))
        
        return arr_ramsey

    
    def ramsey_pt2_evolve(self, idx_tfp):
        #
        # Function used alongside vary_freeprecession, when sweeping free precession time 
        # between pi/2 pulses.
        #
        
        # Index correct state after free precession, set offset time for time-dependent coefficient
        state_post_fp = self.ramsey_pt2_args['states_postfp'][idx_tfp]
        t_freepre = self.ramsey_pt2_args['times_fp'][idx_tfp] #time allowed for free precession
        t_offset = t_freepre + self.ramsey_pt2_args['t_prefp'] #offset time from beginning of experiment
        #print('t_offset', t_offset)
        pulse_handler_loop = self.ramsey_pt2_args['pulse_handler'] # pulse sequence handler object
        
        #
        # End pi/2 pulses
        #
        pulse_90_1 = Pulse('ACDrive', duration=self.time_DQhalfpi, starttime=0)
        pulse_90_1.pulse_parser(self.qutrit_sys, self.B_amp1, self.frq_trans1, self.phase_90end_1, self.detuning1, t_offset)
        pulse_handler_loop.add_pulse(pulse_90_1)
        
        pulse_90_2 = Pulse('ACDrive', duration=self.time_DQhalfpi, starttime=0)
        pulse_90_2.pulse_parser(self.qutrit_sys, self.B_amp2, self.frq_trans2, self.phase_90end_2, self.detuning2, t_offset)
        pulse_handler_loop.add_pulse(pulse_90_2)
        
        
        # Execute pulse, returns state after pulse
        output_ramsey = pulse_handler_loop.exec_pulse(state_post_fp)
        return output_ramsey.states[-1]
    
    def single_DQramsey(self, times_fp, montecarlo = False, n_allstates=0):
                #
        # Input: array of times to run free precession between pi/2 pulses
        # If n_allstates is defined as a positive number greater than 2, then break up time taken to do rotation pulses
        # into this number of intervals and return system state at these times
        #
        time_startsim = time.time()
        
        pulse_handler = pulse_seq(self.qutrit_sys, montecarlo=montecarlo)
        
        #
        # Initial pi/2 pulses
        #
        pulse_90_1 = Pulse('ACDrive', self.time_DQhalfpi, starttime=0)
        pulse_90_1.pulse_parser(self.qutrit_sys, self.B_amp1, self.frq_trans1, self.phase_90init_1, self.detuning1, 0)
        pulse_handler.add_pulse(pulse_90_1)
        
        pulse_90_2 = Pulse('ACDrive', duration=self.time_DQhalfpi, starttime=0)
        pulse_90_2.pulse_parser(self.qutrit_sys, self.B_amp2, self.frq_trans2, self.phase_90init_2, self.detuning2, 0)
        pulse_handler.add_pulse(pulse_90_2)

        # for saving all states during Ramsey experiment
        if n_allstates > 1:
            arr_times_allstates_rot = np.linspace(0, pulse_handler.endtime_seq(), n_allstates)
        else:
            arr_times_allstates_rot = []
                
        #
        # Evolve state until the end of pi/2 pulse. Save state(s)
        #
        output_prefp = pulse_handler.exec_pulse(self.state_init, t_list = arr_times_allstates_rot)
        state_prefp = output_prefp.states[-1]

        #
        # Allow state to undergo free precession (no AC field)
        # Save states at time intervals during free precession
        #
        qutrit_sys_fp = copy.deepcopy(self.qutrit_sys)
        evolver_post_fp = qutrit_sys_fp.evolve(state_prefp, times_fp)
        states_post_fp = evolver_post_fp.states

        state_post_fp = states_post_fp[-1]
        t_freepre = pulse_handler.endtime_seq() #time allowed for free precession
        t_offset = t_freepre + times_fp[-1] #offset time from beginning of experiment
        pulse_handler_loop = pulse_seq(self.qutrit_sys) # pulse sequence handler object
        
        #
        # End pi/2 pulses
        #
        pulse_90_1 = Pulse('ACDrive', duration=self.time_DQhalfpi, starttime=0)
        pulse_90_1.pulse_parser(self.qutrit_sys, self.B_amp1, self.frq_trans1, self.phase_90end_1, self.detuning1, t_offset)
        pulse_handler_loop.add_pulse(pulse_90_1)
        
        pulse_90_2 = Pulse('ACDrive', duration=self.time_DQhalfpi, starttime=0)
        pulse_90_2.pulse_parser(self.qutrit_sys, self.B_amp2, self.frq_trans2, self.phase_90end_2, self.detuning2, t_offset)
        pulse_handler_loop.add_pulse(pulse_90_2)
        
        # Execute pulse, returns state after pulse        
        output_ramsey = pulse_handler_loop.exec_pulse(state_post_fp, t_list = arr_times_allstates_rot)
        
        if len(arr_times_allstates_rot):
            ls_allstates = np.concatenate((output_prefp.states, states_post_fp[1:], output_ramsey.states[1:]))
        else:
            ls_allstates = output_ramsey.states[-1]
        ls_alltimes = np.concatenate((arr_times_allstates_rot, times_fp[1:] + arr_times_allstates_rot[-1], arr_times_allstates_rot[1:] + t_offset))

            
        return ls_allstates, ls_alltimes

    
class Hahn():
    def __init__(self):
        self.B0 = np.zeros(3)
        self.B_amp = None
        self.detuning = None
        self.frq_trans = None
        self.time_pi = None
        
        self.phase_90_1 = 0
        self.phase_180 = 0
        self.phase_90_2 = 0
        
        self.qutrit_sys = None
        self.state_init = None
        
        self.t_offset_global = 0
    
    def vary_freeprecession(self, times_fp, parallel = True, montecarlo = False, progressbar = True):
        #
        # Input: array of times to run free precession between pi/2 pulses
        #
        time_startsim = time.time()
        
        pulse_handler = pulse_seq(self.qutrit_sys, montecarlo=montecarlo)
            
        #
        # 1st pi/2 pulse
        #
        pulse_90_1 = Pulse('ACDrive', self.time_pi/2)
        pulse_handler.tail_pulse_edit(pulse_90_1)
        pulse_90_1.pulse_parser(self.qutrit_sys, self.B_amp, self.frq_trans, self.phase_90_1, self.detuning, 0)
        pulse_handler.add_pulse(pulse_90_1)

        #
        # Evolve state until the end of pi/2 pulse. Save state at end of pulse
        #
        output_prefp = pulse_handler.exec_pulse(self.state_init)
        state_prefp = output_prefp.states[-1]

        #
        # Allow state to undergo free precession (no AC field)
        # Save states at time intervals during free precession
        #
        qutrit_sys_fp = copy.deepcopy(self.qutrit_sys)
        evolver_post_fp = qutrit_sys_fp.evolve(state_prefp, times_fp)
        states_post_fp = evolver_post_fp.states
        
        #
        # Define useful parameters for 2nd pi/2 pulse 
        # (this makes parallelization faster if we don't have to redefine these during each loop)
        #
        self.hahn_pt2_args = {}
        self.hahn_pt2_args['pulse_handler'] = pulse_seq(self.qutrit_sys)
        self.hahn_pt2_args['states_postfp'] = states_post_fp
        self.hahn_pt2_args['t_prefp'] = pulse_handler.endtime_seq()
        #print("self.ramsey_pt2_args['t_prefp']", self.ramsey_pt2_args['t_prefp'])
        self.hahn_pt2_args['times_fp'] = times_fp
        
        time_postfp = time.time()
        print('Time taken to find states after 1st free precession:', round(time_postfp - time_startsim, 3), 's')
        if parallel:
            if progressbar:
                arr_hahn = parallel_map(self.hahn_pt2_evolve, np.arange(len(times_fp)), progress_bar=progressbar)
            else:
                arr_hahn = parallel_map(self.hahn_pt2_evolve, np.arange(len(times_fp)))
        else:
            arr_hahn = []
            for idx_tfp in np.arange(len(times_fp)):
                arr_hahn.append(self.hahn_pt2_evolve(idx_tfp))
        
        del self.hahn_pt2_args
        #if not parallel:
        #    print('2nd pi/2 pulse applied. Time taken:', round(time.time() - time_postfp, 3))
        return arr_hahn

    
    def hahn_pt2_evolve(self, idx_tfp):
        #
        # Function used alongside vary_freeprecession, when sweeping free precession time 
        # between pi/2 pulses.
        #
        
        # Index correct state after free precession, set offset time for time-dependent coefficient
        state_post_fp = self.hahn_pt2_args['states_postfp'][idx_tfp]
        t_freepre = self.hahn_pt2_args['times_fp'][idx_tfp] #time allowed for free precession
        t_offset = t_freepre + self.hahn_pt2_args['t_prefp'] #offset time from beginning of experiment
        alltimes_fp = self.hahn_pt2_args['times_fp']
        times_fp2_loop = alltimes_fp[alltimes_fp <= t_freepre]
        
        
        
        #print('t_offset', t_offset)
        pulse_handler_loop = self.hahn_pt2_args['pulse_handler'] # pulse sequence handler object
        pulse_handler_loop2 = pulse_seq(self.qutrit_sys)
        
        #
        # Create pi pulse
        #
        pulse_180 = Pulse('ACDrive', self.time_pi)
        pulse_handler_loop.tail_pulse_edit(pulse_180)
        #print('self.phase_90_2', self.phase_90_2)
        pulse_180.pulse_parser(self.qutrit_sys, self.B_amp, self.frq_trans, self.phase_180, self.detuning, t_offset)
        pulse_handler_loop.add_pulse(pulse_180)
        state_postpi = pulse_handler_loop.exec_pulse(state_post_fp).states[-1]
        
        #
        # Allow state to undergo 2nd free precession (no AC field)
        # Save state at the end of free precession
        #
        qutrit_sys_fp2 = copy.deepcopy(self.qutrit_sys)
        evolver_post_fp2 = qutrit_sys_fp2.evolve(state_postpi, times_fp2_loop)
        state_post_fp2 = evolver_post_fp2.states[-1] 
        t_offset2 = t_offset + self.time_pi + times_fp2_loop[-1] 
        #print('yay 2nd free precession pulse')
        
        #
        # Create pi/2 pulse
        #
        pulse_90 = Pulse('ACDrive', self.time_pi/2)
        pulse_handler_loop2.tail_pulse_edit(pulse_90)
        #print('self.phase_90_2', self.phase_90_2)
        pulse_90.pulse_parser(self.qutrit_sys, self.B_amp, self.frq_trans, self.phase_90_2, self.detuning, t_offset2)
        pulse_handler_loop2.add_pulse(pulse_90)
        
        # Execute pulse, returns state after pulse
        output_hahn = pulse_handler_loop2.exec_pulse(state_post_fp2)
        return output_hahn.states[-1]
    
    def test_echo(self, times_fp, parallel = True, montecarlo = False, progressbar = True):
        #
        # Input: array of times to run free precession between pi/2 pulses
        #
        realtime_startsim = time.time()
        
        pulse_handler = pulse_seq(self.qutrit_sys, montecarlo=montecarlo)
            
        #
        # 1st pi/2 pulse
        #
        pulse_90_1 = Pulse('ACDrive', self.time_pi/2)
        pulse_handler.tail_pulse_edit(pulse_90_1)
        pulse_90_1.pulse_parser(self.qutrit_sys, self.B_amp, self.frq_trans, self.phase_90_1, self.detuning, 0)
        pulse_handler.add_pulse(pulse_90_1)

        #
        # Evolve state until the end of pi/2 pulse. Save state at end of pulse
        #
        output_prefp = pulse_handler.exec_pulse(self.state_init)
        state_prefp = output_prefp.states[-1]

        #
        # Allow state to undergo free precession (no AC field)
        # Save states at time intervals during free precession
        #
        qutrit_sys_fp = copy.deepcopy(self.qutrit_sys)
        evolver_post_fp = qutrit_sys_fp.evolve(state_prefp, times_fp)
        state_post_fp = evolver_post_fp.states[-1]
        t_postfp1 = pulse_handler.endtime_seq() + times_fp[-1]
        
        pulse_handler2 = pulse_seq(self.qutrit_sys, montecarlo=montecarlo)
        
        #
        # Create pi pulse
        #
        pulse_180 = Pulse('ACDrive', self.time_pi)
        pulse_handler2.tail_pulse_edit(pulse_180)
        #print('self.phase_90_2', self.phase_90_2)
        pulse_180.pulse_parser(self.qutrit_sys, self.B_amp, self.frq_trans, self.phase_180, self.detuning, t_postfp1)
        pulse_handler2.add_pulse(pulse_180)        
        state_postpi = pulse_handler2.exec_pulse(state_post_fp).states[-1]
        #print('yay pi pulse')
        
        #
        # Allow state to undergo free precession (no AC field)
        # Save states at time intervals during free precession
        #
        times_fp2 = np.linspace(0, 2*times_fp[-1], round(2*len(times_fp)))
        qutrit_sys_fp2 = copy.deepcopy(self.qutrit_sys)
        evolver_post_fp2 = qutrit_sys_fp2.evolve(state_postpi, times_fp2)
        states_post_fp2 = evolver_post_fp2.states
        t_offset2 = t_postfp1 + pulse_handler.endtime_seq() + times_fp2
        #print('yay 2nd free precession pulse')
       
        #
        # Define useful parameters for 2nd pi/2 pulse 
        # (this makes parallelization faster if we don't have to redefine these during each loop)
        #
        self.hahn_pt2_args = {}
        self.hahn_pt2_args['pulse_handler'] = pulse_seq(self.qutrit_sys)
        self.hahn_pt2_args['states_postfp'] = states_post_fp2
        self.hahn_pt2_args['t_prefp'] = pulse_handler2.endtime_seq() + t_postfp1
        self.hahn_pt2_args['times_fp2'] = times_fp2
        
        realtime_postfp = time.time()
        print('Time taken to find states after 1st free precession:', round(realtime_postfp - realtime_startsim, 3), 's')
        if parallel:
            if progressbar:
                arr_hahn = parallel_map(self.testecho_pt2, np.arange(len(times_fp2)), progress_bar=progressbar)
            else:
                arr_hahn = parallel_map(self.testecho_pt2, np.arange(len(times_fp2)))
        else:
            arr_hahn = []
            for idx_tfp in np.arange(len(times_fp)):
                arr_hahn.append(self.testecho_pt2(idx_tfp))
        
        del self.hahn_pt2_args
        #if not parallel:
        #    print('2nd pi/2 pulse applied. Time taken:', round(time.time() - time_postfp, 3))
        return arr_hahn

    
    def testecho_pt2(self, idx_tfp):
        
        # Index correct state after free precession, set offset time for time-dependent coefficient
        state_post_fp = self.hahn_pt2_args['states_postfp'][idx_tfp]
        t_freepre = self.hahn_pt2_args['times_fp2'][idx_tfp] #time allowed for free precession
        t_offset = t_freepre + self.hahn_pt2_args['t_prefp'] #offset time from beginning of experiment
        times_fp2 = self.hahn_pt2_args['times_fp2']
        pulse_handler_loop = self.hahn_pt2_args['pulse_handler'] # pulse sequence handler object
        
        #
        # Create pi/2 pulse
        #
        pulse_90 = Pulse('ACDrive', self.time_pi/2)
        pulse_handler_loop.tail_pulse_edit(pulse_90)
        pulse_90.pulse_parser(self.qutrit_sys, self.B_amp, self.frq_trans, self.phase_90_2, self.detuning, t_offset)
        pulse_handler_loop.add_pulse(pulse_90)
        
        # Execute pulse, returns state after pulse
        output_hahn = pulse_handler_loop.exec_pulse(state_post_fp)
        return output_hahn.states[-1]
    
    
    
    

        
#
# Import Python Packages
#
from qutip import *
import os
import glob
import numpy as np
import scipy as scipy
from scipy import optimize
import matplotlib
import matplotlib.pyplot as plt
import time
import pickle
import scipy.signal as scs
import copy
import sympy
import math

#
# Custom python objects and functions JJ created for simulations
#
from pulsehandler import *
from Ham_coeffs import *
from dqdifdetun import *
from fit_functions import *
from jupyterplotparams import *
from nvbloch import *
from pulsemods import *

#
# Define gyromagnetic ratios (MHz/Gauss)
#
gammaNV = 2.8024 #Gyromagnetic ratio NV (MHz/G)
gammaN14 = 307.7e-6 #Gyromagnetic ratio N14 (MHz/G)
gammaN15 = -431.6e-6 #Gyromagnetic ratio N15 (MHz/G)

#
# Hyperfine tensor, quadrupole constant, zero field splitting
# All units in MHz
#
A_N15 = np.array([[3.65,0,0],[0,3.65,0],[0,0,3.03]]) # Nitrogen-15 HF tensor
A_N14 = np.array([[-2.62,0,0],[0,-2.62,0],[0,0,-2.162]]) # Nitrogen-14 HF tensor
Q_N14 = -4.945 # Quadrupole Splitting
ZFS = 2.87e3 #MHz



#
# 2-Spin System Object Class
#
class qutrit_coupled():
    def __init__(self, s2tot=1/2):  # N14 is spin 1, N15 is spin 1/2, change s2tot as needed
        self.s1tot = 1 # NV spin s=1
        self.Nms1vals = round(self.s1tot*2 + 1) # Number of ms values 2s+1 
        self.s2tot = s2tot 
        self.Nms2vals = round(self.s2tot*2 + 1)

        # Spin operators for coupled system
        self.Sx = tensor(jmat(self.s1tot,'x'), identity(self.Nms2vals))
        self.Sy = tensor(jmat(self.s1tot,'y'), identity(self.Nms2vals))
        self.Sz = tensor(jmat(self.s1tot,'z'), identity(self.Nms2vals))

        self.Ix = tensor(identity(self.Nms1vals), jmat(self.s2tot,'x'))
        self.Iy = tensor(identity(self.Nms1vals), jmat(self.s2tot,'y'))
        self.Iz = tensor(identity(self.Nms1vals), jmat(self.s2tot,'z'))
        
        # Dictionary of functions used for different types of pulses
        self.pulsefuncs = {"ACD": self.ACDrive, "ACB": self.ACBfield, "NON": self.NON_pulse,
                           "AMP": self.AMPmodpulse, "FRQ": self.FRQmodpulse, "ARB": self.ARBmodpulse}
        self.H_args = {}
        
        # Define constants depending on exact spin system
        self.define_constants()
        
    def define_constants(self):
        self.gammaNV = gammaNV
        self.ZFS = ZFS
        if self.s2tot == 1:
            self.gammaNuc = gammaN14
            self.A = A_N14
            self.Q = Q_N14
        elif self.s2tot == 1/2:
            self.gammaNuc = gammaN15
            self.A = np.copy(A_N15)
            self.Q = 0 
        
        # Assuming that central spin is NV spin-1, define population operators for different ms states
        if self.s1tot == 1:
            self.pop_msplus = tensor(basis(self.Nms1vals, 0)*basis(self.Nms1vals, 0).dag(), identity(self.Nms2vals))
            self.pop_mszero = tensor(basis(self.Nms1vals, 1)*basis(self.Nms1vals, 1).dag(), identity(self.Nms2vals))
            self.pop_msminus = tensor(basis(self.Nms1vals, 2)*basis(self.Nms1vals, 2).dag(), identity(self.Nms2vals))

        
        # MAXIMUM timestep for qutip evolution (set to 0.01 => 10ns as default) 
        # In general, leave this as is unless working with energies/matrix elements MUCH greater than GHz (1E3) 
        self.max_timestep  = 0.01
        
        # Default QuTiP solver options, change nsteps (max number of timesteps allowed by solver)
        self.opts_default = Options(rhs_reuse = False, nsteps = 1000000, atol=1e-9, rtol=1e-9, tidy=False)     
        
    def setup_Hstat(self, B0=np.zeros(3), bool_HF=True, bool_Q=True):
        #
        # Static Hamiltonian terms
        #
        self.B0 = B0
        
        H_zfs = 2 * np.pi * self.ZFS * self.Sz * self.Sz
        H_zeeNV = 2 * np.pi * self.gammaNV * tensor(sum(self.B0dotjmats(self.s1tot)), identity(self.Nms2vals))
        H_zeeNuc = 2 * np.pi * self.gammaNuc * tensor(identity(self.Nms1vals), sum(self.B0dotjmats(self.s2tot)))
        H_hf = 2 * np.pi * (self.A[2,2]*self.Sz*self.Iz + self.A[0,0]*(self.Sx*self.Ix + self.Sy*self.Iy))
        H_quad = 2 * np.pi * self.Q * self.Iz*self.Iz #Note this becomes zero automatically if nuclear spin 1
        
        self.H_stat = H_zfs + H_zeeNV + H_zeeNuc
        self.H_NVonaxis = H_zfs + H_zeeNV

        if bool_HF:
            self.H_stat += H_hf
        if bool_Q:
            self.H_stat += H_quad
        self.H_ls = [self.H_stat]
        
    def B0dotjmats(self, jtot, B0=None):
        #
        # Dot product between magnetic field vector B=(Bx,By,Bz) and spin matrices S=(Sx,Sy,Sz)
        #
        if B0 is None:
            return [B0comp*jcomp for B0comp, jcomp in zip(self.B0, jmat(jtot))]
        else:
            return [B0comp*jcomp for B0comp, jcomp in zip(B0, jmat(jtot))]

        

         
    def evolve(self, init_state, t_list, c_ops=[], e_ops=[], args = {}, opts = None):
        #
        # Given initial state and Hamiltonian, time evolve state and compute
        #
        
        # Collapse operators
        if not c_ops:
            c_ops = []

        # Expectation operators
        if not e_ops:
            e_ops = []

        # Options for dynamic solver
        if not opts:
            opts = self.opts_default
            #
            # Reuse compiled RHS function. Useful for repetitive tasks.
            # nsteps set to a big number to avoid errors stemming small time steps
            # rtol (and atol) affects numerical accuracy. Decrease if errors become significant.
            #
            
        if not len(args):
            args = self.H_args
        
        #H_t = [self.H_stat, [self.H_Bac1, H_Bac1_coeff], [self.H_Bac2, H_Bac2_coeff]]
        #print('H_t')
        #print(H_t)
        if len(self.H_ls) == 1:
            H_ls_ev = self.H_ls[0]
        else:
            H_ls_ev = copy.deepcopy(self.H_ls)
        
        #print('Hamiltonian for mesolve:', H_ls_ev)
        
        output = mesolve(H_ls_ev, init_state, t_list, c_ops, e_ops, args = args, options = opts)
        #print(init_state)
        return output
    
    
    def serial_evolve(self, init_state, pulseobj, Hsupp = None, opts = None, montecarlo=False):
        #
        # Hsupp is either a Qobj matrix or a list containing a matrix and a time-dependent function 
        # e.g. Hsupp = [<matrix>, <function>]
        #
        
        args = pulseobj.Ham_args
        
        # Options for dynamic solver
        if not opts:
            opts = self.opts_default
            
        pulse_duration = pulseobj.duration
        H_ls_ev = [self.H_stat]
        
        if pulse_duration:
            if int(pulse_duration/self.max_timestep) > 2:
                t_list = np.linspace(0, pulse_duration, int(pulse_duration/self.max_timestep)) # maximum timestep = 0.1 us
            else:
                t_list = np.linspace(0, pulse_duration, 2)
            
            # Check that pulse object is not empty
            if pulseobj.Ham_entry is not None:
                H_ls_ev.append(pulseobj.Ham_entry)
            
            # Check if there is additional term(s) to append to Hamiltonian, e.g. a background AC magnetic field
            if Hsupp is not None:
                H_ls_ev.append(Hsupp)
            
            # check if there's only one term in list of Hamiltonians. 
            # QuTiP solver doesn't take a list which consists of only one Hamiltonain
            if len(H_ls_ev) == 1:
                H_ls_ev = H_ls_ev[0]
                        
            if montecarlo:
                return mcsolve(H_ls_ev, init_state, t_list, args = args, options = opts).states[-1]
            else:
                #print(H_ls_ev, init_state, t_list, args)
                return mesolve(H_ls_ev, init_state, t_list, args = args, options = opts).states[-1]
        else:
            return init_state


            # return pulse_handler_og.exec_pulse(state_init, t_list=t_list).states[-1]        
        
        
    def pulse_evolve(self, init_state, t_list, c_ops=[], e_ops=[], arr_pulses = [], opts = None, montecarlo=False):
        #
        # Given initial state and Hamiltonian, time evolve state and compute
        #
        
        # Collapse operators
        if not c_ops:
            c_ops = []

        # Expectation operators
        if not e_ops:
            e_ops = []

        # Options for dynamic solver
        if not opts:
            opts = self.opts_default
            
        H_ls_ev = copy.deepcopy(self.H_ls)
        dict_likepulses = {}
        
        arr_pulsetypes = list(set([inst_pulse.pulsetype[:3] for inst_pulse in arr_pulses]))
        for pulsetype in self.pulsefuncs:
            if pulsetype != "NON" and pulsetype in arr_pulsetypes:
                Ham_likeentries = [inst_pulse.Ham_entry for inst_pulse in arr_pulses if inst_pulse.pulsetype[:3] == pulsetype]
                Ham_likepulses = [inst_pulse for inst_pulse in arr_pulses if inst_pulse.pulsetype[:3] == pulsetype]
                dict_likepulses.update({pulsetype: Ham_likepulses}) 
                H_ls_ev.append(Ham_likeentries[0]) #to avoid repeats, no need to append multiple times
        
        if len(H_ls_ev) == 1:
            H_ls_ev = H_ls_ev[0]
            
        bool_monte = montecarlo
        if bool_monte:
            output = mcsolve(H_ls_ev, init_state, t_list, c_ops, e_ops, args = dict_likepulses, options = opts)
        else:
            output = mesolve(H_ls_ev, init_state, t_list, c_ops, e_ops, args = dict_likepulses, options = opts)
        #print(init_state)
        return output
    
    
    def setup_ACBfield(self, Bvec):
        #
        # Oscillating magnetic field defined by Bvec
        # Requires input inst_pulse, an instance of Pulse() object defined in pulsehandler.py
        #
        
        B0dot_s1mats = self.B0dotjmats(self.s1tot, Bvec)
        B0dot_s2mats = self.B0dotjmats(self.s2tot, Bvec)
        
        H_zeeNV = 2 * np.pi * self.gammaNV * tensor(sum(B0dot_s1mats), identity(self.Nms2vals))
        H_zeeNuc = 2 * np.pi * self.gammaNuc * tensor(identity(self.Nms1vals), sum(B0dot_s2mats))
        H_entry = H_zeeNV + H_zeeNuc
        
        self.H_ls.append([H_entry, coeff_ACBfield])    
    
    ###########################################
    #
    # Define Different Types of Pulses Below
    #    
    ###########################################
    
    def ACBfield(self, inst_pulse, Bvec, frq, phase=0, t_offset=0, serial=False, str_args=False):
        #
        # Oscillating magnetic field defined by Bvec
        # Requires input inst_pulse, an instance of Pulse() object defined in pulsehandler.py
        #
        
        B0dot_s1mats = self.B0dotjmats(self.s1tot, Bvec)
        B0dot_s2mats = self.B0dotjmats(self.s2tot, Bvec)
        
        H_zeeNV = 2 * np.pi * self.gammaNV * tensor(sum(B0dot_s1mats), identity(self.Nms2vals))
        H_zeeNuc = 2 * np.pi * self.gammaNuc * tensor(identity(self.Nms1vals), sum(B0dot_s2mats))
        if serial:
            H_entry = [H_zeeNV + H_zeeNuc, H_ACB_serialcoeff]
        else:
            H_entry = [H_zeeNV + H_zeeNuc, H_ACB_pulsecoeff]
        #print(H_entry[0])      

        H_args = {}
        H_args['ACB_freq1'] = frq
        H_args['ACB_phase1'] = phase
        H_args['ACB_t_offset1'] = t_offset
        
        if str_args:
            if np.abs(Bvec):
                return H_entry[0], "cos(2*pi*{}*(t + {}) + {})".format(frq, t_offset, phase)
            else:
                return None, None
        else:
            return H_entry, H_args
        
        
    def ACDrive(self, inst_pulse, Bamp=0, frq_trans=0, phase=0, detuning=0, t_offset=0, serial=False, str_args=False):
        #
        # Microwave Drive
        # Requires input inst_pulse, an instance of Pulse() object defined in pulsehandler.py
        #

        freq1 = frq_trans + detuning
        
        H_Bac1 = 2 * np.pi * self.gammaNV * self.Sx * np.sqrt(2)
        if serial:
            H_entry = [H_Bac1, H_ACD_serialcoeff]
        else:            
            H_entry = [H_Bac1, H_ACD_pulsecoeff]

        H_args = {}
        H_args['ACD_freq1'] = freq1
        H_args['ACD_phase1'] = phase
        H_args['ACD_Bamp'] = Bamp
        H_args['ACD_t_offset1'] = t_offset

        if str_args:
            if Bamp:
                return H_Bac1, "{}*cos(2*pi*{}*(t + {}) + {})".format(Bamp, freq1, t_offset, phase)
            else:
                return None, None
        else:
            return H_entry, H_args
        
        
    def AMPmodpulse(self, inst_pulse, mod_func=None, mod_args=None, frq_trans=0, phase=0, detuning=0, t_offset=0, serial=False):
        #
        # Amplitude-modulated Microwave Drive
        # Requires input inst_pulse, an instance of Pulse() object defined in pulsehandler.py
        # mod_func is a time-dependent function with other arguments given by input dictionary mod_args
        #

        freq1 = frq_trans + detuning
        
        H_Bac1 = 2 * np.pi * self.gammaNV * self.Sx * np.sqrt(2)
        if serial:
            H_entry = [H_Bac1, H_AMP_serialcoeff] ##not done
        else:            
            H_entry = [H_Bac1, H_AMP_pulsecoeff]

        H_args = {}
        H_args['AMP_freq1'] = freq1
        H_args['AMP_phase1'] = phase
        H_args['AMP_modfunc'] = mod_func
        H_args['AMP_modargs'] = mod_args
        H_args['AMP_t_offset1'] = t_offset
        
        return H_entry, H_args
        #print('self.H_Bac1')
        #print(self.H_Bac1)
        

    def FRQmodpulse(self, inst_pulse, modexpr=None, Bamp=0, frq_trans=0, phase=0, detuning=0, t_offset=0, serial=False):       
        #
        # Frequency-modulated microwave drive with constant amplitude
        # Requires input inst_pulse, an instance of Pulse() object defined in pulsehandler.py
        # modexpr is a time-dependent Sympy expression using time symbol given by sympy.symbols('t')
        #
        
        freq1 = frq_trans + detuning
        
        H_Bac1 = 2 * np.pi * self.gammaNV * self.Sx * np.sqrt(2)
        if serial:
            H_entry = [H_Bac1, H_FRQ_serialcoeff] ##not done
        else:            
            H_entry = [H_Bac1, H_FRQ_pulsecoeff]
        
        H_args = {}
        H_args['FRQ_freq1'] = frq_trans
        H_args['FRQ_Bamp'] = Bamp
        H_args['FRQ_phasemodexpr'] = modexpr
        H_args['FRQ_t_offset1'] = t_offset
        H_args['FRQ_phase0'] = phase0
        
        print(H_args)
        return H_entry, H_args
        #print('self.H_Bac1')
        #print(self.H_Bac1)        
        

    def ARBmodpulse(self, inst_pulse, AMPmodexpr, PHASEmodexpr, frq_trans, phase0=0, detuning=0, t_offset=0, serial=False):
        #
        # Microwave Drive with arbitrary frequency and/or amplitude modulation
        # Requires input inst_pulse, an instance of Pulse() object defined in pulsehandler.py
        #

        freq1 = frq_trans + detuning
        
        H_Bac1 = 2 * np.pi * self.gammaNV * self.Sx * np.sqrt(2)
        if serial:
            H_entry = [H_Bac1, H_ARB_serialcoeff] ##not done
        else:            
            H_entry = [H_Bac1, H_ARB_pulsecoeff]

        H_args = {}
        H_args['ARB_freq1'] = frq_trans
        H_args['ARB_t_offset1'] = t_offset
        H_args['ARB_phase0'] = phase0

        H_args['ARB_phasemodfunc'] = lambda x: sympy.lambdify(sympy.symbols('t'), PHASEmodexpr, 
                                                              modules=["math", "mpmath", "scipy", "numpy"])(x)
        H_args['ARB_ampmodfunc'] = lambda x: sympy.lambdify(sympy.symbols('t'), AMPmodexpr, 
                                                            modules=["math", "mpmath", "scipy", "numpy"])(x)
        
        
        return H_entry, H_args
        #print('self.H_Bac1')
        #print(self.H_Bac1)
        
    
    def NON_pulse(self, inst_pulse, str_args=False):
        #
        # Empty pulse
        # Requires input inst_pulse, an instance of Pulse() object defined in pulsehandler.py
        #
        if str_args:
            return None, None
        else:
            return None, {}
    
    def ret_transition_frq(self, idx_mi0, idx_mif, idx_ms0=1, idx_msf=0, Ham=None):
        #
        # Return frequency difference (MHz) between two eigenstates 
        # of coupled NV- system with coupled N nuclear spin
        #
        
        if not Ham:
            Ham = self.H_stat
        eigenvals, eigenstates = Ham.eigenstates()
        state0 = tensor(basis(self.Nms1vals, idx_ms0), basis(self.Nms2vals, idx_mi0))
        statef = tensor(basis(self.Nms1vals, idx_msf), basis(self.Nms2vals, idx_mif))
        
        idx_e0 = np.argmax([np.sum(np.abs(eigenstates[i].full())*state0.full()) for i in np.arange(len(eigenvals))])
        idx_ef = np.argmax([np.sum(np.abs(eigenstates[i].full())*statef.full()) for i in np.arange(len(eigenvals))])

        frq_trans = (eigenvals[idx_ef] - eigenvals[idx_e0])/2/np.pi 
        frq_trans = np.abs(frq_trans)
        return frq_trans #MHz

    def diagHam(self, Ham):
        eigenvals, eigenstates = Ham.eigenstates()
        arr_eigenvals = []
        for msbasis in np.arange(3):
            for mibasis in np.arange(2):
                idx_eigenval = np.argmax([np.sum(np.abs(eigenstates[i].full())*tensor(basis(3,msbasis), basis(
                    qutrit_sys.Nms2vals, mibasis)).full()) for i in np.arange(len(eigenvals))])
                #print(idx_eigenval)
                arr_eigenvals.append(idx_eigenval)
        eigenvals_basissort = [eigenvals[idx_evs] for idx_evs in arr_eigenvals]
        return np.diag(eigenvals_basissort)
    #return scipy.sparse.diags(eigenvals_basissort, dtype=numpy.complex128, format='csr')

    
    ###########################################
    #
    # Outdated functions below, to be removed in the future
    #    
    ###########################################    
    
    def drive_cw(self, Bamps, phases, detunings): # Outdated. Delete if not using but note that earlier simulations may depend on this
        #drive transitions (continuous wave)
        #Task for later: make it easy to do any number of drives, not just 2. Change allowed input to include arrays.

        # unpack inputs
        self.B1 = Bamps[0]
        self.B2 = Bamps[1]
        self.phase1 = phases[0]
        self.phase2 = phases[1]

        # define resonance frequencs (MHz) of 0 -> +1 and 0 -> -1 transitions
        #self.freq1 = ZFS + gammaNV * self.B0[2] + detunings[0] # 0 -> +1
        #self.freq2 = ZFS - gammaNV * self.B0[2] + detunings[1] # 0 -> -1
        #print(freq1/(2 * np.pi), freq2/(2 * np.pi ))
        e_trans = self.calc_transitions()
        self.freq1 = e_trans[2] + detunings[0]
        self.freq2 = e_trans[0] + detunings[1]
        
        self.H_Bac1 = 2 * np.pi * self.gammaNV * self.B1 * self.Sx * np.sqrt(2) 
        self.H_Bac2 = 2 * np.pi * gammaNV * self.B2 * self.Sx * np.sqrt(2)
        
        self.H_ls.append([self.H_Bac1, H_Bac1_coeff])
        self.H_ls.append([self.H_Bac2, H_Bac2_coeff])
        
        self.H_args = {'freq1':self.freq1, 'phase1':self.phase1, 'freq2':self.freq2, 'phase2':self.phase2}

        #print('self.H_Bac1')
        #print(self.H_Bac1)    

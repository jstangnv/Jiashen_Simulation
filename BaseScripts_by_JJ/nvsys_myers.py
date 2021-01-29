# Define Constants, Class for Coupled NV system (including Hyperfine Interaction with N15)
from qutip import *
import glob
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import matplotlib
import time
import pickle
import numpy.fft
import os
import scipy.signal as scs
import copy
#import progressbar
from importlib import reload
from scipy import optimize
from pulsehandler import *
from custom_pulse_sequences import *
from dqdifdetun import *
from fit_functions import *
from mpl_toolkits.mplot3d import Axes3D
import cmath
from nvbloch import *

# Define gyromagnetic ratios
gammaNV = 2.8024                   # Gyromagnetic ratio NV (MHz/G)
gammaN14 = 307.7e-6                # Gyromagnetic ratio N14 (MHz/G)
gammaN15 = -431.6e-6               # Gyromagnetic ratio N15 (MHz/G)
magnetonNuc = 762.2593285e-6       # Nuclear magneton (MHz/Gauss)
magnetonBohr = 1.3996              # Bohr magneton (MHz/Gauss)

#hyperfine tensor
A_N15 = np.array([[3.65,0,0],[0,3.65,0],[0,0,3.03]]) # Nitrogen-15 HF tensor
A_N14 = np.array([[-2.62,0,0],[0,-2.62,0],[0,0,-2.162]]) # Nitrogen-14 HF tensor

Q_N14 = -4.945 # Quadrupole Splitting
ZFS = 2.87e3 #MHz

def parallelize_var(var, exp_shape = [1], sweep_dim = 1):
    #
    # Given np array or Qobj, 
    # use np.full() to create arrays of correct size for running parfor
    #
    #
    # INPUTS:
    # var: input array (np.array or Qobj)
    # exp_shape: expected shape of array or Qobj if not using parfor
    # sweep_dim: sweep dimension i.e. number of copies of var required
    #
    
    if isinstance(var, Qobj):
        varsize = np.prod(var)
    else:
        var = np.asarray(var)
        varsize = var.size

    # Check that var is not already the right shape
    if len(var.shape) > 1:
        if varsize != int(sweep_dim*np.prod(exp_shape)):
            return np.full((sweep_dim,) + tuple(np.asarray(var.shape)[1:]), var)
        else:
            return var
    else:
        if var.shape != tuple(exp_shape):
            raise Exception
        else:
            return np.full((sweep_dim, var.size), var)

#################################
# 
# Legacy functions. Delete if not used for a while - 20190921 
#
#def H_Bac1_coeff(t, args):
#    freq1 = args['freq1']
#    phase1 = args['phase1']
#    return np.cos( 2*np.pi*freq1*t + phase1)

#def H_Bac2_coeff(t, args):
#    freq2 = args['freq2']
#    phase2 = args['phase2']
#    return np.cos( 2*np.pi*freq2*t + phase2)

#We shouldn't need two separate functions to do this^

#def H_Bac1_pulsecoeff(t, args):
#    freq1 = args['ACD_freq1']
#    phase1 = args['ACD_phase1']
#    starttime1 = args['starttime1']
#    endtime1 = args['endtime1']
#    t_offset1 = args['ACD_t_offset1']
#    #print('t_offset1:', t_offset1)
#    #print('endtime1', endtime1)
#    if t >= starttime1 and t <= endtime1:
#        return np.cos( 2*np.pi*freq1*(t + t_offset1) + phase1)
#    else:
#        return 0
#################################

def coeff_ACBfield(t, args):
    freq1 = args['ACB_freq1']
    phase1 = args['ACB_phase1']
    t_offset1 = args['ACB_t_offset1']
    return np.cos( 2*np.pi*freq1*(t + t_offset1) + phase1)

#
# Oscillating Magnetic Field
#
def H_ACB_pulsecoeff(t, args):
    coeff_timedep = 0
    for inst_pulse in args:
        if hasattr(inst_pulse, 'pulsetype'):
            if inst_pulse.pulsetype[:3] == "ACB":
                if t >= inst_pulse.starttime and t <= inst_pulse.endtime:
                    freq1 = inst_pulse.Ham_args['ACB_freq1']
                    phase1 = inst_pulse.Ham_args['ACB_phase1']
                    t_offset1 = inst_pulse.Ham_args['ACB_t_offset1']
                    coeff_timedep += np.cos( 2*np.pi*freq1*(t + t_offset1) + phase1)
    return coeff_timedep

#
# Microwave Field
#
def H_ACD_pulsecoeff(t, args):
    coeff_timedep = 0
    for inst_pulse in args:
        if hasattr(inst_pulse, 'pulsetype'):
            if inst_pulse.pulsetype[:3] == "ACD":
                if t >= inst_pulse.starttime and t <= inst_pulse.endtime:
                    freq1 = inst_pulse.Ham_args['ACD_freq1']
                    phase1 = inst_pulse.Ham_args['ACD_phase1']
                    t_offset1 = inst_pulse.Ham_args['ACD_t_offset1']
                    B_amp1 = inst_pulse.Ham_args['ACD_Bamp']
                    coeff_timedep += B_amp1*np.cos( 2*np.pi*freq1*(t + t_offset1) + phase1)
    return coeff_timedep

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
        
        # Dictionary of functions used for different pulses
        self.pulsefuncs = {"ACD": self.ACDrive, "ACB": self.ACBfield, "NON": self.NON_pulse}
        self.H_args = {}
        
        # Define constants depending on exact spin system
        self.define_constants()
        
    def define_constants(self):
        self.gammaNV = gammaNV
        self.ZFS = ZFS
        self.magNuc = magnetonNuc
        self.magBohr = magnetonBohr 

        if self.s2tot == 1:
            self.gammaNuc = gammaN14
            self.A = A_N14
            self.Q = Q_N14
        elif self.s2tot == 1/2:
            self.gammaNuc = gammaN15
            self.A = np.copy(A_N15)
            self.Q = 0 

            
        
    def setup_Hstat(self, B0=np.zeros(3)):
        #
        # Static Hamiltonian terms
        #
        self.B0 = B0
                
        H1 = self.A[2,2]*self.Iz*self.Sz - 3*self.gammaNV/self.ZFS*self.A[0,0]*B0[0]*self.Ix*self.Sz
        H2 = ZFS*self.Sz*self.Sz + self.gammaNV * B0[2] * self.Sz
        H3 = - self.magNuc* (B0[2] * self.Iz + B0[0]*(1 - 2*self.gammaNV/self.magNuc/self.ZFS*self.A[0,0]) * self.Ix)
        
        self.H_stat = H1 + H2 + H3
        self.H_stat *= 2*np.pi
        
        self.H_NVonaxis = H2*2*np.pi
        
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
            opts = Options(rhs_reuse = False, nsteps = 1000000, atol=1e-9, rtol=1e-9, tidy=False) 
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
    
    def pulse_evolve(self, init_state, t_list, c_ops=[], e_ops=[], arr_pulses = {}, opts = None, montecarlo=False):
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
            opts = Options(rhs_reuse = False, nsteps = 1000000, atol=1e-9, rtol=1e-9, tidy=False, method='bdf') 

        H_ls_ev = copy.deepcopy(self.H_ls)
        
        arr_pulsetypes = [inst_pulse.pulsetype[:3] for inst_pulse in arr_pulses]
        for pulsetype in self.pulsefuncs:
            if pulsetype != "NON" and pulsetype in arr_pulsetypes:
                Ham_entries = [inst_pulse.Ham_entry for inst_pulse in arr_pulses if inst_pulse.pulsetype[:3] == pulsetype]
                H_ls_ev.append(Ham_entries[0]) #to avoid repeats, no need to append multiple times
        
        if len(H_ls_ev) == 1:
            H_ls_ev = H_ls_ev[0]
            
        bool_monte = montecarlo
        if bool_monte:
            output = mcsolve(H_ls_ev, init_state, t_list, c_ops, e_ops, args = arr_pulses, options = opts)
        else:
            output = mesolve(H_ls_ev, init_state, t_list, c_ops, e_ops, args = arr_pulses, options = opts)
        #print(init_state)
        return output
    
    
    def par_drivecw_evolve(self, Bamps, phases, detunings, init_state, t_list, c_ops=[], e_ops=[], opts = None):
        self.drive_cw(Bamps, phases, detunings)
        return self.evolve(init_state, t_list, c_ops=[], e_ops=[tensor(identity(self.Nms1vals), identity(self.Nms2vals)) - self.Sz*self.Sz], opts = None)
    
    
    
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
    
    
    #
    # Define Different Types of Pulses
    #    
    
    def ACBfield(self, inst_pulse, Bvec, frq, phase=0, t_offset=0):
        #
        # Oscillating magnetic field defined by Bvec
        # Requires input inst_pulse, an instance of Pulse() object defined in pulsehandler.py
        #
        
        B0dot_s1mats = self.B0dotjmats(self.s1tot, Bvec)
        B0dot_s2mats = self.B0dotjmats(self.s2tot, Bvec)
        
        H_zeeNV = 2 * np.pi * self.gammaNV * tensor(sum(B0dot_s1mats), identity(self.Nms2vals))
        H_zeeNuc = 2 * np.pi * self.gammaNuc * tensor(identity(self.Nms1vals), sum(B0dot_s2mats))
        H_entry = [H_zeeNV + H_zeeNuc, H_ACB_pulsecoeff]
        #print(H_entry[0])      
            
        H_args = {}
        H_args['ACB_freq1'] = frq
        H_args['ACB_phase1'] = phase
        H_args['ACB_t_offset1'] = t_offset
        
        return H_entry, H_args
        
    
    def ACDrive(self, inst_pulse, Bamp, frq_trans, phase=0, detuning=0, t_offset=0):
        #
        # Microwave Drive
        # Requires input inst_pulse, an instance of Pulse() object defined in pulsehandler.py
        #

        freq1 = frq_trans + detuning
        
        H_Bac1 = 2 * np.pi * self.gammaNV * self.Sx * np.sqrt(2)
        H_entry = [H_Bac1, H_ACD_pulsecoeff]

        H_args = {}
        H_args['ACD_freq1'] = freq1
        H_args['ACD_phase1'] = phase
        H_args['ACD_Bamp'] = Bamp
        H_args['ACD_t_offset1'] = t_offset
        
        return H_entry, H_args
        #print('self.H_Bac1')
        #print(self.H_Bac1)
    
    def NON_pulse(self, inst_pulse):
        return [], {}
    
    
    def drive_cw(self, Bamps, phases, detunings): # Outdated. Delete if not using but note that earlier simulations may need this.
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
    
    #def RAMsey_seq(self, inst_pulse, t_freepre, axis_rot = ["x","-x"]):
    #    #
    #    # Executes Ramsey sequence with given parameters:
    ##    #   inst_pulse - Pulse() object
    #    #   t_freepre - free precession time  array/list of numbers
    #   #   axis_rot - 2-element list of strings containing "x", "-x", "y" or "-y". 
    #   #              e.g. ["X", "Y"] phase shifts 1st pi/2 pulse phase from 2nd pi pulse by 90 degrees
    #    #
    #
    #    ls_axis_rot = [str(elem).replace(" ", "").lower()[0] for elem in axis_rot]
        
    #def calc_transitions(self):
    #    energies = self.H_stat.eigenenergies()
    #    e_trans = np.array(energies)
    ##    if self.s2tot == 1/2:
    #        e_trans[2:4] -= energies[0:2]
    #        e_trans[4:6] -= energies[0:2]
    #        e_trans = e_trans[2:]
    #    elif self.s2tot == 1:
    #        # Index for N14 (spin 1):
    #        # 0: ms=0, mI=-1 <=> ms=-1, mI=-1
    #        # 1: ms=0, mI=+1 <=> ms=-1, mI=+1
    #        # 2: ms=0, mI=0  <=> ms=-1, mI=0
    #        # 3: ms=0, mI=-1 <=> ms=+1, mI=-1
    #        # 4: ms=0, mI=-1 <=> ms=+1, mI=-1
    #        # 5: ms=0, mI= 0 <=> ms=+1, mI=0
    #        e_trans[3:6] -= energies[0:3]
    #       e_trans[6:9] -= energies[0:3]
    #        e_trans = e_trans[3:]
    #
    #    return np.array(e_trans)/2/np.pi
    

        

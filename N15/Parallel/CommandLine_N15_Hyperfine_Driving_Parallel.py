
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import sys
commandline = sys.argv
import os
dirpath_pers = "/home/walsworth1/Jiashen_Simulation/NV_HyperfineDriving/N15/Parallel/" #Change to personal directory
if dirpath_pers not in sys.path:
    sys.path.insert(0, dirpath_pers)


import addbase #Change to correct addbase_* file in your directory
from nvsys import * # This is the main workhorse of the NV simulations codebase

state0 = tensor(basis(3,1), (basis(2, 0)+basis(2, 1))/(2**0.5)) # ms=0 superposition of mi=+1/2,-1/2

Bx = 0; By = 0; Bz = float(commandline[1]) # B field components (G)
B0_init = np.array([Bx, By, Bz]) # B field vector (G)
theta = 0 # angle in degrees for rotation around y axis
theta *= np.pi/180
Ry = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]]) 
B0 = np.dot(Ry, B0_init)

#print(B0)


# ### Define NV system

# The nvsys.py, pulsehandler.py files in the "Base Scripts" folder are the most commonly used scripts in our simulations. Some parts are not so well commented (yet) so please feel free to ask JJ if anything is unclear.


qutrit_sys = qutrit_coupled(s2tot=1/2) #N15
qutrit_sys.setup_Hstat(B0)
#print([elem for elem in qutrit_sys.pulsefuncs]) #Types of pulses available

#
# Determine frequency of excitation
#
frq_trans = qutrit_sys.ret_transition_frq(0,0,1,2)*0.5+qutrit_sys.ret_transition_frq(1,1,1,2)*0.5
#print('Carreier Frq', frq_trans)

frq_carrier_central = frq_trans
frq_carrier_detuningsigma = float(commandline[2]) #MHz
frq_carrier_detuningpts = int(commandline[3])
frq_carrier_ls = np.linspace(frq_carrier_central-frq_carrier_detuningsigma, frq_carrier_central+frq_carrier_detuningsigma,frq_carrier_detuningpts)
#print('frq_carrier_ls', frq_carrier_ls)

frq_sidebanddetuning_central = float(commandline[4]) # MHz
frq_sidebanddetuning_detuningsigma = float(commandline[5]) #MHz
frq_sidebanddetuning_detuningpts = int(commandline[6])
frq_sidebanddetuning_ls = np.linspace(frq_sidebanddetuning_central-frq_sidebanddetuning_detuningsigma, frq_sidebanddetuning_central+frq_sidebanddetuning_detuningsigma,frq_sidebanddetuning_detuningpts)

#print('frq_sidebanddetuning_ls',frq_sidebanddetuning_ls)

Bcarrier = float(commandline[7]) #MHz
Bsideband = float(commandline[8]) #MHz
B_amp_carrier = Bcarrier/qutrit_sys.gammaNV
B_amp_sideband = Bsideband/qutrit_sys.gammaNV

t_pulse = 1/Bsideband/2 #pi pulse on resonant transition
#print('pi pulse in us',t_pulse)


ls_pulsehandlers_HFDrive = []
bool_montecarlo=False #Set to False if executing multiple pulse sequences in parallel

print('Building pulse handler list')
for sidebanddetuning in frq_sidebanddetuning_ls:
    for frq_carrier in frq_carrier_ls:
        frq_sideband1 = frq_carrier+sidebanddetuning
        frq_sideband2 = frq_carrier-sidebanddetuning
        
        pulse_handler_HFDrive = pulse_seq(qutrit_sys, montecarlo = bool_montecarlo)
        
        # carrier
        pulse_carrier = Pulse('ACDrive', duration=t_pulse, starttime=0)
        pulse_carrier.pulse_parser(qutrit_sys, B_amp_carrier, frq_carrier,0,0,0,False)
        pulse_handler_HFDrive.add_pulse(pulse_carrier)
        
        # 1st sideband
        pulse_sideband1 = Pulse('ACDrive', duration = t_pulse, starttime = 0)
        pulse_sideband1.pulse_parser(qutrit_sys, B_amp_sideband, frq_sideband1,0,0,0,False)
        pulse_handler_HFDrive.add_pulse(pulse_sideband1)
        
        # 2nd sideband
        pulse_sideband2 = Pulse('ACDrive', duration = t_pulse, starttime = 0)
        pulse_sideband2.pulse_parser(qutrit_sys, B_amp_sideband, frq_sideband2,0,0,0,False)
        pulse_handler_HFDrive.add_pulse(pulse_sideband2)      
        
        ls_pulsehandlers_HFDrive.append(pulse_handler_HFDrive)
print('Completed pulse handler list')        


ntimes = 5
t_list = np.linspace(0, t_pulse, ntimes)
#print('t_list for saving state',t_list)


# ### Execute pulse sequences in parallel
import math
partitionflag = int(commandline[9])
partitionnumber = int(commandline[10])
if partitionflag == 1:
    if len(ls_pulsehandlers_HFDrive) <= partitionnumber:
        raise Exception("Too much partition, reduce the number")
    else:
        print('QuTip Solving....')
        sim_starttime = time.time()
        portionsize = math.floor(len(ls_pulsehandlers_HFDrive)/partitionnumber)
        states_HFDrive = []
        final_states_HFDrive = []
        for index in np.arange(0,partitionnumber):
            print("Run partition:",index,'/',partitionnumber)
            parallel_handlers = parallel_pulseseq(ls_pulsehandlers_HFDrive[index*portionsize : (index+1)*portionsize], state0, t_list=t_list)
            temp = parallel_handlers.exec_pulseseqs(bool_parallel=True)
            states_HFDrive = states_HFDrive + temp
            final_states_HFDrive = final_states_HFDrive + [each[-1] for each in temp]
        if math.floor(len(ls_pulsehandlers_HFDrive)/partitionnumber) != math.ceil(len(ls_pulsehandlers_HFDrive)/partitionnumber):
            # some objects in ls_pulsehandlers_HFDrive are not simulated
            print("Finishing up leftover portions...")
            parallel_handlers = parallel_pulseseq(ls_pulsehandlers_HFDrive[partitionnumber*portionsize : len(ls_pulsehandlers_HFDrive)], state0, t_list=t_list)
            temp = parallel_handlers.exec_pulseseqs(bool_parallel=True)
            states_HFDrive = states_HFDrive + temp
            final_states_HFDrive = final_states_HFDrive + [each[-1] for each in temp]
        sim_endtime = time.time()
        print('Time taken (s)', round(sim_endtime- sim_starttime, 3))
else:
    print('QuTip Solving....')
    sim_starttime = time.time()
    parallel_handlers = parallel_pulseseq(ls_pulsehandlers_HFDrive, state0, t_list=t_list)
    states_HFDrive = parallel_handlers.exec_pulseseqs(bool_parallel=True)
    final_states_HFDrive = [each[-1] for each in states_HFDrive]
    sim_endtime = time.time()
    print('Time taken (s)', round(sim_endtime- sim_starttime, 3))
#print('Find Expectation Value...')
final_states_expect_HFDrive = np.array(expect(qutrit_sys.pop_mszero, final_states_HFDrive))
# ### Save data and relevant  parameters
#print('Expectation value completed....')

#print('plotting...')
levels = MaxNLocator(nbins=100).tick_values(final_states_expect_HFDrive.min(), final_states_expect_HFDrive.max())
cmap = plt.get_cmap('magma')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
plt.rcParams['figure.figsize'] = [20,6]
plt.contourf(frq_carrier_ls, frq_sidebanddetuning_ls, final_states_expect_HFDrive.reshape(len(frq_sidebanddetuning_ls),len(frq_carrier_ls)),levels=levels,cmap=cmap)
plt.colorbar()
plt.title(r'carrier $\Omega$ '+str(Bcarrier)+'MHz side $\Omega$ '+str(Bsideband)+'MHz',fontsize=48)
plt.xlabel('Carrier Frequency MHz', fontsize = 30)
plt.ylabel('Detuning MHz',fontsize =30)
plt.xlim(np.min(frq_carrier_ls),np.max(frq_carrier_ls))
plt.ylim(np.min(frq_sidebanddetuning_ls),np.max(frq_sidebanddetuning_ls))
pathpic = dirpath_pers+'/pic'
pathqu = dirpath_pers+'/qu'
fname_savepic = "Bz_"+str(Bz)+"_csigma_"+str(frq_carrier_detuningsigma)+"_cpts_"+str(frq_carrier_detuningpts)+"_sbdecentral_"+str(frq_sidebanddetuning_central)+"_sbdesigma_"+str(frq_sidebanddetuning_detuningsigma)+"_sbdepts_"+str(frq_sidebanddetuning_detuningpts)+"_cRabi_"+str(Bcarrier)+'_sbRabi_'+str(Bsideband)+'_simtime_'+str(t_pulse)+'.png'
fname_savequtip = "Bz_"+str(Bz)+"_csigma_"+str(frq_carrier_detuningsigma)+"_cpts_"+str(frq_carrier_detuningpts)+"_sbdecentral_"+str(frq_sidebanddetuning_central)+"_sbdesigma_"+str(frq_sidebanddetuning_detuningsigma)+"_sbdepts_"+str(frq_sidebanddetuning_detuningpts)+"_cRabi_"+str(Bcarrier)+'_sbRabi_'+str(Bsideband)+'_simtime_'+str(t_pulse)
plt.savefig(os.path.join(pathpic,fname_savepic),dpi=100,bbox_inches = 'tight')
#plt.show()

dict_save = {}
dict_save['ls_pulsehandlers_HFDrive'] = ls_pulsehandlers_HFDrive
dict_save['states_HFDrive'] = states_HFDrive
dict_save['qutrit_sys'] = qutrit_sys
dict_save['frq_carrier_ls'] = frq_carrier_ls
dict_save['frq_sidebanddetuning_ls'] = frq_sidebanddetuning_ls
dict_save['t_pulse'] = t_pulse
dict_save['state0'] = state0

#Define local filename for qutip save
qsave(dict_save, os.path.join(pathqu,fname_savequtip)) #uncomment to save data
print("Saved QuTiP file to:", fname_savequtip)
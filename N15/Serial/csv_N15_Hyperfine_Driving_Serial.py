# The second dressing drive has a phase shift, no shift for this time

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import sys
commandline = sys.argv

import os
dirpath_pers = r'C:\Users\walsworth1\Desktop\Jiashen_Simulation\NV_HyperfineDriving\N15\Serial'
if dirpath_pers not in sys.path:
    sys.path.insert(0, dirpath_pers)

import addbase #Change to correct addbase_* file in your directory
from nvsys import * # This is the main workhorse of the NV simulations codebase

def N15HyperfineDrivingMain(commandline):
    # ### Define system's initial state for solver
    #miinit = 1
    #state0 = tensor(basis(3,1), basis(2, miinit)) # ms = 0, mi = -1/2 state
    state0 = tensor(basis(3,1), (basis(2, 0)+basis(2, 1))/(2**0.5)) # ms=0 superposition of mi=+1/2,-1/2
    
    
    # ### Applied magnetic field
    
    Bx = 0; By = 0; Bz = float(commandline[0]) # B field components (G)
    B0_init = np.array([Bx, By, Bz]) # B field vector (G)
    theta = 0 # angle in degrees for rotation around y axis
    theta *= np.pi/180
    Ry = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]]) 
    B0 = np.dot(Ry, B0_init)
    #print(B0)
    
    
    # ### Define NV system
    
    
    
    qutrit_sys = qutrit_coupled(s2tot=1/2) #N15
    qutrit_sys.setup_Hstat(B0)
    #print([elem for elem in qutrit_sys.pulsefuncs]) #Types of pulses available
    
    
    # ### Determine frequency of excitation
    # The function
    # 
    # ret_transition_frq(i0, if, s0, sf) 
    # 
    # calculates the difference between energy eigenvalues $E_\text{sf,if}-E_\text{s0,i0}$.
    # 
    # $E_\text{i,j}$ corresponds to energy eigenvalue of electronic basis index $i$ (0, 1 or 2) and nuclear basis index $j$ (0 or 1 for 15N. 0,1 or 2 for 14N)
    
    # In[8]:
    
    
    #
    # Determine frequency of excitation
    #
    frq_trans = qutrit_sys.ret_transition_frq(0,0,1,2)*0.5+qutrit_sys.ret_transition_frq(1,1,1,2)*0.5
    #print('Carreier Frq', frq_trans)
    #frq_trans1 = qutrit_sys.ret_transition_frq(miinit, miinit, idx_ms0=1, 
                                                 #idx_msf=2, Ham=None) # ms=0 <=> ms=-1 transition
    #frq_trans2 = qutrit_sys.ret_transition_frq(miinit, miinit, idx_ms0=1, 
                                                 #idx_msf=0, Ham=None) # ms=0 <=> ms=+1 transition
    #print('frq_trans1', frq_trans1)
    #print('frq_trans2', frq_trans2)
    
    frq_carrier_central = frq_trans
    frq_carrier_detuningsigma = float(commandline[1]) #MHz
    frq_carrier_detuningpts = int(commandline[2])
    frq_carrier_ls = np.linspace(frq_carrier_central-frq_carrier_detuningsigma, frq_carrier_central+frq_carrier_detuningsigma,frq_carrier_detuningpts)
    #print('frq_carrier_ls', frq_carrier_ls)
    
    frq_sidebanddetuning_central = float(commandline[3]) # MHz
    frq_sidebanddetuning_detuningsigma = float(commandline[4]) #MHz
    frq_sidebanddetuning_detuningpts = int(commandline[5])
    frq_sidebanddetuning_ls = np.linspace(frq_sidebanddetuning_central-frq_sidebanddetuning_detuningsigma, frq_sidebanddetuning_central+frq_sidebanddetuning_detuningsigma,frq_sidebanddetuning_detuningpts)
    #print('frq_sidebanddetuning_ls',frq_sidebanddetuning_ls)
    # Pick whether to drive to ms = -1 or +1
    #frq_sweep_init = frq_trans1 
    #frq_stat = frq_trans2
    
    
    # ### Drive magnetic field amplitude (Rabi strength)
    
    # The numbers Bac1MHz & Bac1MHz correspond approximately to rabi freq in MHz (exact if no detuning is present)
    
    # In[9]:
    
    Bcarrier = float(commandline[6]) #MHz
    Bsideband = float(commandline[7]) #MHz
    B_amp_carrier = Bcarrier/qutrit_sys.gammaNV
    B_amp_sideband = Bsideband/qutrit_sys.gammaNV
    
    #Bac1MHz = 5   # ms=-1 transition
    #Bac2MHz = 5   # ms=+1 transition
    #B_amp1 = Bac1MHz / qutrit_sys.gammaNV
    #B_amp_stat = Bac2MHz / qutrit_sys.gammaNV
    
    #
    # AC Drive detuning from resonance (MHz)
    # (Uncomment the following to probe the middle of the *N15 hyperfine levels)
    #
    #detuning1 = -abs((qutrit_sysDQ.ret_transition_frq(0, 0, idx_ms0=1, idx_msf=2, Ham=None) -
    #                               qutrit_sysDQ.ret_transition_frq(1, 1, idx_ms0=1, idx_msf=2, Ham=None))/2)
    #detuning2 = +abs((qutrit_sysDQ.ret_transition_frq(0, 0, idx_ms0=1, idx_msf=0, Ham=None) -
    #                               qutrit_sysDQ.ret_transition_frq(1, 1, idx_ms0=1, idx_msf=0, Ham=None))/2)
    #detuning_stat = +1.5015 # ms = +1 detuning
    
    
    # ## Build DQ ESR Pulse Sequence
    
    # ### Sweep detuning from frq_sweep_init
    
    # In[10]:
    
    
    #detun_start = -10
    #detun_end = 10
    #n_detun = 51
    #arr_detun = np.linspace(detun_start, detun_end, n_detun)
    #print(arr_detun)
    
    
    # ### Vary power of microwave drive
    
    # In[11]:
    
    
    #arr_Bamp_stat = np.linspace(1, 10, 21)/qutrit_sys.gammaNV
    #print('Array of stationary Rabis:', arr_Bamp_stat*qutrit_sys.gammaNV)
    
    
    # ### Define pulse duration
    
    # In[12]:
    
    
    #t_pulse = 1 #microsecs
    t_pulse = 1/Bsideband/2 #pi pulse on resonant transition
    #print('pi pulse in us',t_pulse)
    
    
    # ### Vary parameters, append each pulse_seq() object to list for execution
    
    # In[13]:
    
    
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
    
    #for B_amp_stat in arr_Bamp_stat:
    #    for detuning1 in arr_detun:
    #
            #print('Rabi freq:', frq_rabi)
            #print('Pi pulse duration:', time_pipulse)
    
            # pulse sequence handler
    #        pulse_handler_DQESR = pulse_seq(qutrit_sys, montecarlo=bool_montecarlo)
            
            # First tone (note detuning is swept here)
    #        pulse_90_1 = Pulse('ACDrive', duration=t_pulse, starttime=0)
    #        pulse_90_1.pulse_parser(qutrit_sys, B_amp1, frq_sweep_init, 0, detuning1, 0, False)
    #        pulse_handler_DQESR.add_pulse(pulse_90_1)
            
            # 2nd tone (fixed microwave frequency, power is varied)
    #        pulse_stat = Pulse('ACDrive', duration=t_pulse, starttime=0)
    #        pulse_stat.pulse_parser(qutrit_sys, B_amp_stat, frq_stat, 0, detuning_stat, 0, False)
    #        pulse_handler_DQESR.add_pulse(pulse_stat)
    
    #        ls_pulsehandlers_DQESR.append(pulse_handler_DQESR)
    
    
    # In[14]:
    
    
    #len(ls_pulsehandlers_DQESR)
    
    
    # ### Define times at which to evaluate system state during evolution
    
    # In[15]:
    
    
    ntimes = 5
    t_list = np.linspace(0, t_pulse, ntimes)
    #print('t_list for saving state',t_list)
    
    
    # ### Execute pulse sequences in parallel
    
    # In[16]:
    
    print('QuTip Solving....')
    states_HFDrive = []
    sim_starttime = time.time()
    for index in np.arange(len(ls_pulsehandlers_HFDrive)):
        print(index,'/',len(ls_pulsehandlers_HFDrive))
        states_HFDrive.append(ls_pulsehandlers_HFDrive[index].exec_pulse(state0, t_list=t_list).states)
    #parallel_handlers = parallel_pulseseq(ls_pulsehandlers_HFDrive, state0, t_list=t_list)
    #states_HFDrive = parallel_handlers.exec_pulseseqs(bool_parallel=True)
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
    pathpic = r'C:\Users\walsworth1\Desktop\Jiashen_Simulation\NV_HyperfineDriving\N15\Local\pic'
    pathqu = r'C:\Users\walsworth1\Desktop\Jiashen_Simulation\NV_HyperfineDriving\N15\Local\qu'
    fname_savepic = "Bz_"+str(Bz)+"_csigma_"+str(frq_carrier_detuningsigma)+"_cpts_"+str(frq_carrier_detuningpts)+"_sbdecentral_"+str(frq_sidebanddetuning_central)+"_sbdesigma_"+str(frq_sidebanddetuning_detuningsigma)+"_sbdepts_"+str(frq_sidebanddetuning_detuningpts)+"_cRabi_"+str(Bcarrier)+'_sbRabi_'+str(Bsideband)+'_simtime_'+str(t_pulse)+'.png'
    fname_savequtip = "Bz_"+str(Bz)+"_csigma_"+str(frq_carrier_detuningsigma)+"_cpts_"+str(frq_carrier_detuningpts)+"_sbdecentral_"+str(frq_sidebanddetuning_central)+"_sbdesigma_"+str(frq_sidebanddetuning_detuningsigma)+"_sbdepts_"+str(frq_sidebanddetuning_detuningpts)+"_cRabi_"+str(Bcarrier)+'_sbRabi_'+str(Bsideband)+'_simtime_'+str(t_pulse)
    plt.savefig(os.path.join(pathpic,fname_savepic),dpi=100,bbox_inches = 'tight')
    # In[ ]:
    
    
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

import pandas as pd
configuration = pd.read_csv(r'C:\Users\walsworth1\Desktop\Jiashen_Simulation\NV_HyperfineDriving\N15\Local\configdata.csv')
totalrows = list(configuration.shape)[0]
for index in np.arange(0,totalrows):
    commandline = list(configuration.loc[index,:])
    print("Running Configuration: ",index+1,"/",totalrows)
    print(configuration.loc[index,:])
    N15HyperfineDrivingMain(commandline)

# ### Load Qutip data

# In[17]:


#fname_load = dir_qsave + "DQESR_2021-01-14-12-51-37"
#bool_load = False # Set to True if loading data
#if bool_load:
#    dict_load = qload(fname_load)
#    print([key for key in dict_load])
#    ls_pulsehandlers_DQESR = dict_load['ls_pulsehandlers_DQESR']
#   states_DQESR = dict_load['states_DQESR']
#    qutrit_sys = dict_load['qutrit_sys']
#    arr_detun = dict_load['arr_detun']
#    arr_Bamp_stat = dict_load['arr_Bamp_stat']
#    t_pulse = dict_load['t_pulse']
#    state0 = dict_load['state0']


# ### Extrate system state at the end of each pulse sequence

# In[ ]:


# #finalstates_DQESR = [states[-1] for states in states_DQESR]
# finalpopmszero_2D = np.array(expect(qutrit_sys.pop_mszero, finalstates_DQESR)).reshape((len(arr_Bamp_stat), len(arr_detun)))
# finalpopmszero_2D.shape


# # ### Plot ESR spectra

# # In[ ]:


# plotpalette_plt = init_plotpalette(5, l=0.5, s=0.5)
# plt.rcParams['figure.figsize'] = [20,8]

# #
# # Find expectation value of ms=0 population operator (Sz^2 operator)
# #
# #arr_pop0 = expect(qutrit_sys.pop_mszero, finalstates_DQESR)
# for idx in np.arange(finalpopmszero_2D.shape[0]):
#     arr_xfreqs = arr_detun# + frq_sweep_init
#     plt.plot(arr_xfreqs, finalpopmszero_2D[idx], '.-', lw=2)

#     plt.xlim([min(arr_xfreqs),max(arr_xfreqs)])
#     plt.ylim([0,1.01])
#     plt.xlabel("Detuning (MHz)")
#     plt.ylabel("$m_s = 0$ population")
#     plt.axvline(3.03, c='k', ls='--', lw=2.5, label='Off-resonant hyperfine state')
#     plt.title('Pulsed ESR simulations')
#     #plt.legend(fontsize=20, bbox_to_anchor=(1, 0.5))
#     #plt.legend(fontsize=20,loc='lower left')
#     #plt.savefig(dir_figs + 'test.png', dpi=100, bbox_inches='tight')
#     plt.show()


# # ### Create 2D color plot

# # In[ ]:


# default_rcparams()
# axes_ends = (np.amin(arr_detun), np.amax(arr_detun), np.amin(arr_Bamp_stat*qutrit_sys.gammaNV), np.amax(arr_Bamp_stat*qutrit_sys.gammaNV))
# plt.rcParams['figure.figsize'] = [15,14]
# plt.imshow(finalpopmszero_2D, cmap = plt.get_cmap('magma'), extent = axes_ends, aspect='auto', origin='lower', vmax=np.amax(finalpopmszero_2D))
# #, norm=matplotlib.colors.LogNorm(vmin=np.amin(arr_int_2d_norm), 
# plt.xlabel("Detuning (MHz)", fontsize = 20)
# plt.ylabel('Stationary Drive Rabi (MHz)', fontsize = 20)
# plt.title('DQ Pulsed ESR\n Sweeping Drive Rabi Freq = ' + "{:1f}".format(B_amp1*qutrit_sys.gammaNV)[:3] + " MHz", fontsize=40)

# cbar = plt.colorbar(label="$m_s = 0$ population", orientation='horizontal', pad=.1)
# #plt.clim(min(arr_int_2d),max(arr_int_2d))
# #plt.clim(np.amin(arr_int_2d),np.amax(arr_int_2d)*1.1)
# #plt.clim(np.amin(arr_int_2d_norm),np.amax(arr_int_2d_norm))
# #plt.clim(1, 5)
# #plt.clim(np.amin(np.log10(arr_int_2d_norm)),np.amax(np.log10(arr_int_2d_norm)))
# #plt.clim(.2, .5)

# fname_fig = dir_figs + 'pulsed_DQESR_detun-1.5MHz_' +  "{:1f}".format(B_amp1*qutrit_sys.gammaNV)[:3] + "MHz_" + time.strftime("%Y-%m-%d-%H-%M-%S") + '.png'
# print(fname_fig)
# #plt.savefig(fname_fig, dpi=100, bbox_inches='tight')
# plt.show()


# # In[ ]:





# # ### Miscellaneous bits of code

# # In[ ]:


# fname_txt_2D = dir_qsave + "DQESR_2D_2" + ".txt"
# np.savetxt(fname_txt_2D, finalpopmszero_2D)
# fname_detun = dir_qsave + "DQESR_nondriven_detunings_2" + ".txt"
# np.savetxt(fname_detun, arr_detun)
# fname_rabistat = dir_qsave + "DQESR_rabistationary_2" + ".txt"
# np.savetxt(fname_rabistat, arr_Bamp_stat*qutrit_sys.gammaNV)


# # In[ ]:


# #print(finalpopmszero_2D.shape)
# finalpopmszero_2D_norm = np.array([finalpopmszero_2D[idx]- finalpopmszero_2D[idx][0] for idx in np.arange(finalpopmszero_2D.shape[0]) ])
# finalpopmszero_2D_norm.shape


# # In[ ]:


# default_rcparams()
# axes_ends = (np.amin(arr_detun), np.amax(arr_detun), np.amin(arr_Bamp_stat*qutrit_sys.gammaNV), np.amax(arr_Bamp_stat*qutrit_sys.gammaNV))
# plt.rcParams['figure.figsize'] = [15,14]
# plt.imshow(finalpopmszero_2D_norm, cmap = plt.get_cmap('magma'), extent = axes_ends, aspect='auto', origin='lower', vmax=np.amax(finalpopmszero_2D_norm))
# #, norm=matplotlib.colors.LogNorm(vmin=np.amin(arr_int_2d_norm), 
# plt.xlabel("Detuning (MHz)", fontsize = 20)
# plt.ylabel('Stationary Drive Rabi (MHz)', fontsize = 20)
# #plt.title('Hermite Pulse\n Improvement in Integrated ESR Linewidth \n For Varying Gaussian ' r'$\sigma_1$ & Polynomial Coefficient $\alpha$', fontsize=30)
# plt.title('Corrected DQ Pulsed ESR\n Sweeping Drive Rabi Freq = ' + "{:1f}".format(B_amp1*qutrit_sys.gammaNV)[:3] + " MHz", fontsize=40)

# cbar = plt.colorbar(label="$m_s = 0$ population", orientation='horizontal', pad=.1)

# #plt.clim(min(arr_int_2d),max(arr_int_2d))
# #plt.clim(np.amin(arr_int_2d),np.amax(arr_int_2d)*1.1)
# #plt.clim(np.amin(arr_int_2d_norm),np.amax(arr_int_2d_norm))
# #plt.clim(1, 5)
# #plt.clim(np.amin(np.log10(arr_int_2d_norm)),np.amax(np.log10(arr_int_2d_norm)))
# #plt.clim(0, 5)

# fname_fig = dir_figs + 'pulsed_DQESR_detun-1.5MHznorm_' +  "{:1f}".format(B_amp1*qutrit_sys.gammaNV)[:3] + "MHz_" + time.strftime("%Y-%m-%d-%H-%M-%S") + '.png'
# print(fname_fig)
# #plt.savefig(fname_fig, dpi=100, bbox_inches='tight')
# plt.show()


# # In[ ]:


# idx_1d = np.argmax(finalpopmszero_2D_norm)
# idx_2d = np.array([idx_1d//finalpopmszero_2D_norm.shape[1], idx_1d%finalpopmszero_2D_norm.shape[1]])


# # In[ ]:





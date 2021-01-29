import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#def DQRabi_frqdiff(t, rabi, detun):
#    #rabi *= 2*np.pi
#    detun *= 2*np.pi
#    alpha = (2**0.5)*rabi*(2*np.pi)
#    k = (alpha**2 + 4*detun**2)**0.5

#    term1 = (4*detun**2*((alpha**2*(k - 2*detun)**2)/(alpha**2 - 2*(k - 2*detun)*detun)**2 + (alpha**2*(k + 2*detun)**2)/((alpha**2 + 4*detun**2)*(alpha**2 + 4*detun*(k + 2*detun))))*np.cos((k*t)/2))/(alpha**2 *(1 + (4*detun**2)/alpha**2))
#    term2 = (alpha**4*(k - 2*detun)**4)/(4*(alpha**2 - 2*(k - 2*detun)*detun)**4) + (16*detun**4)/(alpha**4 *(1 + (4 *detun**2)/alpha**2)**2) + (alpha**4 *(k + 2 *detun)**4)/(4 *(alpha**2 + 4* detun**2)**2 *(alpha**2 + 4 *detun *(k + 2 *detun))**2)
#    term3 = (alpha**4 *(k**2 - 4 *detun**2)**2 *np.cos((k *t)))/(2 *(alpha**2 -2* (k - 2* detun)* detun)**2 *(alpha**2 + 4* detun**2)* (alpha**2 + 4* detun* (k + 2* detun)))

#    return (term1+term2+term3), term1, term3, term2

#def DQRabi_frqdiff(t, rabi, detun):
#    detun *= 2*np.pi
#    alpha = (2**0.5)*rabi*(2*np.pi)
#    k = (alpha**2 + 4*detun**2)**0.5
#
#    term1 = np.cos((k *t))*alpha**4/2/((alpha**2 + 4*detun**2)**2)
#    term2 = np.cos((k *t/2))*8*alpha**2 * detun**2 /((alpha**2 + 4*detun**2)**2)
#    term3 = (alpha**4 + 32*detun**4)/2/((alpha**2 + 4*detun**2)**2)

#    return (term1+term2+term3), term2, term1, term3


def DQRabi_frqdiff(t, sqrabi, detun):
    detun *= 2*np.pi
    sqrabi *= 2*np.pi
    
    freqwd = ((2*detun**2 + sqrabi**2)/2)**0.5
    return ((2*detun**2 + sqrabi**2 * np.cos(freqwd * t))/(2*detun**2 + sqrabi**2))**2

def get_DQhalfpi(rabi, detun):
    alpha = (2**0.5)*rabi*(2*np.pi)
    k = (alpha**2 + 4*(detun*2*np.pi)**2)**0.5
    t = np.linspace(0, 2*np.pi*10/k, 10000)
    termtot = DQRabi_frqdiff(t, rabi, detun)
    idx_mins = find_peaks(termtot*(-1))[0]
    t_mins = t[idx_mins]
    pop_mins = termtot[idx_mins]
    #plt.plot(t_mins, pop_mins, 'o')
    #plt.plot(t, termtot, ':')
    #plt.show()
    
    return t_mins

def DQRabi_frqcomm(t, rabi, detun):
    rabi *= 2*np.pi
    detun *= 2*np.pi
    k = (2*rabi**2 + detun**2)**0.5

    denom = detun**2 + 2*rabi**2
    numer = rabi**2 + detun**2 + (rabi**2)*np.cos(k *t)

    return numer/denom





# 
# Delete - redundant ABC comparisons
#
def make_ABCplots():
    import seaborn as sns
    plotpalette = sns.hls_palette(16, l=.4, s=.8).as_hex()

    arr_SQrabi = np.linspace(0.1, 5, 1000)
    arr_terms = []
    for rabi in arr_SQrabi:
        termfinal, B, A, C = DQRabi_frqdiff(0, rabi, 1.515)
        arr_terms.append(np.asarray([termfinal, A, B, C]))
    arr_terms = np.array(arr_terms)

    fig, axs = plt.subplots(3)
    fig.suptitle('$A, B, C$ for Various Drive Strengths $\Omega$\n Differential Detunings $\Delta= 1.515$ MHz', fontsize = 25)

    #axs[0].plot(arr_SQrabi, arr_terms[:,0], 'rx:', label='$|c_0(t)|^2$')

    axs[0].plot(arr_SQrabi, arr_terms[:,1], '-', c=plotpalette[0], lw=3, label='$A$')
    axs[0].set_ylabel('$A$', fontsize=20)
    axs[0].axvline(2.14, c= 'k', ls=':', lw=2, label = '$\Omega_F = 2.14$ MHz')
    axs[0].legend(loc=4, fontsize=16)

    axs[1].plot(arr_SQrabi, arr_terms[:,2], '-', c=plotpalette[0], lw=3, label='$B$')
    axs[1].set_ylabel('$B$', fontsize=20)
    axs[1].axvline(2.14, c= 'k', ls=':', lw=2)
    axs[1].legend(loc=4, fontsize=16)

    #axs[2].plot(arr_SQrabi, arr_terms[:,3], 'k-', lw=2, label='$C$')
    #axs[2].set_ylabel('$C$', fontsize=20)
    #axs[2].axvline(2.14, c= 'k', ls=':', lw=2)
    #axs[2].legend(loc=4, fontsize=16)
    #axs[2].set_xlabel('Drive Strength $\Omega$', fontsize=20)

    axs[2].plot(arr_SQrabi, arr_terms[:,1]-arr_terms[:,2]+arr_terms[:,3], 'k-', lw=3, label='$A-B+C$')
    axs[2].axvline(2.14, c= 'k', ls=':', lw=2)
    axs[2].legend(loc=4, fontsize=16)
    axs[2].set_xlabel('Drive Strength $\Omega$', fontsize=20)

    for ax in axs:
        ax.label_outer()
    fig.tight_layout()
    fig.subplots_adjust(top=.84)

    plt.savefig('./Figures/Magic Rabi Paper Plots/ABC.png', dpi=100, bbox_inches='tight')

    plt.show()

#arr_t = np.linspace(0,.5,100)
#arr_detunings = np.array([1.515])#np.linspace(0, 10, 11)
#for idx_detun in np.arange(arr_detunings.size):
#    plt.figure(idx_detun)
#    rabi = 2.15
#    detuning = arr_detunings[idx_detun]
#    termfinal, term1, term2, term3 = DQRabi_frqdiff(arr_t, rabi, detuning)
#    plt.plot(arr_t, termfinal, 'k--', lw=2, label='DQ Rabi')
#    plt.axvline(get_DQhalfpi(rabi, detuning)[0], c='r', lw=2, label = '$t_1$')
#    plt.axvline(get_DQhalfpi(rabi, detuning)[1], c='b', lw=2, label = '$t_2$')
    
#    #termfinal, term1, term2, term3 = DQRabi_frqdiff2(arr_t, 2.15, detuning)
#    #plt.plot(arr_t, termfinal, 'rx:', label='total')
#    #plt.plot(arr_t, term1, '--', label='term 1')
#    #plt.plot(arr_t, term2*np.ones(arr_t.size), ':', label='term 2')
#    #plt.plot(arr_t, term3, '-.', label='term 3')
#    plt.ylim([0,1])
#    plt.xlabel('Pulse Duration ($\mu s$)', fontsize=20)
#    plt.ylabel('$m_s = 0$ Population', fontsize=20)
#    str_title = 'DQ Rabi \n'
#    str_title += 'Driving Amp: ' + str(round(rabi, 4)) + ' MHz, '
#    str_title += 'Detunings: $\pm$' + str(round(detuning, 4)) + ' MHz'
#    plt.title(str_title , fontsize = 25)
#    #plt.savefig('./difdetun/' + str(int(idx_detun)) + '.png',dpi=100)
#    plt.legend(loc=3, fontsize=15)
#    #plt.close()
#    plt.show()
#    #plt.close()


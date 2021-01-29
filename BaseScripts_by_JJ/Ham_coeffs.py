import numpy as np
import sympy

#
# Oscillating Magnetic Field
#
def H_ACB_pulsecoeff(t, args):
    coeff_timedep = 0
    for pulse in args["ACB"]:
        if t >= pulse.starttime and t <= pulse.endtime:
            coeff_timedep += np.cos( 2*np.pi*pulse.Ham_args['ACB_freq1']*(
                t + pulse.Ham_args['ACB_t_offset1']) + pulse.Ham_args['ACB_phase1'])
    return coeff_timedep

def H_ACB_serialcoeff(t, args):
    return np.cos(2*np.pi*args['ACB_freq1']*(t + args['ACB_t_offset1']) + args['ACB_phase1'])


#
# Microwave Field
#
def H_ACD_pulsecoeff(t, args):
    coeff_timedep = 0
    for pulse in args["ACD"]:
        if t >= pulse.starttime and t <= pulse.endtime:
            coeff_timedep += pulse.Ham_args['ACD_Bamp']*np.cos(2*np.pi*pulse.Ham_args['ACD_freq1']*(
                t + pulse.Ham_args['ACD_t_offset1']) + pulse.Ham_args['ACD_phase1'])
    return coeff_timedep

def H_ACD_serialcoeff(t, args):
    return args['ACD_Bamp']*np.cos(2*np.pi*args['ACD_freq1']*(t + args['ACD_t_offset1']) + args['ACD_phase1'])




#
# AMPlitude-modulated Microwave Pulse
#
def H_AMP_pulsecoeff(t, args):
    coeff_timedep = 0
    for pulse in args["AMP"]:
        if t >= pulse.starttime and t <= pulse.endtime:
            # B_amp1 is amplitude/shape of pulse
            B_amp1 = pulse.Ham_args['AMP_modfunc'](t-pulse.starttime, **pulse.Ham_args['AMP_modargs'])
            
            coeff_timedep += B_amp1*np.cos(2*np.pi*pulse.Ham_args['AMP_freq1']*(
                t + pulse.Ham_args['AMP_t_offset1']) + pulse.Ham_args['AMP_phase1'])
    return coeff_timedep

def H_AMP_serialcoeff(t, args):
    B_amp1 = args['AMP_modfunc'](t-pulse.starttime, **pulse.Ham_args['AMP_modargs'])
    
    return B_amp1*np.cos(2*np.pi*pulse.Ham_args['AMP_freq1']*(
                t + pulse.Ham_args['AMP_t_offset1']) + pulse.Ham_args['AMP_phase1'])

#
# FReQuency-modulated (chirped) Microwave Pulse
#
def H_FRQ_pulsecoeff(t, args):
    coeff_timedep = 0
    for pulse in args["FRQ"]:
        if t >= pulse.starttime and t <= pulse.endtime:
            # Phase/frequency modulation manifests as time-dependent phase via phaset variable
            # In general, this is the integral of the function used for frequency modulation
            phaset = np.float(pulse.Ham_args['FRQ_phasemodexpr'].subs(sympy.symbols('t'), t-pulse.starttime)) 

            coeff_timedep += pulse.Ham_args['FRQ_Bamp']*np.cos(2*np.pi*pulse.Ham_args['FRQ_freq1']*(
                t + pulse.Ham_args['FRQ_t_offset1']) + phaset + pulse.Ham_args['FRQ_phase0'])
    return coeff_timedep


#
# ARBitrary waveform (amp- and/or freq-modulated) Microwave Pulse
#
def H_ARB_pulsecoeff(t, args):
    coeff_timedep = 0
    for pulse in args["ARB"]:
        if t >= pulse.starttime and t <= pulse.endtime:
            # B_amp1 is amplitude/shape of pulse
            B_amp1 = np.float(pulse.Ham_args['ARB_ampmodfunc'](t-pulse.starttime))

            # Phase/frequency modulation manifests as time-dependent phase via phaset variable
            # In general, this is the integral of the function used for frequency modulation
            phase1 = np.float(pulse.Ham_args['ARB_phasemodfunc'](t-pulse.starttime)) 

            coeff_timedep += B_amp1*np.cos(2*np.pi*pulse.Ham_args['ARB_freq1']*(
                t + pulse.Ham_args['ARB_t_offset1']) + phase1 + pulse.Ham_args['ARB_phase0'])
    return coeff_timedep             



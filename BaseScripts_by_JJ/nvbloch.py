#!/usr/bin/env python
# coding: utf-8

# In[8]:


from nvsys import *


# In[4]:


def nvblochvecs(states, proj0, proj1, projx, times, frq_trans1, frq_trans2, Nmivals, sphere=True):
    #------
    # Returns bloch vectors
    #------
    # states - an array of column vectors
    # proj0, proj1 and projx must be row vectors
    # times - an array of times
    # H_lab - square matrix. For conversion into rotating frame.
    #
    #------
    if sphere:
        factor_sphere = 2
    else:
        factor_sphere = 1
        
    w1 = frq_trans1*2*np.pi 
    w2 = frq_trans2*2*np.pi
    
    #opR = [(1j*H_lab*time_loop).expm() for time_loop in times]
    #opR = [tensor(Qobj([[np.exp(- 1j*w1*time_loop), 0, 0],[0, 1, 0],[0, 0, np.exp(-1j*w2*time_loop)]]), identity(Nmivals)) for time_loop in times]
    opR = [tensor(Qobj([[np.exp(- 1j*w1*time_loop), 0, 0],[0, 1, 0],[0, 0, 1]]), identity(Nmivals)) for time_loop in times]
    states_rot = [opR[idx]*states[idx] for idx in np.arange(len(times))]

    # Convert complex coefficients to polar form 
    c0 = [(proj0*elem)[0][0][0] for elem in states_rot]
    r0, phase0 = np.absolute(c0), np.angle(c0)
    c1 = [(proj1*elem)[0][0][0] for elem in states_rot]
    r1, phase1 = np.absolute(c1), np.angle(c1)

    # Find magntiude of coefficient of other states, for normalization
    if projx is None:
        rx = 0
    else:
        cx = [(projx*elem)[0][0][0] for elem in states_rot]
        rx = np.absolute(cx)

    # Calculate 3D polar and azimuthal angles
    angle_polar1 = factor_sphere*np.arcsin(r1/((1 - rx**2)**0.5))
    angle_polar2 = factor_sphere*np.arccos(r0/((1 - rx**2)**0.5))
    angle_polar3 = factor_sphere*np.arctan2(r1, r0)
    angle_polar = angle_polar1
    angle_azi = phase1 - phase0

    xb = np.sin(angle_polar)*np.cos(angle_azi)
    yb = np.sin(angle_polar)*np.sin(angle_azi)
    zb = np.cos(angle_polar)
    
    return xb, yb, zb



def nvblochvecsDQ(states, proj0, projplus, projminus, times, frq_trans1, frq_trans2, Nmivals, sphere=False):
    #------
    # Returns bloch vectors
    #------
    # states - an array of column vectors
    # proj0, proj1 and projx must be row vectors
    # times - an array of times
    # H_lab - square matrix. For conversion into rotating frame.
    #
    #------
    if sphere:
        factor_sphere = 2
    else:
        factor_sphere = 1
        
    w1 = frq_trans1*2*np.pi 
    w2 = frq_trans2*2*np.pi
    
    #opR = [(1j*H_lab*time_loop).expm() for time_loop in times]
    opR = [tensor(Qobj([[np.exp(- 1j*w1*time_loop), 0,0],[0, 1, 0],[0, 0, np.exp(-1j*w2*time_loop)]]), identity(Nmivals)) for time_loop in times]
    states_rot = [opR[idx]*states[idx] for idx in np.arange(len(times))]

    # Convert complex coefficients to polar form 
    c0 = [(proj0*elem)[0][0][0] for elem in states_rot]
    r0, phase0 = np.absolute(c0), np.angle(c0)
    c1 = [(projplus*elem)[0][0][0] for elem in states_rot]
    r1, phase1 = np.absolute(c1), np.angle(c1)
    c2 = [(projminus*elem)[0][0][0] for elem in states_rot]
    r2, phase2 = np.absolute(c2), np.angle(c2)
    
    # Calculate 3D polar and azimuthal angles
    angle_polar = np.arcsin(r1/((1 - r2**2)**0.5))
    angle_azi = phase1 - phase2

    xb = np.sin(angle_polar)*np.cos(angle_azi)
    yb = np.sin(angle_polar)*np.sin(angle_azi)
    zb = np.cos(angle_polar)
    
    return xb, yb, zb





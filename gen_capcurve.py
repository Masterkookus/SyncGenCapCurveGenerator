# -*- coding: utf-8 -*-
####################################################################################################
# SYNCHRONOUS GENERATOR CAPABILITY CURVES GENERATOR
####################################################################################################
# This program was designed to create the capability curves of a synchronous generator
# given the machines' nominal parameters.
# It comprises two functions:
# 1) Capability Curve Generator:
#       generates the capability curve from the current limit curves for the rotor and stator
# 2) Generator Capability Verification:
#       verify wheter a certain load condition can be achieved by a generator
####################################################################################################
# This code was developed using the concepts and electric machinery modeling from Chapman's
# "Electric Machinery Fundamentals", 5th edition and Uman's "Fitzgerald & Kingsley's Electric
#  Machinery", 7th edition.
# The sections used were, respectively, 4.11 (Synchronous Generator Ratings - Synchronous Generator
# Capability Curves, p.254) and 5.5 (Steady-state Operating Characteristics, p.293)
# Along with several mathematical and programming ideas that will be cited when approriate
#  throught the code.
####################################################################################################
# Developed by
#       Felipe Baldner (https://github.com/fbaldner) and
#       Júlia Avellar ()
####################################################################################################
# version 0.1 (2021-12-09):
#      - Initial release
#      - Defines generator input parameters and calculates both curves' data
#      - Plot both curves overlapping each other and prime-mover max power
####################################################################################################
# version 0.1mk (2022-8-15):
#   - Added basic calculations for the End Region Heating Limit
#   - Revised to build a composite curve with tripping regions for generator protection
#   - Added Generator Statistics on graph
#   - Results produced exceptable results in comparison to actual test data
####################################################################################################
# Needed libraries
import math as mt
import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
import intersect as intc
import pandas as pd

# Generator input parameters
Vterminal   = 480       # Terminal voltage, in volt (line voltage)
freq        = 60        # Electrical frequency, in hertz
Srated      = 1700e3      # Rated apparent power, in volt-ampere
poles       = 4         # Number of poles
PowerFactor = 0.8       # Power factor, lagging (inductive)
Xsync       = 0.4882         # Synchronous reactance, in ohm per phase

# Maximum Load parameters
Pload       = 1360e3      # Nameplate maximum load of generator, in watt

### PT/CT Ratios ###
ctr=500
ptr=2.31

### PQ max and Q Min ###
Rmax        = 1500e3        # Max of P or Q
Qmin        = -0.4*Srated   #Min Value of Q
Ifl         = Srated/(Vterminal*np.sqrt(3))

#Show Reverse VAR tripping times and zones
show_Qlim   = [1,1,1,1] #Specify which tripping curves to show
Qlim        = [Qmin,Qmin*1.25,Qmin*1.5,Qmin*2]
trip_labels=['5s Trip ' + str(int(Qlim[0]/1e3)) + ' kVAR (' + str(int(Qlim[0]/(ctr*ptr))) + ' VAR sec)',
             '1s Trip ' + str(int(Qlim[1]/1e3)) + ' kVAR (' + str(int(Qlim[1]/(ctr*ptr))) + ' VAR sec)',
             '0.2s Trip ' + str(int(Qlim[2]/1e3)) + ' kVAR (' + str(int(Qlim[2]/(ctr*ptr))) + ' VAR sec)',
             '0.1s Trip ' + str(int(Qlim[3]/1e3)) + ' kVAR (' + str(int(Qlim[3]/(ctr*ptr))) + ' VAR sec)']

# Generator's Losses
Pmech_loss  = 1.5e3     # Mechanical (friction and windage) Losses, in watt
Pcore_loss  = 1e3       # Core Losses, in watt

# Calculations

#P and Q End Points
Qend        = (Qmin**2-Rmax**2)/(2*(Qmin+(0.31*Rmax)))
Rend        = Qend-Qmin

# Phase voltage, in volt
Vphase          = Vterminal/mt.sqrt(3)

# Maximum armature current, in ampere - absolute value and phasor
Iarm_max        = Srated/(3 * Vphase)
Iarm_max_phasor = Iarm_max*cm.exp(-1j * mt.acos(PowerFactor))

# Origin of rotor current curve
Qrotor          = -(3 * Vphase**2)/Xsync

# Generator's internal generated voltage, in volt
Ea              = Vphase + 1j * Xsync * Iarm_max_phasor

# Apparent power that is the radius of the rotor curve, in volt-ampere
D_E             = (3 * abs(Ea) * Vphase)/Xsync

# Prime-mover maximum output real power, in watt
Pmax_out        = Pload - Pmech_loss - Pcore_loss

# Stator current limit curve
# A circle centered in (0,0) with radius Srated
x_stator_o = 0
y_stator_o = 0
r_stator   = Srated

# Rotor current limit curve
# A circle centered in (0,Q) with radius D_E
x_stator_o = 0
y_stator_o = Qrotor
r_stator   = D_E

# Prime-mover power limit
x_prime = Pmax_out

#Create the individual curves
for i in range(-90,91):
    if i == -90:
        stator_curveA=np.array([Srated*np.cos(i*(np.pi/180)),
                             Srated*np.sin(i*(np.pi/180))])
        rotor_curveA=np.array([D_E*np.cos(i*(np.pi/180)),
                             Qrotor+D_E*np.sin(i*(np.pi/180))])
        end_curveA=np.array([Rend*np.cos(i*(np.pi/180)),
                             Qend+Rend*np.sin(i*(np.pi/180))])
    else:
        stator_curveA = np.vstack(
            (
                stator_curveA,
                np.array([Srated*np.cos(i*(np.pi/180)),Srated*np.sin(i*(np.pi/180))])
            ))
        rotor_curveA = np.vstack(
            (
                rotor_curveA,
                np.array([D_E*np.cos(i*(np.pi/180)), Qrotor+D_E*np.sin(i*(np.pi/180))])
            ))
        end_curveA=np.vstack(
            (
                end_curveA,
                np.array([Rend*np.cos(i*(np.pi/180)),Qend+Rend*np.sin(i*(np.pi/180))])
            ))

#Calculate the point of interception and return the nearest actually point for the stator and end
intx,inty = intc.intersection(stator_curveA[:,0], stator_curveA[:,1], end_curveA[:,0], end_curveA[:,1])
df = pd.DataFrame(stator_curveA,columns=['X','Y'])
df2 = (df['X']-intx[0]).abs() + (df['Y']-inty[0]).abs()
#Return the stator starting point
stator_start=df.loc[df2.idxmin()].name
df = pd.DataFrame(end_curveA,columns=['X','Y'])
df2 = (df['X']-intx[0]).abs() + (df['Y']-inty[0]).abs()
#Return the end stopping point
end_stop=df.loc[df2.idxmin()].name
#Calculate the point of interception and return the nearest actually point for the stator and rotor
intx,inty = intc.intersection(stator_curveA[:,0], stator_curveA[:,1], rotor_curveA[:,0], rotor_curveA[:,1])
df = pd.DataFrame(stator_curveA,columns=['X','Y'])
df2 = (df['X']-intx[0]).abs() + (df['Y']-inty[0]).abs()
#Return the stator stopping point
stator_stop=df.loc[df2.idxmin()].name
df = pd.DataFrame(rotor_curveA,columns=['X','Y'])
df2 = (df['X']-intx[0]).abs() + (df['Y']-inty[0]).abs()
#Return the rotor starting point
rotor_start=df.loc[df2.idxmin()].name

#Build the composite curve using the start/stop points above
comp_curveA=end_curveA[0:end_stop,:]
comp_curveA = np.vstack(
    (
         comp_curveA,
         stator_curveA[stator_start:stator_stop,:]
    ))
comp_curveA = np.vstack(
    (
         comp_curveA,
         rotor_curveA[rotor_start:,:]
    ))

stator_curve = plt.Line2D(stator_curveA[:,1],stator_curveA[:,0],color='m')
rotor_curve = plt.Line2D(rotor_curveA[:,1],rotor_curveA[:,0],color='g')
end_curve = plt.Line2D(end_curveA[:,1],end_curveA[:,0],color='c')
comp_curve = plt.Line2D(comp_curveA[:,1],comp_curveA[:,0],color='r')
                                                       
########
fig = plt.figure(figsize=(8, 5), dpi=80, facecolor='w', edgecolor='k')                                              
ax=fig.add_subplot(111)

# Plot prime-mover real power limit
maxpower_curve = plt.axhline(y=Pmax_out , color='b' )

ylim_pos = 1.6*Srated
xlim_neg = -1.2*Srated
xlim_pos =  1.2*Srated

ax.set_ylim((0        , ylim_pos))
ax.set_xlim((xlim_neg , xlim_pos))
ax.set_aspect('equal')

#plt.legend((stator_curve           , rotor_curve           , end_curve, maxpower_curve) ,
#           ('Stator current limit' , 'Rotor current limit' , 'End Regoin Heating Limit', 'Prime-mover maximum power') ,
#           numpoints=1,
#           loc=1)

plt.legend(
    (comp_curve,maxpower_curve),
    ('Capability','Max Power'),
    numpoints=1,
    loc=1)

plt.ylabel("P, kW/kVA")
plt.xlabel("Q, kVAR/kVA")

for pf in [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]:
    ax.add_line(plt.Line2D([0,np.sqrt(Srated**2-(Srated*pf)**2)*2],[0,(Srated*pf)*2],linestyle='dotted',color='y'))
    ax.add_line(plt.Line2D([0,-1*np.sqrt(Srated**2-(Srated*pf)**2)*2],[0,(Srated*pf)*2],linestyle='dotted',color='y'))
for pf in [0.8,0.6,0.4,0.2]:
    ax.annotate(str(pf),xy=(np.sqrt(Srated**2-(Srated*pf)**2),(Srated*pf)),horizontalalignment='left')
    ax.annotate(str(pf),xy=(-1*np.sqrt(Srated**2-(Srated*pf)**2),(Srated*pf)),horizontalalignment='right')
for mvapu in [0.2,0.4,0.6,0.8,1]:
    ax.add_line(plt.Line2D(stator_curveA[:,1]*mvapu,stator_curveA[:,0]*mvapu,linestyle='dotted',color='k'))
for qlvals in range(0,4):
    if show_Qlim[qlvals]:
        ax.add_line(plt.Line2D([Qlim[qlvals],Qlim[qlvals]],[0,Srated+(Srated*0.1*(qlvals+1))],linestyle='dashed',color='orange'))
        ax.annotate(trip_labels[qlvals],xy=(Qlim[qlvals],Srated+(Srated*0.1*(qlvals+1))),horizontalalignment='left')

ax.add_line(comp_curve)
ax.set_yticks([0,0.2*Srated,0.4*Srated,0.6*Srated,0.8*Srated,1.0*Srated,1.2*Srated],['0','0.2','0.4','0.6','0.8','1.0','1.2'])
ax.set_xticks([-1.2*Srated,-1.0*Srated,-0.8*Srated,-0.6*Srated,-0.4*Srated,-0.2*Srated,0,0.2*Srated,0.4*Srated,0.6*Srated,0.8*Srated,1.0*Srated,1.2*Srated],['-1.2','-1.0','-0.8','-0.6','-0.4','-0.2','0','0.2','0.4','0.6','0.8','1.0','1.2'])

fig.text(0.1, 0.985, 'Generator Data', fontsize=8)
fig.text(0.1, 0.965, 'Power: '+str(np.round(Srated/1e3,1))+'kVA', fontsize=8)
fig.text(0.1, 0.945, 'Voltage: '+str(np.round(Vterminal,1))+'V', fontsize=8)
fig.text(0.1, 0.925, 'Poles: '+str(int(poles)), fontsize=8)
fig.text(0.1, 0.905, 'Power Factor : '+str(PowerFactor), fontsize=8)
fig.text(0.25, 0.965, 'Synchronous Reactance: '+str(Xsync), fontsize=8)
fig.text(0.25, 0.945, 'Max Power: '+str(np.round(Rmax/1e3,1))+'kW', fontsize=8)
fig.text(0.25, 0.925, 'Full Load Current: '+str(np.round(Ifl,1))+'A', fontsize=8)
fig.text(0.6, 0.965, 'PT Ratio: ' + str(ptr), fontsize=8)
fig.text(0.6, 0.945, 'CT Ratio: ' + str(ctr), fontsize=8)
fig.text(0.6, 0.925, 'Full Load Current Secondary: '+str(np.round(Ifl/ctr,1))+'A', fontsize=8)

plt.savefig('CapabilityCurve.jpg', dpi=600, facecolor='w', edgecolor='w',
        orientation='portrait', format='jpg',
        transparent=False, bbox_inches=None, pad_inches=0.1)
#plt.close()

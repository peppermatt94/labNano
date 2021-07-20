# -*- coding: utf-8 -*-
"""
Created on Mon May 17 22:50:42 2021

@author: pepermatt94
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import matplotlib as mpl
from LabLibrary import sci_notation_as_Benini_want
import LabLibrary as myLab
import sympy as sp
from sympy.solvers import solve
mpl.style.use('classic')

path = "C:\\Users\\pepermatt94\OneDrive\\Libri Magistrale\\nanoscience and nanotechnology\\LabNano\\AFM\\"
#the data of the "Resonance1.csv" are connected to the calibration measure (resonance frequency)
data = np.genfromtxt(path + "Resonance1.csv", delimiter = ",", skip_header = 4)

def FreeDampOscillation(omega, omega0, u, Q):
    return u*omega0**2/np.sqrt((omega0**2-omega**2)**2 +(omega*omega0/Q)**2)

FreeResonance = pd.DataFrame({"Pulse": data[:,0], "Amplitude": data[:,1]})
FreeResonance.Pulse = FreeResonance.Pulse*1000
FreeResonance.Amplitude = FreeResonance.Amplitude*1e-9
plt.plot(FreeResonance.Pulse/1000, FreeResonance.Amplitude*1e9)

#Some initial condition
Max = FreeResonance.sort_values("Amplitude", ascending = False).head(1)

parameters, cov = curve_fit(FreeDampOscillation,FreeResonance.Pulse, 
                            FreeResonance.Amplitude,p0=[Max.Pulse.iloc[0], 10,27.3])



plt.plot(FreeResonance.Pulse/1000, FreeDampOscillation(FreeResonance.Pulse, *parameters)*1e9)
plt.title("FREE RESONANCE CURVE")
plt.xlabel("FREQUENCY (kHz)")
plt.ylabel("AMPLITUDE (V)")
plt.legend(["Experimental", "Fit"])
FitFormula = r"$A(\omega)=\frac{u_0\omega_0}{\sqrt{(\omega^2_0-\omega^2)^2+\frac{\omega\omega_0}{Q}}}$"
plt.xlim([6700/1000,8200/1000])
#plt.annotate(FitFormula, xy = (7550,1.9e-9), fontsize =14)
ParametersFit1 = r"$\omega_0$ = "+ sci_notation_as_Benini_want(parameters[0]/1000)+ " kHz"
ParametersFit2 = r"$u_0$ = " + sci_notation_as_Benini_want(parameters[1]*1e9) + " V" 
ParametersFit3 = r"$Q$ = " + sci_notation_as_Benini_want(parameters[2]) 
#plt.annotate(ParametersFit1 +"\n" +ParametersFit2 + "\n" + ParametersFit3,
#             xy = (7630,1.1e-9), fontsize = 14)

plt.text(6790/1000, 1.80,  ParametersFit1 +"\n" +ParametersFit2 +
         "\n" + ParametersFit3, style='italic', fontsize= 16,
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})
plt.savefig(path + "FreeResonance.png", dpi=400, bbox_inches = "tight")


#data for the Resonance in contact with the sample
path = "C:\\Users\\pepermatt94\OneDrive\\Libri Magistrale\\nanoscience and nanotechnology\\LabNano\\AFM\\"
data = np.genfromtxt(path+"ResonanceSpectroscopy.csv", delimiter =",", skip_header = 4)
ContactResonance = pd.DataFrame({"ZScanner": data[:,0], "Amplitude": data[:,1]})

tmp = ContactResonance
ContactResonance.ZScanner =  ContactResonance.ZScanner -57

Force = plt.figure(figsize = (14,7))
Grid = Force.add_gridspec(2, 1)
Amplitude_z = Force.add_subplot(Grid[0,0])
Force_z = Force.add_subplot(Grid[1,0])

Amplitude_z.plot(ContactResonance.ZScanner, ContactResonance.Amplitude, ".")
Amplitude_z.set_title("CONTACT RESONANCE CURVE")
Amplitude_z.set_xlabel("Z SCANNER (nm)")
Amplitude_z.set_ylabel("NORMALIZED AMPLITUDE")
Amplitude_z.vlines(0, ymin = 0,ymax = 1)

Amplitude_z.set_xlim(-60,45)
Amplitude_z.set_ylim(0.3,0.65)
Amplitude_z.fill_between(np.arange(0,70,0.1), 1, -1 , color ="pink", alpha =0.4,zorder = 1)
Amplitude_z.annotate("REPULSIVE\nREGION", xy = (8,0.55))

#found dfdz with an equation

dfdz = sp.Symbol("x")
omega0 = parameters[0]
u=parameters[1]
Q = parameters[2]
omega = omega0

#ContactResonance.ZScanner =  ContactResonance.ZScanner +210

def F(omega0, A, omega, Q):
    return np.sqrt((omega0**4/A**2)-(omega*omega0/Q)**2) - omega0**2+omega**2

#SOLVING EQUATION:
dforcedz  = [F(omega0, i, omega,Q) for i in ContactResonance.Amplitude]
index = np.empty(394)
integration=np.empty(394)

df = np.flip(dforcedz)
for count  in range(390):
    #dz = index[count]-index[count-1]
    index[count] =  (ContactResonance.ZScanner.iloc[(-1)]-ContactResonance.ZScanner[(393-count)])
    dz = -(ContactResonance.ZScanner.iloc[(count)]-ContactResonance.ZScanner.iloc[(count+1)])
    integration[count] = dforcedz[count+1]*dz*12e-9

integration[np.where(integration>1.4)[0][0]] = 0.6
integration = np.flip(integration)
Force_z.plot(index[4:388],integration[4:388], ".")

Force_z.set_ylabel("F (nN)")
Force_z.set_xlabel(r"$\delta$ (nm)")
plt.savefig("Force.png", dpi =250, bbox_tight = True)
plt.show()
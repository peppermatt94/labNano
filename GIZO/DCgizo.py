# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:56:03 2021

@author: pepermatt94
"""
import pylab as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
import glob
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib as mpl
import sys
from LabLibrary import sci_notation_as_Benini_want
import LabLibrary as myLib
import os 
from uncertainties import unumpy as unp
from uncertainties import ufloat

#mpl.style.use('classic')

def Func_UnoSuS(Vsg, S, Von):
    return S*Vsg+Von
    
    
def CurrentDrain(Vsg, Vt, mu):
    Vd = transfer[1]
    c = 54*10**(-9)
    W = 320*10**(-6)
    L = 20*10**(-6)
    return (1/L)*c*mu*W*(Vsg-Vt)*Vd 
    
#IN d we put all the data from the file . See LabLibrary for this method
d = myLib.charge_ALL_file_in_dir(os.getcwd(),1)
direction = 0
#I need some set up on the data.
for dati in d.keys():
    a = np.array_split(d[dati],2)
    d[dati] = a[direction]                            #I SELECT ONLY THE GO BRANCHES. TO SELECT THE BACK BRANCHES; WRITE a[1]. 
    d[dati].columns = [c.replace(' - ', '') for c in d[dati].columns]    

"""
THE FOLLOWING BLOCK OF CODE IS DEDICATED TO THE CREATION OF THE GRAPHICAL 
ASPECT. 
I CREATE A GLOBAL FIGURE (characteristic) THEN I CREATE THE SUBPLOT IN THE 
BIG GRAPH: GENERAL WILL CONTAIN ALL THE NOT ANALYZED DATA, THE OTHERS THE 
ANALYZED DATA AND THE RESULTS

"""


characteristic = plt.figure(figsize = (14,7))
Grid = characteristic.add_gridspec(3, 4)
Generale = characteristic.add_subplot(Grid[0:3,0:2])
Zero_1_log = characteristic.add_subplot(Grid[0,2])
Cinque_log = characteristic.add_subplot(Grid[1,2])
Dieci_log = characteristic.add_subplot(Grid[2,2])
tag_01 = characteristic.add_subplot(Grid[0,3])
tag_5 = characteristic.add_subplot(Grid[1,3])
tag_10 = characteristic.add_subplot(Grid[2,3])
characteristic.subplots_adjust(hspace=0)
characteristic.subplots_adjust(wspace=0)

#general plots of the transfer function
parametri = {"0.1 V" : {"params": [], "cov" : []}, "5 V" : {"params": [], "cov" : []}, "10 V" : {"params": [], "cov" : []}}
parametri = pd.DataFrame(parametri)


for i in [(d["transfer0_1"], "blue"),  (d["transfer5"], "green"),(d["transfer10"],"red")]:
    Generale.semilogy("Ch1Volts", "Ch2Amps", data = i[0], color = i[1] )     

Generale.legend(["VD = 0.1 V", "VD = 5 V", "VD = 10 V"], loc = "upper left")
Generale.get_yaxis().get_major_formatter().labelOnlyBase = True
if direction == 0:
    #Generale.set_yticklabels(["",-12,-11,-10,-9,-8,-7,-6,-5,-4])
    Generale.set_yticklabels(["","",-11,-9,-7,-5])
else:
    Generale.set_yticklabels(["","",-13,-11,-9,-7,-5])
for transfer in [(d["transfer0_1"], 0.1, "blue", 10**(-7.5), -4.3,-3.6,10**(-14), 10**(-7.5), Zero_1_log,tag_01), (d["transfer5"],5,"green",10**(-7.5),-4.1,-3.4,10**(-12.5), 10**(-7), Cinque_log,tag_5), (d["transfer10"], 10, "red", 10**(-7.5),-4.2,-3.2,10**(-12.5), 10**(-7), Dieci_log,tag_10)]:
   
    transfer[8].semilogy("Ch1Volts", "Ch2Amps", data = transfer[0], c=transfer[2])
    
    """
UNCEERTAINTY CALCULATION
THE IDEA IS TO CREATE A SERIES OF IF IN ORDER TO FULLFILL THE SIGMA ARRAY WITH DIFFERENT
ERRORS RESPECT TO THE VALUE OF THE MEASURE.

AFTER THIS COMPUTATION, THE sigma ARRAY IS ADD AS A COLUMN TO DATA DATAFRAME 
WITH transfer.assign() METHOD
"""    


    sigma = np.zeros(len(transfer[0].Ch2Amps))
    count = 0
    for dato in transfer[0].Ch2Amps:
        if abs(dato) < 100e-9:
            sigma[count] = 50e-12
            count+=1
        if abs(dato) < 1000e-9 and abs(dato) > 100e-9:
            sigma[count] = 100e-12
            count+=1

        if abs(dato) < 10e-6 and  abs(dato)>1e-6:
            sigma[count] = 0.5e-9
            count+=1  

        if abs(dato) < 100e-6 and  abs(dato)>10e-6:
            sigma[count] = 1.5e-9
            count+=1   
        if abs(dato)>100e-6:
            sigma[count] = 25e-9
            count+=1   
 
    Data_With_error = transfer[0].assign(Errors =  sigma) ### I NEED ANOTHER DATAFRAME data_with_error since i cannot overlap tranfer[0]

    #data for the fit 
    xy = Data_With_error.query(f"Ch1Volts>{transfer[4]}&Ch1Volts<{transfer[5]}")   ### WITH .query METHOD I CAN SELECT ONLY CERTAIN DATA (THAT I NEED)

    param = np.polyfit( xy.Ch1Volts, np.log10(xy.Ch2Amps), 1)      #I NEED FIRST POLYFIT AS STARTING PARAMETER; THEN I WILL USE curve_fit METHOD (NO MORE polyfit, it haven't Covariances calculation)
    params, cov = curve_fit(Func_UnoSuS, xy.Ch1Volts, np.log10(xy.Ch2Amps), sigma = np.log10(xy.Errors), absolute_sigma=True, p0 = [param[1],param[0]]) 
    x = np.arange(transfer[4],transfer[5]+1,0.1)
    transfer[8].semilogy(x, 10**(Func_UnoSuS(x, *params)), c = "black")   ##PLOT OF THE RESULT OF THE FIT
    transfer[8].set_yticks([])
    transfer[8].set_xticks([])

    #Parameters found  
    S = ufloat(param[0], cov[0][0])
    UnoSuS = S**(-1)
    Ioff = np.mean(transfer[0].Ch2Amps[transfer[0].Ch1Volts<-4.4])
    Ioff_std = np.std(transfer[0].Ch2Amps[transfer[0].Ch1Volts<-4.4])
    Ioff = ufloat(Ioff, Ioff_std)

    # Ion as a mean
    # Ion = np.mean(transfer[0].Ch2Amps[transfer[0].Ch1Volts>5])
    # Ion_std = np.std(transfer[0].Ch2Amps[transfer[0].Ch1Volts>5])
    # Ion = ufloat(Ion, Ion_std)

    #Ion as last current information
    if direction == 1:
        Ion = transfer[0].Ch2Amps.iloc[0]
        Ion_std = abs(transfer[0].Ch2Amps.iloc[0]-transfer[0].Ch2Amps.iloc[1])
    else:
        Ion = transfer[0].Ch2Amps.iloc[-1]
        Ion_std = abs(transfer[0].Ch2Amps.iloc[-1]-transfer[0].Ch2Amps.iloc[-2])

    Ion = ufloat(Ion, Ion_std)*10**6
    Von = transfer[0].query(f"Ch2Amps>{4*Ioff.n}&Ch1Volts<{transfer[5]}") #I FOUND Von AS THE FIRST V IN THE REGION WHERE THE I IS SIGNIFICATLY DIFFERENT FROM Ioff (4*Ioff)
    Von_error = 0.1   #error as division
    Von = Von["Ch1Volts"].iloc[0]
    Von = ufloat(Von, Von_error)
    Ioff = Ioff*10**12
    
    ##STRING TO PRINT IN THE GRAPH
    if direction == 1:
        if transfer[2] == "blue":
            Volt_Per_Decade = r"S = " +  sci_notation_as_Benini_want(UnoSuS.n, error = UnoSuS.s, cifre = 3) + r"$ \frac{V}{dec}$"
            I_on = r"$I_{on} =$"+  sci_notation_as_Benini_want(Ion.n, error = Ion.s) + r"$ \mu A$"
            I_off = r"$I_{off} =$" +  sci_notation_as_Benini_want(Ioff.n, error = Ioff.s, cifre = 1) + " pA"
        elif transfer[2] == "green":
            Volt_Per_Decade = r"$S = $" +  sci_notation_as_Benini_want(UnoSuS.n, error = UnoSuS.s, cifre = 3) + r"$ \frac{V}{dec}$"
            I_on = r"$I_{on} =$"+  sci_notation_as_Benini_want(Ion.n, error = Ion.s, cifre = 1) + r" $ \mu A$"
            I_off = r"$I_{off} =$" +  sci_notation_as_Benini_want(Ioff.n, error = Ioff.s, cifre = 1) + " pA"
        else:
            Volt_Per_Decade = r"$S = $" +  sci_notation_as_Benini_want(UnoSuS.n, error = UnoSuS.s, cifre = 2) + r"$ \frac{V}{dec}$"
            I_on = r"$I_{on} =($"+  "99"+r"$\pm$"+"2)" + r" $ \mu A$"
            I_off = r"$I_{off} =$" +  sci_notation_as_Benini_want(Ioff.n, error = Ioff.s, cifre = 1) + " pA"
      
        #I_off = r"$I_{off} =$" +  sci_notation_as_Benini_want(Ioff.n, error = Ioff.s, cifre = 2) + " pA"
        V_on = r"$V_{on}$ =" + sci_notation_as_Benini_want(Von.n, error = Von.s, cifre = 1) + " V"
    else:
        if transfer[2] == "green":
            Volt_Per_Decade = r"$S = $" +  sci_notation_as_Benini_want(UnoSuS.n, error = UnoSuS.s, cifre = 2) + r"$ \frac{V}{dec}$"
            I_on = r"$I_{on} =$"+  sci_notation_as_Benini_want(Ion.n, error = Ion.s, cifre = 0) + r" $ \mu A$"
        elif transfer[2] == "red":
            Volt_Per_Decade = r"$S = $" +  sci_notation_as_Benini_want(UnoSuS.n, error = UnoSuS.s, cifre = 2) + r"$ \frac{V}{dec}$"
            I_on = r"$I_{on} =$"+  sci_notation_as_Benini_want(Ion.n, error = Ion.s, cifre = 0) + r" $\mu A$"
        elif transfer[2] == "blue":
            Volt_Per_Decade = r"$S = $" +  sci_notation_as_Benini_want(UnoSuS.n, error = UnoSuS.s, cifre = 2) + r"$ \frac{V}{dec}$"
            I_on = r"$I_{on} =$"+  sci_notation_as_Benini_want(Ion.n, error = Ion.s, cifre = 2) + r" $\mu A$"
        
        I_off = r"$I_{off} =$" +  sci_notation_as_Benini_want(Ioff.n, error = Ioff.s, cifre = 0) + " pA"
        V_on = r"$V_{on}$ =" + sci_notation_as_Benini_want(Von.n, error = Von.s, cifre = 1) + " V"

    transfer[9].plot()
    transfer[9].annotate(Volt_Per_Decade + "\n"+I_on + "\n"+I_off + "\n"+ V_on, xy = (-0.01, 0.1), fontsize = 14) ##INSET OF THE RESULTS
    transfer[9].set_yticks([])
    transfer[9].set_xticks([])

    params = [round(i,2) for i in params]
    params = [round(i,2) for i in params]
    cov = [cov[0][0], cov[1][1]]
    cov = [round(i,4) for i in cov]
    parametri[str(transfer[1]) + " V"]["params"] = params 
    parametri[str(transfer[1]) + " V"]["cov"] = cov 


Generale.set_xlabel("GATE VOLTAGE (V)", size = 20)
Generale.set_ylabel( r"$log_{10}[I(A)]$", size = 20)
if direction == 0:
    characteristic.suptitle("SEMILOGARITHMIC CHARACTERISTICS, FORWARD SWEEP", size = 20) 
    characteristic.savefig(f"semilogTransfer.png", dpi = 250, bbox_inches = "tight") 
elif direction==1:
    characteristic.suptitle("SEMILOGARITHMIC CHARACTERISTICS, BACKWARD SWEEP", size = 20) 
    characteristic.savefig(f"semilogTransferBACK.png", dpi = 250, bbox_inches = "tight") 
plt.show()

for i in [d["output1"], d["output2"],d["output3"],d["output4"]]:
    i.Ch2Amps = i.Ch2Amps*10**6
    plt.plot(i.columns[0], i.columns[1], data = i)
   
plt.legend(["VG = 1 V", "VG = 2 V", "VG = 3 V", "VG = 4 V", "VG = 5 V"], loc = "upper left")
plt.xlabel("DRAIN VOLTAGE (V)")
plt.ylabel(r"DRAIN CURRENT ( $\mu $A)")
plt.title("OUTPUT VOLTAGE IN THE DRAIN")
plt.savefig("outputRitorno.png", dpi =250)
plt.show()

"""
THE FOLLOWING IS THE PLOT OF THE DECIMAL CHARACTERISTIC
TO UNDERSTAND, SEE PREVIOUS COMMENTS. THE PROCEDURE IS THE SAME
"""

characteristic = plt.figure(figsize = (14,7))
Grid = characteristic.add_gridspec(3, 4)
Generale = characteristic.add_subplot(Grid[0:3,0:2])
Zero_1_dec = characteristic.add_subplot(Grid[0,2])
Cinque_dec = characteristic.add_subplot(Grid[1,2])
Dieci_dec = characteristic.add_subplot(Grid[2,2])
tag_01 = characteristic.add_subplot(Grid[0,3])
tag_5 = characteristic.add_subplot(Grid[1,3])
tag_10 = characteristic.add_subplot(Grid[2,3])
characteristic.subplots_adjust(hspace=0)
characteristic.subplots_adjust(wspace=0)

for i in [(d["transfer0_1"], "blue"), (d["transfer5"], "green"), (d["transfer10"],"red")]:
    Generale.plot("Ch1Volts", "Ch2Amps", data = i[0], color = i[1] )     

Generale.legend(["VD = 0.1 V", "VD = 5 V", "VD = 10 V"], loc = "upper left")
Generale.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

for transfer in [(d["transfer0_1"], 0.1,"blue", 0.0000020,Zero_1_dec,tag_01), (d["transfer5"],5,"green",0.000020,Cinque_dec,tag_5), (d["transfer10"], 10, "red", 0.00002,Dieci_dec, tag_10)]:

    ##Uncertainty 
    sigma = np.zeros(len(transfer[0].Ch2Amps))
    count = 0
    for dato in transfer[0].Ch2Amps:
        if abs(dato) < 100e-9:
            sigma[count] = 50e-12
            count+=1
        if abs(dato) < 1000e-9 and abs(dato) > 100e-9:
            sigma[count] = 100e-12
            count+=1

        if abs(dato) < 10e-6 and  abs(dato)>1e-6:
            sigma[count] = 0.5e-9
            count+=1  

        if abs(dato) < 100e-6 and  abs(dato)>10e-6:
            sigma[count] = 1.5e-9
            count+=1   
        if abs(dato)>100e-6:
            sigma[count] = 25e-9
            count+=1   

    Data_With_error = transfer[0].assign(Errors =  sigma)

    #data for the fit 
    transfer[4].plot("Ch1Volts", "Ch2Amps", data = transfer[0], c = transfer[2])
    xy = Data_With_error.query(f"Ch1Volts>7&Ch1Volts<10")

    params, cov = curve_fit(CurrentDrain, xy.Ch1Volts, xy.Ch2Amps, p0=[3, 50])
    x = np.arange(4,10,0.1)
    transfer[4].plot(x, CurrentDrain(x, *params), c = "black")
    transfer[4].set_yticks([])
    transfer[4].set_xticks([])

    if direction == 0:
        if transfer[2]!="blue":
            mu = r"$\mu = $ "+ sci_notation_as_Benini_want(params[1], error = cov[1][1], cifre = 3) +r" $\frac{cm^2}{Vs}$" 
            Vt = r"$V_{t} =$" + sci_notation_as_Benini_want(params[0], error = cov[0][0], cifre = 3) + " V"
        else:
            mu = r"$\mu = $ "+ sci_notation_as_Benini_want(params[1], error = cov[1][1], cifre = 4) +r" $\frac{cm^2}{Vs}$" 
            Vt = r"$V_{t} =$" + sci_notation_as_Benini_want(params[0], error = cov[0][0], cifre = 4) + " V"
    else:
        if transfer[2]!="red":
            mu = r"$\mu = $ "+ sci_notation_as_Benini_want(params[1], error = cov[1][1], cifre = 3) +r" $\frac{cm^2}{Vs}$" 
            Vt = r"$V_{t} =$" + sci_notation_as_Benini_want(params[0], error = cov[0][0], cifre = 3) + " V"
        else :
            mu = r"$\mu = $ "+ sci_notation_as_Benini_want(params[1], error = cov[1][1], cifre = 4) +r" $\frac{cm^2}{Vs}$" 
            Vt = r"$V_{t} =$" + sci_notation_as_Benini_want(params[0], error = cov[0][0], cifre = 4) + " V"

    transfer[5].plot()
    transfer[5].annotate( mu + "\n"+Vt, xy = (-0.025,0.3), fontsize = 15)
    transfer[5].set_yticks([])
    transfer[5].set_xticks([])


    params = [round(i,2) for i in params]
    cov = [cov[0][0], cov[1][1]]
    cov = [round(i,4) for i in cov]
    parametri[str(transfer[1]) + " V"]["params"] = params 
    parametri[str(transfer[1]) + " V"]["cov"] = cov 

Generale.set_xlim(-10, 11)
Generale.set_ylim(-1e-5, 1.1e-4)
Generale.set_xlabel("GATE VOLTAGE (V)", size = 20)
Generale.set_ylabel("DRAIN CURRENT (A)", size = 20)

if direction == 0:
    characteristic.suptitle("CHARACTERISTICS OF MOSFET, FORWARD BRANCHES", size = 20)
    characteristic.savefig(f"decimalTransfer.png", dpi = 250, bbox_inches = "tight")
elif direction==1:
    characteristic.suptitle("CHARACTERISTICS OF MOSFET, BACKWARD BRANCHES", size = 20) 
    characteristic.savefig(f"decimalTransferBACK.png", dpi = 250, bbox_inches = "tight") 
plt.show()
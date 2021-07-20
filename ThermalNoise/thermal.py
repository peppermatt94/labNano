# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:05:59 2021

@author: pepermatt94
"""
import pandas as pd
import numpy as np 
import glob
import matplotlib.pylab as plt
import sympy as sp
from sympy.solvers import solve
import sys
import os
from LabLibrary import sci_notation_as_Benini_want
import LabLibrary as myLib
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable

Path = "C:\\Users\\pepermatt94\\OneDrive\\Libri Magistrale\\nanoscience and nanotechnology\\labNano\\ThermalGroup2\\Data-20210710T162629Z-001\\Data"
d = myLib.charge_ALL_file_in_dir(Path,0, sep = ",")
for dati in d.keys(): 
    d[dati].columns = [c.replace(' ', '') for c in d[dati].columns]    
    d[dati].columns = [c.replace('(', '') for c in d[dati].columns]
    d[dati].columns = [c.replace(')', '') for c in d[dati].columns]


d["whiteNOISE_FFT_amp_4mV"].Trace1VHz = 20*np.log10(d["whiteNOISE_FFT_amp_4mV"].Trace1VHz/d["whiteNOISE_FFT_noAmp_4mV"].Trace1VHz)

#plt.plot(d["whiteNOISE_FFT_amp_4mV"].FrequencyHz, d["whiteNOISE_FFT_amp_4mV"].Trace1VHz, c = "brown")
xyConst = d["whiteNOISE_FFT_amp_4mV"].query(f"FrequencyHz>200&FrequencyHz<2200")   ### WITH .query METHOD I CAN SELECT ONLY CERTAIN DATA (THAT I NEED)
xyLinear = d["whiteNOISE_FFT_amp_4mV"].query(f"FrequencyHz>5000&FrequencyHz<10000")   ### WITH .query METHOD I CAN SELECT ONLY CERTAIN DATA (THAT I NEED)
   
Gain = np.polyfit(xyConst.FrequencyHz, xyConst.Trace1VHz, deg = 0)
Linear = np.polyfit(xyLinear.FrequencyHz, xyLinear.Trace1VHz, deg = 1)
x = sp.Symbol("x")
f = (Gain - Linear[0]*x-Linear[1])
fitUtility = solve(f)

xConst = np.arange(-10, fitUtility[x], 0.1)
xLin = np.arange(fitUtility[x], 30000, 0.1)
fitUtility = fitUtility[x]
plt.plot(d["whiteNOISE_FFT_amp_4mV"].FrequencyHz, d["whiteNOISE_FFT_amp_4mV"].Trace1VHz, c = "brown")
plt.plot(xConst, xConst*0 +Gain,linestyle = "dashed", alpha = 1)
plt.plot(xLin, xLin*Linear[0] +Linear[1],linestyle = "dashed", alpha = 1)
plt.xlabel("FREQUENCY (HZ)")
plt.ylabel("AMPLIFICATION (dB)")
plt.title("BANDWIDTH AND GAIN OF THE OPAMP 4mV")

x = sp.Symbol("x")
f = (Gain-3 - Linear[0]*x-Linear[1])
bandwidth = solve(f)
bandwidthText = "Bandwidth = " + sci_notation_as_Benini_want(bandwidth[x]/1000, cifre = 2) + " kHz"
plt.ylim(50, 60)
plt.vlines(bandwidth[x], -10,510,linestyle = "dashed", alpha = 0.3)
constant = lambda x: x*0+Gain-3
bandwidth = bandwidth[x]
x = np.arange(-10, bandwidth, 0.1)
plt.xlim(-1, 20000)
plt.plot(x, constant(x), linestyle = "dashed", alpha = 0.3)

gainText = "Gain = " + sci_notation_as_Benini_want(Gain[0], cifre =3) + " dB"
x1 = np.arange(0,fitUtility,0.1)
x2 = np.arange(fitUtility, bandwidth, 0.1)
plt.annotate("", xy=(2500,57.5),xytext=(2400,54.5), arrowprops=dict(arrowstyle="<->", color="0.5",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle="arc3,rad=-0.3",
                                ))

plt.annotate("-3dB", xy = (2000,55.5), fontsize = 14)
plt.text(0.98*10**4, 58, gainText +"\n" +bandwidthText
        , style='italic', fontsize= 14,
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})
plt.savefig("bandwidth4.png", dpi = 250)
plt.show()

d["whiteNOISE_FFT_amp_5mV"].Trace1VHz = 20*np.log10(d["whiteNOISE_FFT_amp_5mV"].Trace1VHz/d["whiteNOISE_FFT_noAmp_5mV"].Trace1VHz)

xyConst = d["whiteNOISE_FFT_amp_5mV"].query(f"FrequencyHz>200&FrequencyHz<2200")   ### WITH .query METHOD I CAN SELECT ONLY CERTAIN DATA (THAT I NEED)
xyLinear = d["whiteNOISE_FFT_amp_5mV"].query(f"FrequencyHz>5000&FrequencyHz<10000")   ### WITH .query METHOD I CAN SELECT ONLY CERTAIN DATA (THAT I NEED)
Gain = np.polyfit(xyConst.FrequencyHz, xyConst.Trace1VHz, deg = 0)
Linear = np.polyfit(xyLinear.FrequencyHz, xyLinear.Trace1VHz, deg = 1)

x = sp.Symbol("x")
f = (Gain - Linear[0]*x-Linear[1])
fitUtility = solve(f)

xConst = np.arange(-10, fitUtility[x], 0.1)
xLin = np.arange(fitUtility[x], 30000, 0.1)
fitUtility = fitUtility[x]

plt.plot(d["whiteNOISE_FFT_amp_5mV"].FrequencyHz, d["whiteNOISE_FFT_amp_5mV"].Trace1VHz, c = "brown")
plt.plot(xConst, xConst*0 +Gain,  alpha = 1)
plt.plot(xLin, xLin*Linear[0] +Linear[1], alpha = 1)
plt.xlabel("FREQUENCY (HZ)")
plt.ylabel("AMPLIFICATION (dB)")
plt.title("BANDWIDTH AND GAIN OF THE OPAMP 5mV")



gainText = "Gain = " + sci_notation_as_Benini_want(Gain[0], cifre =3) + " dB"

x = sp.Symbol("x")
f = (Gain-3 - Linear[0]*x-Linear[1])
bandwidth = solve(f)

bandwidthText = "Bandwidth = " + sci_notation_as_Benini_want(bandwidth[x]/1000, cifre = 2) + " kHz"

plt.ylim(50, 60)
plt.vlines(bandwidth[x], -10,100, linestyle = "dashed", alpha = 0.3)
constant = lambda x: x*0+Gain-3
bandwidth = bandwidth[x]
x = np.arange(-10, bandwidth , 0.1)
plt.annotate("", xy=(2500,57.1),xytext=(2400,54.1), arrowprops=dict(arrowstyle="<->", color="0.5",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle="arc3,rad=-0.3",
                                ))

plt.annotate("-3dB", xy = (2000,55), fontsize = 14)
plt.text(0.98*10**4, 58, gainText +"\n" +bandwidthText
        , style='italic', fontsize= 14,
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})
plt.plot(x, constant(x), linestyle = "dashed", alpha = 0.3)
plt.xlim(-1, 20000)
plt.savefig("bandwidth5.png", dpi = 250)
plt.show()



RMS = {}

RMSgraphs = plt.figure()#figsize = (7,7))
each = RMSgraphs.add_gridspec(3, 1)
RMS["1kGraph"] = RMSgraphs.add_subplot(each[0,0])
RMS["10kGraph"] = RMSgraphs.add_subplot(each[1,0])
RMS["200kGraph"] = RMSgraphs.add_subplot(each[2,0])
RMSgraphs.subplots_adjust(wspace=0)
RMSgraphs.subplots_adjust(hspace=0.5)

RMS["1kGraph"].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
RMS["10kGraph"].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
RMS["200kGraph"].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

RMS["10kGraph"].plot(d["rms_10k"].Times, (1000000)*d["rms_10k"].Channel1V/10**(57.49/20))
#RMS["10k"] = (sum(pow(d["rms_10k"].Channel1V,2)))/(d["rms_10k"].Times.iloc[-1]-d["rms_10k"].Times.iloc[0])
RMS["10k"] = np.sqrt((np.mean(pow(d["rms_10k"].Channel1V,2))))/10**(57.49/20)
RMS["10kb"] = (RMS["10k"])**2/(4*300*10000*bandwidth)
RMS["10kGraph"].set_xticklabels({})
RMS["1kGraph"].set_xticklabels({})
RMS["1kGraph"].set_xticks([])
RMS["10kGraph"].set_xticks([])
title = RMS["10kGraph"].set_title("R = 10 K\t"+r" $V_{RMS}$ = "+f"{sci_notation_as_Benini_want(RMS['10k'])} V",position=(.55, 0.92),
                          bbox=(dict(facecolor = "pink", alpha = 0.5)),
                          fontfamily = "monospace", verticalalignment="bottom", horizontalalignment="center")
title._bbox_patch._mutation_aspect = 0.08
title.get_bbox_patch().set_boxstyle("square", pad=1.9)

RMS["1kGraph"].plot(d["rms_1k"].Times, (1000000)*d["rms_1k"].Channel1V/10**(57.49/20))
#RMS["1k"] = (sum(pow(d["rms_1k"].Channel1V,2)))/(d["rms_1k"].Times.iloc[-1]-d["rms_1k"].Times.iloc[0])
RMS["1k"] = np.sqrt((np.mean(pow(d["rms_1k"].Channel1V,2))))/10**(57.49/20)
RMS["1kb"] = (RMS["1k"])**2/(4*300*1000*bandwidth)

title = RMS["1kGraph"].set_title("R = 1 K \t" +r" $V_{RMS}$ = "+f"{sci_notation_as_Benini_want(RMS['1k'])} V", position=(.56, 0.942),
             bbox=(dict(facecolor = "pink", alpha = 0.5)),
             verticalalignment="bottom", horizontalalignment="center")
title._bbox_patch._mutation_aspect = 0.06
title.get_bbox_patch().set_boxstyle("square", pad=3.7)

RMS["200kGraph"].plot(d["rms_200k"].Times, (1000000)*d["rms_200k"].Channel1V/10**(57.49/20))
#RMS["200k"] = (sum(pow(d["rms_200k"].Channel1V,2)))/(d["rms_200k"].Times.iloc[-1]-d["rms_200k"].Times.iloc[0])
RMS["200k"] = np.sqrt((np.mean(pow(d["rms_200k"].Channel1V,2))))/10**(57.49/20)

RMS["200kb"] = (RMS["200k"])**2/(4*300*200000*bandwidth)
title = RMS["200kGraph"].set_title("R = 200 K\t" +r" $V_{RMS}$ = "+f"{sci_notation_as_Benini_want(RMS['200k'])} V",position=(.55, 0.995),
                          bbox=(dict(facecolor = "pink", alpha = 0.5, zorder = 0)), fontfamily = "monospace")
title._bbox_patch._mutation_aspect = 0.07
title.get_bbox_patch().set_boxstyle("square", pad=1.9)

RMSgraphs.suptitle("TIME SIGNAL AND RMS FOR 3 RESISTORS", y = 1.02)
RMS["200kGraph"].set_xlabel("TIME (s)")
RMS["10kGraph"].set_ylabel("SIGNAL (mV)")
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig("RMStot.png", dpi = 250, bbox_inches = "tight")
plt.show()

FFT={}
FFTgraphs = plt.figure()#figsize = (7,7))
each = FFTgraphs.add_gridspec(3, 1)
FFT["1kGraph"] = FFTgraphs.add_subplot(each[0,0])
FFT["10kGraph"] = FFTgraphs.add_subplot(each[1,0])
FFT["200kGraph"] = FFTgraphs.add_subplot(each[2,0])
#FFTgraphs.subplots_adjust(wspace=0)
FFTgraphs.subplots_adjust(hspace=0.5)

FFT["effective1k"] = (d["fft_1k"]).query(f"FrequencyHz>1000&FrequencyHz<10000")
FFT["effective10k"] = (d["fft_10k"]).query(f"FrequencyHz>1000&FrequencyHz<10000")
FFT["effective200k"] = (d["fft_200k"]).query(f"FrequencyHz>1000&FrequencyHz<10000")


FFT["1k"] =  np.mean(10**((FFT["effective1k"].Trace1dBV)/10))*(10**(57.49/20))**2
FFT["10k"] = np.mean(10**((FFT["effective10k"].Trace1dBV/10)))*(10**(57.49/20))**2
FFT["200k"] =  np.mean(10**((FFT["effective200k"].Trace1dBV)/10))*(10**(57.49/20))**2

FFT["1kb"] = FFT["1k"]/(4*300*1000)
FFT["10kb"] = FFT["10k"]/(4*300*10000)
FFT["200kb"] = FFT["200k"]/(4*300*200000)

FFT["1kGraph"].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
FFT["10kGraph"].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
FFT["200kGraph"].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

FFT["10kGraph"].set_xticklabels({})
FFT["1kGraph"].set_xticklabels({})
FFT["1kGraph"].set_xticks([])
FFT["10kGraph"].set_xticks([])

FFT["10kGraph"].semilogy(d["fft_10k"].FrequencyHz/1000, (10**(d["fft_10k"].Trace1dBV/10))/(749**2))
FFT["1kGraph"].semilogy(d["fft_1k"].FrequencyHz/1000, (10**(d["fft_1k"].Trace1dBV/10))/(749**2))
FFT["200kGraph"].semilogy(d["fft_200k"].FrequencyHz/1000, (10**(d["fft_200k"].Trace1dBV/10))/(749**2))

FFT["10kGraph"].semilogy(FFT["effective10k"].FrequencyHz/1000, (10**(FFT["effective10k"].Trace1dBV/10))/(749**2))
FFT["1kGraph"].semilogy(FFT["effective1k"].FrequencyHz/1000, (10**(FFT["effective1k"].Trace1dBV/10))/(749**2))
FFT["200kGraph"].semilogy(FFT["effective200k"].FrequencyHz/1000, (10**(FFT["effective200k"].Trace1dBV/10))/(749**2))

title = FFT["10kGraph"].set_title(("R = 10 K\t"+r" $v^2_n$"+
                                   r" = 1.53 $10^{-15}$ "+ r"$V^2/Hz$" 
                                   ),#+ r" $K_b = $" +f" = {sci_notation_as_Benini_want(FFT['10kb'])} J/k"),
                                  position=(.55, 0.92),
                          bbox=(dict(facecolor = "pink", alpha = 0.5)),
                          fontfamily = "monospace", verticalalignment="bottom", horizontalalignment="center")
title._bbox_patch._mutation_aspect = 0.08
title.get_bbox_patch().set_boxstyle("square", pad=1.9)

title = FFT["1kGraph"].set_title(("R = 1 K \t"+r" $v^2_n$"+
                                  r" = 0.68 $10^{-15}$ " + r"$V^2/Hz$"
                                  ),#+ r" $K_b = $" +f" = {sci_notation_as_Benini_want(FFT['1kb'])} J/k"),
                                 position=(.55, 0.93),
                          bbox=(dict(facecolor = "pink", alpha = 0.5)),
                          fontfamily = "monospace", verticalalignment="bottom", horizontalalignment="center")
title._bbox_patch._mutation_aspect = 0.08
title.get_bbox_patch().set_boxstyle("square", pad=1.9)

title = FFT["200kGraph"].set_title(("R = 200 K\t"+r" $v^2_n$"+
                                   r" = 9.07 $10^{-15}$ "+ r"$V^2/Hz$"),
                                   #+ "" +r" $K_b = $"+f" {sci_notation_as_Benini_want(FFT['200kb'])} J/k"),
                                   position=(.55, 0.93),
                          bbox=(dict(facecolor = "pink", alpha = 0.5)),
                          fontfamily = "monospace", verticalalignment="bottom", horizontalalignment="center")
title._bbox_patch._mutation_aspect = 0.08
title.get_bbox_patch().set_boxstyle("square", pad=1.9)

FFTgraphs.suptitle("FFT SIGNAL FOR 3 RESISTORS")
FFTgraphs.suptitle("TIME SIGNAL AND RMS FOR 3 RESISTORS", y = 1.02)
FFT["200kGraph"].set_xlabel("FREQUENCY (kHz)")
FFT["10kGraph"].set_ylabel("POWER SPECTRAL " + r"$(V^2/Hz)$")

plt.savefig("fft.png", dpi = 250, bbox_inches = "tight")
plt.show()


solution = pd.DataFrame( {"1k" : pd.Series(
    [FFT["1kb"], RMS["1kb"]],
    index=['FFT', 'RMS']),
    "10k": pd.Series(
    [FFT["10kb"], RMS["10kb"]],
    index=['FFT', 'RMS']),
    "200k": pd.Series(
    [FFT["200kb"], RMS["200kb"]],
    index=['FFT', 'RMS'])
    })

Table_solution = solution.to_latex()
print(Table_solution)
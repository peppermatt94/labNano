import matplotlib.pylab as plt
import pandas as pd
import scipy as sp
import glob
import matplotlib.pylab as plt
import ffmpeg
from scipy.optimize import curve_fit
import matplotlib as mpl
import sys
import sympy as sp
import numpy as np
from LabLibrary import sci_notation_as_Benini_want
mpl.style.use('classic')


dataFromDir = [pd.read_csv(filename, sep = ";") for filename in glob.glob("*.csv")]
filenames = glob.glob("*.csv")
filenames = [k[:-4] for k in filenames]

d = {}

for i in range(len(dataFromDir)):
    d[filenames[i]] = dataFromDir[i]
    #d[filenames[i]].columns = [c.replace(' ', '') for c in d[filenames[i]].columns]
    
    
dataFromDir = [pd.read_csv(filename,  sep = ";") for filename in glob.glob("*.txt")]
filenames = glob.glob("*.txt")
filenames = [k[:-4] for k in filenames]



for i in range(len(dataFromDir)):
    d[filenames[i]] = dataFromDir[i]
    d[filenames[i]].columns = [c.replace(' ', '') for c in d[filenames[i]].columns]

def video(dictionary):
    count = 1
    for key, values in dictionary.items():
        for i in values.columns:
            for j in values.columns:
                if i!=j:
                    try:
                        values.plot(i,j)
                        plt.title(f"{key} and {i} {j}")
                        plt.savefig(f"C:/Users/pepermatt94/Desktop/image/frame{count:03d}.jpg", dpi=250)
                        plt.show()
                        count +=1
                    except:
                        print(f"problem in the {key} and {i}{j}")
    d["FRA_dummy"].plot("Frequency (Hz)", "Z' (\u03A9)")
    import ffmpeg
    (
        ffmpeg
        .input('C:/Users/pepermatt94/Desktop/image/%03d.jpg', pattern_type='glob', framerate=25)
        .output('movie.mp4')
        .run()
    )



# =============================================================================
# DUMMY CELL CALIBRATION
# =============================================================================


#HERE i create the function that is needed to do the fit : znorm and zphase. 
#I took the shape of the function from the program QsapecNG, free, open source
#and lightweight (12 mb). Here RC parallel plus a resistance in series

c1,r2,r1,x = sp.symbols('c1,r2,r1,x', real = True)
expr = ((c1*r2*sp.I*x +1)/(c1*r1*r2*sp.I*x +(r2+r1)))**(-1) #Z of the circuit in question 
f = abs(expr) #Znorm 
znorm=sp.simplify(expr.as_real_imag()[0]**2+expr.as_real_imag()[1]**2)#other way for znorm
zphase=sp.simplify(sp.atan(expr.as_real_imag()[0]/expr.as_real_imag()[1])) #Zphase 
print(sp.latex(znorm))
print(sp.latex(zphase))

#printing the previous function, one can recognize that the following one are the correct choice for the fit
def phase(x, r1,r2,c1):
    return -np.arctan(c1*r1*x + r1/(c1*r2**2*x) + 1/(c1*r2*x))+np.pi/2

def fit(x, Z1,Z2):
    return Z1/np.sqrt((1 + (4*np.pi**2*x**2*Z2**2)))

def norm(x, r1,r2,c1):
    return (np.sqrt((c1**2*r2**2*x**2 + 1)/(c1**2*r1**2*r2**2*x**2 + r1**2 + 2*r1*r2 + r2**2)))**(-1)

Ztot = ((d["FRA_dummy"].iloc[:,3])**(-1) + (d["FRA_dummy"].iloc[:,4])**(-1))**(-1) + d["FRA_dummy"].iloc[:,2]

param, cov = curve_fit(norm, d["FRA_dummy"].iloc[:,1], d["FRA_dummy"].iloc[:,4], p0=[100, 10**6, 9.3e-7])
paramP, cov = curve_fit(phase, d["FRA_dummy"].iloc[:,1], d["FRA_dummy"].iloc[:,5]*2*np.pi/360 , p0=[param[0], param[1], param[2]])
#TEXT FOR THE GRAPHIC
ParametersFit1="R1 = " + sci_notation_as_Benini_want(float(d['FRA_dummy_fitresult'].Value[0].replace(",","."))) + "\u03A9"
ParametersFit2="C = " + sci_notation_as_Benini_want(float(d['FRA_dummy_fitresult'].Value[1].replace(",","."))) + "F"
ParametersFit3="R2 = " + sci_notation_as_Benini_want(float(d['FRA_dummy_fitresult'].Value[2].replace(",","."))) + "\u03A9"
ParametersFit4=r"$\chi^2$ = " + sci_notation_as_Benini_want(float(d['FRA_dummy_fitresult'].Value[4].replace(",",".")))


fig, ax1 = plt.subplots()
#plt.title("DUMMY CELL")

ax1.semilogx(d["FRA_dummy"].iloc[:,1], norm(d["FRA_dummy"].iloc[:,1], *param), color = "blue")
ax1.semilogx(d["FRA_dummy"].iloc[:,1], d["FRA_dummy"].iloc[:,4], "^", color = "blue")
ax1.set_ylabel('|Z| (\u03A9)', color="blue")
ax1.tick_params(axis='y', labelcolor="blue")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid(True)
plt.text(1.5*10**3, 0.65*10**6,  ParametersFit1 +"\n" +ParametersFit3 +"\n" + ParametersFit2 + "\n" + ParametersFit4
         , style='italic', fontsize= 16,
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})
ax2 = ax1.twinx()
ax2.semilogx(d["FRA_dummy"].iloc[:,1], phase(d["FRA_dummy"].iloc[:,1], *paramP), color = "red")
ax2.semilogx(d["FRA_dummy"].iloc[:,1],  d["FRA_dummy"].iloc[:,5]*2*np.pi/360, "^", color = "red")
#ax2.set_ylabel('PHASE (rad)', color="red")
ax2.tick_params(axis='y', labelcolor="red")

ax1.set_xlabel("FREQUENCY (Hz)")
ax1.set_ylabel(" Z (\u03A9)", color = "blue")
ax2.set_ylabel("PHASE (rad)", color = "red")

plt.grid(True)
plt.savefig("DummyFit.png",bbox_inches = "tight")
plt.show()



# =============================================================================
# ITO IMPEDANCE SPECTROSOPY
# =============================================================================

#HERE i create the function that is needed to do the fit : znorm and zphase. 
#I took the shape of the function from the program QsapecNG, free, open source
#and lightweight (12 mb). Here RC parallel

x, R,C = sp.symbols("x,R,C", real = True)
TransferRC = R/(C * R * x*sp.I + 1)
f = abs(TransferRC) #Znorm 
znorm=sp.simplify(TransferRC.as_real_imag()[0]**2+TransferRC.as_real_imag()[1]**2)#other way for znorm
zphase=sp.simplify(sp.atan(TransferRC.as_real_imag()[0]/TransferRC.as_real_imag()[1])) #Zphase 
print(sp.latex(znorm))
print(sp.latex(zphase))

def phaseRC(w,r,c):
    return -np.arctan(c*np.pi*w)+np.pi/2

def normRC(w,r,c):
    return r/np.sqrt(c**2*r**2*w**2+1)

# #WRONG IN THE FOLLOWING  R-(R//C)
# def phaseRC(x, r1,r2,c1):
#     return -np.arctan(c1*r1*x + r1/(c1*r2**2*x) + 1/(c1*r2*x))
# def normRC(x, r1,r2,c1):
#     return (np.sqrt((c1**2*r2**2*x**2 + 1)/(c1**2*r1**2*r2**2*x**2 + r1**2 + 2*r1*r2 + r2**2)))**(-1)

param,cov= curve_fit(normRC, d["FRA_ITO"].iloc[:,1], d["FRA_ITO"].iloc[:,4], p0=[ 38.121, 1.4e-5])
paramP,cov= curve_fit(phaseRC, d["FRA_ITO"].iloc[:,1], d["FRA_ITO"].iloc[:,5]*2*np.pi/360 , p0=[ 38.121, 1.4e-5])
#TEXT FOR THE GRAPHIC
ParametersFit1="R = " + sci_notation_as_Benini_want(float(d['FRA_ITO_fitresults'].Value[0].replace(",","."))) + "\u03A9"
ParametersFit2="C = " + sci_notation_as_Benini_want(float(d['FRA_ITO_fitresults'].Value[1].replace(",","."))) + "F"
ParametersFit4=r"$\chi^2$ = " + sci_notation_as_Benini_want(float(d['FRA_ITO_fitresults'].Value[3].replace(",",".")))

fig, ax1 = plt.subplots()
#plt.title("ITO IN PBS SOLUTION")

ax1.semilogx(d["FRA_ITO"].iloc[:,1],d["FRA_ITO"].iloc[:,4], "^", color = "blue")
ax1.semilogx(d["FRA_ITO"].iloc[:,1],normRC(d["FRA_ITO"].iloc[:,1], *param), color = "blue")

plt.grid(True)
#ax1.set_ylabel('|Z| (\u03A9)', color="blue")
ax1.tick_params(axis='y', labelcolor="blue")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.text(1.5*10**3, 0.75*10**5,  ParametersFit1 +"\n" + ParametersFit2 + "\n" + ParametersFit4
        , style='italic', fontsize= 16,
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})

ax2 = ax1.twinx()
ax2.semilogx(d["FRA_ITO"].iloc[:,1],d["FRA_ITO"].iloc[:,5]*2*np.pi/360, "^", color = "red")
ax2.semilogx(d["FRA_ITO"].iloc[:,1], phaseRC(d["FRA_ITO"].iloc[:,1], *paramP), color = "red")

ax1.set_xlabel("FREQUENCY (Hz)")
ax1.set_ylabel(" Z (\u03A9)", color = "blue")
ax2.set_ylabel("PHASE (rad)", color = "red")

plt.grid(True)
#ax2.set_ylabel('PHASE (rad)', color="red")
ax2.tick_params(axis='y', labelcolor="red")

plt.savefig("ITOspectroscopy.png", dpi =250, bbox_inches = "tight")
plt.show()

# =============================================================================
# PEDOT spectroscopy
# =============================================================================
#One could create a class in order to avoid copy and paste
class spectroscopy:
    def __init__(self, norm, phase, *impedencesInCircuit, NameInTheDictionaryd):
        pass
    #etc etc, i haven't time


s, R1,C1,R2,C2 = sp.symbols("s,R1,C1,R2,C2", real = True)
TransferRC = (C1 * C2 * R1 * R2 * (sp.I*s)**2 + ( C1 * R2 + C2 * R2 + C1 * R1 ) * sp.I*s + 1)/(C1 * C2 * R2  * (sp.I*s)**2 + C1 * sp.I*s)

f = abs(TransferRC) #Znorm 
znorm=sp.simplify(TransferRC.as_real_imag()[0]**2+TransferRC.as_real_imag()[1]**2)#other way for znorm
zphase=sp.simplify(sp.atan(TransferRC.as_real_imag()[0]/TransferRC.as_real_imag()[1])) #Zphase 
print(sp.latex(znorm))
print(sp.latex(zphase))


def phaseRCRC(w,r1,r2,c1,c2):
    return -np.arctan(c1*w*(c2**2*r1*r2**2*w**2 + r1 + r2)/(c1*c2*r2**2*w**2 + c2**2*r2**2*w**2 + 1)) +np.pi/2

def normRCRC(w,r1,r2,c1,c2):
    return np.sqrt((w**2*(c1*r1 + c1*r2 + c2*r2*(c1*c2*r1*r2*w**2 - 1) + c2*r2)**2 + (-c1*c2*r1*r2*w**2 + c2*r2*w**2*(c1*r1 + c1*r2 + c2*r2) + 1)**2)/(c1**2*w**2*(c2**2*r2**2*w**2 + 1)**2))
param,cov= curve_fit(normRCRC, d["FRA_PEDOT2"].iloc[:,1], d["FRA_PEDOT2"].iloc[:,4], p0=[40.058, 923.47,  0.00035848, 1.6084E-05])
#paramP,cov= curve_fit(phaseRCRC, d["FRA_PEDOT2"].iloc[:,1], d["FRA_PEDOT2"].iloc[:,5]*2*np.pi/360 -np.pi/2, p0=[40.058, 923.47,  0.00035848, 1.6084E-05])
paramP,cov= curve_fit(phaseRCRC, d["FRA_PEDOT2"].iloc[:,1], d["FRA_PEDOT2"].iloc[:,5]*2*np.pi/360, p0=[40.058, 923.47,  0.00035848, 1.6084E-05])

#TEXT FOR THE GRAPHIC
ParametersFit1="R1 = " + sci_notation_as_Benini_want(float(d['FRA_PEDOT2_fitresults'].Value[0].replace(",","."))) + "\u03A9"
ParametersFit2="C1 = " + sci_notation_as_Benini_want(float(d['FRA_PEDOT2_fitresults'].Value[1].replace(",","."))) + "F"
ParametersFit3="C2 = " + sci_notation_as_Benini_want(float(d['FRA_PEDOT2_fitresults'].Value[2].replace(",","."))) + "F"
ParametersFit4="R2 = " + sci_notation_as_Benini_want(float(d['FRA_PEDOT2_fitresults'].Value[3].replace(",","."))) + "\u03A9"

ParametersFit5=r"$\chi^2$ = " + sci_notation_as_Benini_want(float(d['FRA_PEDOT2_fitresults'].Value[5].replace(",",".")))

fig, ax1 = plt.subplots()
#plt.title("ITO IN PBS SOLUTION")


ax1.semilogx(d["FRA_PEDOT2"].iloc[:,1],d["FRA_PEDOT2"].iloc[:,4], "^", color = "blue")
ax1.semilogx(d["FRA_PEDOT2"].iloc[:,1],normRCRC(d["FRA_PEDOT2"].iloc[:,1], *param), color = "blue")


plt.grid(True)
#ax1.set_ylabel('|Z| (\u03A9)', color="blue")
ax1.tick_params(axis='y', labelcolor="blue")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.text(1*10**3, 3.2*10**3,  ParametersFit1 +"\n" + ParametersFit2 + "\n" + ParametersFit3 + "\n" + ParametersFit4 + "\n" + ParametersFit5
        , style='italic', fontsize= 16,
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})

ax2 = ax1.twinx()
ax2.semilogx(d["FRA_PEDOT2"].iloc[:,1],d["FRA_PEDOT2"].iloc[:,5]*2*np.pi/360 , "^", color = "red")
ax2.semilogx(d["FRA_PEDOT2"].iloc[:,1], phaseRCRC(d["FRA_PEDOT2"].iloc[:,1], *paramP), color = "red")

plt.grid(True)
ax2.set_ylabel('PHASE (rad)', color="red")
ax2.tick_params(axis='y', labelcolor="red")

ax1.set_xlabel("FREQUENCY (Hz)")
ax1.set_ylabel(" Z (\u03A9)", color = "blue")
ax2.set_ylabel("PHASE (rad)", color = "red")

plt.savefig("PEDOT2spectroscopy.png", dpi =250, bbox_inches = "tight")
plt.show()

# =============================================================================
# CYCLO INTEGRAL      
# =============================================================================

def integrate(x,y):
    result = 0
    for i in  range(1,len(x)):
        result+=(y[i]+y[i-1])*(x[i]-x[i-1])/2
    return result

from scipy.constants import e
for dati in d.keys(): 
    d[dati].columns = [c.replace(' ', '') for c in d[dati].columns]    
    d[dati].columns = [c.replace('(', '') for c in d[dati].columns]
    d[dati].columns = [c.replace(')', '') for c in d[dati].columns]
    d[dati].columns = [c.replace('.', '') for c in d[dati].columns]

ExcludedLastBranches = d["CYC_PEDOT"].query("PotentialappliedV==0")
Start_Index_to_exclude = ExcludedLastBranches.index[-1]
ExcludedLastBranches = d["CYC_PEDOT"].query(f"index<{Start_Index_to_exclude}")

Q = integrate(ExcludedLastBranches.iloc[:,1],ExcludedLastBranches.iloc[:,2])
N = Q/(2*e)

ParametersFit="N = " + sci_notation_as_Benini_want(N)

plt.plot(ExcludedLastBranches["PotentialappliedV"], ExcludedLastBranches["WE1CurrentA"]*1000, ".")
plt.title("CYCLE IN THE PEDOT:PSS SOLID LIQUID INTERFACE")
plt.grid(True)
plt.legend(["V vs I characteristic for deposition"])
plt.text(1*10**(-1), 1.5,  ParametersFit +"\n" 
        , style='italic', fontsize= 16,
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

plt.xlabel("VOLTAGE (V)")
plt.ylabel("CURRENT (mA)")
plt.savefig("CYC_PEDOT.png", dpi=250)
plt.show()

plt.plot(ExcludedLastBranches["Times"],ExcludedLastBranches["WE1CurrentA"], ".")
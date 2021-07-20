
import tkinter as tk
import pandas as pd
from PIL import Image, ImageTk
from tkinter.font import Font
from tabulate import tabulate
import tkinter.scrolledtext as tkscrolled
import io
from time import sleep
import os
import glob
from tkinter import filedialog
from tkinter.ttk import Progressbar
from tkscrolledframe import ScrolledFrame
from SerialSMU import SMU
from itertools import cycle
import time
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def start(Compliance, SweepStart, SweepFinish, baudrate, port):
    global Results
    text = tkscrolled.ScrolledText(UserLabx)#grid(row=10, column=17, rowspan = 5,columnspan= 16, sticky= "WE", pady=5, padx=5)
    breakpoint()
    smu = SMU(baudrate, port)
    
    # SWEEP TEST
    smu.sendCommands([
						'*RST', # Reset instrument to default parameters.
						':SYST:BEEP:STAT 0', # Disable the fricking beep
						':SENS:FUNC "VOLT"', # Select volt measurement function.
						':SENS:VOLT:PROT 1', # Set 1V compliance limit.
						':SOUR:FUNC CURR', # Select current source function.
						f':SOUR:CURR:STAR {SweepStart}',
						f':SOUR:CURR:STOP {SweepFinish}',
						':SOUR:CURR:MODE SWE',
						':SOUR:SWE:POIN 10',
						':TRIG:COUN 10',
						':FORM:ELEM VOLT,CURR', # Set to output volt reading to PC.
		 				':OUTP ON',
		 				':INIT'
					])
    readings = smu.getReadingsArray()
    text.insert(tk.END, readings)
    text.grid(row=10, column=17, rowspan = 5,columnspan= 16, sticky= "WE", pady=5, padx=5)
    Results=Results.append(readings)
    sleep(1)
    smu.sendCommand(':OUTP OFF')
    smu.close()
    
def show_plot():
    global Results
    #frame = tk.Frame(UserLabx, height=300, width=450, bd=1, relief="sunken").grid(row=10, columns=1,rowspan = 5,columnspan= 16,pady=1)
    Results=np.array(Results)
    figure = plt.Figure(figsize=(6,5), dpi=100)
    ax = figure.add_subplot(1,1,1)
    data = np.genfromtxt("Resonance1.csv" ,skip_header=3, delimiter= ", ")
    dataframe = pd.DataFrame({"X": data[:,0], "Y": data[:,1]})
    chart_type = FigureCanvasTkAgg(figure, UserLabx)
    chart_type.get_tk_widget().grid(row=10, columns=1,rowspan = 5,columnspan= 16,pady=1)
    ax.plot(dataframe.X, dataframe.Y)
    ax.set_title('Title')
    
    
def export_csv():
    global Results
    with open("myfile.csv", "w") as f:
        Results.to_csv(f)
    
def progress(iterator):
    cycling = cycle("⡇⣆⣤⣰⢸⠹⠛⠏")
    for element in iterator:
        print(next(cycling), end="\r")
        yield element
    print(" \r", end='')

def init():
    warn = tk.Toplevel()
    warn.title("Abandon the current session?")
    label = tk.Label(warn,"Are you you to want to leave the current session?\nThe data not saved will be lost!!!!!").pack()
    ok = tk.Button(warn, text="Ok", command = initializer).pack()
    null = tk.Button(warn, text="Cancel", command = warn.destroy).pack()
def initializer():    
    Compliance.set()
    SweepStart.set()
    SweepFinish.set()
    altro.set()
    Baudrate.set()
    Port.set()
    Results=[]
    

# for idx in progress(range(10000)):
#     time.sleep(0.5)
# print("finished!")
UserLabx = tk.Tk()
UserLabx.title("SMU Interface")


SMUimage= Image.open("SMU.jpg")
SMUimage = SMUimage.resize((480, 350), Image.ANTIALIAS)
SMUimage = ImageTk.PhotoImage(SMUimage)
#SMUimage = tk.Label(image=SMUimage).grid(row=0, column=3, columnspan =1, rowspan = 1)


#Variable Initialization
Compliance = tk.DoubleVar()
SweepStart = tk.DoubleVar()
SweepFinish = tk.DoubleVar()
altro = tk.DoubleVar()
Baudrate = tk.IntVar()
baudrate = [110, 300, 600, 1200, 2400, 4800, 9600, 14400, 19200, 38400, 57600, 115200, 128000 , 256000]
Port = tk.IntVar()
port = [1,2,3,4,5,6,7,8,9,10]
Results=list()

#let's make the window responsive
n_rows =8
n_columns =32
for i in range(n_rows):
    UserLabx.grid_rowconfigure(i,  weight =1)
for i in range(n_columns):
    UserLabx.grid_columnconfigure(i,  weight =1)
    
 
smuimage = tk.Label(UserLabx,image=SMUimage).grid(row=0, column=16, columnspan =16, rowspan = 10)

sep = tk.Frame(UserLabx, height=3, width=450, bd=1, relief="sunken").grid(row=0, column=1, columnspan = 15, pady=1)
init = tk.Label(UserLabx, text = "Intializing setup", font=("Courier", 7)).grid(row=0, column=0,sticky = "WE", columnspan = 1)

baudLab = tk.Label(UserLabx, text = "Baudrate").grid(row=1, column=0,sticky = "WE", padx=5)
baud = tk.OptionMenu(UserLabx, Baudrate, *baudrate)
baud.config(width=10)
baud.grid(row=1, column = 1, sticky = "W")

portLab = tk.Label(UserLabx, text = "Port", height =5, width=5).grid(row=1, column=2,sticky = "WE", padx=5)
port = tk.OptionMenu(UserLabx, Port, *port)
port.config(width=1)
port.grid(row=1, column = 3, sticky = "W")

sep = tk.Frame(UserLabx, height=3, width=450, bd=1, relief="sunken").grid(row=2, column=1, columnspan = 15, padx=5, pady=2)
init = tk.Label(UserLabx, text = "Measure setup", font=("Courier", 7)).grid(row=2, column=0,sticky = "WE", columnspan = 1,padx=2)

compl = tk.Label(UserLabx, text = "Compliance").grid(row=3, column=0,sticky = "WE", columnspan=1, padx=2)
complValue = tk.Entry(UserLabx, textvariable=Compliance).grid(row=3, column=1,sticky = "WE",columnspan=1, padx=2)
    
sweep = tk.Label(UserLabx, text = "Sweep start").grid(row=4, column=0,sticky = "WE",columnspan=1, padx=2)
sweepValue = tk.Entry(UserLabx, textvariable=SweepStart).grid(row=4, column=1,sticky = "WE",columnspan=1, padx=2)
 
finish = tk.Label(UserLabx, text = "Sweep end").grid(row=5, column=0,sticky = "WE",columnspan=1, padx=2)
finishValue = tk.Entry(UserLabx, textvariable=SweepFinish).grid(row=5, column=1,sticky = "WE",columnspan=1, padx=2)

tk.Button(UserLabx, text = "Start serial comunication", command =lambda: start(Compliance,SweepStart,SweepFinish, Baudrate,Port )).grid(row=7,column=0, columnspan = 16,sticky = "WE", padx=10, pady = 10)


menu = tk.Menu(UserLabx)
UserLabx.config(menu=menu)
fileMenu = tk.Menu(menu)
menu.add_cascade(label="File", menu=fileMenu)
fileMenu.add_command(label = "Reset all data" ,command = init)
fileMenu.add_command(label = "Export data", command = export_csv) 
   
editMenu = tk.Menu(menu)
menu.add_cascade(label="Measure", menu=editMenu)
editMenu.add_command(label="Choose X-Y measure")
editMenu.add_command(label="Show plot", command = show_plot)
UserLabx.mainloop()
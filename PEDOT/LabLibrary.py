# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:13:23 2021

@author: pepermatt94
"""
import matplotlib.pylab as plt
import pandas as pd
import scipy as sp
import ffmpeg
from ffmpeg import Error as FFmpegError
import io

def sci_notation_as_Benini_want(number,  error = None, cifre = 2):
    if cifre == 0:
        better = cifre +1
    else:
        better = cifre
    ret_string = ("{:."+f"{better}" +"e}").format(number)
    a, b = ret_string.split("e")
    a = ret_string.split("e")
    # remove leading "+" and strip leading zeros
    arg = float(a[0])
    exp = int(a[1])
    errorString =""
    openString = ""
    if error != None:
         ret_string = ("{:."+f"{cifre}" +"e}").format(error)
         a, b = ret_string.split("e")
         a = ret_string.split("e")
         # remove leading "+" and strip leading zeros
         argErr = float(a[0])
         expErr = int(a[1])
         if exp == 1:
             argErr = error
         elif exp == 0:
             argErr = error
         else:
             expMultiplication = abs(expErr-exp)
             argErr = argErr*10**(-(expMultiplication))
             
         errorString = r"$\pm$"+ ("{:."+f"{cifre}" +"f})").format(argErr)
         openString = "("
    if exp == 1:
        #number = round(arg*10,cifre)
        number = arg*10
        return openString + ("{:."+f"{cifre}" +"f}").format(number) + errorString 
    elif exp == 0:
        #number = round(arg,cifre)
        number = arg
        return openString + ("{:."+f"{cifre}" +"f}").format(number)+ errorString
    else:
        number = arg
        return openString + ("{:."+f"{cifre}" +"f}").format(number)  + errorString + r" $10^{}$".format("{" + str(exp) + "}")

#yields all the csv files in the path in a dictionary. Use "\\" to divide path (os.norm)

def charge_ALL_file_in_dir(PathTOdirectory, skip = 0, form = ".csv",sep = ";"):
    import glob
    files = []
    file = ""
    for filename in glob.glob(PathTOdirectory + "\\*" +form):
        with open(filename, "r") as f:
            text = f.read()
            text = text.split("\n")
            for line in text:
                if len(line) != 0:
                    if line[0] == "#":
                        pass
                    else:
                        file += line + "\n"
                else:
                    pass
        files.append(io.StringIO(file))
        file = ""
                  
    #dataFromDir = [pd.read_csv(filename, sep = sep, skiprows=skip) for filename in glob.glob(PathTOdirectory + "\\*" +form)]
    dataFromDir = [pd.read_csv(filename, sep = sep, skiprows=skip) for filename in files]
    filenames = glob.glob(PathTOdirectory + "\\*"+form)
    filenames = [k.split("\\")[-1] for k in filenames]
    filenames = [k[:-len(form)] for k in filenames]
    
    d = {}
    
    for i in range(len(dataFromDir)):
        d[filenames[i]] = dataFromDir[i]
    
    return d

def All_possible_graphs(dictionary):
    count = 1
    for key, values in dictionary.items():
        values = values.astype(str).apply(lambda x: x.str.replace(",", "."))
        values = values.astype(float)
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

def video():
    output_options = {
    'crf': 20,
    'preset': 'slower',
    'movflags': 'faststart',
    'pix_fmt': 'yuv420p'
    }
    try:
        (
        ffmpeg
        .input("C:/Users/pepermatt94/Desktop/image/frame%03d.jpg",  framerate=3) 
        .output("movie.mp4", **output_options) 
        .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e

def integrate(x,y):
    result = 0
    for i in  range(1,len(x)):
        result+=(y[i]+y[i-1])*(x[i]-x[i-1])/2
    return result
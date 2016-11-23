import numpy as np
from skimage import data
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filter import rank
from skimage import io
from skimage import exposure
from skimage import data
import os,fnmatch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import gzip

import sys


ImageFilePath = "/home/elliotnam/project/mammography/pilot_images/"
DataFilePath = "/home/elliotnam/project/mammography/"
def listdir_fullpath(d,type):
    dd = [os.path.join(d, f) for f in os.listdir(d)]
    return fnmatch.filter(dd,"*." + type)
    #return [os.path.join(d, f) for f in os.listdir(d)]

def runPatientInfoWithHist(patientDataFrame,idx):
    print(idx)
    global df_patient
    #print(patientDataFrame)
    fName = ImageFilePath+ patientDataFrame['filename'][idx].replace(".dcm.gz",".csv2")
    print(fName)
    dFrame = pd.read_csv(fName,header=None)

    df_patient = df_patient.append(dFrame)
    print("df patient")
    print(len(df_patient))

def runNormalize(fileName):
    print(" run normalize....")
    dataFrames = pd.read_csv(fileName)
    X = dataFrames.values
    # separate array into input and output components
#    X = array[:, 0:8]
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(X)

    fName = fileName.replace("csv", "csv2")

    print(fName)
    np.savetxt(fName, rescaledX.T,delimiter=",", fmt="%s")

def runHistgram(fileName):
    img = io.imread(fileName, as_grey=True)
    print(img)
    hist, bin_cen = exposure.histogram(img)

    fName = fileName.replace("jpg","csv")
    hist[0] =0
    hist[255] =0
    print(fName)
    np.savetxt(fName, hist, fmt="%s")
#    print("bin_cen")
#    print(type(bin_cen))
    print(np.sum(hist))
#    print(bin_cen)
#    bin_cen = np.arange(256)


tt = listdir_fullpath(ImageFilePath,"csv")

def startHistogram():
    i = 0
    for jj in tt:
        print(tt)
        runHistgram(jj)
    #    runNormalize(jj)
        i += 1
        if i == 2 :
            break

def startNormilize():

    i = 0
    for jj in tt:
        print(tt)
        runNormalize(jj)
        i += 1
        if i == 2 :
            break

def rebuildPatientInfo():
    cIndex = list()
    pFile = pd.read_csv(DataFilePath + "patient_info.csv2")

    for i in range(len(pFile)):
        print(pFile.ix[i])
        if pFile.ix[i]['imageView'] == 'R':
            if pFile.ix[i]['cancerR'] == 1:
                cIndex.append('1')
            else :
                cIndex.append('0')
        else :
            if pFile.ix[i]['cancerL'] == 1:
                cIndex.append('1')
            else:
                cIndex.append('0')
    #print(series)
    pFile['isCancer'] = cIndex
    print(pFile)
    pFile.to_csv(DataFilePath+"patient_info.csv2")

def startPatientInfoWithHist():
    global df_patient
    pFile = pd.read_csv(DataFilePath+"patient_info.csv")
    print(len(pFile))
    j = 0
    #for i in range(len(pFile)):
    #    runPatientInfoWithHist(pFile,i)
        #j+=1
        #if j ==2 :
        #    break
    #df_patient.to_csv("/home/elliotnam/project/mammography/patient_info_hist.csv")

    dd_patient = pd.read_csv(DataFilePath+"patient_info_hist.csv")
    xx = dd_patient.columns.values.tolist()

    for ii in range(1,256):
        pFile[xx[ii]] = dd_patient[xx[ii]]


    pFile.to_csv(DataFilePath+"patient_info.csv2")
    print(pFile)


def startDataunZip(dir):
    tt = listdir_fullpath(dir, "gz")
    for dn in tt:
        print(dn)
        inF = gzip.GzipFile(dn, 'rb')
        s = inF.read()
        inF.close()
        print("end inf")
        outF = file(dn.replace(".dcm.gz",".dcm"), 'wb')
        outF.write(s)
        outF.close()


#runHistgram("/home/elliotnam/PycharmProjects/MaMa/output.jpg")
#startHistogram()
#startNormilize()
#df_patient = pd.DataFrame()
#startPatientInfoWithHist()
#rebuildPatientInfo()

startDataunZip(ImageFilePath)

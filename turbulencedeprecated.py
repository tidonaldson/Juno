# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:33:51 2020

@author: YUNG TI
"""

from dataClasses import getFiles
from dataClasses import FGMData
import numpy as np
import pandas as pd
import os,datetime,logging,pathlib,struct
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.linear_model import LinearRegression


 
def gyro(bx, by, bz, m, z, q):
    mean_b = np.mean(np.sqrt(bx**2 + by**2 + bz**2))
    gyrofreq = (z*q/m)*(mean_b/(2*np.pi))
    return gyrofreq

#------------------------------------------------------------------------------------

def q(psd_perp,freq, bx, by, bz, b1, b2, m):
    delta_b_perp3 = (psd_perp*freq)**(3/2)
    v_rel_x = -400e3   #m/s
    v_rel_y = -100e3
    v_rel_mag = 300e3
    avgbx = np.mean(bx)*np.ones(len(bx))
    avgby = np.mean(by)*np.ones(len(bx))
    avgbz = np.mean(bz)*np.ones(len(bz))
    mean_b_mag = np.sqrt(avgbx**2 + avgby**2 + avgbz**2)
    #dotfactor = (avgbx*v_rel_x + avgby*v_rel_y)/(mean_b_mag*v_rel_mag)
    #theta = np.arccos((dotfactor))
    n_density = .1*(100**3)
    density = m*n_density
    mu_0 = np.pi*4*1e-7
    kperp = (2*np.pi*freq)/(v_rel_mag*np.sin(np.pi/2))
    rho_i = 1e7
    
    qkaw = (0.5*(delta_b_perp3[b2])*kperp[b2]/np.sqrt(mu_0**3*density))*(1+kperp[b2]**2*rho_i**2)**0.5*(1+(1/(1+kperp[b2]**2*rho_i**2))*(1/(1+1.25*kperp[b2]**2*rho_i**2))**2)
    qmhd = (delta_b_perp3[b1])*kperp[b1]/(np.sqrt((mu_0**3)*density))
    
    return qmhd, qkaw
#--------------------------------------------------------------------------------
    
def freqrange(f, gyro):
    """finds ranges of freqencies for MHD and KAW scales.\n  
    Returns the two frequency arrays and indices for the two arrays"""
    b1 = np.where((f>3E-5) & (f<(gyro)))
    freq_mhd = f[b1]
    
    b2 = np.where((f>(gyro*1.5)) & (f<.5))
    freq_kaw = f[b2]
    
    b3 = np.where((f>0)&(f<0.5))
    return freq_mhd, freq_kaw, b1, b2, b3 

#-------------------------------------------------------------------------------

def PSDfxn(cwt,freq,sig,fs):
    psd_list = []
    for i in range(len(sig)):
        psd = (2*fs/len(sig))*(sum(abs(cwt[i]**2)))/freq[i]
        psd_list.append(psd)
    psd = np.array(psd_list)
    return psd        


#------------------------------------------------------------------------------
x = getFiles('2016-09-20T00:00:00.000','2016-09-22T00:39:29.490','csv', 'C:\\Users\\YUNG TI\\Desktop\\Juno\\Programming\\Data\\', 'fgm')
#filelist = getFiles(startTime, endTime, fileType, dataFolder, instrument)

class turbulence(FGMData):
    """calculates """
    def __init__(self, dateList, filelist, startTime, endTime, step, window, interval):
        super().__init__(filelist, startTime, endTime)
        self.dateList = dateList
        self.avgBx = 0
        self.avgBy = 0
        self.avgBz = 0
        self.step = step
        self.window = window
        self.interval = interval
        self.bx = 0
        self.by = 0
        self.bz = 0
        self.time = 0
        self.delta_bx = 0
        self.delta_by = 0
        self.delta_bz = 0
        self.delta_bx_par = 0
        self.delta_by_par = 0
        self.delta_bz_par = 0
        self.dotfactor = 0
        self.delta_bx_perp1 = 0
        self.delta_by_perp1 = 0
        self.delta_bz_perp1 = 0
        self.delta_bx_perp2 = 0
        self.delta_by_perp2 = 0
        self.delta_bz_perp2 = 0
        self.delta_b_perp2 = 0
        self.delta_b_perp2list = []
        self.delta_b_parlist = []
        self.delta_b_perp1list = []
        self.delta_b_perp1 = 0
        self.delta_b_par = 0
        self.gyrofreq = 0
        self.mean_b = 0
        self.m = 23/6.0229e26
        self.z = 1.6
        self.q = 1.6e-19
        self.par = 0
        self.perp1 = 0
        self.perp2 = 0
        self.bxloop = 0
        self.byloop = 0
        self.bzloop = 0
        self.freq = 0
        self.fs = 0
        self.widths = 0
        self.w0 = 0
        self.cwt_par = 0
        self.cwt_perp1 = 0
        self.cwt_perp2 = 0
        self.psd_par = 0
        self.psd_perp1 = 0
        self.psd_perp2 = 0
        self.psd_perp = 0
        self.freq_mhd = 0
        self.freq_kaw = 0
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        self.mhd_index = 0
        self.kaw_index = 0
        self.mhd_slopelist = []
        self.kaw_slopelist = []
        self.qmhd_list = []
        self.qkaw_list = []
        self.q_time = []
        self.getB()
        self.meanB()
        self.deltaB()
        self.PSD()
        self.graphq()
        
        
        
    def getB(self):
        for i in self.dateList:
            bx = self.dataDict[i]['BX']
            by = self.dataDict[i]['BY']
            bz = self.dataDict[i]['BZ']
            time = np.array(self.dataDict[i]['TIME_ARRAY'])*3600
            self.bx = np.append(self.bx,bx)
            self.by = np.append(self.by,by)
            self.bz = np.append(self.bz,bz)
            self.time = np.append(self.time,time)
        return self.bx, self.by, self.bz, self.time
        
    def meanB(self):
        avgBx_list = []
        avgBy_list = []
        avgBz_list = []
        for i in range(len(self.bx)):
            self.avgBx = np.mean(self.bx[int(i*self.step):self.window+int(i*self.step)])
            self.avgBy = np.mean(self.by[int(i*self.step):self.window+int(i*self.step)])
            self.avgBz = np.mean(self.bz[int(i*self.step):self.window+int(i*self.step)])
            avgBx_list.append(self.avgBx)
            avgBy_list.append(self.avgBy)
            avgBz_list.append(self.avgBz)
        self.avgBx = np.array(avgBx_list)
        self.avgBy = np.array(avgBy_list)
        self.avgBz = np.array(avgBz_list)
        return self.avgBx, self.avgBy, self.avgBz
    
    def deltaB(self):       
        self.delta_bx = np.subtract(self.bx,self.avgBx)
        self.delta_by = np.subtract(self.by,self.avgBy)
        self.delta_bz = np.subtract(self.bz,self.avgBz)
        
        #decompose delta b vector into parallel
        self.dotfactor = (self.delta_bx*self.avgBx+self.delta_by*self.avgBy+self.delta_bz*self.avgBz)/(self.avgBx**2+self.avgBy**2+self.avgBz**2)
        
        self.delta_bx_par = self.dotfactor*self.avgBx
        self.delta_by_par = self.dotfactor*self.avgBy
        self.delta_bz_par = self.dotfactor*self.avgBz
    
        self.delta_bx_perp1 = np.subtract(self.delta_bx,self.delta_bx_par)
        self.delta_by_perp1 = np.subtract(self.delta_by,self.delta_by_par)
        self.delta_bz_perp1 = np.subtract(self.delta_bz,self.delta_bz_par)
        
        for i in range(len(self.delta_bx)):                                             #cross product of delta b par and delta b perp1 to get a vector perpendicular to both
            self.delta_b_perp2 = np.cross([self.delta_bx_par[i],self.delta_by_par[i],self.delta_bz_par[i]],[self.delta_bx_perp1[i],self.delta_by_perp1[i],self.delta_bz_perp1[i]])
            self.delta_b_perp2list.append(self.delta_b_perp2)
            
        self.delta_b_perp2 = np.array(self.delta_b_perp2list)
        self.delta_b_perp2 = np.sqrt(self.delta_b_perp2[:,0]**2+self.delta_b_perp2[:,1]**2,self.delta_b_perp2[:,2]**2)
        self.delta_b_perp1 = np.sqrt(self.delta_bx_perp1**2 + self.delta_by_perp1**2 + self.delta_bz_perp1**2)
        self.delta_b_par = np.sqrt(self.delta_bx_par**2 + self.delta_by_par**2 +self.delta_bz_par**2)
        
        return self.delta_b_perp2, self.delta_b_perp1, self.delta_b_par
    
    
        
    def PSD(self):
        for i in range(int(len(self.bx)/self.interval)):
            self.par = self.delta_b_par[self.interval*(i):self.interval*(i+1)]          #extract delta b components for each interval
            self.perp1 = self.delta_b_perp1[self.interval*i:self.interval*(1+i)]
            self.perp2 = self.delta_b_perp2[self.interval*i:self.interval*(1+i)]
            self.fs = int(1/self.step)                                                  #sampling frequency
            self.freq = np.arange(0.1,len(self.par))*(self.fs/len(self.par))        #frequency range
            self.widths = self.fs/(1.03*self.freq)                                  #widths for CWT with 1.03 factor        
            self.w0 = 6
            #self.cwt_par = signal.cwt(self.par, signal.morlet2,self.widths,w = self.w0)         #take morlet wavelet transform of each component
            self.cwt_perp1 = signal.cwt(self.perp1, signal.morlet2,self.widths, w = self.w0)
            self.cwt_perp2 = signal.cwt(np.reshape(self.perp2, len(self.perp2),), signal.morlet2,self.widths, w = self.w0)
        
            #self.psd_par = PSDfxn(self.cwt_par,self.freq,self.par,self.fs)          #obtain power spectral density from wavelet transform
            self.psd_perp1 = PSDfxn(self.cwt_perp1, self.freq, self.par,self.fs)
            self.psd_perp2 = PSDfxn(self.cwt_perp2, self.freq, self.par,self.fs)
            self.psd_perp = (self.psd_perp1 + self.psd_perp2)*(1e-18)               #summing perpendicular components, convert to T^2/freq
            
            self.bxloop = self.bx[self.interval*i:self.interval*(1+i)]*1e-9         #extract b data for interval and convert to T
            self.byloop = self.by[self.interval*i:self.interval*(1+i)]*1e-9
            self.bzloop = self.bz[self.interval*i:self.interval*(1+i)]*1e-9
            
            self.gyrofreq = gyro(self.bxloop,self.byloop,self.bzloop,self.m,self.z,self.q)      #calculate gyro freqency for interval
           
            self.freq_mhd, self.freq_kaw, self.b1, self.b2, self.b3 = freqrange(self.freq,self.gyrofreq)     #exract frequencies and ranges for MHD and KAW scales
            
            #r = LinearRegression()
            #r.fit(np.reshape(np.log10(self.freq_mhd),(-1,1)),np.reshape(np.log10(self.psd_perp[self.b1]),(-1,1)))       #find slopes of MHD and KAW ranges
            #self.mhd_slopelist.append(r.coef_)
            
            #r.fit(np.reshape(np.log10(self.freq_kaw),(-1,1)),np.reshape(np.log10(self.psd_perp[self.b2]),(-1,1)))
            #self.kaw_slopelist.append(r.coef_)
            
            q_mhd, q_kaw = q(self.psd_perp,self.freq,self.bxloop,self.byloop,self.bzloop,self.b1,self.b2,self.m)    #calculates qMHD and qKAW for each point in interval
            
            mean_q_mhd = np.mean(q_mhd)                                         #takes average q over interval
            self.qmhd_list.append(mean_q_mhd)
            
            mean_q_kaw = np.mean(q_kaw)
            self.qkaw_list.append(mean_q_kaw)                                   #produces list of one q value for each interval
            
            dateloop = self.startTime + datetime.timedelta(seconds = self.interval*i)
            
            #fig = plt.figure()
            #ax1 = fig.add_subplot(211)
            #ax1.loglog(self.freq_mhd,q_mhd,'r')
            #ax1.loglog(self.freq_kaw,q_kaw,'b')
            #ax1.loglog(self.freq_mhd,np.linspace(mean_q_mhd,mean_q_mhd,len(self.freq_mhd)),'black')
            #ax1.loglog(self.freq_kaw,np.linspace(mean_q_kaw,mean_q_kaw,len(self.freq_kaw)),'black')
            #ax1.set_title(dateloop)
            #ax1.set_ylabel('heating rate density \n [W/$m^2$]')
            #ax1.set_ylim(1e-20,1e-13)
            
            #ax2 = fig.add_subplot(212, sharex =ax1)
            #ax2.loglog(self.freq[self.b3],self.psd_perp[self.b3]*1e18, linewidth = 1)
            #ax2.loglog(self.freq_mhd,self.psd_perp[self.b1]*1e18,'r', linewidth = 0.5)
            #ax2.loglog(self.freq_kaw,self.psd_perp[self.b2]*1e18,'b', linewidth = 0.5)
            #ax2.loglog((self.gyrofreq,self.gyrofreq),(0,1e6))
            #ax2.set_xlabel('frequency [Hz]')
            #ax2.set_ylabel('Power Density \n [$nT^2$/Hz]')
            
            
        
    def graphq(self):
        #plt.figure()    
        #plt.hist(np.reshape(np.array(self.mhd_slopelist),(len(self.mhd_slopelist),)),6)
        #plt.plot((-5/3,-5/3),(0,100))
        #plt.ylabel('counts')
        #plt.xlabel('power law')
        #plt.title('MHD')
        #plt.show()
        
        #plt.figure()
        #plt.hist(np.reshape(np.array(self.kaw_slopelist),(len(self.kaw_slopelist),)),6)
        #plt.plot((-7/3,-7/3),(0,100))
        #plt.ylabel('counts')
        #plt.xlabel('power law')
        #plt.title('KAW')
        #plt.show()
        
        #y= np.log10(np.array(self.qmhd_list))-np.log10(np.array(self.qkaw_list))
        #plt.figure()
        #plt.hist(y,8)
        #plt.ylabel('counts')
        #plt.xlabel('$log_1$$_0$($q_M$$_H$$_D$)-$log_1$$_0$(q$_K$$_A$$_W$)')
        #plt.show()
        
        
        
        avg_q = (np.array(self.qmhd_list)+np.array(self.qkaw_list))/2
        #for i in range(len(avg_q)):
            #q_array = avg_q[i]*np.ones(self.interval)
            #self.q_time.append(q_array)
        #self.q_time = np.concatenate(np.array(self.q_time))
        time = np.linspace(self.time[0],len(self.time),len(avg_q))
        
        for i in range(len(self.dateList)):
            timeloop = time[i*int(len(time)/len(self.dateList)):(i+1)*int(len(time)/len(self.dateList))]
            qloop = avg_q[i*int(len(avg_q)/len(self.dateList)):(i+1)*int(len(avg_q)/len(self.dateList))]
            
            plt.figure()
            plt.plot(timeloop,qloop)
            plt.yscale('log')
            plt.ylabel('mean heating rate density [W/$m^2$]')
            plt.title(str(self.dateList[i]))
            plt.xticks([0+i*3600*24,6*3600+i*24*3600,12*3600+i*24*3600,18*3600+i*3600*24,24*3600+i*3600*24],['00:00','6:00','12:00','18:00','24:00'])
            plt.xlabel('time of day')
            
            
         
        
        
        #plt.show()
        
        
        
            
        
        #plt.figure()
        #plt.plot(time,avg_q)    
        
           
        
        
        
        #for date in self.dateList:
            #fgmStart = 0
            #for i in range(1,5):
                
                #if date in self.dataDict.keys():
                #    fgmData = self.dataDict[date]
                    
                    #fgmIndex = min(range(len(fgmData['TIME_ARRAY'])), key=lambda j: abs(fgmData['TIME_ARRAY'][j]-i*6))
                    
                    #plt.figure()
                    #plt.plot(fgmData['TIME_ARRAY'][fgmStart:fgmIndex+1], self.q_time[fgmStart:fgmIndex+1])
                
            
                
                
                
h = turbulence(x[1],x[2],'2016-09-20T00:00:00.000','2016-09-22T00:09:29.490',1,30,1800)
        
#-----------------------------------------------------------------------------------------------------------------------------
        
def runturbulence(startDate,endDate,filetype,datatype,filelist,step,window,interval):
    
    y = getFiles(startDate,endDate,filetype,filelist,datatype)
    
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:46:39 2020

@author: Ti Donaldson
"""


from dataClasses import getFiles, FGMData
import numpy as np
import pandas as pd
import os,datetime,logging,pathlib,struct
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.linear_model import LinearRegression


 
def gyro(bx, by, bz, m, z, q):
    """finds a gyrofreqency given magnetosphere properties.  \n All inputs must be in
    fundamental units (T, m, etc.) \n Returns scalar qyrofrequency corresponding to given
    range of B"""
    mean_b = np.mean(np.sqrt(bx**2 + by**2 + bz**2))
    gyrofreq = (z*q/m)*(mean_b/(2*np.pi))
    return gyrofreq

#------------------------------------------------------------------------------------

def q(psd_perp,freq, bx, by, bz, b1, b2, m):
    """Takes PSD of perpendicular component and other parameters to find q MHD 
    and q KAW.  \n Every parameter must be in base units (T, kg, m, etc).  \n Empirical
    parameters are subject to adjustment below.  \n Outputs ranges of q MHD and q KAW 
    over freqency domains, according to b1 and b2 (respectively MHD and KAW freqency 
    domains. \n MAG vector components used only to find theta for k perp.  """
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
    n_density = 0.1*(100**3)
    density = m*n_density
    mu_0 = np.pi*4*1e-7
    kperp = (2*np.pi*freq)/(v_rel_mag*np.sin(np.pi/2))
    rho_i = 1e7
    
    qkaw = (0.5*(delta_b_perp3[b2])*kperp[b2]/np.sqrt(mu_0**3*density))*(1+kperp[b2]**2*rho_i**2)**0.5*(1+(1/(1+kperp[b2]**2*rho_i**2))*(1/(1+1.25*kperp[b2]**2*rho_i**2))**2)
    qmhd = (delta_b_perp3[b1])*kperp[b1]/(np.sqrt((mu_0**3)*density))
    
    return qmhd, qkaw
#--------------------------------------------------------------------------------
    
def freqrange(f, gyro):
    """Finds ranges of freqencies for MHD and KAW scales.\n  
    Inputs: f is frequency range for PSD, and gyro is the gyrofreqency for given domain. \n
    Returns the two frequency arrays and indices (tuples) for the two arrays. \n b1
    corresponds to MHD, b2 to KAW, and b3 to all real points of freqency range."""
    b1 = np.where((f>3E-4) & (f<(gyro)))
    freq_mhd = f[b1]
    
    b2 = np.where((f>(gyro*1.5)) & (f<0.1))
    freq_kaw = f[b2]
    
    b3 = np.where((f>0)&(f<0.5))
    return freq_mhd, freq_kaw, b1, b2, b3 

#-------------------------------------------------------------------------------

def PSDfxn(cwt,freq,sig,fs):
    """Finds PSD as per Tao et. al. 2015 given a morlet wavelet transform, frequency 
    range, the signal, and sampling frequency. \n Outputs an array of length of signal"""
    psd_list = []
    for i in range(len(sig)):
        psd = (2*fs/len(sig))*(sum(abs(cwt[i]**2)))/freq[i]
        psd_list.append(psd)
    psd = np.array(psd_list)
    return psd        


#------------------------------------------------------------------------------

class turbulence(FGMData):
    """calculates heating rate density and power spectral density then creates various plots.  \n Step, window, and interval must be in seconds as an integer.  \n
    Inherits FGMData class to extract BX, BY, and BZ data. \n
    dateList must be from getFiles, and startTime and endTime must be in UTC 
    i.e., "2016-09-22T00:00:00.0000"""
    def __init__(self, dateList, filelist, startTime, endTime, step, window, interval,savePath):
        super().__init__(filelist, startTime, endTime)
        self.dateList = dateList
        self.step = step                                        
        self.window = window                                                   #for average B field
        self.interval = interval                                               #interval over which PSD is found
        self.savePath = savePath                                               
        self.m = 23/6.0229e26
        self.z = 1.6
        self.q = 1.6e-19
        self.mhd_slopelist = []
        self.kaw_slopelist = []
        self.qmhd_biglist = []
        self.qkaw_biglist = []
        self.get_q()
        
    def get_q(self):
        for date in self.dateList:
            bx = np.array(self.dataDict[date]['BX'])                           #extract MAG data for each day, compile into arrays         
            by = np.array(self.dataDict[date]['BY'])
            bz = np.array(self.dataDict[date]['BZ'])
            qmhd_list = []
            qkaw_list = []
            avgbx = []
            avgby = []
            avgbz = []
            freq = 0
            for i in range(len(bx)):
                avebx = np.mean(bx[i*self.step:i*self.step + self.window])     #find average B field, given the window for taking running average 
                aveby = np.mean(by[i*self.step:i*self.step +self.window])
                avebz = np.mean(bz[i*self.step:i*self.step +self.window])
                avgbx.append(avebx)
                avgby.append(aveby)
                avgbz.append(avebz)
            avgbx = np.array(avgbx)
            avgby = np.array(avgby)
            avgbz = np.array(avgbz)
                
            for i in range(int(len(bx)/self.interval)):                        #find PSD and q over given interval                 
                
                avgbxloop = avgbx[i*self.interval:(1+i)*self.interval]         #extracts average B field into arrays of size of interval
                avgbyloop = avgby[i*self.interval:(1+i)*self.interval]
                avgbzloop = avgbz[i*self.interval:(1+i)*self.interval] 
                
                delta_bx = bx[i*self.interval:(1+i)*self.interval] - avgbxloop #B - B0 to get delta B
                delta_by = by[i*self.interval:(1+i)*self.interval] - avgbyloop
                delta_bz = bz[i*self.interval:(1+i)*self.interval] - avgbzloop
                
                dotfactor = (delta_bx*avgbxloop + delta_by*avgbyloop +delta_bz*avgbzloop)/(avgbxloop**2 + avgbyloop**2 + avgbzloop**2)
                
                delta_bx_par = dotfactor*avgbxloop                             #decomposition of delta B alligned to average B field
                delta_by_par = dotfactor*avgbyloop
                delta_bz_par = dotfactor*avgbzloop
                
                delta_bx_perp1 = delta_bx - delta_bx_par                       #subtract parallel component from delta B to get first perpendicular vector
                delta_by_perp1 = delta_by - delta_by_par
                delta_bz_perp1 = delta_bz - delta_bz_par
                
                delta_bx_perp2 = []
                delta_by_perp2 = []
                delta_bz_perp2 = []
                
                for j in range(len(delta_bx_perp1)):                           #difficult to perform numpy cross without involvement of lists
                    cross = np.cross([delta_bx_par[j],delta_by_par[j],delta_bz_par[j]],[delta_bx_perp1[j],delta_by_perp1[j],delta_bz_perp1[j]])
                    delta_bx_perp2.append(cross[0])                            #extracts x, y, z components of 2nd perpendicular vector
                    delta_by_perp2.append(cross[1])
                    delta_bz_perp2.append(cross[2])
                delta_bx_perp2 = np.array(delta_bx_perp2)
                delta_by_perp2 = np.array(delta_by_perp2)
                delta_bz_perp2 = np.array(delta_bz_perp2)
                
                delta_b_perp1 = np.sqrt(delta_bx_perp1**2 + delta_by_perp1**2 + delta_bz_perp1**2)   #find magnitudes of each vector
                delta_b_perp2 = np.sqrt(delta_bx_perp2**2 + delta_by_perp2**2 + delta_bz_perp2**2)
                delta_b_par = np.sqrt(delta_bx_par**2 + delta_by_par**2 + delta_bz_par**2)
                
                fs = int(1/(self.step))                                        #sampling freqency, based off of given step size
                freq = np.arange(0.1,len(delta_b_par))*fs/len(delta_b_par)     #finds freqency range (this is divided by 2 in freqrange function)
                widths = 1/(1.03*freq)                                         #parameter for cwt
                w0 = 6                                                         #as per Tao et. al. 2015
                
                #psd_par = PSDfxn(signal.cwt(delta_b_par,signal.morlet2, widths,w = w0), freq, delta_b_par, fs)
                psd_perp1 = PSDfxn(signal.cwt(delta_b_perp1, signal.morlet2, widths, w = w0), freq, delta_b_perp1, fs)
                psd_perp2 = PSDfxn(signal.cwt(delta_b_perp2, signal.morlet2, widths, w = w0), freq, delta_b_perp2, fs)
                psd_perp = (psd_perp1 + psd_perp2)*1e-18                       #calls PSDfxn to find PSD given morlet wavelet transforms of each vector component.  
                                                                                #the two perpendicular compoenents are summed to find total perpendicular
                
                gyrofreq = gyro(bx[self.interval*i:(1+i)*self.interval]*1e-9, by[self.interval*i:(i+1)*self.interval]*1e-9, bz[self.interval*i:(1+i)*self.interval]*1e-9, self.m, self.z, self.q)
                                                                               #finds gyrofreqency according to B values on given interval (which is why the above line is so long)
                freq_mhd, freq_kaw, b1, b2, b3 = freqrange(freq, gyrofreq)     #uses freqrange function to find MHD and KAW frequency ranges 
                
                q_mhd, q_kaw = q(psd_perp, freq, bx[i*self.interval:(1+i)*self.interval]*1e-9, by[i*self.interval:(1+i)*self.interval]*1e-9, bz[i*self.interval:(1+i)*self.interval]*1e-9, b1, b2, self.m)
                                                                               #finds q MHD and q KAW, which have length of b1 and b2 respectively
                mean_q_mhd = np.mean(q_mhd)                                    #finds average q values for given interval
                mean_q_kaw = np.mean(q_kaw)
                
                qmhd_list.append(mean_q_mhd)
                qkaw_list.append(mean_q_kaw)
                
                self.qmhd_biglist.append(mean_q_mhd)                           #adds mean q values to list of every q value over the date range for histogram
                self.qkaw_biglist.append(mean_q_kaw)
                
                r = LinearRegression()                                         #perform linear regression to find power law fits
                r.fit(np.reshape(np.log10(freq_mhd),(-1,1)), np.reshape(np.log10(psd_perp[b1]),(-1,1)))
                self.mhd_slopelist.append(r.coef_)                             #create list of slopes for KAW and MHD per each interval
                
                r.fit(np.reshape(np.log10(freq_kaw),(-1,1)), np.reshape(np.log10(psd_perp[b2]),(-1,1)))
                self.kaw_slopelist.append(r.coef_)
                
               
                
                fig = plt.figure()                                             #plot heating rate densities on frequency domain
                ax1 = fig.add_subplot(211)
                ax1.loglog(freq_mhd,q_mhd,'r')
                ax1.loglog(freq_kaw,q_kaw,'b')
                ax1.loglog(freq_mhd,np.linspace(mean_q_mhd,mean_q_mhd,len(freq_mhd)),'black')
                ax1.loglog(freq_kaw,np.linspace(mean_q_kaw,mean_q_kaw,len(freq_kaw)),'black')
                ax1.set_title(date + '  ' + str(datetime.timedelta(seconds = i*self.interval)))
                ax1.set_ylabel('heating rate density \n [W/$m^2$]')
                ax1.set_yticks([1e-19,1e-17,1e-15,1e-13,1e-11])
                
                
                ax2 = fig.add_subplot(212, sharex =ax1)                        #plot power spectral density and gyrofrequency
                ax2.loglog(freq[b3],psd_perp[b3]*1e18, linewidth = 1)
                ax2.loglog(freq_mhd,psd_perp[b1]*1e18,'r', linewidth = 0.5)
                ax2.loglog(freq_kaw,psd_perp[b2]*1e18,'b', linewidth = 0.5)
                ax2.loglog((gyrofreq,gyrofreq),(0,np.max(psd_perp)*1e20),'g--')
                ax2.set_xlabel('frequency [Hz]')
                ax2.set_ylabel('Power Density \n [$nT^2$/Hz]')
                ax2.set_yticks([1e-3,1e-1,1e1,1e3,1e5])
                plt.tight_layout()
                
                
                saveName = f'PSD_and_q_{date}_{i}'                             #create file name with date and interval on that day
                plt.savefig(f'{self.savePath}\\{saveName}')
               
            mean_q = (np.array(qmhd_list) + np.array(qkaw_list))/2
            time = np.linspace(np.min(self.dataDict[date]['TIME_ARRAY']),np.max(self.dataDict[date]['TIME_ARRAY']),len(mean_q))
            
            for i in range(4):
                
                xtick = np.arange(i*6,(1+i)*6+1)
                plt.figure(figsize = (10,4))                                                         #plot q over time on a 6 hour window
                index = np.where((i*6<=time) & ((i+1)*6>=time))
                
                timeloop = time[index]
                qloop = mean_q[index]
                
                for k in range(len(qloop)-1):
                    plt.plot((timeloop[k],timeloop[k+1]),(qloop[k],qloop[k]),'b')            
                
                
                plt.xticks(xtick, [str(xtick[0])+':00',str(xtick[1])+':00',str(xtick[2])+':00',str(xtick[3])+':00',str(xtick[4])+':00',str(xtick[5])+':00',str(xtick[6])+':00'])
                plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom']=True
                plt.rcParams['xtick.top'] = True
                
                plt.title(date)
                plt.yscale('log')
                plt.xlabel('time [hours]')
                plt.ylabel('mean heating rate density [W/$m^2$]')
        
                timeFormat = {0:'0000',1:'0600',2:'1200',3:'1800'}
                saveName = f'q_time_domain_{date}_{timeFormat[i]}'             #produces file name with date and start time
                
                plt.savefig(f'{self.savePath}/{saveName}')
        
        mhd_slope = np.array(self.mhd_slopelist)
        kaw_slope = np.array(self.kaw_slopelist)
        
        plt.figure()                                                           #create histograms of power laws and KAW and MHD differences
        plt.hist(np.reshape(mhd_slope,(-1,1)),8)
        plt.plot((-5/3,-5/3),(0,len(self.mhd_slopelist)))                       #theoretical power law
        plt.xlabel('MHD Power Law')
        plt.ylabel('count')
        plt.title('MHD')
        saveName = f'hist_MHD_power_law_{min(self.dateList)}_{max(self.dateList)}'
        plt.savefig(f'{self.savePath}/{saveName}')
        
        plt.figure()
        plt.hist(np.reshape(kaw_slope,(-1,1)),8)
        plt.plot((-7/3,-7/3),(0,len(self.kaw_slopelist)))                      #theoretical power law
        plt.xlabel('KAW Power Law')
        plt.ylabel('count')
        plt.title('KAW')
        saveName = f'hist_KAW_power_law_{min(self.dateList)}_{max(self.dateList)}'
        plt.savefig(f'{self.savePath}/{saveName}')
        
        q_diff = np.log10(np.array(self.qmhd_biglist))-np.log10(np.array(self.qkaw_biglist))
        plt.figure()
        plt.hist(q_diff,8)
        plt.ylabel('count')
        plt.xlabel('$log_1$$_0$($q_M$$_H$$_D$)-$log_1$$_0$(q$_K$$_A$$_W$)')
        saveName = f'hist_scale_diff_{min(self.dateList)}_{max(self.dateList)}'
        plt.savefig(f'{self.savePath}/{saveName}')        
                
                
#--------------------------------------------------------------------------------------------------

def run_turbulence(startTime,endTime,fileType,dataType,fileList,step,window,interval,savePath):
    """perform turbulence analysis on given MAG data.  \n startTime and endTime in UTC,
    for example: '2016-09-20T00:00:00.000'.  \n step, window, and interval all must be given 
    as integers in seconds"""
    
    a = getFiles(startTime,endTime,fileType, fileList,dataType)
    b = turbulence(a[1],a[2],startTime,endTime,step,window,interval,savePath)
    return b                
#--------------------------------------------------------------------------------------------------                
h1 = run_turbulence('2016-11-15T00:00:00.000','2016-11-16T00:00:00.000','csv','fgm','C:\\Users\\YUNG TI\\Desktop\\Juno\\Programming\\Data\\',1,60,1800,'C:\\Users\YUNG TI\Desktop\Juno\Figures')                
#h2 = run_turbulence('2016-09-20T00:00:00.000','2016-09-25T00:00:00.000','csv','fgm','C:\\Users\\YUNG TI\\Desktop\\Juno\\Programming\\Data\\',1,30,1200)                
#h3 = run_turbulence('2016-09-20T00:00:00.000','2016-09-25T00:00:00.000','csv','fgm','C:\\Users\\YUNG TI\\Desktop\\Juno\\Programming\\Data\\',1,30,1800)                              
             

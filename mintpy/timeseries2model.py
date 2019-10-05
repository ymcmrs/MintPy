#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright(c) 2019, Yunmeng Cao                           #
# Author:  Yunmeng Cao                                     #
############################################################
# This script is modified from timeseries2velocity.py 
# Written by Heresh Fattahi, Zhang Yunjun
#

import sys
import os
import argparse
import numpy as np
from mintpy.objects import timeseries, giantTimeseries, HDFEOS
from mintpy.utils import readfile, writefile, ptime, utils as ut
from numpy.linalg import inv

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


from scipy.optimize import curve_fit
from datetime import datetime as dt

dataType = np.float32
# key configuration parameter name
key_prefix = 'mintpy.velocity.'
configKeys = ['excludeDate']


############################################################################
EXAMPLE = """example:
  timeseries2model.py  timeSeries_ECMWF_demErr.h5 --variance variogramTsModel.h5
  timeseries2model.py  timeSeries_ECMWF_demErr.h5 --variance variogramTsModel.h5 --model linear
  timeseries2model.py  timeSeries_ECMWF_demErr.h5 --model linear_season
  timeseries2model.py  timeSeries_ECMWF_demErr.h5 --model linear_season_semiseason
  timeseries2model.py  timeSeries_ECMWF_demErr.h5 --model linear_season_semiseason --parallel 8
  
"""

TEMPLATE = """
## estimate linear velocity from timeseries, and from tropospheric delay file if exists.
mintpy.velocity.excludeDate = auto   #[exclude_date.txt / 20080520,20090817 / no], auto for exclude_date.txt
mintpy.velocity.startDate   = auto   #[20070101 / no], auto for no
mintpy.velocity.endDate     = auto   #[20101230 / no], auto for no
"""

DROP_DATE_TXT = """exclude_date.txt:
20040502
20060708
20090103
"""


def create_parser():
    parser = argparse.ArgumentParser(description='Inverse velocity from time-series using weighted least-squares method.',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=TEMPLATE+'\n'+EXAMPLE)

    parser.add_argument('timeseries_file',
                        help='Time series file for velocity inversion.')
    
    parser.add_argument('--model', dest='model',choices = {'linear','linear_season','linear_season_semiseason'}, default='linear', help='Velocity model of time-series observations')
    parser.add_argument('--variance', dest='variance', help='Variances of time-series observations')
    parser.add_argument('--start-date','-s', dest='startDate',
                        help='start date for velocity estimation')
    parser.add_argument('--end-date','-e', dest='endDate',
                        help='end date for velocity estimation')
    parser.add_argument('--exclude', '--ex', dest='excludeDate', nargs='+', default=[],
                        help='date(s) not included in velocity estimation, could be list of string or text file, i.e.:\n' +
                             '--exclude 20040502 20060708 20090103\n' +
                             '--exclude exclude_date.txt\n'+DROP_DATE_TXT)
    parser.add_argument('--template', '-t', dest='template_file',
                        help='template file with the following items:'+TEMPLATE)
    parser.add_argument('--parallel', dest='parallelNumb', type=int, default=1, help='Enable parallel processing and Specify the number of processors.')
    parser.add_argument('-o', '--output', dest='outfile',
                        help='output file name')
    parser.add_argument('--update', dest='update_mode', action='store_true',
                        help='Enable update mode, and skip estimation if:\n'+
                             '1) output velocity file already exists, readable '+
                             'and newer than input timeseries file\n' +
                             '2) all configuration parameters are the same.')
    return parser


def cmd_line_parse(iargs=None):
    """Command line parser."""
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    inps.key = readfile.read_attribute(inps.timeseries_file)['FILE_TYPE']
    if inps.key not in ['timeseries', 'giantTimeseries', 'HDFEOS']:
        raise Exception('input file is {}, NOT timeseries!'.format(inps.key))
    return inps


def read_template2inps(template_file, inps=None):
    """Read input template file into inps.excludeDate"""
    if not inps:
        inps = cmd_line_parse()
    inpsDict = vars(inps)
    print('read options from template file: '+os.path.basename(template_file))
    template = readfile.read_template(inps.template_file)
    template = ut.check_template_auto_value(template)

    # Read template option
    prefix = 'mintpy.velocity.'
    keyList = [i for i in list(inpsDict.keys()) if prefix+i in template.keys()]
    for key in keyList:
        value = template[prefix+key]
        if value:
            if key in ['startDate', 'endDate']:
                inpsDict[key] = ptime.yyyymmdd(value)
            elif key in ['excludeDate']:
                inpsDict[key] = ptime.yyyymmdd(value.replace(',', ' ').split())
    return inps


def run_or_skip(inps):
    print('update mode: ON')
    flag = 'skip'

    # check output file
    if not os.path.isfile(inps.outfile):
        flag = 'run'
        print('1) output file {} NOT found.'.format(inps.outfile))
    else:
        print('1) output file {} already exists.'.format(inps.outfile))
        ti = os.path.getmtime(inps.timeseries_file)
        to = os.path.getmtime(inps.outfile)
        if ti > to:
            flag = 'run'
            print('2) output file is NOT newer than input file: {}.'.format(inps.timeseries_file))
        else:
            print('2) output file is newer than input file: {}.'.format(inps.timeseries_file))

    # check configuration
    if flag == 'skip':
        atr = readfile.read_attribute(inps.outfile)
        if any(str(vars(inps)[key]) != atr.get(key_prefix+key, 'None') for key in configKeys):
            flag = 'run'
            print('3) NOT all key configration parameters are the same: {}.'.format(configKeys))
        else:
            print('3) all key configuration parameters are the same: {}.'.format(configKeys))

    # result
    print('run or skip: {}.'.format(flag))
    return flag


############################################################################
def read_exclude_date(inps, dateListAll):
    # Merge ex_date/startDate/endDate into ex_date
    yy_list_all = ptime.yyyymmdd2years(dateListAll)
    exDateList = []
    # 1. template_file
    if inps.template_file:
        print('read option from template file: '+inps.template_file)
        inps = read_template2inps(inps.template_file, inps)

    # 2. ex_date
    exDateList += ptime.read_date_list(list(inps.excludeDate), date_list_all=dateListAll)
    if exDateList:
        print('exclude date:'+str(exDateList))

    # 3. startDate
    if inps.startDate:
        print('start date: '+inps.startDate)
        yy_min = ptime.yyyymmdd2years(ptime.yyyymmdd(inps.startDate))
        for i in range(len(dateListAll)):
            date = dateListAll[i]
            if yy_list_all[i] < yy_min and date not in exDateList:
                print('  remove date: '+date)
                exDateList.append(date)

    # 4. endDate
    if inps.endDate:
        print('end date: '+inps.endDate)
        yy_max = ptime.yyyymmdd2years(ptime.yyyymmdd(inps.endDate))
        for i in range(len(dateListAll)):
            date = dateListAll[i]
            if yy_list_all[i] > yy_max and date not in exDateList:
                print('  remove date: '+date)
                exDateList.append(date)
    exDateList = list(set(exDateList))
    return exDateList


def read_date_info(inps):
    """Get inps.excludeDate full list
    Inputs:
        inps          - Namespace, 
    Output:
        inps.excludeDate  - list of string for exclude date in YYYYMMDD format
    """
    if inps.key == 'timeseries':
        tsobj = timeseries(inps.timeseries_file)
    elif inps.key == 'giantTimeseries':
        tsobj = giantTimeseries(inps.timeseries_file)
    elif inps.key == 'HDFEOS':
        tsobj = HDFEOS(inps.timeseries_file)
    tsobj.open()
    inps.excludeDate = read_exclude_date(inps, tsobj.dateList)
    
    # Date used for estimation inps.dateList
    inps.dateList = [i for i in tsobj.dateList if i not in inps.excludeDate]
    
    date_list = inps.dateList
    dt_list = [dt.strptime(i, '%Y%m%d') for i in date_list]
    yr_list = [i.year + (i.timetuple().tm_yday - 1) / 365.25 for i in dt_list]
    yr_diff = np.array(yr_list)
    yr_diff -= yr_diff[0]
    inps.yr_diff = yr_diff
    
    inps.numDate = len(inps.dateList)
    print('-'*50)
    print('dates from input file: {}\n{}'.format(tsobj.numDate, tsobj.dateList))
    print('-'*50)
    if len(inps.dateList) == len(tsobj.dateList):
        print('using all dates to calculate the velocity')
    else:
        print('dates used to estimate the velocity: {}\n{}'.format(inps.numDate, inps.dateList))
    print('-'*50)

    # flag array for ts data reading
    inps.dropDate = np.array([i not in inps.excludeDate for i in tsobj.dateList], dtype=np.bool_)

    # output file name
    if not inps.outfile:
        outname = 'velocity'
        if inps.key == 'giantTimeseries':
            prefix = os.path.basename(inps.timeseries_file).split('PARAMS')[0]
            outname = prefix + outname
        outname += '.h5'
        inps.outfile = outname
    return inps


def dateList2yr(dateList):
    dt_list = [dt.strptime(i, '%Y%m%d') for i in date_list]
    yr_list = [i.year + (i.timetuple().tm_yday - 1) / 365.25 for i in dt_list]
    yr_diff = np.array(yr_list)
    yr_diff -= yr_diff[0]
    
    return yr_diff


def def_linear(t,a0,k0):
    #a0 = m[0]
    #k0 = m[1] 
    return a0 + k0*t


def def_linearSeason(t,a0,k0,p0,pc0):    
    #a0 = m[0]
    #k0 = m[1]
    #p0 = m[2]
    #ph0 = m[3]
    return a0 + k0*t + p0*np.sin(2*np.pi*t ) + pc0*np.cos(2*np.pi*t )

def def_linearSeasonSemi(t,a0,k0,p0,pc0,p01,pc01):    
    #a0 = m[0]
    #k0 = m[1]
    #p0 = m[2]
    #ph0 = m[3]
    #p01 = m[4]
    #ph01 = m[5]
    return a0 + k0*t + p0*np.sin(2*np.pi*t ) + pc0*np.cos(2*np.pi*t ) + p01*np.sin(np.pi*t) + pc01*np.sin(np.pi*t)

def model_linear(data0):
    dt_yr, ts_data, ts_vari = data0
    ts_vari[ts_vari==0]=0.01 # consider the zero nugget situtation
    
    rr0, cc0 = ts_vari.shape    
    vel = np.zeros((cc0,),dtype = np.float32)
    vel_std = np.zeros((cc0,),dtype = np.float32)
    
    my = max(dt_yr)
    k0 = 0
    
    for i in range(cc0):
        ts_data0 = ts_data[:,i]
        ts_vari0 = ts_vari[:,i]

        #print(ts_vari0.shape)
        #print(ts_data0.shape)
        v0 = max(ts_data0)/my
        p0 = k0,v0
        popt2, pcov2 = curve_fit(def_linear, dt_yr, ts_data0, p0, sigma=ts_vari0, absolute_sigma=False)
        vel[i] = popt2[1]
        vel_std[i] = np.sqrt(pcov2[1,1])
        #print(vel)
    return vel, vel_std

def model_linearSeason(data0):
    dt_yr, ts_data, ts_vari = data0
    ts_vari[ts_vari==0]=0.01 # consider the zero nugget situtation
    
    rr0, cc0 = ts_vari.shape    
    vel = np.zeros((cc0,),dtype = np.float32)
    amp = np.zeros((cc0,),dtype = np.float32)
    pha = np.zeros((cc0,),dtype = np.float32)
    
    
    vel_std = np.zeros((cc0,),dtype = np.float32)
    amp_std = np.zeros((cc0,),dtype = np.float32)
    pha_std = np.zeros((cc0,),dtype = np.float32)
    
    my = max(dt_yr)
    k0 = 0
    amp0 = 0.001
    pha0 = 0
    
    for i in range(cc0):
        ts_data0 = ts_data[:,i]
        ts_vari0 = ts_vari[:,i]

        #print(ts_vari0.shape)
        #print(ts_data0.shape)
        v0 = max(ts_data0)/my
        p0 = k0,v0, amp0, pha0
        popt2, pcov2 = curve_fit(def_linearSeason, dt_yr, ts_data0, p0, sigma=ts_vari0, absolute_sigma=False)
        vel[i] = popt2[1]
        amp[i] = np.sqrt(popt2[2]**2 + popt2[3]**2)
        pha[i] = np.arctan((popt2[3]/popt2[2]))
        
        vel_std[i] = np.sqrt(pcov2[1,1])
        #amp_std[i] = np.sqrt(pcov2[2,2])
        #pha_std[i] = np.sqrt(pcov2[3,3])
        #print(vel)
    return vel, vel_std, amp, pha

  
def model_linearSeasonSemi(data0):
    dt_yr, ts_data, ts_vari = data0
    ts_vari[ts_vari==0]=0.01 # consider the zero nugget situtation
    
    rr0, cc0 = ts_vari.shape    
    vel = np.zeros((cc0,),dtype = np.float32)
    amp = np.zeros((cc0,),dtype = np.float32)
    pha = np.zeros((cc0,),dtype = np.float32)
    ampt = np.zeros((cc0,),dtype = np.float32)
    phat = np.zeros((cc0,),dtype = np.float32)
    
    vel_std = np.zeros((cc0,),dtype = np.float32)
    amp_std = np.zeros((cc0,),dtype = np.float32)
    pha_std = np.zeros((cc0,),dtype = np.float32)
    ampt_std = np.zeros((cc0,),dtype = np.float32)
    phat_std = np.zeros((cc0,),dtype = np.float32)
    
    
    my = max(dt_yr)
    k0 = 0
    amp0 = 0.001
    pha0 = 0
    ampt0 = 0.00001
    phat0 = 0
    
    for i in range(cc0):
        ts_data0 = ts_data[:,i]
        ts_vari0 = ts_vari[:,i]

        #print(ts_vari0.shape)
        #print(ts_data0.shape)
        v0 = max(ts_data0)/my
        p0 = k0,v0, amp0, pha0, ampt0, phat0
        popt2, pcov2 = curve_fit(def_linearSeasonSemi, dt_yr, ts_data0, p0, sigma=ts_vari0, absolute_sigma=False)
        vel[i] = popt2[1]
        amp[i] = np.sqrt(popt2[2]**2 + popt2[3]**2)
        pha[i] = np.arctan((popt2[3]/popt2[2]))
        
        ampt[i] = np.sqrt(popt2[4]**2 + popt2[5]**2)
        phat[i] = np.arctan((popt2[5]/popt2[4]))
        
        
        vel_std[i] = np.sqrt(pcov2[1,1])
        
        #print(vel)
    return vel, vel_std, amp, pha, ampt, phat
    

def split_list(nn, split_numb=4):

    dn = round(nn/int(split_numb))
    
    idx = []
    for i in range(split_numb):
        a0 = i*dn
        b0 = (i+1)*dn
        
        if i == (split_numb - 1):
            b0 = nn
        
        if not a0 > b0:
            idx0 = np.arange(a0,b0)
            #print(idx0)
            idx.append(idx0)
            
    return idx

def parallel_process(array, function, n_jobs=16, use_kwargs=False):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return [function(**a) if use_kwargs else function(a) for a in tqdm(array[:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[:]]
        else:
            futures = [pool.submit(function, a) for a in array[:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return out    

def velocity_wls(data0):
    A, ts_data, ts_vari = data0
    ts_vari[ts_vari==0]=1 # consider the zero nugget situtation
    rr0, cc0 = ts_vari.shape    
    vel = np.zeros((cc0,),dtype = np.float32)
    for i in range(cc0):
        AA0 = np.dot(np.transpose(A),inv(np.diag(ts_vari[:,i])))
        AA1 = np.dot(AA0,A)
        yy0 = np.dot(AA0,ts_data[:,i])
        X = np.dot(np.linalg.pinv(AA1), yy0)
        vel[i] = X[0]    
    return vel


def estimate_model_para(inps):
    # read time-series data
    print('reading data from file {} ...'.format(inps.timeseries_file))
    ts_data, atr = readfile.read(inps.timeseries_file)
    dt_yr = inps.yr_diff
    if not inps.variance:
        ts_vari = np.ones((ts_data.shape))
    else:
        ts_vari, atr0 = readfile.read(inps.variance)
        
    ts_data = ts_data[inps.dropDate, :, :].reshape(inps.numDate, -1)
    ts_vari = ts_vari[inps.dropDate, :, :].reshape(inps.numDate, -1)
    
    if atr['UNIT'] == 'mm':
        ts_data *= 1./1000.
    length, width = int(atr['LENGTH']), int(atr['WIDTH'])
    #A = timeseries.get_design_matrix4average_velocity(inps.dateList)
     
    split_numb = 2000
    idx_list = split_list(int(length*width),split_numb = split_numb)
    
    data_parallel = []
    for i in range(split_numb):
        data0 = (dt_yr,ts_data[:,idx_list[i]],ts_vari[:,idx_list[i]])
        data_parallel.append(data0)
    
    if inps.model =='linear':
        future = parallel_process(data_parallel, model_linear, n_jobs=inps.parallelNumb, use_kwargs=False)
    elif inps.model =='linear_season':
        future = parallel_process(data_parallel, model_linearSeason, n_jobs=inps.parallelNumb, use_kwargs=False) 
    elif inps.model =='linear_season_semiseason':
        future = parallel_process(data_parallel, model_linearSeasonSemi, n_jobs=inps.parallelNumb, use_kwargs=False) 
    
    zz = np.zeros((length*width,),dtype = np.float32)
    zz_std = np.zeros((length*width,),dtype = np.float32)
    
    for i in range(split_numb):
        id0 = idx_list[i]
        gg = future[i]
        zz[id0] = gg[0]
        zz_std[id0] = gg[1]
        
    vel_all = zz.reshape(length,width)
    vel_std = zz_std.reshape(length,width)
    
    if inps.model == 'linear_season': 
        aa = np.zeros((length*width,),dtype = np.float32)
        aa_std = np.zeros((length*width,),dtype = np.float32)
        
        pp = np.zeros((length*width,),dtype = np.float32)
        pp_std = np.zeros((length*width,),dtype = np.float32)
        
        for i in range(split_numb):
            id0 = idx_list[i]
            gg = future[i]
            aa[id0] = gg[2]         
            pp[id0] = gg[3]
        
        amp_all = aa.reshape(length,width)
        pha_all = pp.reshape(length,width)
        
    if inps.model == 'linear_season_semiseason': 
        aa = np.zeros((length*width,),dtype = np.float32)
        pp = np.zeros((length*width,),dtype = np.float32)
        
        aat = np.zeros((length*width,),dtype = np.float32)
        ppt = np.zeros((length*width,),dtype = np.float32)
        
        for i in range(split_numb):
            id0 = idx_list[i]
            gg = future[i]
            aa[id0] = gg[2]         
            pp[id0] = gg[3]
            
            aat[id0] = gg[4] 
            ppt[id0] = gg[5]
        
        amp_all = aa.reshape(length,width)
        pha_all = pp.reshape(length,width)
        
        ampt_all = aat.reshape(length,width)
        phat_all = ppt.reshape(length,width)
        #phat_std = ppt_std.reshape(length,width) 
    
    #vel = np.zeros((length,width))
    #vel = vel.flatten()
    #rr0, cc0 = ts_vari.shape
    #for i in range(cc0):
    #    print_progress(i+1, cc0, prefix='point: ', suffix=str(i+1))
    #    AA0 = np.dot(np.transpose(A),inv(np.diag(ts_vari[:,i])))
    #    AA1 = np.dot(AA0,A)
    #    yy0 = np.dot(AA0,ts_data[:,i])
    #    X = np.dot(np.linalg.pinv(AA1), yy0)
    #    vel[i] = X[0]
    
    #vel = vel.reshape(length, width)
    # prepare attributes
    atr['FILE_TYPE'] = 'velocity'
    atr['UNIT'] = 'm/year'
    atr['START_DATE'] = inps.dateList[0]
    atr['END_DATE'] = inps.dateList[-1]
    atr['DATE12'] = '{}_{}'.format(inps.dateList[0], inps.dateList[-1])
    # config parameter
    print('add/update the following configuration metadata:\n{}'.format(configKeys))
    for key in configKeys:
        atr[key_prefix+key] = str(vars(inps)[key])

    # write to HDF5 file
    dsDict = dict()
    dsDict['velocity'] = vel_all
    dsDict['velStd'] = vel_std
    
    if inps.model =='linear_season':
        dsDict['annualAmp'] = amp_all
        dsDict['annualPha'] = pha_all
    
    if inps.model =='linear_season_semiseason':    
        dsDict['annualAmp'] = amp_all        
        dsDict['annualPha'] = pha_all
        dsDict['semiAmp'] = ampt_all
        dsDict['semiPha'] = phat_all
        #dsDict['semiseaPhaStd'] = phat_std   
        
    writefile.write(dsDict, out_file=inps.outfile, metadata=atr)
    return inps.outfile


############################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)
    inps = read_date_info(inps)

    # --update option
    if inps.update_mode and run_or_skip(inps) == 'skip':
        return inps.outfile

    inps.outfile = estimate_model_para(inps)
    return inps.outfile


############################################################################
if __name__ == '__main__':
    main()

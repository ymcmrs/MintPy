#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright(c) 2013-2019, Heresh Fattahi, Zhang Yunjun     #
# Author:  Heresh Fattahi, Zhang Yunjun                    #
############################################################
# Modified from timeseries2velocity.py written by Yunjun Zhang & Heresh Fattahi
# July 15, 2019 

# Modified Sep. 13, 2019, support weighted-least-squares (WlS) and parallel processing
# By Yunmeng Cao

import sys
import os
import argparse
import numpy as np
from mintpy.objects import timeseries, giantTimeseries, HDFEOS
from mintpy.utils import readfile, writefile, ptime, utils as ut
from numpy.linalg import inv

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

dataType = np.float32
# key configuration parameter name
key_prefix = 'mintpy.velocity.'
configKeys = ['excludeDate']


############################################################################
EXAMPLE = """example:
  timeseries2velocity_wls.py  timeSeries_ECMWF_demErr.h5 variogramTsModel.h5
  timeseries2velocity_wls.py  timeseries_ECMWF_demErr_ramp.h5  -t smallbaselineApp.cfg --update
  timeseries2velocity_wls.py  timeseries_ECMWF_demErr_ramp.h5  -t KyushuT73F2980_2990AlosD.template
  timeseries2velocity_wls.py  timeseries.h5  --start-date 20080201
  timeseries2velocity_wls.py  timeseries.h5  --start-date 20080201  --end-date 20100508
  timeseries2velocity_wls.py  timeseries.h5  --exclude-date exclude_date.txt

  timeseries2velocity_wls.py  LS-PARAMS.h5
  timeseries2velocity_wls.py  NSBAS-PARAMS.h5
  timeseries2velocity_wls.py  TS-PARAMS.h5
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
    parser.add_argument('timeseries_variance',
                        help='Time series file of the variance data.')
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


def estimate_linear_velocity(inps):
    # read time-series data
    print('reading data from file {} ...'.format(inps.timeseries_file))
    ts_data, atr = readfile.read(inps.timeseries_file)
    ts_data = ts_data[inps.dropDate, :, :].reshape(inps.numDate, -1)
    if atr['UNIT'] == 'mm':
        ts_data *= 1./1000.
    length, width = int(atr['LENGTH']), int(atr['WIDTH'])

    # The following is equivalent
    # X = scipy.linalg.lstsq(A, ts_data, cond=1e-15)[0]
    # It is not used because it can not handle NaN value in ts_data
    A = timeseries.get_design_matrix4average_velocity(inps.dateList)
    X = np.dot(np.linalg.pinv(A), ts_data)
    vel = np.array(X[0, :].reshape(length, width), dtype=dataType)

    # velocity STD (Eq. (10), Fattahi and Amelung, 2015)
    ts_diff = ts_data - np.dot(A, X)
    t_diff = A[:, 0] - np.mean(A[:, 0])
    vel_std = np.sqrt(np.sum(ts_diff ** 2, axis=0) / np.sum(t_diff ** 2)  / (inps.numDate - 2))
    vel_std = np.array(vel_std.reshape(length, width), dtype=dataType)

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
    dsDict['velocity'] = vel
    dsDict['velocityStd'] = vel_std
    writefile.write(dsDict, out_file=inps.outfile, metadata=atr)
    return inps.outfile

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

def estimate_linear_velocity_wls(inps):
    # read time-series data
    print('reading data from file {} ...'.format(inps.timeseries_file))
    ts_data, atr = readfile.read(inps.timeseries_file)
    ts_vari, atr0 = readfile.read(inps.timeseries_variance)
    ts_data = ts_data[inps.dropDate, :, :].reshape(inps.numDate, -1)
    ts_vari = ts_vari[inps.dropDate, :, :].reshape(inps.numDate, -1)
    
    if atr['UNIT'] == 'mm':
        ts_data *= 1./1000.
    length, width = int(atr['LENGTH']), int(atr['WIDTH'])
    A = timeseries.get_design_matrix4average_velocity(inps.dateList)
    
    split_numb = 2000
    idx_list = split_list(int(length*width),split_numb = split_numb)
    
    data_parallel = []
    for i in range(split_numb):
        data0 = (A,ts_data[:,idx_list[i]],ts_vari[:,idx_list[i]])
        data_parallel.append(data0)
    
    future = parallel_process(data_parallel, velocity_wls, n_jobs=inps.parallelNumb, use_kwargs=False)
    
    zz = np.zeros((length*width,),dtype = np.float32)
    for i in range(split_numb):
        id0 = idx_list[i]
        gg = future[i]
        zz[id0] = gg
        
    vel_all = zz.reshape(length,width)
    
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
    #dsDict['velocityStd'] = vel_std
    writefile.write(dsDict, out_file=inps.outfile, metadata=atr)
    return inps.outfile


############################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)
    inps = read_date_info(inps)

    # --update option
    if inps.update_mode and run_or_skip(inps) == 'skip':
        return inps.outfile

    inps.outfile = estimate_linear_velocity_wls(inps)
    return inps.outfile


############################################################################
if __name__ == '__main__':
    main()

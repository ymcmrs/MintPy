#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright(c) 2013-2019, Heresh Fattahi, Zhang Yunjun     #
# Author:  Heresh Fattahi, Zhang Yunjun, Yunmeng Cao       #
############################################################

import h5py
import os
import argparse
import numpy as np
from mintpy.objects import timeseries, giantTimeseries, HDFEOS
from mintpy.utils import readfile, writefile, ptime, utils as ut
import sys
from joblib import Parallel, delayed



dataType = np.float32
# key configuration parameter name
key_prefix = 'mintpy.velocity.'
configKeys = ['excludeDate']


############################################################################
EXAMPLE = """example:
  timeseries2velocity_interative.py  timeSeries_ECMWF_demErr.h5
  timeseries2velocity_interative.py  timeseries_ECMWF_demErr_ramp.h5  -t smallbaselineApp.cfg --update
  timeseries2velocity_interative.py  timeseries_ECMWF_demErr_ramp.h5  -t KyushuT73F2980_2990AlosD.template
  timeseries2velocity_interative.py  timeseries.h5  --start-date 20080201
  timeseries2velocity_interative.py  timeseries.h5  --start-date 20080201  --end-date 20100508
  timeseries2velocity_interative.py  timeseries.h5  --exclude-date exclude_date.txt

  timeseries2velocity_interative.py  LS-PARAMS.h5
  timeseries2velocity_interative.py  NSBAS-PARAMS.h5
  timeseries2velocity_interative.py  TS-PARAMS.h5
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
    parser = argparse.ArgumentParser(description='Inverse velocity from time-series.',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=TEMPLATE+'\n'+EXAMPLE)

    parser.add_argument('timeseries_file', help='Time series file for velocity inversion.')
    parser.add_argument('--start-date','-s', dest='startDate',help='start date for velocity estimation')
    parser.add_argument('--end-date','-e', dest='endDate',help='end date for velocity estimation')
    parser.add_argument('--exclude', '--ex', dest='excludeDate', nargs='+', default=[],
                        help='date(s) not included in velocity estimation, could be list of string or text file, i.e.:\n' +
                             '--exclude 20040502 20060708 20090103\n' +
                             '--exclude exclude_date.txt\n'+DROP_DATE_TXT)
    parser.add_argument('--template', '-t', dest='template_file',
                        help='template file with the following items:'+TEMPLATE)
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

def estimate_linear_velocity_interative(dateList,ts_data,remove_num = 10, UNIT ='m'):
    # read time-series data
    #print('reading data from file {} ...'.format(inps.timeseries_file))
    #ts_data, atr = readfile.read(inps.timeseries_file)
    #ts_data = ts_data[inps.dropDate, :, :].reshape(inps.numDate, -1)
    if UNIT == 'mm':
        ts_data *= 1./1000.

    A = timeseries.get_design_matrix4average_velocity(dateList)
    X = np.dot(np.linalg.pinv(A), ts_data)
    t_diff = A[:, 0] - np.mean(A[:, 0])
    #vel = np.array(X[0, :].reshape(length, width), dtype=dataType)
    vel0 = X[0]
    # velocity STD (Eq. (10), Fattahi and Amelung, 2015)
    ts_diff = ts_data - np.dot(A, X)
    sort_diff = sorted(list(np.abs(ts_diff)))
    k0 = sort_diff[len(sort_diff)-remove_num-1] + 0.0001
    
    fg = np.where(abs(ts_diff)<k0)
    fg = np.asarray(fg,dtype=int)
    #print(dateList)
    #print(fg.shape)
    dateList = np.asarray(dateList)
    fg = fg.reshape(fg.shape[1],)
    dateList1 = dateList[fg]
    ts_data1 = ts_data[fg]
    N_left = len(fg)
    
    A1 = timeseries.get_design_matrix4average_velocity(dateList1)
    X1 = np.dot(np.linalg.pinv(A1), ts_data1)
    #print(X1)
    vel1 = X1[0]
    ts_diff1 = ts_data1 - np.dot(A1, X1)
    t_diff1 = A1[:, 0] - np.mean(A1[:, 0])
    vel_std = np.sqrt(np.sum(ts_diff1 ** 2, axis=0) / np.sum(t_diff1 ** 2)  / (len(dateList) - 2))

    return vel1, vel_std

def print_progress(iteration, total, prefix='calculating:', suffix='complete', decimals=1, barLength=50, elapsed_time=None):
    """Print iterations progress - Greenstick from Stack Overflow
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int) 
        barLength   - Optional  : character length of bar (Int) 
        elapsed_time- Optional  : elapsed time in seconds (Int/Float)
    
    Reference: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    if elapsed_time:
        sys.stdout.write('%s [%s] %s%s    %s    %s secs\r' % (prefix, bar, percents, '%', suffix, int(elapsed_time)))
    else:
        sys.stdout.write('%s [%s] %s%s    %s\r' % (prefix, bar, percents, '%', suffix))
    sys.stdout.flush()
    if iteration == total:
        print("\n")

    '''
    Sample Useage:
    for i in range(len(dateList)):
        print_progress(i+1,len(dateList))
    '''
    return

def write_variogram_h5(datasetDict, out_file, metadata=None, ref_file=None, compression=None):
    #output = 'variogramStack.h5'
    'lags                  1 x N '
    'semivariance          M x N '
    'sills                 M x 1 '
    'ranges                M x 1 '
    'nuggets               M x 1 '
    
    if os.path.isfile(out_file):
        print('delete exsited file: {}'.format(out_file))
        os.remove(out_file)

    print('create HDF5 file: {} with w mode'.format(out_file))
    with h5py.File(out_file, 'w') as f:
        for dsName in datasetDict.keys():
            data = datasetDict[dsName]
            ds = f.create_dataset(dsName,
                              data=data,
                              compression=compression)
        
        for key, value in metadata.items():
            f.attrs[key] = str(value)
            #print(key + ': ' +  value)
    print('finished writing to {}'.format(out_file))
        
    return out_file

#########################################################################

INTRODUCTION = '''
#############################################################################
   Copy Right(c): 2019, Yunmeng Cao   @PySME v1.0
   
   Stochastic model estimation for single or time-series of interferograms.
'''

EXAMPLE = '''
    Usage:
            variogram_insar.py ifgramStack.h5
            variogram_insar.py ifgramStack.h5 -m maskTempCoh.h5 --sample_numb 5000
            variogram_insar.py ifgramStack.h5 -m maskTempCoh.h5 --bin_numb 30
            
            variogram_insar.py timeseries.h5
            variogram_insar.py timeseries.h5 --bin_numb 20
            variogram_insar.py timeseries.h5 -m maskTempCoh.h5 --bin_numb 30
##############################################################################
'''


def cmdLineParse():
    parser = argparse.ArgumentParser(description='Check common busrts for TOPS data.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('timeseries_file',help='input file name (ifgramStack.h5 or timeseires.h5).')
    parser.add_argument('-m','--mask', dest='mask_file', metavar='FILE',
                      help='mask file for masking those large deforming or low-coherence pixels')
    parser.add_argument('-o','--out_file', dest='outfile', metavar='FILE',
                      help='name of the output file')
    parser.add_argument('--variogram_model', dest='variogram_model', default='spherical',
                      help='variogram model used to fit the variance samples')
    parser.add_argument('--sample_numb', dest='sample_numb',type=int,default=3000,metavar='NUM',
                      help='number of samples used to calculate the variance sample')
    parser.add_argument('--bin_numb', dest='bin_numb',type=int,default=30, metavar='NUM',
                      help='number of bins used to fit the variogram model')
    parser.add_argument('--used_bin_ratio', dest='used_bin_ratio',type=float,default=1.0, metavar='NUM',
                      help='used bin ratio for mdeling the structure model.')

    inps = parser.parse_args()

    return inps

################################################################################    
    
    
def main(argv):
    
    inps = cmdLineParse() 

    ts_data, atr = readfile.read(inps.timeseries_file)
    data = ts_data.reshape(ts_data.shape[0],ts_data.shape[1]*ts_data.shape[2])
    UNIT = atr['UNIT']
    tsobj = timeseries(inps.timeseries_file)
    date_list = tsobj.get_date_list()
    velocity = np.zeros(ts_data.shape[1]*ts_data.shape[2],)
    velocity_std = np.zeros(ts_data.shape[1]*ts_data.shape[2],)
    #numb_used = np.zeros(ts_data.shape[1]*ts_data.shape[2],)
    
    for i in range(ts_data.shape[1]*ts_data.shape[2]):
        print_progress(i+1, ts_data.shape[1]*ts_data.shape[2], prefix='Pixel Number: ', suffix=i)
        y0 = data[:,i]
        vel1, vel_std = estimate_linear_velocity_interative(date_list,y0,remove_num = 10, UNIT = UNIT)
        velocity[i] = vel1
        velocity_std[i] = vel_std
        #numb_used[i] = N_left
    velocity = velocity.reshape(ts_data.shape[1], ts_data.shape[2])
    velocity_std = velocity_std.reshape(ts_data.shape[1], ts_data.shape[2])
    
    datasetDict = dict()
    datasetDict['velocity'] = velocity
    datasetDict['velocityStd'] = velocity_std
    write_variogram_h5(datasetDict, inps.outfile, metadata=atr, ref_file=None, compression=None)
       
    
    sys.exit(1)

if __name__ == '__main__':
    main(sys.argv[:])

#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright(c) 2018-2019, Zhang Yunjun                     #
# Author:  Yunmeng Cao, 2019                               #
############################################################

import os
import h5py
import argparse
import numpy as np
from mintpy.objects import timeseries
from mintpy.utils import ptime, readfile


################################################################################
EXAMPLE = """example:
  2to3_ifgramStack.py  unwrapIfgram.h5 coherence.h5 -o ifgramStack.h5
"""

def unify_date(date0):
    date0 = str(date0)
    if len(date0)==8:
        date1 = date0
    else:
        if int(date0[0:2]) > 50:
            date1 = '19' + date0
        else:
            date1 = '20' + date0
    return date1

def create_parser():
    """ Command line parser """
    parser = argparse.ArgumentParser(description='Convert interferogram file from py2-PySAR to py3-MintPy format.',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)

    parser.add_argument('unw_file', help='file to be converted')
    parser.add_argument('cor_file', help='file to be converted')
    parser.add_argument('-o', '--output', dest='outfile', required=True, help='output file name')
    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    return inps

def write_h5(datasetDict, out_file, metadata=None, ref_file=None, compression=None):
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
################################################################################
def run_2to3_timeseries(py2_file, py3_file):
    """Convert timeseries file from py2-MintPy format to py3-MintPy format"""
    # read data from py2_file
    atr = readfile.read_attribute(py2_file)
    length, width = int(atr['LENGTH']), int(atr['WIDTH'])
    with h5py.File(py2_file, 'r') as f:
        date_list = list(f['timeseries'].keys())
        num_date = len(date_list)
        ts_data = np.zeros((num_date, length, width), np.float32)
        print('reading time-series ...')
        prog_bar = ptime.progressBar(maxValue=num_date)
        for i in range(num_date):
            ts_data[i, :, :] = f['timeseries/{}'.format(date_list[i])][:]
            prog_bar.update(i+1, suffix=date_list[i])
        prog_bar.close()

    # prepare metadata
    bperp = np.array([float(i) for i in atr['P_BASELINE_TIMESERIES'].split()], dtype=np.float32)
    dates = np.array(date_list, np.string_)
    atr['REF_DATE'] = date_list[0]
    for key in ['P_BASELINE_TIMESERIES', 
                'P_BASELINE_TOP_TIMESERIES',
                'P_BASELINE_BOTTOM_TIMESERIES']:
        try:
            atr.pop(key)
        except:
            pass

    # write to py3_file
    ts_obj = timeseries(py3_file)
    ts_obj.write2hdf5(data=ts_data, dates=dates, bperp=bperp, metadata=atr)
    return py3_file

def run_2to3_ifgramStack(py2_unwfile, py2_corfile, py3_file):
    """Convert timeseries file from py2-MintPy format to py3-MintPy format"""
    # read data from py2_file
    atr0 = readfile.read_attribute(py2_unwfile)
    length, width = int(atr0['LENGTH']), int(atr0['WIDTH'])
    #with h5py.File(py2_file, 'r') as f:
    #    date_list = list(f['timeseries'].keys())
    #    num_date = len(date_list)
    #    ts_data = np.zeros((num_date, length, width), np.float32)
    #    print('reading time-series ...')
    #    prog_bar = ptime.progressBar(maxValue=num_date)
    #    for i in range(num_date):
    #        ts_data[i, :, :] = f['timeseries/{}'.format(date_list[i])][:]
    #        prog_bar.update(i+1, suffix=date_list[i])
    #    prog_bar.close()

    with h5py.File(py2_unwfile, 'r') as f:
        data_name_list = list(f['interferograms'].keys())
        num_date = len(data_name_list)
        ts_unw_data = np.zeros((num_date, int(length), int(width)), np.float32)
        
        bperp_top = np.zeros((num_date,),np.float32)
        bperp_bottom = np.zeros((num_date,),np.float32)
        bperp = np.zeros((num_date,),np.float32)
        
        date = np.zeros((num_date,2),np.string_)
        date00 = np.ones((num_date,2)) #3x4数组
        date = date00.astype(np.string_)
        date = np.asarray(date, dtype='<S8')
        
        dropIfgram = np.ones((num_date,),dtype=bool)
        
        print('reading unwrapIfgram ...')
        prog_bar = ptime.progressBar(maxValue=num_date)
        for i in range(num_date):
            STR  = '/interferograms/' + data_name_list[i]
            STR0  = 'interferograms/' + data_name_list[i] + '/' +data_name_list[i]
            ts_unw_data[i, :, :] = f[STR0][:]
            #atr = readfile.read_attribute(py2_unwfile, datasetName = STR)
            atr = f[STR].attrs
            DATE12 = str(atr['DATE12'], encoding='utf-8')
            #STR00 = str(atrr['DATE12'], encoding='utf-8')
            #print(DATE12)
            date1 = DATE12.split('-')[0]
            date2 = DATE12.split('-')[1]
            #print(unify_date(date1))
            date1 = unify_date(date1)
            date2 = unify_date(date2)
            #print(date1)
            #print(date2)
            date[i,0] = unify_date(date1)
            date[i,1] = unify_date(date2)
            bperp_bottom[i] = float(atr['P_BASELINE_BOTTOM_HDR'])
            bperp_top[i] = float(atr['P_BASELINE_TOP_HDR'])
            bperp[i] = 1/2*(bperp_bottom[i]+bperp_top[i])
            #ts_data[i, :, :] = f['interferograms/{}'.format][:]
            prog_bar.update(i+1, suffix=data_name_list[i])
        prog_bar.close()
        
    with h5py.File(py2_corfile, 'r') as f:
        data_name_list = list(f['coherence'].keys())
        num_date = len(data_name_list)
        ts_cor_data = np.zeros((num_date, int(length), int(width)), np.float32)
        print('reading coherence ...')
        prog_bar = ptime.progressBar(maxValue=num_date)
        for i in range(num_date):
            STR  = '/' + data_name_list[i].split('.')[0]
            STR0  = 'coherence/' + data_name_list[i] + '/' +data_name_list[i]
            ts_cor_data[i, :, :] = f[STR0][:]
            #ts_cor_data[i, :, :] = f['interferograms/{}'.format][:]
            prog_bar.update(i+1, suffix=data_name_list[i])
        prog_bar.close()
    # prepare metadata
    #bperp = np.array([float(i) for i in atr['P_BASELINE_TIMESERIES'].split()], dtype=np.float32)
    #dates = np.array(date_list, np.string_)
    #atr['REF_DATE'] = date_list[0]
    
    #atr0['REF_DATE'] = date_list[0]
    atr0['FILE_TYPE'] = 'ifgramStack'
    atr0['UNIT'] = 'radian'
    
    datasetDict = dict()
    datasetDict['unwrapPhase'] = ts_unw_data
    datasetDict['coherence'] = ts_cor_data
    datasetDict['bperp_top'] = bperp_top
    datasetDict['bperp_bottom'] = bperp_bottom
    datasetDict['bperp'] = bperp
    datasetDict['date'] = date
    datasetDict['dropIfgram'] = dropIfgram
    #print(date)  
    # write to py3_file
    #ts_obj = timeseries(py3_file)
    #ts_obj.write2hdf5(data=ts_data, dates=dates, bperp=bperp, metadata=atr0)
    write_h5(datasetDict, py3_file, metadata=atr0, ref_file=None, compression=None)
    
    return py3_file
################################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)
    run_2to3_ifgramStack(inps.unw_file, inps.cor_file, inps.outfile)
    print('Done.')
    return inps.outfile


################################################################################
if __name__ == '__main__':
    main()

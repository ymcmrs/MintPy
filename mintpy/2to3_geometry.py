#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright(c) 2018-2019, Zhang Yunjun                     #
# Author:   Yunmeng Cao, 2019                              #
############################################################


import h5py
import argparse
import numpy as np
from mintpy.objects import timeseries
from mintpy.utils import ptime, readfile
import os

################################################################################
EXAMPLE = """example:
  2to3_geometry.py  demRadar.h5 -o geometryRadar.h5    [incidence_angle.h5 and range_distance.h5 should be available]
  2to3_geometry.py  demGeo.h5 -o geomeGeo.h5      [geo2rdc.h5 should be available]
"""


def create_parser():
    """ Command line parser """
    parser = argparse.ArgumentParser(description='Convert geometry file from py2-PySAR to py3-MintPy format.',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)

    parser.add_argument('file', help='file to be converted')
    parser.add_argument('-o', '--output', dest='outfile', required=True, help='output file name')
    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    return inps


################################################################################
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

def run_2to3_geometryRadar(py2_file, py3_file):
    """Convert geometry file from py2-MintPy format to py3-MintPy format"""
    # read data from py2_file
    inc_file = 'incidenceAngle.h5'
    range_file = 'rangeDistance.h5'
    atr0 = readfile.read_attribute(inc_file)
    length, width = int(atr0['LENGTH']), int(atr0['WIDTH'])
    with h5py.File(py2_file, 'r') as f:
        dem_data = np.zeros((length, width), np.float32)
        dem_data = f['dem/dem'][:]    
    with h5py.File(inc_file, 'r') as f:
        inc_data = np.zeros((length, width), np.float32)
        inc_data = f['mask'][:]      
    with h5py.File(range_file, 'r') as f:
        range_data = np.zeros((length, width), np.float32)
        range_data = f['mask'][:]  
        
    datasetDict = dict()
    datasetDict['height'] = np.asarray(dem_data,dtype='float32')
    datasetDict['incidenceAngle'] = inc_data
    datasetDict['slantRangeDistance'] = range_data

    write_h5(datasetDict, py3_file, metadata=atr0, ref_file=None, compression=None)
   
    return py3_file

def run_2to3_geometryGeo(py2_file, py3_file):
    """Convert geometry file from py2-MintPy format to py3-MintPy format"""
    # read data from py2_file
    lt_file = 'geo2rdc.h5'
    atr0 = readfile.read_attribute(py2_file)
    length, width = int(atr0['LENGTH']), int(atr0['WIDTH'])
    
    with h5py.File(py2_file, 'r') as f:
        dem_data = np.zeros((length, width), np.float32)
        dem_data = f['dem/dem'][:]

    with h5py.File(lt_file, 'r') as f:
        lt_data = np.zeros((length, width), np.complex64)
        lt_data = f['lt/lt'][:]
        range_data = (lt_data.real).astype(np.float32)
        azimuth_data = (lt_data.imag).astype(np.float32)
   
        
    datasetDict = dict()
    datasetDict['height'] = np.asarray(dem_data,dtype='float32')
    datasetDict['rangeCoord'] = range_data
    datasetDict['azimuthCoord'] = azimuth_data

    write_h5(datasetDict, py3_file, metadata=atr0, ref_file=None, compression=None)

    return py3_file
################################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)
    if not inps.file =='demGeo.h5':
        run_2to3_geometryRadar(inps.file, inps.outfile)
    else:
        run_2to3_geometryGeo(inps.file, inps.outfile)
        
    print('Done.')
    return inps.outfile


################################################################################
if __name__ == '__main__':
    main()

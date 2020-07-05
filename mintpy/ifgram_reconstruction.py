#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Heresh Fattahi, 2013               #
############################################################
import os
import h5py
import argparse
import numpy as np
from mintpy.objects import ifgramStack
from mintpy.utils import readfile, writefile

def write_h5(datasetDict, out_file, metadata=None, ref_file=None, compression=None):

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

#####################################################################################
EXAMPLE = """example:
  ifgram_reconstruction.py  timeseries.h5  inputs/ifgramStack.h5  
  ifgram_reconstruction.py  timeseries_ECWMF_ramp_demErr.h5  inputs/ifgramStack.h5  -d reconCorUnwrapPhase
"""

def create_parser():
    parser = argparse.ArgumentParser(description='Reconstruct network of interferograms from time-series',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)

    parser.add_argument('timeseries_file', type=str, help='time-series file.')
    parser.add_argument('-r', dest='ifgram_file', type=str, default='./inputs/ifgramStack.h5',
                        help='reference interferograms stack file')
    parser.add_argument('-o','--output', dest='out_file', default='reconUnwrapIfgram.h5',
                        help='output filename for the reconstructed interferograms.')
    return parser

def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    return inps


#####################################################################################
def timeseries2ifgram(ts_file, ifgram_file, out_file='reconUnwrapIfgram.h5'):
    # read time-series
    atr = readfile.read_attribute(ts_file)
    atr0 = readfile.read_attribute(ifgram_file)
    range2phase = -4.*np.pi / float(atr['WAVELENGTH'])
    print('reading timeseries data from file {} ...'.format(ts_file))
    ts_data = readfile.read(ts_file)[0] * range2phase
    num_date, length, width = ts_data.shape
    ts_data = ts_data.reshape(num_date, -1)

    # reconstruct unwrapPhase
    print('reconstructing the interferograms from timeseries')
    stack_obj = ifgramStack(ifgram_file)
    stack_obj.open(print_msg=False)
    A1 = stack_obj.get_design_matrix4timeseries(stack_obj.get_date12_list(dropIfgram=True))[0]
    date_list = stack_obj.get_date_list(dropIfgram=True)
    date12 = stack_obj.get_date12_list(dropIfgram=True)
    print(date12)
   
    with h5py.File(ifgram_file, 'r') as f:
        pbaseIfgram = f['bperp'][:]
        pbaseIfgram = pbaseIfgram[f['dropIfgram'][:]]

    num_ifgram = A1.shape[0]
    A0 = -1.*np.ones((num_ifgram, 1))
    A = np.hstack((A0, A1))
    ifgram_est = np.dot(A, ts_data).reshape(num_ifgram, length, width)
    ifgram_est = np.array(ifgram_est, dtype=ts_data.dtype)
    del ts_data
    
    date12m = np.zeros((len(date12),2),dtype = '<S8')
    for i in range(len(date12)):
        k0 = date12[i]
        date12m[i,0]=k0.split('_')[0] 
        date12m[i,1] = k0.split('_')[1]

    dropIfgram = np.zeros((len(date12),),dtype = bool) 
    print(np.asarray(date12m,dtype = '<S8'))
    # write to ifgram file
    dsDict = {}
    dsDict['unwrapPhase'] = ifgram_est
    dsDict['dropIfgram'] = dropIfgram
    dsDict['date_list'] = np.asarray(date_list,dtype = '<S8')
    dsDict['date'] = np.asarray(date12m,dtype = '<S8')
    dsDict['bperp'] = np.asarray(pbaseIfgram)
    write_h5(dsDict, out_file = out_file, metadata=atr0, ref_file=None, compression=None)
    
    #writefile.write(dsDict, out_file=out_file, ref_file=ifgram_file)
    return ifgram_file


def main(iargs=None):
    inps = cmd_line_parse(iargs)
    timeseries2ifgram(inps.timeseries_file, inps.ifgram_file, inps.out_file)
    return


#####################################################################################
if __name__ == '__main__':
    main()

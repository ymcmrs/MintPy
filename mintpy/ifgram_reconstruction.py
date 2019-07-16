#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright(c) 2013-2019, Zhang Yunjun, Heresh Fattahi     #
# Author:  Zhang Yunjun, Heresh Fattahi, Yunmeng Cao       #
############################################################


import argparse
import numpy as np
from mintpy.objects import ifgramStack
from mintpy.utils import readfile, writefile

# Modify 2019-07-09 By Yunmeng Cao
# reference reconstruct consider the dropifgrams in the original ifgram_file

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
    range2phase = -4.*np.pi / float(atr['WAVELENGTH'])
    print('reading timeseries data from file {} ...'.format(ts_file))
    ts_data = readfile.read(ts_file)[0] * range2phase
    num_date, length, width = ts_data.shape
    ts_data = ts_data.reshape(num_date, -1)

    # reconstruct unwrapPhase
    print('reconstructing the interferograms from timeseries')
    stack_obj = ifgramStack(ifgram_file)
    stack_obj.open(print_msg=False)  
    date12 = readfile.read(ifgram_file, datasetName='date')[0]
    bperp = readfile.read(ifgram_file, datasetName='bperp')[0]
    droplist = readfile.read(ifgram_file, datasetName='dropIfgram')[0]
    date = stack_obj.get_date_list(dropIfgram=True)
    date = np.asarray(date,dtype='|S8')
    print(date)
    meta = readfile.read_attribute(ifgram_file,datasetName=None)
    
    #A1 = stack_obj.get_design_matrix4timeseries(stack_obj.get_date12_list(dropIfgram=False))[0]
    A1 = stack_obj.get_design_matrix4timeseries(stack_obj.get_date12_list(dropIfgram=True))[0]
    num_ifgram = A1.shape[0]
    A0 = -1.*np.ones((num_ifgram, 1))
    A = np.hstack((A0, A1))
    ifgram_est = np.dot(A, ts_data).reshape(num_ifgram, length, width)
    ifgram_est = np.array(ifgram_est, dtype=ts_data.dtype)
    del ts_data
    
    
    # write to ifgram file
    dsDict = {}
    dsDict['unwrapPhase'] = ifgram_est
    dsDict['date'] = date12[droplist]
    dsDict['date_list'] = date
    dsDict['bperp'] = bperp[droplist]
    dsDict['dropIfgram'] = droplist[droplist]
    
    writefile.write(dsDict, out_file=out_file, metadata=meta)
    return ifgram_file


def main(iargs=None):
    inps = cmd_line_parse(iargs)
    timeseries2ifgram(inps.timeseries_file, inps.ifgram_file, inps.out_file)
    return


#####################################################################################
if __name__ == '__main__':
    main()

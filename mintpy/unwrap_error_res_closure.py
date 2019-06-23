#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright(c) 2013-2019, Zhang Yunjun, Heresh Fattahi     #
# Author:  Zhang Yunjun, Heresh Fattahi                    #
############################################################


import os
import argparse
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

try:
    from cvxopt import matrix
except ImportError:
    raise ImportError('Cannot import cvxopt')
try:
    from skimage import measure
except ImportError:
    raise ImportError('Could not import skimage!')

from mintpy.objects import ifgramStack
from mintpy.objects.conncomp import connectComponent
from mintpy.utils import ptime, readfile, utils as ut, plot as pp
from mintpy.utils.solvers import l1regls
from mintpy import ifgram_inversion as ifginv


def read_template2inps(template_file, inps=None):
    """Read input template options into Namespace inps"""
    if not inps:
        inps = cmd_line_parse()
    inpsDict = vars(inps)
    print('read options from tempalte file: '+os.path.basename(inps.template_file))
    template = readfile.read_template(template_file)
    template = ut.check_template_auto_value(template)

    key_list = [i for i in list(inpsDict.keys()) if key_prefix+i in template.keys()]
    for key in key_list:
        value = template[key_prefix+key]
        if value:
            if key in ['waterMaskFile']:
                inpsDict[key] = value
    return inps


def run_or_skip(inps):
    print('-'*50)
    print('update mode: ON')
    flag = 'skip'

    # check output dataset
    with h5py.File(inps.ifgram_file, 'r') as f:
        if inps.datasetNameOut not in f.keys():
            flag = 'run'
            print('1) output dataset: {} NOT found.'.format(inps.datasetNameOut))
        else:
            print('1) output dataset: {} exists.'.format(inps.datasetNameOut))
            to = float(f[inps.datasetNameOut].attrs.get('MODIFICATION_TIME', os.path.getmtime(inps.ifgram_file)))
            ti = float(f[inps.datasetNameIn].attrs.get('MODIFICATION_TIME', os.path.getmtime(inps.ifgram_file)))
            if ti > to:
                flag = 'run'
                print('2) output dataset is NOT newer than input dataset: {}.'.format(inps.datasetNameIn))
            else:
                print('2) output dataset is newer than input dataset: {}.'.format(inps.datasetNameIn))

    # result
    print('run or skip: {}.'.format(flag))
    return flag


##########################################################################################
def write_hdf5_file_patch(ifgram_file, data, box=None, dsName='unwrapPhase_phaseClosure'):
    """Write a patch of 3D dataset into an existing h5 file.
    Parameters: ifgram_file : string, name/path of output hdf5 file
                data : 3D np.array to be written
                box  : tuple of 4 int, indicating of (x0, y0, x1, y1) of data in file
                dsName : output dataset name
    Returns:    ifgram_file
    """
    num_ifgram, length, width = ifgramStack(ifgram_file).get_size(dropIfgram=False)
    if not box:
        box = (0, 0, width, length)
    num_row = box[3] - box[1]
    num_col = box[2] - box[0]

    # write to existing HDF5 file
    print('open {} with r+ mode'.format(ifgram_file))
    f = h5py.File(ifgram_file, 'r+')

    # get h5py.Dataset
    msg = 'dataset /{d} of {t:<10} in size of {s}'.format(d=dsName, t=str(data.dtype),
                                                          s=(num_ifgram, box[3], box[2]))
    if dsName in f.keys():
        print('update '+msg)
        ds = f[dsName]
    else:
        print('create '+msg)
        ds = f.create_dataset(dsName, (num_ifgram, num_row, num_col),
                              maxshape=(None, None, None),
                              chunks=True, compression=None)

    # resize h5py.Dataset if current size is not enough
    if ds.shape != (num_ifgram, length, width):
        ds.resize((num_ifgram, box[3], box[2]))

    # write data to file
    ds[:, box[1]:box[3], box[0]:box[2]] = data

    ds.attrs['MODIFICATION_TIME'] = str(time.time())
    f.close()
    print('close {}'.format(ifgram_file))
    return ifgram_file

##########################################################################################
INTRODUCTION = '''
#############################################################################
   Copy Right(c): 2019, Yunmeng Cao   This script is part of mintPy software.
   
   Correcting the unwrap error for an indendent island or land.
'''


EXAMPLE = """
    
    Example:
     
      unwrap_error_res_closure.py  ifgramStack.h5  IfgInvRes.h5  maskResRegion.h5  
      unwrap_error_res_closure.py  ifgramStack.h5  IfgInvRes.h5  maskResRegion.h5  -o ifgramStackCor.h5
"""


NOTE = """
  Correcting the unwrap error for cases: the research region is divided by water or forest e.g., islands
  
"""

###########################################################################################

def cmdLineParse():
    parser = argparse.ArgumentParser(description='Unwrap error correction based on the inversion error',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('ifgram_file',help='input ifgramStack file.')
    parser.add_argument('inversion_res', help='inversion residual file.')
    parser.add_argument('mask_file', help='common area that unwrapping error occurs (i.e., could occurs)')
    parser.add_argument('-o','--out_file', dest='out_file', metavar='FILE',
                      help='name of the output file')
   
    inps = parser.parse_args()

    return inps

################################################################################    
    
    
def main(argv):
    
    inps = cmdLineParse() 
    ifgram = inps.ifgram_file
    invRes = inps.inversion_res
    mask = inps.mask_file
    
    if inps.out_file:
        OUT = inps.out_file
    else:
        OUT = 'ifgramStackCor.h5'

    sliceList = readfile.get_slice_list(ifgram)
    N_list =len(sliceList)
        
    g_list = []
    for i in range(N_list):
        if 'unwrapPhase' in sliceList[i]:
            g_list.append(sliceList[i])
        
    
    
    sys.exit(1)



if __name__ == '__main__':
    main(sys.argv[:])


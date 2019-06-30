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
from scipy import linalg   # more effieint than numpy.linalg
from numpy.linalg import matrix_rank
from mintpy.utils.solvers import l1regls


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

    
def write__h5(datasetDict, out_file, metadata=None, ref_file=None, compression=None):
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


def get_good_pair_closure(C,ResC):
    N = len(ResC)
    ResC = np.asarray(ResC);ResC = ResC.reshape(len(ResC),)
    ResC = np.abs(ResC)
    ResC[ResC<0.5]=0
    good_pair = []
    for i in range(len(ResC)):
        if ResC[i]==0:
            y0 = C[i,:]
            y0 = np.abs(y0)
            bb = np.where(y0==1)
            bb = list(bb[0])
            good_pair.append(bb[0])
            good_pair.append(bb[1])
            good_pair.append(bb[2])
    good_pair = sorted(good_pair)
    good_pair = np.asarray(good_pair)
    good_pair = np.unique(good_pair)
    
    return good_pair

def union_good_pairs(gp1,gp2):
    gp1 = list(gp1)
    gp2 = list(gp2)
    
    for k0 in gp2:
        gp1.append(k0)
    gp1 = sorted(gp1)
    gp1 = np.asarray(gp1)
    gp1 = np.unique(gp1)    
    gp1 = list(gp1)
    gp_union = gp1
    
    return gp_union
    

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

def estimate_timeseries(A, ifgram, weight_sqrt=None, rcond=1e-5):

    ifgram = ifgram.reshape(A.shape[0], -1)
    if weight_sqrt is not None:
        weight_sqrt = weight_sqrt.reshape(A.shape[0], -1)
    else:
        weight_sqrt = np.ones((A.shape[0],1))
        weight_sqrt = weight_sqrt.reshape(A.shape[0], -1)
    
    num_date = A.shape[1]
    num_pixel = ifgram.shape[1]

    # Initial output value
    ts = np.zeros((num_date, num_pixel), np.float32)
    temp_coh = 0.
    num_inv_ifg = 0

    #if weight_sqrt is not None:
    
    A_w = np.multiply(A, weight_sqrt)
    ifgram_w = np.multiply(ifgram, weight_sqrt)
    X = linalg.lstsq(A_w, ifgram_w, cond=rcond)[0]
    
    #else:
    #    X = linalg.lstsq(A, ifgram, cond=rcond)[0]
    
    ts = X
    ifgram_diff = ifgram - np.dot(A, X)
    
    # calculate temporal coherence
    #num_inv_ifg = A.shape[0]
    #temp_coh = np.abs(np.sum(np.exp(1j*ifgram_diff), axis=0)) / num_inv_ifg
    sigma0 = np.sum(ifgram_w*(ifgram_diff**2))/(num_date-1)
    
    P = np.diag(weight_sqrt.reshape(A.shape[0],))
    A0 = np.dot(np.transpose(A),P)
    QQ = np.linalg.inv(np.dot(A0,A))
    #print(QQ)
    var_ts = np.diag(sigma0*QQ)
    var_ts = var_ts.reshape(A.shape[1], -1)

    return ts, var_ts

def get_bad_pair(good_pair,N_ifg):
    bad_pair = []
    for i in range(N_ifg):
        if i not in good_pair:
            bad_pair.append(i)
    
    return bad_pair

def get_design_matrix_unwrap_error(C,bad_pair,y):
    
    row,col = C.shape
    row_keep = []
    y = np.asarray(y)
    y = y.reshape(len(y),)
    for i in range(row):
        y0 = C[i,:]
        y0 = np.abs(y0)
        bb = np.where(y0==1)
        bb = list(bb[0])
        if (bb[0] in bad_pair) or (bb[1] in bad_pair) or (bb[2] in bad_pair):
            row_keep.append(i)
    
    Nr = len(row_keep)
    A = np.zeros((Nr,len(bad_pair)))        
    L = np.zeros((Nr,1)) 
    
    for i in range(Nr):
        Ifg0 = y
        ya = C[row_keep[i],:]
        y1 = C[row_keep[i],:]
        y0 = C[row_keep[i],:]
        y0 = np.abs(y0)
        bb = np.where(y0==1)
        bb = list(bb[0])
        if bb[0] in bad_pair:
            A[i,idx0] = ya[bb[0]]
            idx0 = bad_pair.index(bb[0])
            y1[bb[0]]=0
            
        if bb[1] in bad_pair:
            idx0 = bad_pair.index(bb[1])
            A[i,idx0] = ya[bb[1]]
            y1[bb[1]]=0

        if bb[2] in bad_pair:
            idx0 = bad_pair.index(bb[2])
            A[i,idx0] = ya[bb[2]]
            y1[bb[2]]=0

           
        y1 = y1.reshape(1,len(y1))
        L0 = np.dot(y1,y)
        L[i] = L0    
        
    return A, L

def read_unwrap_phase(stack_obj, box, ref_phase, unwDatasetName='unwrapPhase', dropIfgram=True,
                      print_msg=True):
    """Read unwrapPhase from ifgramStack file
    Parameters: stack_obj : ifgramStack object
                box : tuple of 4 int
                ref_phase : 1D array or None
    Returns:    pha_data : 2D array of unwrapPhase in size of (num_ifgram, num_pixel)
    """
    # Read unwrapPhase
    num_ifgram = stack_obj.get_size(dropIfgram=dropIfgram)[0]
    if print_msg:
        print('reading {} in {} * {} ...'.format(unwDatasetName, box, num_ifgram))
    pha_data = stack_obj.read(datasetName=unwDatasetName,
                              box=box,
                              dropIfgram=dropIfgram,
                              print_msg=False).reshape(num_ifgram, -1)

    # read ref_phase
    if ref_phase is not None:
        # use input ref_phase array (for split_file=False)
        if print_msg:
            print('use input reference phase')
    elif 'refPhase' in stack_obj.datasetNames:
        # read ref_phase from file itself (for split_file=True)
        if print_msg:
            print('read reference phase from file')
        with h5py.File(stack_obj.file, 'r') as f:
            ref_phase = f['refPhase'][:]
    else:
        raise Exception('No reference phase input/found on file!'+
                        ' unwrapped phase is not referenced!')

    # reference unwrapPhase
    for i in range(num_ifgram):
        mask = pha_data[i, :] != 0.
        pha_data[i, :][mask] -= ref_phase[i]
    return pha_data

##########################################################################################
INTRODUCTION = '''
#############################################################################
   Copy Right(c): 2019, Yunmeng Cao   This script is part of mintPy software.
   
   Correcting the unwrap error for an indendent island or land.
'''


EXAMPLE = """
    
    Example:
     
      unwrap_error_res_closure.py  ifgramStack.h5  maskResRegion.h5  
      unwrap_error_res_closure.py  ifgramStack.h5  maskResRegion.h5  -o ifgramStackCor.h5
"""


NOTE = """
  Correcting the unwrap error for cases: the research region is divided by water or forest e.g., islands
  
"""

###########################################################################################

def cmdLineParse():
    parser = argparse.ArgumentParser(description='Unwrap error correction based on phase closure and common area',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('ifgram_file',help='input ifgramStack file.')
    parser.add_argument('mask_file', help='common area that unwrapping error occurs (i.e., could occurs)')
    parser.add_argument('-o','--out_file', dest='out_file', metavar='FILE',
                      help='name of the output file')
   
    inps = parser.parse_args()

    return inps

################################################################################    
    
    
def main(argv):
    
    inps = cmdLineParse() 
    ifgram = inps.ifgram_file
    Bperp = readfile.read(ifgram, datasetName='bperp')[0]
    Date = readfile.read(ifgram, datasetName='date')[0]
    DropIfgram = readfile.read(ifgram, datasetName='dropIfgram')[0]
    
    #invRes = inps.inversion_res
    mask_file = inps.mask_file
    mask = readfile.read(mask_file, datasetName='mask')[0]
    meta = readfile.read_attribute(ifgram, datasetName=None) 
    REF_X = int(meta['REF_X'])
    REF_Y = int(meta['REF_Y'])
    
    if inps.out_file:
        OUT = inps.out_file
    else:
        OUT = 'ifgramStackCor.h5'

    sliceList = readfile.get_slice_list(ifgram)
    N_list =len(sliceList)
        
    g_list = []
    for i in range(N_list):
        if 'unwrapPhase-' in sliceList[i]:
            g_list.append(sliceList[i])
    
    N_list = len(g_list)
    Ifg = [] 
    
    print('Start to calculate the integer ambugity for each closure')
    for i in range(N_list):
        print_progress(i+1, N_list, prefix='Data: ', suffix=g_list[i])
        dset = g_list[i]
        ifgram1 = readfile.read(ifgram, datasetName=dset)[0]
        ifgram0 = ifgram1 - ifgram1[REF_Y,REF_X]
        Ifg.append(np.mean(ifgram0[mask==1]))

    
    stack_obj = ifgramStack(ifgram)
    stack_obj.open()
    date12_list = stack_obj.get_date12_list(dropIfgram=True)
    num_ifgram = len(date12_list)
    date12_list = stack_obj.get_date12_list(dropIfgram=True)
    num_ifgram = len(date12_list)
    
    C0 = ifgramStack.get_design_matrix4triplet(date12_list)
    #Ifg = np.asarray(Ifg);Ifg=Ifg.reshape(len(Ifg),)
    #ResC = np.dot(C0,Ifg)
    #good_pair = get_good_pair_closure(C,ResC)
    #good_pair = list(good_pair)
    #bad_pair = get_bad_pair(good_pair,N_list)    
    #print('Bad interferograms: ' + str(bad_pair)) 
    
    
    
    #Ifg_org = []
    #for k in bad_pair:
    #    Ifg_org.append(Ifg[k])

    #Ifg_org = np.asarray(Ifg_org)
    #Ifg_org.reshape(len(bad_pair),)

    #A,L = get_design_matrix_unwrap_error(C,bad_pair,Ifg)
    #Ifg_est, var_ts = estimate_timeseries(A, -L, weight_sqrt=None, rcond=1e-5)

    #Ifg_est = Ifg_est.reshape(len(Ifg_est),)
    #kk = np.round((Ifg_est - Ifg_org)/(2*np.pi))
    #print('Cycle shift of the bad interferogram: ' + str(kk))

    
    C = matrix(ifgramStack.get_design_matrix4triplet(date12_list).astype(float)) 
    ResC = np.dot(C,Ifg)
    L = matrix(np.round(ResC/(2*np.pi)))

    U = l1regls(-C, L, alpha=1e-2, show_progress=0)
    kk = np.round(U)
 
    num_row = stack_obj.length
    num_col = stack_obj.width
    box = [0,0,stack_obj.width,stack_obj.length]
    ref_phase = stack_obj.get_reference_phase(unwDatasetName='unwrapPhase',
                                                      skip_reference=None,
                                                       dropIfgram=True)
    pha_data = read_unwrap_phase(stack_obj,box,ref_phase,unwDatasetName='unwrapPhase',dropIfgram=True)
    data = pha_data.reshape(N_list,num_row,num_col)
    data0 = data
    
    print('Start to correct and write new unwrapIfg file ...')
    for i in range(N_list):  
        
        ifgram1 = data[i,:,:]
        ifgram0 = ifgram1 - ifgram1[REF_Y,REF_X]
        #print(ifgram0.shape)
        #ifgram0 = ifgram0 + mask*(kk*(2*np.pi)) 
        ifgram0[mask==1] = ifgram0[mask==1] + kk[i]*(2*np.pi) 
        data0[i,:,:] = ifgram0
    
    datasetDict = dict()
    datasetDict['unwrapPhase'] = data0
    datasetDict['IntAmbiguity'] = kk
    datasetDict['bperp'] = Bperp
    datasetDict['date'] = Date
    datasetDict['dropIfgram'] = DropIfgram
    datasetDict['C'] = C0
    write_variogram_h5(datasetDict, OUT, metadata=meta, ref_file=ifgram, compression=None)

    sys.exit(1)



if __name__ == '__main__':
    main(sys.argv[:])


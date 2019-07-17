#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright(c) 2013-2019, Yunmeng Cao                      #
# Author:  Yunmeng Cao                                     #
############################################################
# Recommend import:
#   import mintpy.view as view


import os
import sys
import argparse
from datetime import datetime as dt
import numpy as np
#import matplotlib; matplotlib.use("TkAgg")
#import matplotlib; matplotlib.use("macosx")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mintpy.objects import (geometryDatasetNames,
                            geometry,
                            ifgramDatasetNames,
                            ifgramStack,
                            timeseriesKeyNames,
                            timeseries)
from mintpy.objects.gps import GPS
from mintpy.utils import (ptime,
                          readfile,
                          utils as ut,
                          plot as pp)
from mintpy.multilook import multilook_data
from mintpy import subset, version



##########################################################################################
INTRODUCTION = '''
#############################################################################
   Copy Right(c): 2019, Yunmeng Cao   This script is part of mintPy software.
   
   Get specific datelist text for velocity or IfgInversion estimation.
'''


EXAMPLE = """
    
    Example:
     
      get_datelist_txt.py ifgramStack.h5              [default step =1 ]
      get_datelist_txt.py timeseries.h5 --step 1     [Select the datelist every 1 acquisitions] 
      get_datelist_txt.py timeseries.h5 --step 2     [Select the datelist every 2 acquisitions] 
      get_datelist_txt.py ifgramStack.h5 --step 2 -o datelistStep2.txt
"""

###########################################################################################

def cmdLineParse():
    parser = argparse.ArgumentParser(description='Unwrap error correction based on the inversion error',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('input_file',help='input file, [ifgramStack.h5 or timeseries.h5]')
    parser.add_argument('--step',dest='step', metavar='NUM', type=int, default=1, help='steps to select the datelist')
    parser.add_argument('--exclude', dest='exclude_date', action = 'store_true', help='exclude the selected date')
    parser.add_argument('-o','--out_file', dest='out_file', metavar='FILE',help='name of the output file')
   
    inps = parser.parse_args()

    return inps

################################################################################    
    
    
def main(argv):
    
    inps = cmdLineParse() 
    input_file = inps.input_file
    Step = inps.step
    if inps.out_file:
        OUT = inps.out_file
    else:
        OUT = 'datelist_step' + str(inps.step) + '.txt'
    
    if os.path.isfile(OUT):
        print('delete exist output file ...')
        os.remove(OUT)
      
    if 'ifgramStack' in input_file:  
        stack_obj = ifgramStack(ifgram_file)
        stack_obj.open(print_msg=False)
        date_list = stack_obj.get_date_list(dropIfgram=False)
    elif 'timeseries' in input_file:
        tsobj = timeseries(input_file)
        date_list = tsobj.get_date_list()
        #print(date_list)    
    #print(date_list)
    N = len(date_list)
    kk = np.arange(0,N,1)
    k0 = np.arange(0,N,Step)
    k1 = k0
    #print(k1)
    #print(len(k1))

    if inps.exclude_date:
        k1 = [i for i in kk if i not in k0]
         
    for i in range(len(k1)):
        #print(i)
        date0 = date_list[k1[i]]
        call_str ='echo ' + date0 + ' >> ' + OUT
        #print(call_str)
        os.system(call_str)
    print('Generate text file done.')    
    sys.exit(1)
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    

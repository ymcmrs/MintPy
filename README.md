## MintPy ##

[![Language](https://img.shields.io/badge/python-3.5%2B-blue.svg)](https://www.python.org/)
[![Latest version](https://img.shields.io/badge/latest%20version-v1.1.2-green.svg)](https://github.com/insarlab/MintPy/releases)
[![License](https://img.shields.io/badge/license-GPL-yellow.svg)](https://github.com/insarlab/MintPy/blob/master/LICENSE)
[![Forum](https://img.shields.io/badge/forum-Google%20Group-orange.svg)](https://groups.google.com/forum/#!forum/mintpy)

The Miami INsar Time-series software in PYthon (MintPy) is an open-source package for Interferometric Synthetic Aperture Radar time series analysis. It reads the stack of interferograms (coregistered and unwrapped) in [ISCE](https://github.com/isce-framework/isce2), Gamma or ROI_PAC format, and produces three dimensional (2D in space and 1D in time) ground displacement. It includes a routine time series analysis (`smallbaselineApp.py`) and some independent toolbox.

This package was called PySAR before version 1.1.1. For version 1.1.2 and onward, we use MintPy instead.

### 1. [Installation](./docs/installation.md) ###

### 2. Running MintPy ###

MintPy reads a stack of interferograms (unwrapped interferograms, coherence, wrapped interferograms and connecting components from SNAPHU if available) and the geometry files (DEM, lookup table, etc.). You need to give the path to where the files are and MintPy takes care of the rest!

```cfg
smallbaselineApp.py                         #run with default template 'smallbaselineApp.cfg'
smallbaselineApp.py <custom_template>       #run with default and custom templates
smallbaselineApp.py -h / --help             #help
smallbaselineApp.py -H                      #print    default template options
smallbaselineApp.py -g                      #generate default template if it does not exist
smallbaselineApp.py -g <custom_template>    #generate/update default template based on custom template

# Run with --start/stop/dostep options
smallbaselineApp.py GalapagosSenDT128.template --dostep velocity  #run at step 'velocity' only
smallbaselineApp.py GalapagosSenDT128.template --end load_data    #end after step 'load_data'
```

#### [Example](https://github.com/insarlab/MintPy/wiki/Example) on Fernandina volcano, Galápagos with Sentinel-1 data ####

```
wget https://zenodo.org/record/2748487/files/FernandinaSenDT128.tar.xz
tar -xvJf FernandinaSenDT128.tar.xz
cd FernandinaSenDT128/MintPy
smallbaselineApp.py ${MINTPY_HOME}/docs/examples/input_files/FernandinaSenDT128.txt
```

<p align="left">
  <img width="600" src="https://github.com/insarlab/MintPy/blob/master/docs/resources/images/FernandinaSenDT128_POI.jpg">
</p>    

Inside smallbaselineApp.py, it reads the unwrapped interferograms, references all of them to the same coherent pixel (reference point), calculates the phase closure and estimates the unwrapping errors (if it has been asked for), inverts the network of interferograms into time-series, calculates a parameter called "temporal coherence" which can be used to evaluate the quality of inversion, corrects local oscillator drift (for Envisat only), corrects stratified tropospheric delay (using pyaps or phase-elevation-ratio approach), removes phase ramps (if it has been asked for), corrects DEM error,... and finally estimates the velocity.

Check **./pic** folder for auto-generated figures. More details about this test data are in [here](https://github.com/insarlab/MintPy/wiki/Example).

#### 2.1 Some useful scripts for information and visualization: ####

```cfg
info.py                    #check HDF5 file structure and metadata
view.py                    #2D map view
tsview.py                  #1D point time-series (interactive)   
transect.py                #1D profile (interactive)
plot_coherence_matrix.py   #plot coherence matrix for one pixel (interactive)
plot_network.py            #plot network configuration of the dataset    
save_kmz.py                #generate Google Earth KMZ file in raster image
save_kmz_timeseries.py     #generate Goodle Earth KMZ file in points for time-series (interactive)
```

#### 2.2 Build your own processing recipe: [example](./sh/compare_velocity_with_diff_tropcor.sh) ####

MintPy is a toolbox with a lot of individual utility scripts, highly modulized in python. Check its documentation or simply run it with -h to see its usage, you could build your own customized processing recipe! Here is an example to compare the velocities estimated from displacement time-series with different tropospheric delay corrections: [link](./sh/compare_velocity_with_diff_tropcor.sh)

### 3. Documentation ###

+ [Tutorials on Jupyter Notebooks](./docs/tutorials)
+ [Example datasets](https://github.com/insarlab/MintPy/wiki/Example)
+ [Example template files for InSAR processors](./docs/examples/input_files)
+ [Google Earth KMZ file](https://github.com/insarlab/MintPy/wiki/Google-Earth)

### 4. User Forum ###

Join our google group [https://groups.google.com/forum/#!forum/mintpy](https://groups.google.com/forum/#!forum/mintpy) to ask questions, get notice of latest features pushed to you!

### Contributors ###

* Zhang Yunjun
* Heresh Fattahi
* Falk Amelung
* Scott Baker
* Joshua Zahner
* Alfredo Terreco
* David Grossman
* Yunmeng Cao
* [_other community members_](https://github.com/insarlab/MintPy/graphs/contributors)

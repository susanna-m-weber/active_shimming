# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:44:40 2023

@author: Sebastian
"""

import sys
sys.path.append("..")

import shimsimulator
import magsimulator
import numpy as np
import magpylib as magpy
import pandas as pd
import numpy.matlib


filename = sys.argv[1]
best_ledger = pd.read_csv(filename)

# load measured background field
# Assumed to be in mT and mm!
Bfilename = '../measured_magnet_field_w_passive_shim_sphere_filtered/B.csv'
posfilename = '../measured_magnet_field_w_passive_shim_sphere_filtered/col_sensors_pos.csv'
B = np.genfromtxt(Bfilename, delimiter=',', skip_header=True)
coordinates = np.genfromtxt(Bfilename, delimiter=',', skip_header=True)


col_sensors = magpy.Collection(style_label='sensors')
col_sensors.add(magpy.Sensor(position=coordinates,style_size=2))

print(f'Loaded active shims from {filename}...')

B_background = B/1000 #convert mT to Tesla
# B_background is normalized to the mean value in shim_field_w_constraints

B_background_hz = B_background*42580000
print(f'mean B0 = {np.mean(B_background[:,0])*1e3: .1f} mT, inhomogeneity = {np.std(B_background_hz[:,0]): .1f} Hz')


maxtotalcurr = 450  #to match the 10 turns above , originally was 45
maxchannelcurr = 10 #3 was original

solution, stdfield_hz = shimsimulator.shim_field_w_constraints(best_ledger,
                                                col_sensors,
                                                B_background,
                                                maxchannelcurr,
                                                maxtotalcurr,
                                                Bfield_component=0)

print(f'Resulting imhomogeneity: {stdfield_hz: .1f} Hz')
print(f'Maximum absolute current: {np.abs(solution).max(): .1f} A')
print(f'Total absolute current: {np.abs(solution).sum(): .1f} A')

print('Current settings:\n')
with np.printoptions(precision=3):
    print(f'{str(solution[:,0].T)} A')
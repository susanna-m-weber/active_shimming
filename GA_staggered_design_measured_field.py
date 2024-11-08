# -*- coding: utf-8 -*-

import sys
sys.path.append("..")

import shimsimulator
import magsimulator
#import magcadexporter
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from magpylib.magnet import Cuboid, CylinderSegment
import itertools
from scipy.spatial.transform import Rotation as R
import pandas as pd
import cProfile
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool, freeze_support
from os import getpid
import time
#import addcopyfighandler
# import pygad
import numpy.matlib
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
import multiprocessing
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
from multiprocessing.pool import ThreadPool

import pickle
from cvxopt import matrix, solvers

def add_colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

# from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.core.variable import Real, Integer
from pymoo.optimize import minimize



# load measured background field
# Assumed to be in mT and mm!
Bfilename = '../measured_magnet_field_w_passive_shim_sphere_filtered/B.csv'
posfilename = '../measured_magnet_field_w_passive_shim_sphere_filtered/col_sensors_pos.csv'

B = np.genfromtxt(Bfilename, delimiter=',', skip_header=True)
coordinates = np.genfromtxt(Bfilename, delimiter=',', skip_header=True)
                    
data = {'Bfield':  B, 'coordinates': coordinates}
col_sensors = magpy.Collection(style_label='sensors')
col_sensors.add(magpy.Sensor(position=coordinates,style_size=2))


#%%
class MultiObjectiveMixedVariableProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
       

        print("STARTING")
       
        self.B_background =data['Bfield']
        self.coordinates = data['coordinates']
     
        self.B_background =self.B_background.reshape(-1,3)/1000 #convert mT to Tesla
        self.r=76  #changed from 70 - diameter 152
        self.maxchannels = 32 #from 64
        self.refcurr = 1  #to simulate 1 amp per turn originally 1
       
        self.maxtotalcurr = 450  #to match the 10 turns above , originally was 45
        self.maxchannelcurr = 30 #3 was original
        self.maxzrange = 300
       
        variables = dict()
       
        variables["x01"] = Integer(bounds=(0, 90)) #delta dc phase
        variables["x02"] = Integer(bounds=(2,8)) # number of coils azimuthally
        variables["x03"] = Integer(bounds=(15,100)) #coil diameter
        variables["x04"] = Integer(bounds=(4,20)) #number of z rows
        variables["x05"] = Integer(bounds=(25,100)) #delta z
        #variables["x06"] = Integer(bounds=(0, 360)) #delta theta between z increments
       
        # super().__init__(vars=variables,n_ieq_constr=1, n_obj=2, **kwargs)
        super().__init__(vars=variables,n_ieq_constr=3, n_obj=1, **kwargs)


    def _evaluate(self, x, out, *args, **kwargs):
        delta_dc_phase=np.array(x["x01"]).reshape((1,1)).astype(float)
        delta_dc_phase=int(delta_dc_phase[0])
       
        num_coils_azi = np.array(x["x02"]).reshape((1,1)).astype(int)
        num_coils_azi=int(num_coils_azi[0])
       
        # print(num_coils_azi)
       
        coil_diam = np.array(x["x03"]).reshape((1,1)).astype(int)
        coil_diam=int(coil_diam[0])

        nz = np.array(x["x04"]).reshape((1,1)).astype(int)
        nz=int(nz[0])

        dz = np.array(x["x05"]).reshape((1,1)).astype(int)
        dz=int(dz[0])

        #delta_theta = np.array(x["x06"]).reshape((1,1)).astype(float)
        #delta_theta=int(delta_theta[0])        
       

        delta_theta = int(180 / num_coils_azi)


        # print(delta_dc_phase,num_coils_azi,dz,nz,coil_diam,delta_theta)
# running quadradic programming optimization

        ledg= shimsimulator.generate_shim_coils_on_cylinder(self.r,
                                                            num_coils_azi,
                                                            coil_diam,
                                                            dz,
                                                            delta_theta,
                                                            nz,
                                                            delta_dc_phase,
                                                            self.refcurr,
                                                            plot=False)
       
        # ledg.head()
       
        # col_sensors = magpy.Collection(style_label='sensors')
        # sensor1 = magsimulator.define_sensor_points_on_filled_sphere(200,70,5,[0,0,0])
        # col_sensors.add(sensor1)
        col_sensors = magpy.Collection(style_label='sensors')
        sensor1 = magpy.Sensor(position=self.coordinates)
        col_sensors.add(sensor1)
               
        # solution,stdfield_hz=shimsimulator.shim_least_squares(self.B_background,ledg,col_sensors,Bfield_component=0)
        solution, stdfield_hz = shimsimulator.shim_field_w_constraints(ledg,col_sensors,
                                                self.B_background,
                                                self.maxchannelcurr,
                                                self.maxtotalcurr,
                                                Bfield_component=0)

        g = nz*num_coils_azi-self.maxchannels
        g2 = dz*(nz-1) + coil_diam - self.maxzrange # maximum z-range
        g3 = 1.2 * coil_diam - dz
        print('unshimmed:'+ str(np.round(np.std(self.B_background[:,0]*42580000),5)) + ' ; shimmed:' + str(np.round(stdfield_hz,5)) +
                '; n-coils:' + str(nz*num_coils_azi) + ' ; delta z:' + str(dz) + '; diameter:' + str(coil_diam) )
        # out["F"] = np.column_stack([stdfield_hz,nz*num_coils_azi])
        out["F"] = stdfield_hz
        #out["G"] = g
        out["G"] = np.column_stack([g,g2,g3])
     
   
   
   
    def get_coils(self, x):
        delta_dc_phase=np.array(x["x01"]).reshape((1,1)).astype(float)
        delta_dc_phase=float(delta_dc_phase[0])
       
        num_coils_azi = np.array(x["x02"]).reshape((1,1)).astype(int)
        num_coils_azi=int(num_coils_azi[0])
       
        # print(num_coils_azi)
       
        coil_diam = np.array(x["x03"]).reshape((1,1)).astype(int)
        coil_diam=int(coil_diam[0])

        nz = np.array(x["x04"]).reshape((1,1)).astype(int)
        nz=int(nz[0])

        dz = np.array(x["x05"]).reshape((1,1)).astype(int)
        dz=int(dz[0])

        #delta_theta = np.array(x["x06"]).reshape((1,1)).astype(float)
        #delta_theta=float(delta_theta[0])        

        delta_theta = int( 180 / num_coils_azi)
       
        # print(delta_dc_phase,num_coils_azi,dz,nz,coil_diam,delta_theta)
# running quadradic programming optimization

        ledg= shimsimulator.generate_shim_coils_on_cylinder(self.r,
                                                            num_coils_azi,
                                                            coil_diam,
                                                            dz,
                                                            delta_theta,
                                                            nz,
                                                            delta_dc_phase,
                                                            self.refcurr,
                                                            plot=False)
       
        # ledg.head()
       
        # col_sensors = magpy.Collection(style_label='sensors')
        # sensor1 = magsimulator.define_sensor_points_on_filled_sphere(200,70,5,[0,0,0])
        # col_sensors.add(sensor1)
        col_sensors = magpy.Collection(style_label='sensors')
        sensor1 = magpy.Sensor(position=self.coordinates)
        col_sensors.add(sensor1)

        # solution,stdfield_hz=shimsimulator.shim_least_squares(self.B_background,ledg,col_sensors,Bfield_component=0)
        solution, stdfield_hz = shimsimulator.shim_field_w_constraints(ledg,col_sensors,
                                                self.B_background,
                                                self.maxchannelcurr,
                                                self.maxtotalcurr,
                                                Bfield_component=0)

        return ledg,solution,stdfield_hz
   
#%%
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
import multiprocessing
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
from multiprocessing.pool import ThreadPool
import pickle
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.gradient.automatic import AutomaticDifferentiation
from pymoo.core.mixed import MixedVariableGA
from cvxopt import matrix, solvers

import warnings
warnings.filterwarnings("ignore")

solvers.options['show_progress'] = False #used to kill the output and just display the resulting homogeneity

n_threads = 2
pool = ThreadPool(n_threads)
runner = StarmapParallelization(pool.starmap)

#problem = MultiObjectiveMixedVariableProblem(elementwise_runner=runner)

problem = MultiObjectiveMixedVariableProblem(data=data)

algorithm = MixedVariableGA(pop_size=10, survival=RankAndCrowdingSurvival())
# algorithm = NSGA2(pop_size=80,sampling=MixedVariableSampling(),                 mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),                  eliminate_duplicates=MixedVariableDuplicateElimination(),)

# algorithm = MixedVariableGA(pop=20)

res = minimize(problem,
               algorithm,
               ('n_gen', 1),
               seed=1,
               verbose=True)


# filename = 'opt_shimming_layers1_maxzpos_50.xlsx'

# fileObj = open(filename[:-4]+'pkl', 'wb')
# pickle.dump(res,fileObj)
# fileObj.close()
#%%


print (res.F)
#plt.figure()
#plt.scatter(res.F[:,0], res.F[:,1],s=100)
#plt.title("Tradeoff B0 and Homogeneity", fontsize=18, weight='bold')
#plt.xlabel("Standard deviation field (Hz)", fontsize=16, weight='bold')
#plt.ylabel("Number of coils", fontsize=16, weight='bold')
#plt.xticks(fontsize=14, weight='bold')
#plt.show()
#%%
#B_rescaled = B_background /1000
 
xx=res.X

print(xx)


ledger,solution,stdfield_hz = problem.get_coils(xx)
ledger.to_excel('GA_staggered_50pops_250gens.xlsx', index=False)
ledger.to_csv('GA_staggered_50pops_250gens.csv')

#%%

B_loops,loops = shimsimulator.simulate_shim_ledger(ledger,col_sensors,0,True)

#print('std shimmed B0=' + str((np.std(Btar[:,0] - np.dot(A,m[0]).reshape(-1))*42580000)))

#outfileName='./output.xlsx'
#%%
#magcadexporter.export_magnet_ledger_to_cad('mean B0=' + str(np.round(meanB0,4)) + ' homogen=' + str(np.round(eta,4)) + ' ' + outfileName,[0.2,0.2,0.2])



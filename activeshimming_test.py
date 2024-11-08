import sys
sys.path.append("..")
import magsimulator
import numpy as np
from scipy.linalg import solve
from scipy.linalg import lstsq
from cvxopt import matrix, solvers
import magpylib as magpy
import shimsimulator

filename = 'optimization_after_neonate_magnet_Rmin_132p7mm_extrinsic_rot_DSV140mm_maxh60_maxlayers6_maxmag990.xlsx'
mag_vect = [1270,0,0]
ledger, magnets = magsimulator.load_magnet_positions(filename, mag_vect)

col_sensors = magpy.Collection(style_label='sensors')
# sensor1 = magsimulator.define_sensor_points_on_sphere(1000,70,[0,0,0])
sensor1 = magsimulator.define_sensor_points_on_filled_sphere(200,70,5,[0,0,0])
col_sensors.add(sensor1)

magnets = magpy.Collection(style_label='magnets')

eta, meanB0, col_magnet, B_mag = magsimulator.simulate_ledger(magnets,col_sensors,mag_vect,ledger,0.06,4,True,False,None,False)
print('mean B0='+str(round(meanB0,3)) +  ' homogeneity=' + str(round(eta,3)))

data=magsimulator.extract_3Dfields(col_magnet,xmin=-70,xmax=70,ymin=-70,ymax=70,zmin=-70, zmax=70, numberpoints_per_ax = 33,filename=None,plotting=True,Bcomponent=0) #homogeneity figure
# magsimulator.plot_3D_field(data['Bfield'],Bcomponent=0) # does the same thing as the previous function - from an older version? 
B=data['Bfield']
col_sensors = magpy.Collection(style_label='sensors')
sensor1 = magpy.Sensor(position=data['coordinates'])
col_sensors.add(sensor1)
#%%
r=76
number_mags_azimuthal= 6
coil_diam=35
delta_z=55.8
delta_theta=191
nz= 5
Btar = B.reshape(-1,3)
dc_phase=66

ledg= shimsimulator.generate_shim_coils_on_cylinder(r,number_mags_azimuthal,coil_diam,delta_z,delta_theta,nz,dc_phase,ref_curr=1,plot=True,tofile=None)
A = shimsimulator.simulate_shim_elements(ledg,col_sensors,Bfield_component=0) # shim array figure 
# shim_field, loops = shimsimulator.simulate_shim_ledger(ledg,col_sensors,Bfield_component=0,plotting=True) # (x, 3)
# shim_field_3D_test = shim_field.reshape(11, 11, 11, 3)

# shim_field_3D = shimsimulator.extract_3D_shim_field(loops,xmin=-70,xmax=70,ymin=-70,ymax=70,zmin=-70, zmax=70, numberpoints_per_ax = 11,Bcomponent=0) #(11, 11, 11, 3)
# shim_field_tar = shim_field_3D.reshape(-1,3)
# shim_field_3D_mT = shim_field_3D * 1000
#%%

#%%
b = Btar[:,0] - np.mean(Btar[:,0]) # T
b_3D = B - np.mean(B) # mT
m = lstsq(A, b)

# plotting shim field 
magsimulator.plot_3D_field(B,Bcomponent=0)
# magsimulator.plot_3D_field(shim_field_3D_mT,Bcomponent=0)
# magsimulator.plot_3D_field(B - np.mean(B) - shim_field_3D,Bcomponent=0)

print(m[0])
print('std unshimmed B0=' + str(np.std(b)*42580000))
# print('std shimmed B0=' + str(np.std(np.dot(A,m[0]).reshape(-1)*42580000)))
print('std shimmed B0=' + str((np.std(Btar[:,0] - np.dot(A,m[0]).reshape(-1))*42580000)))

#%%
# ledg= shimsimulator.generate_shim_coils_on_cylinder(r,
#                                                     number_mags_aximuthal,
#                                                     coil_diam,
#                                                     delta_z,
#                                                     delta_theta,
#                                                     nz,
#                                                     ref_curr=1,
#                                                     plot=True)

# A = shimsimulator.simulate_shim_elements(ledg,col_sensors,Bfield_component=0)
# b = Btar[:,0] - Btar[-1,0]
b = Btar[:,0] - np.mean(Btar[:,0])

Q = np.dot(A.T,A)
c = np.dot(-A.T,b)

# G_maxtotalcurr = np.ones((1,A.shape[1]))
# G_max_curr = np.eye(A.shape[1])
# G_min_curr = -np.eye(A.shape[1])
# G = np.concatenate((G_maxtotalcurr,G_max_curr,G_min_curr),axis=0)
# h = np.concatenate((45,3*np.ones((A.shape[1],1)), 3*np.ones((A.shape[1],1))), axis=None)

# G = np.ones((1,c.shape[0]))*G_maxtotalcurr
# h = np.ones((1,1))*100

# sol=solvers.qp(matrix(Q), matrix(c), matrix(G), matrix(h))
sol=solvers.qp(matrix(Q), matrix(c))
solution=np.array(sol['x'])

print(solution.T)
print('std unshimmed B0=' + str(np.std(b)*42580000))
# print('std shimmed B0=' + str(np.std(np.dot(A,m[0]).reshape(-1)*42580000)))
print('std shimmed B0=' + str((np.std(Btar[:,0] - np.dot(A,solution).reshape(-1))*42580000)))

#%%
solution,resultingfieldhomogeneity=shimsimulator.shim_field_w_constraints(ledg,col_sensors,Btar,maxcur_per_chan=3,max_tot_cur=45,Bfield_component=0)

# print(solution,resultingfieldhomogeneity)
# shimsimulator.shim_least_squares(Btar,ledg,col_sensors,Bfield_component=0)
# shimsimulator.shim_field_no_constraints(ledg,col_sensors,Btar,Bfield_component=0)

#%%

solution,resultingfieldhomogeneity=shimsimulator.shim_least_squares(Btar,ledg,col_sensors,Bfield_component=0)

print(solution,resultingfieldhomogeneity)
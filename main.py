# stolen from here: https://amor.cms.hu-berlin.de/~rodrigus/Jupyter%20Notebooks/Solar_System_N-Body_Simulation.html
import numpy as np
import scipy as sp
import time
import scipy.constants as cs
import matplotlib.pyplot as plt
from numba import jit
from numba import cuda
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from copy import deepcopy
from random import random

#Vectorial acceleration function
def a_t(r, m, epsilon, numnatural):
    """
    Function of matrices returning the gravitational acceleration
    between N-bodies within a system with positions r and masses m
    -------------------------------------------------
    r  is a N x 3 matrix of object positions
    m is a N x 1 vector of object masses
    epsilon is the softening factor to prevent numerical errors
    a is a N x 3 matrix of accelerations
    -------------------------------------------------
    """  
    G = cs.gravitational_constant
    # positions r = [x,y,z] for all planets in the N-Body System
    px = r[:numnatural,0:1]
    py = r[:numnatural,1:2]
    pz = r[:numnatural,2:3]
    
    # matrices that store each pairwise body separation for each [x,y,z] direction: r_j - r_i
    pdx = px.T - px
    pdy = py.T - py
    pdz = pz.T - pz
    #matrix 1/r^3 for the absolute value of all pairwise body separations together and 
    inv_pr3 = (pdx**2 + pdy**2 + pdz**2 + epsilon**2)**(-1.5)
    #resulting acceleration components in each [x,y,z] direction  
    pax = G * (pdx * inv_pr3) @ m[:numnatural]
    pay = G * (pdy * inv_pr3) @ m[:numnatural]
    paz = G * (pdz * inv_pr3) @ m[:numnatural]
    # pack together the three acceleration components
    pa = np.hstack((pax,pay,paz))
    
    # positions r = [x,y,z] for all satellites in the N-Body System
    sx = r[numnatural:,0:1]
    sy = r[numnatural:,1:2]
    sz = r[numnatural:,2:3]    
    
    # between planets and satellites
    sdx = px.T - sx
    sdy = py.T - sy
    sdz = pz.T - sz
    
    #matrix 1/r^3 for the absolute value of all pairwise body separations together and 
    inv_sr3 = (sdx**2 + sdy**2 + sdz**2 + epsilon**2)**(-1.5)
    #resulting acceleration components in each [x,y,z] direction  
    # XXX i don't understand mass list here
    sax = G * (sdx * inv_sr3) @ m[:numnatural]
    say = G * (sdy * inv_sr3) @ m[:numnatural]
    saz = G * (sdz * inv_sr3) @ m[:numnatural]
    
    sa = np.hstack((sax,say,saz))

    a = np.vstack((pa, sa))
    return a

def omega(N, t_max, dt):
    """
    Dummy acceleration function to give an estimate of the total memory
    consumption for a simulation with N bodies, total simulation time 
    t_max and timestep dt
    -------------------------------------------------
    N is the amount of bodies in the simulation
    t_max is the total simulation time in years
    dt is the timestep for each integration in days
    -------------------------------------------------
    """
    epsilon = 1
    G = 1
    r = np.ones((N,3))
    v = np.ones((N,3)) 
    """although the velocities aren't actually taken into account for computing the 
    acceleration, they will be stored as Nx3 matrices for exactly as many iterations 
    in the integration loop itself and take up memory accordingly"""
    m = np.ones((N,1))
    # positions r = [x,y,z] for all bodies in the N-Body System
    x = r[:,0:1]
    y = r[:,1:2]
    z = r[:,2:3]

    # matrices that store each pairwise body separation for each [x,y,z] direction: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z
    #matrix 1/r^3 for the absolute value of all pairwise body separations together and 
    #resulting acceleration components in each [x,y,z] direction 
    inv_r3 = (dx**2 + dy**2 + dz**2 + epsilon**2)**(-1.5)
    ax = G * (dx * inv_r3) @ m
    ay = G * (dy * inv_r3) @ m
    az = G * (dz * inv_r3) @ m
    # pack together the three acceleration components
    a = np.hstack((ax,ay,az))
    # sum the memory usage of each matrix storing the positions, distances and accelerations
    memory_usage_per_iteration = r.nbytes + v.nbytes + x.nbytes + y.nbytes + z.nbytes + dx.nbytes + dy.nbytes + dz.nbytes + ax.nbytes + ay.nbytes + az.nbytes + inv_r3.nbytes + a.nbytes 
    total_memory_usage = memory_usage_per_iteration * (t_max)/(dt*1e6) 
    return total_memory_usage #in megabytes
    

#Example for the estimated memory usage from a simulation with N = 13 bodies, t_max = 250 years and dt = 1 day
year = 365*24*60*60
day = 24*60*60
omega(13,250*year,1*day)

#Conversion Units
AU = 149597870700
D = 24*60*60
#Simulation starting date in Y/M/D
t_0 = "2018-10-26"
#Get Starting Parameters for Sun-Pluto from Nasa Horizons
jpl_ids = [0,1,2,399,301,4,5,6,7,8,9]
NUMNATURAL = len(jpl_ids)
r_list = []
v_list = []
m_list = [[1.989e30],[3.285e23],[4.867e24],[5.972e24],[7.34767e22],[6.39e23],[1.8989e27],[5.683e26],[8.681e25],[1.024e26],[1.309e22]] #Object masses for Sun-Pluto

plot_colors = ['green','brown','orange','blue','gray','red','red','orange','cyan','blue','brown']
plot_labels = ['Barycenter Drift','Mercury Orbit','Venus Orbit','Earth Orbit','Moon Orbit','Mars Orbit','Jupiter Orbit','Saturn Orbit','Uranus Orbit','Neptune Orbit','Pluto Orbit']
for i, jpl_id in enumerate(jpl_ids):
    obj = Horizons(id=jpl_id, location="@sun", epochs=Time(t_0).jd, id_type='id').vectors()
    r_obj = [obj['x'][0], obj['y'][0], obj['z'][0]]
    v_obj = [obj['vx'][0], obj['vy'][0], obj['vz'][0]]
    r_list.append(r_obj)
    v_list.append(v_obj)
    print(plot_labels[i], np.linalg.norm(v_obj)*AU/D, "m/s")#np.linalg.norm(v_obj),  
#Get Starting Parameters for any extra object to add into the simulation with input for id/mass/plot_color/plot_label

MAXSPEEDDIFF = 5000/AU*D
print(MAXSPEEDDIFF)

def sample_spherical_many(npoints=1, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def sample_spherical():
	return list(zip(*sample_spherical_many()))[0]
    
def add_simulation_object(Id_obj,m_obj, plot_color, plot_label, n_objects=1):
    ori = Horizons(id=Id_obj, location="@sun", epochs=Time(t_0).jd, id_type='id').vectors()
    print(ori)
    for i in range(n_objects):
        obj = deepcopy(ori)
        r_obj = [obj['x'][0], obj['y'][0], obj['z'][0]]
        v_obj = [obj['vx'][0], obj['vy'][0], obj['vz'][0]]
        
        #print(v_obj)
        speed_before = np.linalg.norm(v_obj)*AU/D
        random_vector = sample_spherical()
        deltas = []
        
        for j in range(3):
        	delta = MAXSPEEDDIFF*random_vector[j]*np.random.uniform(0.5, 1)
        	deltas.append(delta)
        	v_obj[j] += delta

        speed_after = np.linalg.norm(v_obj)*AU/D
        print(f"Added: {plot_label+str(i)}\tSpeed: {speed_after:.2f} m/s\tDelta: {speed_after-speed_before:.2f} m/s\tApplied delta: {np.linalg.norm(deltas)*AU/D}")
        r_list.append(r_obj)
        v_list.append(v_obj)
        m_list.append([m_obj])
        plot_colors.append(plot_color)
        plot_labels.append(plot_label+str(i))

#Sample ID for Neowise
id_neowise = '90004475'
m_neowise = 5e13
#Sample ID for Space X's Starman
id_starman = 'SpaceX Roadster'
m_starman = 1300
#Add NEOWISE and Starman to the simulation
#add_simulation_object('90004479', 5e13, 'black','Comet Neowise')
#add_simulation_object('SpaceX Roadster',1300, 'pink','Starman')
#add_simulation_object('90000033',1e14, 'gray','Comet Halley')
add_simulation_object("-48", 11110, "black", "Hubble", 30)

#Convert object staring value lists to numpy
r_i = np.array(r_list)*AU
v_i = np.array(v_list)*AU/D
m_i = np.array(m_list)
#pack together as list for the simulation function
horizons_data = [r_i,v_i,m_i]

earth_radius = 6371*1000#m

def simulate_solar_system(N,dN,starting_values): #
    t0_sim_start = time.time()
    t = 0
    t_max = 365*24*60*60*N #N year simulation time
    dt = 60*60*24*dN #dN day time step
    epsilon_s = 0.01 #softening default value
    r_i = starting_values[0]
    v_i = starting_values[1]
    m_i = starting_values[2]
    a_i = a_t(r_i, m_i, epsilon_s, NUMNATURAL)
    ram_usage_estimate = omega(len(r_i), t_max, dt) #returns the estimated ram usage for the simulation
    # Simulation Main Loop using a Leapfrog Kick-Drift-Kick Algorithm
    k = int(t_max/dt)
    r_save = np.zeros((r_i.shape[0],3,k+1))
    r_save[:,:,0] = r_i
    for i in range(k):
        if i % 1000 == 0:
            print(f"{i}/{k}")
            earth_position = r_i[3]
            for idx, sat_position in enumerate(r_i[NUMNATURAL:]):
            	dist = np.linalg.norm(earth_position-sat_position)
            	#print(idx, dist)#"earth:", earth_position, "sat:", sat_position, 
            	if dist < earth_radius:
            		print("crashed:", idx)
        # (1/2) kick
        v_i += a_i * dt/2.0
        # drift
        r_i += v_i * dt
        # update accelerations
        a_i = a_t(r_i, m_i, epsilon_s, NUMNATURAL)
        # (2/2) kick
        v_i += a_i * dt/2.0
        # update time
        t += dt
        #update list
        r_save[:,:,i+1] = r_i
    sim_time = time.time()-t0_sim_start
    print('The required computation time for the N-Body Simulation was', round(sim_time,3), 'seconds. The estimated memory usage was', round(ram_usage_estimate,3), 'megabytes of RAM.')
    return r_save

STEP = 0.01
YEARS = 1
print(f"DURATION: {YEARS} years\tSTEP: {STEP*60*60*24} seconds")

#Run simulation for 250 years at a 1 day time-step
r_save = simulate_solar_system(YEARS, STEP, horizons_data)

#Plot for the outer planets

fig2 = plt.figure()
fig2.canvas.manager.set_window_title('YOLO orbits')
ax = plt.axes(projection='3d')
ax.set_xlim3d(-50e11,50e11)
ax.set_ylim3d(-50e11,50e11)
ax.set_zlim3d(-50e11,50e11)
#Input sim. data
for i in range(0,10): #Plots the outer planets
    ax.plot3D(r_save[i,0,:],r_save[i,1,:],r_save[i,2,:], plot_colors[i],label=plot_labels[i])
if len(r_i)>=10:
    for i in range(10,len(r_i)): #Plots any additional objects
        ax.plot3D(r_save[i,0,:],r_save[i,1,:],r_save[i,2,:], plot_colors[i],label=plot_labels[i])
ax.legend(plot_labels[:NUMNATURAL], loc = 'upper right', prop={'size': 6.5})

plt.show()
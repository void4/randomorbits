# code stolen and adapted from here: https://amor.cms.hu-berlin.de/~rodrigus/Jupyter%20Notebooks/Solar_System_N-Body_Simulation.html

import time
from random import random
from copy import deepcopy
import math

import numpy as np
import scipy.constants as cs

from astropy.time import Time
from astroquery.jplhorizons import Horizons

from numba import *

from utils import *

#Example for the estimated memory usage from a simulation with N = 13 bodies, t_max = 250 years and dt = 1 day
year = 365*24*60*60
day = 24*60*60

#Conversion Units
AU = 149597870700
D = 24*60*60

sun_radius = 696_340
mercury_radius = 2_439.7
venus_radius = 6_051.8
earth_radius = 6_371
moon_radius = 1_737.4
mars_radius = 3_389.5
jupiter_radius = 69_911
saturn_radius = 58_232
uranus_radius = 25_362
neptune_radius = 24_622
pluto_radius = 1_188.3



def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = math.sqrt(XsqPlusYsq + z**2)               # r
    elev = math.atan2(z,math.sqrt(XsqPlusYsq))     # theta
    az = math.atan2(y,x)                           # phi
    return r, elev, az

def cart2sphA(pts):
    return np.array([cart2sph(x,y,z) for x,y,z in pts])

# Vectorial acceleration function
@jit
def a_t(r=np.array([[]]), m=np.array([]), epsilon=0.0, numnatural=0):
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
    px = r[:numnatural, 0:1]
    py = r[:numnatural, 1:2]
    pz = r[:numnatural, 2:3]

    # matrices that store each pairwise body separation for each [x,y,z] direction: r_j - r_i
    pdx = np.subtract(px.T, px)
    pdy = py.T - py
    pdz = pz.T - pz
    # matrix 1/r^3 for the absolute value of all pairwise body separations together and
    inv_pr3 = (pdx ** 2 + pdy ** 2 + pdz ** 2 + epsilon ** 2) ** (-1.5)
    # resulting acceleration components in each [x,y,z] direction
    pax = G * (pdx * inv_pr3) @ m[:numnatural]
    pay = G * (pdy * inv_pr3) @ m[:numnatural]
    paz = G * (pdz * inv_pr3) @ m[:numnatural]
    # pack together the three acceleration components
    pa = np.hstack((pax, pay, paz))

    # positions r = [x,y,z] for all satellites in the N-Body System
    sx = r[numnatural:, 0:1]
    sy = r[numnatural:, 1:2]
    sz = r[numnatural:, 2:3]

    # between planets and satellites
    sdx = px.T - sx
    sdy = py.T - sy
    sdz = pz.T - sz

    # matrix 1/r^3 for the absolute value of all pairwise body separations together and
    inv_sr3 = (sdx ** 2 + sdy ** 2 + sdz ** 2 + epsilon ** 2) ** (-1.5)
    # resulting acceleration components in each [x,y,z] direction
    # XXX i don't understand mass list here
    sax = G * (sdx * inv_sr3) @ m[:numnatural]
    say = G * (sdy * inv_sr3) @ m[:numnatural]
    saz = G * (sdz * inv_sr3) @ m[:numnatural]

    sa = np.hstack((sax, say, saz))

    a = np.vstack((pa, sa))
    return a

COLLISIONCHECKEVERY = 10

class SolarSystemSimulator:

    def __init__(self):
        #Simulation starting date in Y/M/D
        self.t_0 = "2023-07-12"
        #Get Starting Parameters for Sun-Pluto from Nasa Horizons
        jpl_ids = [0,1,2,399,301,4,5,6,7,8,9]
        self.numnatural = len(jpl_ids)
        self.r_list = []
        self.v_list = []
        self.m_list = [[1.989e30],[3.285e23],[4.867e24],[5.972e24],[7.34767e22],[6.39e23],[1.8989e27],[5.683e26],[8.681e25],[1.024e26],[1.309e22]] #Object masses for Sun-Pluto

        self.plot_colors = ['green','brown','orange','blue','gray','red','red','orange','cyan','blue','brown']
        self.plot_labels = ['Barycenter Drift','Mercury Orbit','Venus Orbit','Earth Orbit','Moon Orbit','Mars Orbit','Jupiter Orbit','Saturn Orbit','Uranus Orbit','Neptune Orbit','Pluto Orbit']
        self.planet_radii = [sun_radius, mercury_radius, venus_radius, earth_radius, moon_radius, mars_radius, jupiter_radius, saturn_radius, uranus_radius, neptune_radius, pluto_radius]
        self.planet_radii = [pr*1000 for pr in self.planet_radii]
        for i, jpl_id in enumerate(jpl_ids):
            obj = Horizons(id=jpl_id, location="@sun", epochs=Time(self.t_0).jd, id_type='id').vectors()
            r_obj = [obj['x'][0], obj['y'][0], obj['z'][0]]
            v_obj = [obj['vx'][0], obj['vy'][0], obj['vz'][0]]
            self.r_list.append(r_obj)
            self.v_list.append(v_obj)
            print(self.plot_labels[i], np.linalg.norm(v_obj)*AU/D, "m/s")#np.linalg.norm(v_obj),

    def add_object(self, Id_obj, m_obj, plot_color, plot_label, n_objects=1, random_acceleration=None):
        ori = Horizons(id=Id_obj, location="@sun", epochs=Time(self.t_0).jd, id_type='id').vectors()
        print(ori)
        self.random_vectors = []
        for i in range(n_objects):
            obj = deepcopy(ori)
            r_obj = [obj['x'][0], obj['y'][0], obj['z'][0]]
            v_obj = [obj['vx'][0], obj['vy'][0], obj['vz'][0]]

            # print(v_obj)
            speed_before = np.linalg.norm(v_obj) * AU / D
            random_vector = sample_spherical()

            deltas = []

            if random_acceleration is not None:
                for j in range(3):
                    delta = random_acceleration * random_vector[j]# * np.random.uniform(0.5, 1)
                    deltas.append(delta)
                    v_obj[j] += delta

            self.random_vectors.append(deltas)

            speed_after = np.linalg.norm(v_obj) * AU / D
            #print(f"Added: {plot_label + str(i)}\tSpeed: {speed_after:.2f} m/s\tDelta: {speed_after - speed_before:.2f} m/s\tApplied delta: {np.linalg.norm(deltas) * AU / D}")
            self.r_list.append(r_obj)
            self.v_list.append(v_obj)
            self.m_list.append([m_obj])
            self.plot_colors.append(plot_color)
            self.plot_labels.append(plot_label + str(i))

        self.angles = cart2sphA(np.array(self.random_vectors))

    def simulate_solar_system(self, N, dN, saveevery=1):
        # Convert object staring value lists to numpy
        r_i = np.array(self.r_list) * AU
        v_i = np.array(self.v_list) * AU / D
        m_i = np.array(self.m_list)

        t0_sim_start = time.time()
        t = 0
        t_max = 365*24*60*60*N #N year simulation time
        dt = 60*60*24*dN #dN day time step
        epsilon_s = 0.01 #softening default value

        #blocks_per_grid = 64
        #threads_per_block = 64
        #[blocks_per_grid, threads_per_block]

        a_i = a_t(r_i, m_i, epsilon_s, self.numnatural)

        # Simulation Main Loop using a Leapfrog Kick-Drift-Kick Algorithm
        k = int(t_max/dt)

        r_save = np.zeros((r_i.shape[0],3,k//saveevery+1))
        r_save[:,:,0] = r_i
        self.crashed = {}
        self.crashindex = {}

        for i in range(k):
            if i % 1000 == 0:
                print(f"{i}/{k}")

            if i % COLLISIONCHECKEVERY == 0:
                # TODO if not checking often enough, probes could coast through atmosphere for a while
                for planet_index in range(self.numnatural):
                    planet_position = r_i[planet_index]
                    for idx, sat_position in enumerate(r_i[self.numnatural:]):
                        if idx in self.crashed:
                            continue
                        dist = np.linalg.norm(planet_position-sat_position)
                        #print(idx, dist)#"earth:", earth_position, "sat:", sat_position,
                        if dist < self.planet_radii[planet_index]:
                            #print("crashed:", idx)
                            # TODO check other planets
                            self.crashed[idx] = planet_index
                            self.crashindex[idx] = i
            # (1/2) kick
            v_i += a_i * dt/2.0
            # drift
            r_i += v_i * dt
            # update accelerations
            #[blocks_per_grid, threads_per_block]
            a_i = a_t(r_i, m_i, epsilon_s, self.numnatural)
            # (2/2) kick
            v_i += a_i * dt/2.0
            # update time
            t += dt
            #update list
            if i%saveevery == 0:
                index = i//saveevery+1
                if index >= k//saveevery+1:
                    break
                r_save[:,:,index] = r_i
        sim_time = time.time()-t0_sim_start
        print('The required computation time for the N-Body Simulation was', round(sim_time,3), 'seconds.')
        return r_save

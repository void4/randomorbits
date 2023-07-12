# code stolen and adapted from here: https://amor.cms.hu-berlin.de/~rodrigus/Jupyter%20Notebooks/Solar_System_N-Body_Simulation.html

import time
from random import random
from copy import deepcopy

import numpy as np
import scipy.constants as cs

from astropy.time import Time
from astroquery.jplhorizons import Horizons

from utils import *

#Example for the estimated memory usage from a simulation with N = 13 bodies, t_max = 250 years and dt = 1 day
year = 365*24*60*60
day = 24*60*60

#Conversion Units
AU = 149597870700
D = 24*60*60

earth_radius = 6371*1000#m

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
        for i, jpl_id in enumerate(jpl_ids):
            obj = Horizons(id=jpl_id, location="@sun", epochs=Time(self.t_0).jd, id_type='id').vectors()
            r_obj = [obj['x'][0], obj['y'][0], obj['z'][0]]
            v_obj = [obj['vx'][0], obj['vy'][0], obj['vz'][0]]
            self.r_list.append(r_obj)
            self.v_list.append(v_obj)
            print(self.plot_labels[i], np.linalg.norm(v_obj)*AU/D, "m/s")#np.linalg.norm(v_obj),

    # Vectorial acceleration function
    def a_t(self, r, m, epsilon):
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
        px = r[:self.numnatural, 0:1]
        py = r[:self.numnatural, 1:2]
        pz = r[:self.numnatural, 2:3]

        # matrices that store each pairwise body separation for each [x,y,z] direction: r_j - r_i
        pdx = px.T - px
        pdy = py.T - py
        pdz = pz.T - pz
        # matrix 1/r^3 for the absolute value of all pairwise body separations together and
        inv_pr3 = (pdx ** 2 + pdy ** 2 + pdz ** 2 + epsilon ** 2) ** (-1.5)
        # resulting acceleration components in each [x,y,z] direction
        pax = G * (pdx * inv_pr3) @ m[:self.numnatural]
        pay = G * (pdy * inv_pr3) @ m[:self.numnatural]
        paz = G * (pdz * inv_pr3) @ m[:self.numnatural]
        # pack together the three acceleration components
        pa = np.hstack((pax, pay, paz))

        # positions r = [x,y,z] for all satellites in the N-Body System
        sx = r[self.numnatural:, 0:1]
        sy = r[self.numnatural:, 1:2]
        sz = r[self.numnatural:, 2:3]

        # between planets and satellites
        sdx = px.T - sx
        sdy = py.T - sy
        sdz = pz.T - sz

        # matrix 1/r^3 for the absolute value of all pairwise body separations together and
        inv_sr3 = (sdx ** 2 + sdy ** 2 + sdz ** 2 + epsilon ** 2) ** (-1.5)
        # resulting acceleration components in each [x,y,z] direction
        # XXX i don't understand mass list here
        sax = G * (sdx * inv_sr3) @ m[:self.numnatural]
        say = G * (sdy * inv_sr3) @ m[:self.numnatural]
        saz = G * (sdz * inv_sr3) @ m[:self.numnatural]

        sa = np.hstack((sax, say, saz))

        a = np.vstack((pa, sa))
        return a

    def add_object(self, Id_obj, m_obj, plot_color, plot_label, n_objects=1, random_acceleration=None):
        ori = Horizons(id=Id_obj, location="@sun", epochs=Time(self.t_0).jd, id_type='id').vectors()
        print(ori)
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
                    delta = random_acceleration * random_vector[j] * np.random.uniform(0.5, 1)
                    deltas.append(delta)
                    v_obj[j] += delta

            speed_after = np.linalg.norm(v_obj) * AU / D
            print(
                f"Added: {plot_label + str(i)}\tSpeed: {speed_after:.2f} m/s\tDelta: {speed_after - speed_before:.2f} m/s\tApplied delta: {np.linalg.norm(deltas) * AU / D}")
            self.r_list.append(r_obj)
            self.v_list.append(v_obj)
            self.m_list.append([m_obj])
            self.plot_colors.append(plot_color)
            self.plot_labels.append(plot_label + str(i))

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

        a_i = self.a_t(r_i, m_i, epsilon_s)

        # Simulation Main Loop using a Leapfrog Kick-Drift-Kick Algorithm
        k = int(t_max/dt)

        r_save = np.zeros((r_i.shape[0],3,k//saveevery+1))
        r_save[:,:,0] = r_i
        crashed = set()

        for i in range(k):
            if i % 1000 == 0:
                print(f"{i}/{k}")
                earth_position = r_i[3]
                for idx, sat_position in enumerate(r_i[self.numnatural:]):
                    dist = np.linalg.norm(earth_position-sat_position)
                    #print(idx, dist)#"earth:", earth_position, "sat:", sat_position,
                    if dist < earth_radius:
                        #print("crashed:", idx)
                        crashed.add(idx)
            # (1/2) kick
            v_i += a_i * dt/2.0
            # drift
            r_i += v_i * dt
            # update accelerations
            a_i = self.a_t(r_i, m_i, epsilon_s)
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
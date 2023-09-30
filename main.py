from math import degrees
from PIL import Image, ImageDraw
from collections import Counter

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata

from simulation import *

#Get Starting Parameters for any extra object to add into the simulation with input for id/mass/plot_color/plot_label

MAXSPEEDDIFF = 10000/AU*D
print(MAXSPEEDDIFF)

sim = SolarSystemSimulator()

NOBJECTS = 100
STEP = 0.001#0.001#days
YEARS = 1
SAVEEVERY = 25
print(f"DURATION: {YEARS} years\tSTEP: {STEP*60*60*24:.2f} seconds")

#Add NEOWISE and Starman to the simulation
#sim.add_object('90004479', 5e13, 'black','Comet Neowise')
#sim.add_object('SpaceX Roadster',1300, 'pink','Starman')
#sim.add_object('90000033',1e14, 'gray','Comet Halley')
sim.add_object("-48", 11110, "black", "Hubble", n_objects=NOBJECTS, random_acceleration=MAXSPEEDDIFF)

r_save = sim.simulate_solar_system(YEARS, STEP, saveevery=SAVEEVERY)

ex, ey, ez = r_save[3,0,-1], r_save[3,1,-1], r_save[3,2,-1]

w = 360
h = 180

DOIMG = False
DOPLOT = False

if DOIMG:
    img = Image.new("RGB", (w, h))

distances = []

xs = []
ys = []
zs = []

DORANGEPLOT = True

NUMSOLAR = 11

r_save_truncated = list(deepcopy(r_save[:NUMSOLAR]))

for i in range(len(r_save)-NUMSOLAR):
    stopindex = sim.crashindex[i]//SAVEEVERY if i in sim.crashindex else -1
    r_save_truncated.append([r_save[i+NUMSOLAR,0,:stopindex], r_save[i+NUMSOLAR,1,:stopindex], r_save[i+NUMSOLAR,2,:stopindex]])
    edist = ((ex-r_save[i+NUMSOLAR,0,stopindex])**2 + (ey-r_save[i+NUMSOLAR,1,stopindex])**2 + (ez-r_save[i+NUMSOLAR,2,stopindex])**2)**0.5/AU
    acc, phi, theta = sim.angles[i][-3], sim.angles[i][-2], sim.angles[i][-1]
    el, az = degrees(phi), degrees(theta)%360
    x,y = round(az)%w, round(-el+90)%h
    #print(i, edist, phi, theta, x, y)
    if DOIMG:
        if i in sim.crashed:
            #print(i, "crashed")
            color = (255,0,0)
        else:
            color = (0,255,0)#TODO distance
        img.putpixel((x,y), color)
    
    if DORANGEPLOT:
        xs.append(x)
        ys.append(y)
        zs.append(edist)

    distances.append([i, edist])

nfarthest = 5
print(f"{nfarthest} farthest:")
for i, x in enumerate(sorted(distances, key=lambda x:x[1], reverse=True)):
    if i >= nfarthest:
        break
    if i in sim.crashed:
        print(f"{x} crashed into {sim.plot_labels[sim.crashed[i]].split()[0]}")
    else:
        print(x)

#for key, value in Counter(sim.crashed.values()).most_common(5):
#    print(value, "\t", sim.plot_labels[key])

if DOIMG:
    img.show()

#Plot for the outer planets
import pickle
with open("trajectories.pickle", "wb+") as f:
    f.write(pickle.dumps(r_save_truncated))



if DORANGEPLOT:
    xi = np.arange(0,360,1)
    yi = np.arange(0,180,1)

    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((xs,ys), zs, (xi,yi), method="linear")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.contourf(xi,yi,zi,levels=100)#np.arange(0,1.01,0.01))
    plt.plot(xs,ys,'k.')
    plt.xlabel('xi',fontsize=16)
    plt.ylabel('yi',fontsize=16)
    plt.savefig('interpolated.png',dpi=100)


    def fmt(x, y):
        z = zi[round(y)][round(x)]
        result = f"x={x:.5f}  y={y:.5f}"
        if z:
            result += f" z={z:.5f}"
        return result

    plt.gca().format_coord = fmt
    plt.show()
    plt.close(fig)

if DOPLOT:
    fig2 = plt.figure()
    fig2.canvas.manager.set_window_title('YOLO orbits')
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-50e11,50e11)
    ax.set_ylim3d(-50e11,50e11)
    ax.set_zlim3d(-50e11,50e11)
    #Input sim. data

    PLOT_INTERVAL = 10

    for i in range(len(r_save)): #Plots the outer planets
        stopindex = -1
        crashed = False
        if i >= sim.numnatural:
            idx = i-sim.numnatural
            if idx in sim.crashed:
                stopindex = sim.crashindex[idx]
                crashed = True
        ax.plot3D(r_save[i,0,:stopindex:PLOT_INTERVAL],r_save[i,1,:stopindex:PLOT_INTERVAL],r_save[i,2,:stopindex:PLOT_INTERVAL], "red" if crashed else sim.plot_colors[i],label=sim.plot_labels[i])

    ax.legend(sim.plot_labels[:sim.numnatural], loc = 'upper right', prop={'size': 6.5})

    plt.show()

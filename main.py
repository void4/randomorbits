import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from simulation import *

#Get Starting Parameters for any extra object to add into the simulation with input for id/mass/plot_color/plot_label

MAXSPEEDDIFF = 5000/AU*D
print(MAXSPEEDDIFF)

sim = SolarSystemSimulator()

#Add NEOWISE and Starman to the simulation
#sim.add_object('90004479', 5e13, 'black','Comet Neowise')
#sim.add_object('SpaceX Roadster',1300, 'pink','Starman')
#sim.add_object('90000033',1e14, 'gray','Comet Halley')
sim.add_object("-48", 11110, "black", "Hubble", n_objects=100, random_acceleration=MAXSPEEDDIFF)

STEP = 0.001#days
YEARS = 1
print(f"DURATION: {YEARS} years\tSTEP: {STEP*60*60*24} seconds")

r_save = sim.simulate_solar_system(YEARS, STEP)

#Plot for the outer planets

fig2 = plt.figure()
fig2.canvas.manager.set_window_title('YOLO orbits')
ax = plt.axes(projection='3d')
ax.set_xlim3d(-50e11,50e11)
ax.set_ylim3d(-50e11,50e11)
ax.set_zlim3d(-50e11,50e11)
#Input sim. data

PLOT_INTERVAL = 10

for i in range(len(r_save)): #Plots the outer planets
    ax.plot3D(r_save[i,0,::PLOT_INTERVAL],r_save[i,1,::PLOT_INTERVAL],r_save[i,2,::PLOT_INTERVAL], sim.plot_colors[i],label=sim.plot_labels[i])

ax.legend(sim.plot_labels[:sim.numnatural], loc = 'upper right', prop={'size': 6.5})

plt.show()

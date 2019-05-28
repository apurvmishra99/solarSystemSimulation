import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from numpy.linalg import norm
from celestial_body import CelestialBody
from celestial_body import GRAVITATIONAL_CONSTANT as G

PLANET_LIST = ["Mercury", "Venus", "Earth", "Mars"]


class SolarSystem(object):
    def __init__(self, dat_name, time_step=20000, satellite=False):
        self.t = time_step
        self.time = 0
        self.energies = ([], [])
        with open(dat_name + '.json') as json_dat:
            dat = json.load(json_dat)

        self.bodies = [CelestialBody(*[body] + dat[body]) for body in dat]
        self.sat_target = self.bodies[4]
        self.sat_reached = 0
        self.sat_time = 0
        for body in self.bodies:
            body.update_acc(self.bodies, first=True)

    def move(self):
        """Move the bodies. Will first update all the positions, then all the
        velocities and accelerations. Lastly, the energy of the system will be
        calculated and the time increased by the time step.
        """
        t = self.get_time_step()  # Get the timestep

        for body in self.bodies:
            body.update_pos_revs(t, self.time)

        # Once all the positions are updated, update all velocities and
        # accelerations.
        for body in self.bodies:
            body.update_vel_acc(t, self.bodies)

        self.get_energies()  # Get the total energy
        self.time += t  # Increase the time by time step.

    def get_time_step(self):
        """Get a time step depending on the velocity and acceleration of any
        potential satelites. If there is a satelite it will choose a time step
        what will change the speed by one percent, as long as the time step is not
        larger than the user-defined one. If not, the value given by the user
        will be used.
        """
        for body in self.bodies:
            # If body is a Satelite
            if body.name == "Satelite":
                # Assuming that acceleration for a small times step is constant
                t = 0.01 * norm(body.velocity) / norm(body.acc)
                if t < self.t:
                    return t
        return self.t

    def potential_energies(self):
        """Calculates the potential energy of the system in Joules.
        """
        # Create all pairs of planets
        pairs = itertools.combinations(self.bodies, 2)
        # Return the sum of all potential energies.
        return sum([-G * pair[0].mass * pair[1].mass /
                    norm(pair[0].position - pair[1].position) for pair in pairs])

    def kinetic_energies(self):
        """Calculates the potential energy of the system in Jules.
        """
        return sum([body.kinetic_energy
                    for body in self.bodies])

    def get_energies(self):
        """Calculate the total energy of the system and add it to a text file
        named 'energies.txt'.
        """
        # Total energy is the sum of kinetic plus potential energy.
        energy = self.potential_energies() + self.kinetic_energies()
        # self.energies will be used for plotting.
        self.energies[0].append(self.time)
        self.energies[1].append(energy)

        # If it's the first iteration overwrite the file, else append.
        if self.time == 0:
            mode = "w"
        else:
            mode = "a"
        # Write the energy to the file.
        text = ("Time: " + str(self.time) + "s. Energy: "
                + str(energy) + "J.\n")
        with open("energies.txt", mode) as energies_file:
            energies_file.write(text)

    def plot_energies(self):
        """Plot a graph of the total energy of the system over time.
        """
        fig, ax = plt.subplots()  # create a new figure with a default 111 subplot
        ax.plot(self.energies[0], self.energies[1])
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.show()

    def get_obital_periods(self):
        """Get the orbital period of all the planets. Writes the periods to a file
        named 'orbital-periods.txt', only if the planet has done more than one lap.
        """
        s = ""
        for body in self.bodies:
            if body.name in PLANET_LIST:
                try:  # If it has done more than one lap
                    # Get the period in days
                    period = round(
                        body.revs[1] / (body.revs[0] * 3600 * 24), 2)
                    # Get the fracion of a year
                    fraction = round(period / 365, 2)

                    s += (body.name + ": " + str(period) + " days (" +
                             str(fraction) + " times the real Earth's period).\n")
                except:
                    pass
        # Write to text file.
        with open('orbital-periods.txt', 'w') as f:
            f.write(s)

    def write_satellite_time(self):
        with open('satellite_time.txt', 'w') as f:
            f.write(self.sat_time)

    def init(self):
        """Initialize the animation.
        """
        return self.patches

    def get_dimensions(self):
        """Find the maximum distance from one planet to the sun, and return twenty
        percent more of that distance to find a suitable axis size.
        """
        dim = np.amax([body.position for body in self.bodies])
        return dim * 1.2

    def animate(self, i):
        """Call the function move and set the center of the circles to the new
        position of the bodies.
        """
        self.move()
        #Checking if satellite reaches its target
        if self.sat_target.distance(self.bodies[5]) < 7e9 and not self.sat_reached:
            self.sat_time = self.t * i / (3600*24)
            self.sat_reached = 1

        for n, body in enumerate(self.bodies):
            self.patches[n].center = (body.position[0], body.position[1])
        return self.patches

    def run(self, animation=True, energy_graph=False, orbital_periods=True, time_to_mars=True):
        """Run the simulation.
        """
        fig = plt.figure()
        ax = plt.axes()
        lim = self.get_dimensions()
        self.patches = [plt.Circle((body.position[0], body.position[1]),
                                   body.size, color=body.colour)
                        for body in self.bodies]
        for patch in self.patches:
            ax.add_patch(patch)
        ax.set_facecolor('black')
        ax.axis('scaled')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        anim = FuncAnimation(
            fig=fig, func=self.animate, init_func=self.init, frames=30000, interval=1)

        if animation:
            plt.show()

        if orbital_periods:
            self.get_obital_periods()

        if energy_graph:
            self.plot_energies()

        if time_to_mars:
            self.write_satellite_time()

        return "Simulation complete."

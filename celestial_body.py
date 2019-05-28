import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import scipy.constants as const

GRAVITATIONAL_CONSTANT = const.gravitational_constant
MASS_SUN = 1.989 * 10 ** 30


class CelestialBody(object):

    def __init__(self, name, colour, size, mass, r, radius_planet, v=None):
        self.name = name
        self.colour = colour
        self.size = size
        self.mass = mass
        self._position = np.asanyarray(r, dtype='float64')
        self.revs = (0,0)
        self.acc = None
        self.old_acc = self.acc
        radius = np.linalg.norm(r)
        if v is None:  # If it is a planet and has no given initial velocity.
            if radius > 0:  # If its not the Sun
                # Calculate the velocity using the square root of G times the
                # mass of the Sun over the distance to it.
                self._velocity = np.asanyarray(np.sqrt(GRAVITATIONAL_CONSTANT * MASS_SUN / radius) *
                            np.array([-self.position[1] / radius, self.position[0] / radius]), dtype='float64')
            else:
                self._velocity = np.zeros(2, dtype='float64')  # Set Sun's initial velocity to zero.
        else:
            self._velocity = np.asanyarray(
                v * np.array([np.cos(0.9), np.sin(0.9)]), dtype='float64')
    
    def update_pos_revs(self, t, time):
        """Obtaining the new position using the Beeman's algorithm.
        If an orbit is completed, increase the number of laps by one and save
        the time.
        Args:
            param1 (float): a small time step.
            param2 (float): the current time since the start of the simulation.
        """
        new_pos = (self.position + self.velocity * t) + ((t**2 / 6.0) *(4.0 * self.acc - self.old_acc))

        # If the planet is going from 4th to 1st quadrant
        if self.position[1] < 0 and new_pos[1] >= 0:
            self.revs = (self.revs[0]+1, time)  # Increase laps by one
        self.position = new_pos

    def update_vel(self, t, new_acc):
        """Obtaining the new velocity using the Beeman's algorithm.
        Args:
            param1 (float): a small time step needed for numerical integration.
            param2 (float): the acceleration at the next time step.
        """
        self.velocity = self.velocity + ((t / 6.0) * ((2.0 * new_acc) + (5.0 * self.acc) - self.old_acc))

    def update_acc(self, planets, first = False):
        """Calculating the acceleration due to the other planets.
        The sum of all the Gm/r, where m is the mass of the other planet and r
        is the radius separating them.
        Args:
            param1 (list): a list of all the planets.
            param2 (bool) (opt): whether it is the first iteration or not
        """
        acc = np.zeros(2)
        for planet in planets:
            if planet is not self:
                dist = self.position - planet.position  # Distance between planets
                # Using Newton's gravitational formula.
                acc = acc + ((-GRAVITATIONAL_CONSTANT * planet.mass * dist) / np.linalg.norm(dist)**3)

        if first:  # If it is the first iteration
            self.acc = acc
            # Set the previous acceleration equal to the current acceleration
            self.old_acc = acc
        return acc

    def update_vel_acc(self, t, planets):
        """Updates the velocity and the acceleration of the satelite.
        Args:
            param1 (float): a small time step needed for numerical integration.
            param2 (list): a list of all the planets.
        """
        # First calculate the new acceleration.
        new_acc = self.update_acc(planets)
        # Calculate the velocity with it
        self.update_vel(t, new_acc)
        # Lastly, update the accelerations.
        self.old_acc, self.acc = self.acc, new_acc

    def distance(self, other):
        return euclidean(self.position, other.position)

    @property
    def position(self):
        return self._position

    @property
    def velocity(self):
        return self._velocity

    @property
    def kinetic_energy(self):
        return 0.5 * self.mass * np.linalg.norm(self.velocity)**2

    @position.setter
    def position(self, value):
        self._position = np.asanyarray(value, dtype='float64')

    @velocity.setter
    def velocity(self, value):
        self._velocity = np.asanyarray(value, dtype='float64')

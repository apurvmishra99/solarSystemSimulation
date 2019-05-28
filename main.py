from SolarSystem import SolarSystem

if __name__ == "__main__":

    solar_system = SolarSystem("solar-system-data", 20000, satellite=True)
    solar_system.run(animation=True, energy_graph=True, orbital_periods=True, time_to_mars=False)
    
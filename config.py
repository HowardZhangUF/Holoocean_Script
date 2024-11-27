import holoocean
import matplotlib.pyplot as plt
import numpy as np

#### GET SONAR CONFIG
scenario = "OpenWater-TorpedoSidescanSonar"
config = holoocean.packagemanager.get_scenario(scenario)
config = config['agents'][0]['sensors'][-1]["configuration"]
maxR = config['RangeMax']
binsR = config['RangeBins']

#### GET PLOT READY
plt.ion()

# Set up time and range for the plot
t = np.arange(0, 50)
r = np.linspace(-maxR, maxR, binsR)
R, T = np.meshgrid(r, t)
data = np.zeros_like(R)

# Prepare the plot with inverted Y-axis for sonar visualization
plt.grid(False)
plot = plt.pcolormesh(R, T, data, cmap='copper', shading='auto', vmin=0, vmax=1)
plt.tight_layout()
plt.gca().invert_yaxis()
plt.gcf().canvas.flush_events()

#### RUN SIMULATION
command = np.array([0, 0, 0,0, 0, 5000])  # Define movement command for the AUV
with holoocean.make(scenario) as env:
    for i in range(100000):
        env.act("auv0", command)
        state = env.tick()

        # Process the sonar data if available
        if 'SidescanSonar' in state:
            # Roll the array to visualize new data while keeping old data
            data = np.roll(data, 1, axis=0)
            data[0] = state['SidescanSonar']

            # Update the plot with new data
            plot.set_array(data.ravel())

            # Redraw the plot to update the visualization
            plt.draw()
            plt.gcf().canvas.flush_events()

print("Finished Simulation!")
plt.ioff()
plt.show()     
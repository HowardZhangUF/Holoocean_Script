import holoocean
import numpy as np
import cv2
import os

# Create an output directory for images
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

config = {
    "name": "SurfaceNavigator",
    "world": "SimpleUnderwater",
    "package_name": "Ocean",
    "main_agent": "sv",
    "agents": [
        {
            "agent_name": "sv",
            "agent_type": "SurfaceVessel",
            "sensors": [
                {"sensor_type": "GPSSensor"},
                {"sensor_type": "IMUSensor"},
                {"sensor_type": "RGBCamera"}  # Adding an RGB camera sensor
            ],
            "control_scheme": 1,  # PD Control Scheme
            "location": [0, 0, -10],
            "rotation": [0, 0, 0]
        }
    ],
}

# Define waypoints
idx = 0
locations = np.array([[25, 25],
                      [-25, 25],
                      [-25, -25],
                      [25, -25]])

# Image counter
image_counter = 0

# Function to save image
def save_image(image, counter):
    filename = os.path.join(output_dir, f"image_{counter}.png")
    cv2.imwrite(filename, image)
    print(f"Saved image at {filename}")

# Start simulation
with holoocean.make(scenario_cfg=config) as env:
    # Draw waypoints
    for l in locations:
        env.draw_point([l[0], l[1], 0], lifetime=0)

    print("Going to waypoint ", idx)

    while True:
        # Send waypoint to HoloOcean
        state = env.step(locations[idx])

        # Check if we're close to the waypoint
        p = state["GPSSensor"][0:2]
        if np.linalg.norm(p - locations[idx]) < 1e-1:
            # Capture and save image at the waypoint with incrementing filename
            image = state["RGBCamera"]  # Access the RGBCamera sensor data
            save_image(image, image_counter)

            # Increment image counter
            image_counter += 1

            # Move to the next waypoint
            idx = (idx + 1) % len(locations)
            print("Going to waypoint ", idx)

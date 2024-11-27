import holoocean
import numpy as np
import cv2
import os

# Create output directories for RGB and sonar images
output_dir_rgb = "output_images/rgb"
output_dir_sonar = "output_images/sonar"
os.makedirs(output_dir_rgb, exist_ok=True)
os.makedirs(output_dir_sonar, exist_ok=True)

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
                {"sensor_type": "RGBCamera"},
                {"sensor_type": "ImagingSonar"}
            ],
            "control_scheme": 1,
            "location": [0, 0, 2],
            "rotation": [0, 0, 0]
        },
        {
            "agent_name": "torpedo",
            "agent_type": "TorpedoAUV",
            "sensors": [
                {"sensor_type": "IMUSensor"},
                {"sensor_type": "DepthSensor"},
                {"sensor_type": "ImagingSonar"}
            ],
            "control_scheme": 1,
            "location": [0, 0, -5],
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

# Function to save RGB image
def save_rgb_image(image, img_count):
    filename = os.path.join(output_dir_rgb, f"waypoint_{img_count}.png")
    cv2.imwrite(filename, image)
    print(f"Saved RGB image at {filename}")

# Function to save sonar data as an image
def save_sonar_data(sonar_data, img_count, agent_name):
    sonar_image = (sonar_data - sonar_data.min()) / (sonar_data.max() - sonar_data.min()) * 255
    sonar_image = sonar_image.astype(np.uint8)
    filename = os.path.join(output_dir_sonar, f"waypoint_{img_count}_{agent_name}_sonar.png")
    cv2.imwrite(filename, sonar_image)
    print(f"Saved sonar data image for {agent_name} at {filename}")

# Define no-operation actions
no_op_sv = np.zeros(6)  # Adjust as necessary for SurfaceVessel
no_op_torpedo = np.zeros(6)  # Adjust as necessary for TorpedoAUV

# Image count to keep filenames unique
img_count = 0

# Start simulation
with holoocean.make(scenario_cfg=config) as env:
    # Draw waypoints
    for l in locations:
        env.draw_point([l[0], l[1], 0], lifetime=0)

    print("Going to waypoint ", idx)

    while True:
        # Act for each agent separately
        env.act("sv", no_op_sv)
        env.act("torpedo", no_op_torpedo)

        # Step the environment once for SurfaceVessel
        state = env.step(no_op_sv)

        # Check if we're close to the waypoint
        p = state["sv"]["GPSSensor"][0:2]
        if np.linalg.norm(p - locations[idx]) < 1e-1:
            # Capture and save RGB image at the waypoint
            rgb_image = state["sv"]["RGBCamera"]
            save_rgb_image(rgb_image, img_count)

            # Capture and save sonar data for SurfaceVessel
            if "ImagingSonar" in state["sv"]:
                sonar_data_sv = state["sv"]["ImagingSonar"]
                save_sonar_data(sonar_data_sv, img_count, "sv")

            # Capture and save sonar data for TorpedoAUV
            if "ImagingSonar" in state["torpedo"]:
                sonar_data_torpedo = state["torpedo"]["ImagingSonar"]
                save_sonar_data(sonar_data_torpedo, img_count, "torpedo")

            # Increment image count
            img_count += 1

            # Move to the next waypoint
            idx = (idx + 1) % len(locations)
            print("Going to waypoint ", idx)

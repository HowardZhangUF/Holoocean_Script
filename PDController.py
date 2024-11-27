import holoocean
import numpy as np

config = {
    "name": "SurfaceNavigator",
    "world": "SimpleUnderwater",
    "package_name": "Ocean",
    "main_agent": "sv",
    "agents":[
        {
            "agent_name": "sv",
            "agent_type": "SurfaceVessel",
            "sensors": [
            {
                "sensor_type": "GPSSensor",
            },
            {
                "sensor_type":"IMUSensor"
            }
            ],
            "control_scheme": 1, # PD Control Scheme
            "location": [0,0,2],
            "rotation": [0, 0, 0]
        }
    ],
}

# Define waypoints
idx = 0
locations = np.array([[25,25],
                    [-25,25],
                    [-25,-25],
                    [25,-25]])

# Start simulation
with holoocean.make(scenario_cfg=config) as env:
    # Draw waypoints
    for l in locations:
        env.draw_point([l[0], l[1], 0], lifetime=0)

    print("Going to waypoint ", idx)

    while True:
        #send waypoint to holoocean
        state = env.step(locations[idx])

        # Check if we're close to the waypoint
        p = state["GPSSensor"][0:2]
        if np.linalg.norm(p-locations[idx]) < 1e-1:
            idx = (idx+1) % 4
            print("Going to waypoint ", idx)
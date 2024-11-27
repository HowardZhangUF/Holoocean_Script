import holoocean
import numpy as np

cfg = {
    "name": "test_rgb_camera",
    "world": "SimpleUnderwater",
    "package_name": "Ocean",
    "main_agent": "auv0",
    "ticks_per_sec": 60,
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "TorpedoAUV",
            "sensors": [
                {
                    "sensor_type": "IMUSensor"
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -5]
        },
        {
            "agent_name": "auv1",
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "DVLSensor"
                }
            ],
            "control_scheme": 0,
            "location": [0, 2, -5]
        }
    ]
}

env = holoocean.make(scenario_cfg=cfg)
env.reset()

env.act('auv0', np.array([0,0,0,0,75]))
env.act('auv1', np.array([0,0,0,0,20,20,20,20]))
for i in range(3000000):
    states = env.tick()

    # states is a dictionary
    imu = states["auv0"]["IMUSensor"]

    vel = states["auv1"]["DVLSensor"]
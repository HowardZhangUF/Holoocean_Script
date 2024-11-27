import holoocean

cfg = {
    "name": "test_acou_coms",
    "world": "SimpleUnderwater",
    "package_name": "Ocean",
    "main_agent": "auv0",
    "ticks_per_sec": 200,
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "OpticalModemSensor",
                    "location": [0,0,0],
                    "socket": "SonarSocket",
                    "configuration": {
                        "id": 0
                    }
                },
            ],
            "control_scheme": 0,
            "location": [25, 0, -5],
            "rotation": [0, 0, 180]
        },
        {
            "agent_name": "auv1",
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "OpticalModemSensor",
                    "location": [0,0,0],
                    "socket": "SonarSocket",
                    "configuration": {
                        "id": 1
                    }
                },
            ],
            "control_scheme": 0,
            "location": [0, 0, -5]
        }
    ]
}

env = holoocean.make(scenario_cfg=cfg)
env.reset()

# This is how you send a message from one optical com to another
# This sends from id 0 to id 1 (ids configured above)
# with data "my_data_payload"
env.send_optical_message(0, 1, "my_data_payload")

for i in range(300):
    states = env.tick()
    if "OpticalModemSensor" in states['auv1']:
        print(i, "Received:", states['auv1']["OpticalModemSensor"])
        break
    else:
        print(i, "No message received")
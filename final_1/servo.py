import asyncio
from mavsdk import System

async def control_actuator():
    drone = System()
    # await drone.connect(system_address="udp://:14540")
    await drone.connect(system_address="serial:///dev/ttyUSB0:57600")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected!")
            break

    # Set actuator with index 1 to maximum value (e.g., to trigger a switch/release)
    # The value range is typically -1.0 to 1.0
    actuator_index = 1
    actuator_value = 1.0 
    await drone.action.set_actuator(actuator_index, actuator_value)
    print(f"Set actuator {actuator_index} to value {actuator_value}")

    # Wait for a short duration to ensure the command is processed
    await asyncio.sleep(1) 

    # Set actuator value back to neutral/off (e.g., 0.0)
    actuator_value = -1.0
    await drone.action.set_actuator(actuator_index, actuator_value)
    print(f"Set actuator {actuator_index} to value {actuator_value}")

if __name__ == "__main__":
    asyncio.run(control_actuator())

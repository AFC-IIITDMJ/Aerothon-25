import asyncio
from mavsdk import System

connection_string = "serial:///dev/ttyUSB0:57600"
# connection_string = "serial:///dev/ttyACM0:9600"

async def clear_uploaded_mission():
    # Connect to the drone
    drone = System()
    await drone.connect(system_address=connection_string)

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected!")
            break

    print("Clearing mission...")
    try:
        await drone.mission.clear_mission()
        print("Mission cleared successfully!")
    except Exception as e:
        print(f"Failed to clear mission: {e}")

if __name__ == "__main__":
    # Run the asyncio event loop
    asyncio.run(clear_uploaded_mission())

#!/usr/bin/env python3

import asyncio
import sys
from mavsdk import System

async def simple_test():
    drone = System()
    await drone.connect(system_address=sys.argv[1] if len(sys.argv) > 1 else "udp://:14540")
    
    print("Waiting for drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    
    print("Waiting for GPS...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            break
    
    # Monitor altitude
    async def altitude():
        async for pos in drone.telemetry.position():
            print(f"Alt: {pos.relative_altitude_m:.1f}m")
    
    task = asyncio.create_task(altitude())
    
    try:
        # Takeoff
        print("Arming & Takeoff...")
        await drone.action.arm()
        await drone.action.takeoff()
        await asyncio.sleep(20)  # Hover 20 seconds
        
        # RTL
        print("RTL...")
        await drone.action.return_to_launch()
        await asyncio.sleep(30)  # Wait for RTL
        
        # Land (if still flying)
        print("Landing...")
        await drone.action.land()
        
        # Wait for landing
        async for in_air in drone.telemetry.in_air():
            if not in_air:
                break
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        task.cancel()

if __name__ == "__main__":
    asyncio.run(simple_test())

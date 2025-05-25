import rclpy
import yaml
from src.drone_control.vehicle_manager import VehicleManager
from src.drone_control.geofence_manager import GeofenceManager
from src.drone_control.mission_planner import MissionPlanner
from src.drone_control.alignment_controller import AlignmentController
from src.vision.camera_subscriber import CameraSubscriber
from src.utilities.threading_manager import ThreadManager
from threading import Lock

class SharedData:
    def __init__(self):
        self.lock = Lock()
        self.coords = None

def load_config(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load configurations
    geofence_config = load_config('config/geofence_coords.yaml')
    pid_config = load_config('config/pid_params.yaml')
    
    shared_data = SharedData()
    thread_manager = ThreadManager()
    vehicle_manager = VehicleManager()
    vehicle_manager.connect_vehicle()
    
    geofence_manager = GeofenceManager(vehicle_manager.get_vehicle())
    geofence_manager.set_geofence(geofence_config['coordinates'])
   
    mission_planner = MissionPlanner(vehicle_manager.get_vehicle())
    mission_planner.plan_mission(geofence_config['coordinates'])
    
    vehicle_manager.arm_and_takeoff(15)
    
    vehicle_manager.set_mode("AUTO")
    
    camera_subscriber = CameraSubscriber(shared_data, "model/sample_model.pt")
    alignment_controller = AlignmentController(shared_data, vehicle_manager.get_vehicle(), pid_config)
  
    thread_manager.start_task(rclpy.spin, (camera_subscriber,))
    thread_manager.start_task(alignment_controller.alignment_loop)
    
    thread_manager.join_all()
    
    vehicle_manager.close()

if __name__ == "__main__":
    main()
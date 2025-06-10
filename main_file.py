import threading
import rclpy
from detector_node import YoloDetector
from mission_manager import DroneMission


def main():
    """
    Entry point: initialize shared state, start ROS 2 detector and DroneMission threads.
    """
    # Shared state dictionary with locks
    shared_state = {
        'ALTITUDE': 15.0,
        'LOW_ALTITUDE': 10.0,
        'ZIGZAG_SPACING': 18.0,
        'GEOFENCE_POLYGON': [
            (-35.36375764, 149.16612950),
            (-35.36374710, 149.16495300),
            (-35.36235539, 149.16494007),
            (-35.36239757, 149.16592264),
            (-35.36375764, 149.16612950)
        ],
        'CONNECTION_STRING': '127.0.0.1:14550',
        # Shared variables
        'detected_targets': [],       # [(u, v, cx, cy, lat0, lon0), ...]
        'detected_targets_gps': [],   # [(lat, lon), ...]
        'hotspot_dict': {},           # {id: (lat, lon)}
        'hotspot_id_counter': 1,
        'published_hotspot_history': {},  # {((u,v),(cx,cy)): cooldown_counter}
        'latest_target_pixel': None,
        'target_serviced': False,
        # Vehicle reference
        'vehicle': None,
        # Locks
        'detected_targets_lock': threading.Lock(),
        'gps_lock': threading.Lock(),
        'hotspot_lock': threading.Lock(),
        'pixel_lock': threading.Lock(),
        'target_lock': threading.Lock(),
        'vehicle_lock': threading.Lock()
    }

    # Initialize and start detector thread (ROS 2)
    rclpy.init()
    detector_node = YoloDetector(
        config_path='path/to/yolov4-tiny.cfg',
        weights_path='path/to/yolov4-tiny.weights',
        classes_path='path/to/classes.txt',
        shared_state=shared_state
    )
    detector_thread = threading.Thread(target=rclpy.spin, args=(detector_node,), daemon=True)
    detector_thread.start()

    # Initialize and start the DroneMission thread
    mission = DroneMission(shared_state)
    mission_thread = threading.Thread(target=mission.execute_mission)
    mission_thread.start()

    # Wait for both threads to complete
    mission_thread.join()
    detector_node.destroy_node()
    rclpy.shutdown()
    detector_thread.join()
    print("All threads completed.")


if __name__ == '__main__':
    main()

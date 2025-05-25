import threading
from typing import Callable, Any, List
import logging

class ThreadManager:
    
    def __init__(self):
        self.threads: List[threading.Thread] = []
        self.running = True
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def start_task(self, task: Callable, args: tuple = ()) -> threading.Thread:
        thread = threading.Thread(target=self._run_task, args=(task, args))
        thread.daemon = True
        self.threads.append(thread)
        thread.start()
        self.logger.info(f"Started thread for task {task.__name__}")
        return thread
    
    def _run_task(self, task: Callable, args: tuple) -> None:
        try:
            task(*args)
        except Exception as e:
            self.logger.error(f"Error in task {task.__name__}: {str(e)}")
    
    def stop_all(self) -> None:
        self.running = False
        self.logger.info("Stopping all threads")
    
    def join_all(self) -> None:
        for thread in self.threads:
            if thread.is_alive():
                thread.join()
        self.threads.clear()
        self.logger.info("All threads joined")
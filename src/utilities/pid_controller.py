import time
import numpy as np
from typing import Dict

class StablePID:
    
    def __init__(self, Kp: float, Ki: float, Kd: float, max_output: float, tau: float = 0.1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output
        self.tau = tau
        self.reset()
    
    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.prev_time = time.time()
    
    def update(self, error: float) -> float:
        now = time.time()
        dt = now - self.prev_time
        if dt <= 0:
            dt = 1e-16
        self.prev_time = now
        
        self.integral += error * dt
        self.integral = np.clip(self.integral, -1.0, 1.0)
        
        raw_derivative = (error - self.prev_error) / dt
        derivative = (self.tau * self.prev_derivative + dt * raw_derivative) / (self.tau + dt)
        self.prev_derivative = derivative
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        output = np.clip(output, -self.max_output, self.max_output)
        return output
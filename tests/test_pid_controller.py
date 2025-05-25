import unittest
from src.utilities.pid_controller import StablePID

class TestStablePID(unittest.TestCase):
    def test_pid_output(self):
        pid = StablePID(Kp=1.0, Ki=0.1, Kd=0.05, max_output=1.0)
        output = pid.update(10.0)
        self.assertTrue(-1.0 <= output <= 1.0, "PID output out of bounds")

if __name__ == '__main__':
    unittest.main()
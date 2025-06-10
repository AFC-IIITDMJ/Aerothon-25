import time
import numpy as np


class StablePID:
    """
    EliteStablePID - Enhanced PID controller with multiple filters for superior stability

    Attributes:
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        max_output (float): Maximum absolute output value.
        tau (float): Time constant for derivative filter.
        derivative_filter_tau (float): Time constant for derivative smoothing.
        error_ema_alpha (float): Alpha for exponential moving average on error.
        output_smoothing_alpha (float): Alpha for smoothing the PID output.
        integral_limit (float): Limit for integral term to prevent windup.
    """

    def __init__(self, Kp: float, Ki: float, Kd: float,
                 max_output: float, tau: float = 0.1,
                 derivative_filter_tau: float = 0.05,
                 error_ema_alpha: float = 0.2,
                 output_smoothing_alpha: float = 0.1,
                 integral_limit: float = 1.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output
        self.tau = tau
        self.derivative_filter_tau = derivative_filter_tau
        self.error_ema_alpha = error_ema_alpha
        self.output_smoothing_alpha = output_smoothing_alpha
        self.integral_limit = integral_limit
        self.reset()

    def reset(self) -> None:
        """
        Reset all internal state variables to initial conditions.
        """
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_error = 0.0
        self.prev_derivative = 0.0
        self.filtered_derivative = 0.0
        self.prev_output = 0.0
        self.prev_time = time.time()

    def update(self, error: float) -> float:
        """
        Update the PID controller with a new error measurement.

        Args:
            error (float): The current error value.

        Returns:
            float: The smoothed, clipped PID output.
        """
        now = time.time()
        dt = now - self.prev_time
        if dt <= 0:
            dt = 1e-16
        self.prev_time = now

        # Exponential moving average (EMA) on error
        self.filtered_error = (
            self.error_ema_alpha * error
            + (1 - self.error_ema_alpha) * self.filtered_error
        )

        # Integral term with anti-windup
        prev_integral = self.integral
        self.integral += self.filtered_error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        # Raw derivative on filtered error
        raw_derivative = (self.filtered_error - self.prev_error) / dt
        derivative = (
            (self.tau * self.prev_derivative + dt * raw_derivative) / (self.tau + dt)
        )
        self.prev_derivative = derivative

        # Further smoothing on derivative
        self.filtered_derivative = (
            (self.derivative_filter_tau * self.filtered_derivative + dt * derivative)
            / (self.derivative_filter_tau + dt)
        )

        # PID components
        p_term = self.Kp * self.filtered_error
        i_term = self.Ki * self.integral
        d_term = self.Kd * self.filtered_derivative
        raw_output = p_term + i_term + d_term

        # Clip output and anti-windup adjustment
        clipped_output = np.clip(raw_output, -self.max_output, self.max_output)
        if clipped_output != raw_output:
            self.integral = prev_integral

        # Output smoothing
        smoothed_output = (
            self.output_smoothing_alpha * clipped_output
            + (1 - self.output_smoothing_alpha) * self.prev_output
        )
        self.prev_output = smoothed_output
        self.prev_error = self.filtered_error

        # Scale factor for final output (optional tuning)
        return smoothed_output * 0.9

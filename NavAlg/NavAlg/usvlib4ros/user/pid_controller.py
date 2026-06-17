from dataclasses import dataclass


@dataclass
class PIDConfig:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    output_limit: float | None = None
    integral_limit: float | None = None


class PID:
    """Simple PID controller with integral and output clamping."""

    def __init__(self, config: PIDConfig):
        self.config = config
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            dt = 1e-6

        proportional = self.config.kp * error

        self.integral += error * dt
        if self.config.integral_limit is not None:
            self.integral = max(-self.config.integral_limit, min(self.integral, self.config.integral_limit))
        integral = self.config.ki * self.integral

        if self.initialized:
            derivative_raw = (error - self.prev_error) / dt
        else:
            derivative_raw = 0.0
            self.initialized = True
        derivative = self.config.kd * derivative_raw

        self.prev_error = error
        output = proportional + integral + derivative
        if self.config.output_limit is not None:
            output = max(-self.config.output_limit, min(output, self.config.output_limit))
        return output

    def get_terms(self, error: float, dt: float) -> tuple[float, float, float]:
        if dt <= 0:
            dt = 1e-6
        proportional = self.config.kp * error
        integral_state = self.integral + error * dt
        if self.config.integral_limit is not None:
            integral_state = max(-self.config.integral_limit, min(integral_state, self.config.integral_limit))
        integral = self.config.ki * integral_state
        derivative_raw = 0.0 if not self.initialized else (error - self.prev_error) / dt
        derivative = self.config.kd * derivative_raw
        return proportional, integral, derivative

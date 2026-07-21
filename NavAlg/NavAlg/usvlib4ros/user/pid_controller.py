from dataclasses import dataclass


@dataclass
class PIDConfig:
    """
    PID 控制器的配置参数（数据类）。

    Attributes:
        kp: 比例系数（P）。
        ki: 积分系数（I），默认 0.0（不使用积分）。
        kd: 微分系数（D），默认 0.0（不使用微分）。
        output_limit: 输出限幅值（绝对值），例如 10.0 表示输出限制在 [-10.0, 10.0] 内；None 表示不限制。
        integral_limit: 积分项累加值的限幅（绝对值），防止积分饱和；None 表示不限制。
    """
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    output_limit: float | None = None
    integral_limit: float | None = None


class PID:
    """简易 PID 控制器，带积分和输出限幅功能。"""

    def __init__(self, config: PIDConfig):
        """
        初始化 PID 控制器。

        Args:
            config: PIDConfig 实例，包含所有控制参数。
        """
        self.config = config          # 配置参数
        self.integral = 0.0           # 积分累加值（未乘 ki）
        self.prev_error = 0.0         # 上一次的误差，用于计算微分
        self.initialized = False      # 标志位，用于区分第一次更新（微分项需延迟一帧）

    def reset(self) -> None:
        """重置控制器状态（积分累加、历史误差、初始化标志），回到初始状态。"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def update(self, error: float, dt: float) -> float:
        """
        根据当前误差和时间步长更新 PID 输出。

        Args:
            error: 当前误差（设定值 - 测量值，或反之，取决于您的系统定义）。
            dt:    时间步长（秒），必须大于 0。若传入 <=0，会被置为极小正值以避免除零。

        Returns:
            float: 经过限幅后的控制输出值。
        """
        # 防止 dt 为零或负数导致除零或反向积分
        if dt <= 0:
            dt = 1e-6

        # ---- 比例项 ----
        proportional = self.config.kp * error

        # ---- 积分项 ----
        # 累加误差*dt（数值积分）
        self.integral += error * dt
        # 若设置了积分限幅，则对累加值进行钳制
        if self.config.integral_limit is not None:
            self.integral = max(-self.config.integral_limit,
                                min(self.integral, self.config.integral_limit))
        integral = self.config.ki * self.integral

        # ---- 微分项 ----
        # 第一次更新时没有历史误差，微分项视为 0，并初始化 prev_error
        if self.initialized:
            derivative_raw = (error - self.prev_error) / dt
        else:
            derivative_raw = 0.0
            self.initialized = True
        derivative = self.config.kd * derivative_raw

        # 保存当前误差供下一次微分计算
        self.prev_error = error

        # ---- 总输出 ----
        output = proportional + integral + derivative

        # 若设置了输出限幅，则钳制最终输出
        if self.config.output_limit is not None:
            output = max(-self.config.output_limit,
                         min(output, self.config.output_limit))

        return output

    def get_terms(self, error: float, dt: float) -> tuple[float, float, float]:
        """
        计算并返回当前 P、I、D 各分量的值（不修改控制器内部状态，仅用于观察/调试）。

        注意：该函数会基于当前内部状态（integral 和 prev_error）和传入的 error/dt
        计算各分量，但**不会更新**内部状态（积分累加值和历史误差不变）。

        Args:
            error: 当前误差。
            dt:    时间步长（秒），若 <=0 则置为极小值。

        Returns:
            tuple[float, float, float]: (比例项, 积分项, 微分项) 的元组。
        """
        if dt <= 0:
            dt = 1e-6

        # 比例项（直接计算）
        proportional = self.config.kp * error

        # 积分项：先计算“如果更新”后的积分累加值，但不实际赋值
        integral_state = self.integral + error * dt
        if self.config.integral_limit is not None:
            integral_state = max(-self.config.integral_limit,
                                 min(integral_state, self.config.integral_limit))
        integral = self.config.ki * integral_state

        # 微分项：使用当前历史误差计算，若未初始化则视为 0
        derivative_raw = 0.0 if not self.initialized else (error - self.prev_error) / dt
        derivative = self.config.kd * derivative_raw

        return proportional, integral, derivative
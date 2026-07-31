class GuidedHeadingSlewLimiter:
    """Limit PPO heading changes according to its prior guidance error."""

    def __init__(self, min_change_degrees: float, error_margin_degrees: float):
        if min_change_degrees < 0 or error_margin_degrees < 0:
            raise ValueError("heading limit parameters must be non-negative")
        self.min_change_degrees = float(min_change_degrees)
        self.error_margin_degrees = float(error_margin_degrees)
        self._previous_heading = None
        self._previous_guidance_error = None

    @staticmethod
    def normalize_signed(angle: float) -> float:
        return (float(angle) + 180.0) % 360.0 - 180.0

    def reset(self) -> None:
        self._previous_heading = None
        self._previous_guidance_error = None

    def apply(self, proposed_heading: float, guidance_heading: float) -> float:
        proposed_heading = self.normalize_signed(proposed_heading)
        guidance_heading = self.normalize_signed(guidance_heading)
        if self._previous_heading is None:
            limited_heading = proposed_heading
        else:
            max_change = max(
                self.min_change_degrees,
                self._previous_guidance_error + self.error_margin_degrees,
            )
            max_change = min(max_change, 180.0)
            change = self.normalize_signed(proposed_heading - self._previous_heading)
            change = max(-max_change, min(change, max_change))
            limited_heading = self.normalize_signed(self._previous_heading + change)

        self._previous_heading = limited_heading
        self._previous_guidance_error = abs(
            self.normalize_signed(limited_heading - guidance_heading)
        )
        return limited_heading

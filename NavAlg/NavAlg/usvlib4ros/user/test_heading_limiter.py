import unittest

from heading_limiter import GuidedHeadingSlewLimiter


class HeadingSlewLimiterTest(unittest.TestCase):
    def test_uses_110_degree_range_after_90_degree_guidance_error(self):
        limiter = GuidedHeadingSlewLimiter(70.0, 20.0)

        self.assertEqual(limiter.apply(90.0, 0.0), 90.0)
        self.assertEqual(limiter.apply(-90.0, 0.0), -20.0)

    def test_uses_90_degree_range_after_70_degree_guidance_error(self):
        limiter = GuidedHeadingSlewLimiter(70.0, 20.0)

        self.assertEqual(limiter.apply(70.0, 0.0), 70.0)
        self.assertEqual(limiter.apply(-90.0, 0.0), -20.0)

    def test_uses_minimum_70_degree_range_after_40_degree_guidance_error(self):
        limiter = GuidedHeadingSlewLimiter(70.0, 20.0)

        self.assertEqual(limiter.apply(40.0, 0.0), 40.0)
        self.assertEqual(limiter.apply(-90.0, 0.0), -30.0)

    def test_uses_shortest_change_across_signed_boundary(self):
        limiter = GuidedHeadingSlewLimiter(70.0, 20.0)

        self.assertEqual(limiter.apply(170.0, 170.0), 170.0)
        self.assertEqual(limiter.apply(-170.0, -170.0), -170.0)

    def test_reset_makes_next_heading_unrestricted(self):
        limiter = GuidedHeadingSlewLimiter(70.0, 20.0)
        limiter.apply(90.0, 0.0)

        limiter.reset()

        self.assertEqual(limiter.apply(-90.0, 0.0), -90.0)


if __name__ == "__main__":
    unittest.main()

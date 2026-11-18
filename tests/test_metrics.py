import unittest

from libero_saycan.metrics import summarize_episodes


class MetricsTests(unittest.TestCase):
    def test_empty_input(self):
        result = summarize_episodes([])
        self.assertEqual(result["episodes"], 0)
        self.assertEqual(result["successes"], 0)
        self.assertEqual(result["success_rate"], 0.0)

    def test_success_rate_and_suite_breakdown(self):
        episodes = [
            {"suite": "libero_spatial", "episode_success": True},
            {"suite": "libero_spatial", "episode_success": False},
            {"suite": "libero_object", "success": True},
        ]
        result = summarize_episodes(episodes)
        self.assertEqual(result["episodes"], 3)
        self.assertEqual(result["successes"], 2)
        self.assertAlmostEqual(result["success_rate"], 2.0 / 3.0)
        self.assertEqual(result["by_suite"]["libero_spatial"]["episodes"], 2)
        self.assertEqual(result["by_suite"]["libero_spatial"]["successes"], 1)
        self.assertEqual(result["by_suite"]["libero_object"]["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from libero_saycan.observations import camera_frame, observation_shapes


class ObservationTests(unittest.TestCase):
    def test_vertical_flip_returns_contiguous_rgb(self):
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        result = camera_frame({"agentview_image": image})
        np.testing.assert_array_equal(result[0], image[-1])
        self.assertEqual(result.shape, (4, 5, 3))
        self.assertTrue(result.flags["C_CONTIGUOUS"])

    def test_channel_first_input(self):
        image = np.zeros((3, 4, 5), dtype=np.uint8)
        result = camera_frame({"agentview_image": image}, flip_vertical=False)
        self.assertEqual(result.shape, (4, 5, 3))

    def test_rgba_is_converted_to_rgb(self):
        image = np.zeros((4, 5, 4), dtype=np.uint8)
        result = camera_frame({"agentview_image": image}, flip_vertical=False)
        self.assertEqual(result.shape, (4, 5, 3))

    def test_missing_camera_reports_available_keys(self):
        with self.assertRaises(KeyError):
            camera_frame({"robot0_eef_pos": np.zeros(3)})

    def test_observation_shape_summary(self):
        summary = observation_shapes({
            "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        })
        self.assertEqual(summary["agentview_image"]["shape"], [8, 8, 3])
        self.assertEqual(summary["robot0_eef_pos"]["dtype"], "float32")


if __name__ == "__main__":
    unittest.main()

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_ntuviral import align_positions, evaluate_spline, load_prism_offset, load_spline


class EvaluationTest(unittest.TestCase):
    def test_constant_velocity_and_angular_velocity(self):
        # A cubic B-spline does not interpolate its control points. With linear
        # control values its phase is one knot ahead, for both position and SO3.
        start, dt, count = 1609060334.75, .1, 12
        velocity = np.array([.4, -.7, 1.2])
        p0 = np.array([2., 4., 8.])
        omega = np.array([.2, -.1, .3])
        r0 = Rotation.from_rotvec([.5, .3, -.4])
        rotations = r0 * Rotation.from_rotvec(np.arange(count)[:, None] * dt * omega)
        quaternions = rotations.as_quat()
        quaternions[::2] *= -1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spline.csv"
            with path.open("w") as stream:
                stream.write(f"Dt: {dt}, Order: 4, Knots: {count}, MinTime: {start}, MaxTime: {start+(count-3)*dt}, OtrItr: 0\n")
                np.savetxt(stream, np.column_stack((np.arange(count), start+np.arange(count)*dt,
                           p0+np.arange(count)[:, None]*dt*velocity, quaternions)), delimiter=",", fmt="%.17g")
            spline = load_spline(path)
            times = start + np.array([0., .001, .09999, .1, .10001, .413, .899])
            position, orientation = evaluate_spline(spline, times)
        phase = times-start+dt
        np.testing.assert_allclose(position, p0+phase[:, None]*velocity, atol=1e-12, rtol=0)
        expected = r0 * Rotation.from_rotvec(phase[:, None]*omega)
        np.testing.assert_allclose((orientation.inv()*expected).as_rotvec(), 0, atol=1e-12, rtol=0)
        with self.assertRaises(ValueError):
            evaluate_spline(spline, [start-.01])
        with self.assertRaises(ValueError):
            evaluate_spline(spline, [start+1.])

    def test_rigid_alignment_keeps_metric_scale(self):
        rng = np.random.default_rng(35)
        source = rng.normal(size=(30, 3))
        expected_r = Rotation.from_rotvec([.2, -.4, .7])
        expected_t = np.array([5., -3., 7.])
        truth = expected_r.apply(source) + expected_t
        aligned, rotation, translation = align_positions(source, truth)
        np.testing.assert_allclose(aligned, truth, atol=1e-12, rtol=0)
        np.testing.assert_allclose(rotation, expected_r.as_matrix(), atol=1e-12, rtol=0)
        np.testing.assert_allclose(translation, expected_t, atol=1e-12, rtol=0)
        wrong_scale, _, _ = align_positions(2*source, truth)
        self.assertGreater(np.linalg.norm(wrong_scale-truth), 1.)
        with self.assertRaises(ValueError):
            align_positions(np.zeros((10, 3)), np.zeros((10, 3)))

    def test_opencv_prism_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prism.yaml"
            path.write_text("%YAML:1.0\nT_Body_Prism: !!opencv-matrix\n   rows: 4\n   cols: 4\n   dt: d\n   data: [1,0,0,-.3, 0,1,0,.02, 0,0,1,-.2, 0,0,0,1]\n")
            offset = load_prism_offset(path)
            np.testing.assert_allclose(offset, [-.3, .02, -.2], atol=1e-12)
            # The lever arm is rotated by the estimated attitude before adding.
            r = Rotation.from_euler("z", 90, degrees=True)
            np.testing.assert_allclose(r.apply(offset), [-.02, -.3, -.2], atol=1e-12)


if __name__ == "__main__":
    unittest.main()

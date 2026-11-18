"""Compact RGB trajectory recorder for LIBERO runs."""

from pathlib import Path


class TrajectoryRecorder:
    def __init__(self, output_dir, fps=10, stride=8, camera_key="agentview_image"):
        self.output_dir = Path(output_dir)
        self.fps = max(1, int(fps))
        self.stride = max(1, int(stride))
        self.camera_key = str(camera_key)
        self.frames = []
        self.observation_count = 0
        self._last_frame = None

    def capture(self, observation):
        if not observation or self.camera_key not in observation:
            return
        self.observation_count += 1
        if self.observation_count != 1 and self.observation_count % self.stride != 0:
            return
        from libero_saycan.observations import camera_frame
        frame = camera_frame(observation, self.camera_key)
        self._last_frame = frame
        self.frames.append(frame)

    def finalize(self):
        if not self.frames and self._last_frame is not None:
            self.frames.append(self._last_frame)
        if not self.frames:
            return {"recorded": False, "reason": "no_camera_frames"}

        import imageio.v2 as imageio
        self.output_dir.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(str(self.output_dir / "initial_agentview.png"), self.frames[0])
        imageio.imwrite(str(self.output_dir / "final_agentview.png"), self.frames[-1])
        imageio.mimsave(
            str(self.output_dir / "trajectory.gif"),
            self.frames,
            duration=1.0 / float(self.fps),
            loop=0,
        )
        return {
            "recorded": True,
            "frame_count": len(self.frames),
            "observation_count": self.observation_count,
            "fps": self.fps,
            "stride": self.stride,
            "files": [
                "trajectory.gif",
                "initial_agentview.png",
                "final_agentview.png",
            ],
        }

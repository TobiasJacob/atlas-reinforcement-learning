from typing import List
from atlasrl.motions.MotionState import MotionState
import json
from dataclasses import dataclass

import numpy as np

# duration of frame in seconds (1D),
# root position (3D),
# root rotation (4D),
# chest rotation (4D),
# neck rotation (4D),
# right hip rotation (4D),
# right knee rotation (1D),
# right ankle rotation (4D),
# right shoulder rotation (4D),
# right elbow rotation (1D),
# left hip rotation (4D),
# left knee rotation (1D),
# left ankle rotation (4D),
# left shoulder rotation (4D),
# left elbow rotation (1D)

@dataclass
class MotionReader:
    path: str
    loop: str
    frames: List[MotionState]

    @classmethod
    def readClip(cls, path = "data/motions/humanoid3d_walk.txt") -> "MotionReader":
        with open(path) as f:
            data = json.load(f)
        loop = data["Loop"]
        frames: List[MotionState] = []
        time = 0
        for row in data["Frames"]:
            # First index of row containes delta-time, so we have to add it up
            frames.append(MotionState.fromRow(time, row))
            time += frames[-1].deltaTime
        return cls(path, loop, frames)

    def getState(self, time: float) -> MotionState:
        if self.loop == "wrap":
            time = time % self.frames[-1].absoluteTime
        assert time >= self.frames[0].absoluteTime and time <= self.frames[-1].absoluteTime
        for (i, row) in enumerate(self.frames):
            if time < row.absoluteTime:
                break
        row1 = self.frames[i - 1]
        row2 = self.frames[i]
        alpha = (time - row1.absoluteTime) / (row2.absoluteTime - row1.absoluteTime)
        return MotionState.fromInterpolation(row1, row2, alpha)

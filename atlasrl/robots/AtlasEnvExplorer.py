import numpy as np
from atlasrl.robots.AtlasBulletEnv import AtlasBulletEnv
import pybullet as p
import PySimpleGUIQt as sg

env = AtlasBulletEnv(render=True)
layout = [
    [sg.Text("Env explorer")],
    [sg.Button("Set")],
    *[[sg.Slider(range=(-100, 100), orientation="h")] for _ in range(env.action_space.shape[0])]
]

window = sg.Window("Demo", layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == "Set":
        action = np.array([values[k] / 100.0 for k in values])
        resp = env.step(action)
        print(resp)

window.close()

# Atlas reinforcement learning

This project aims to run reinforcement learning models on a real Atlas.

- Python version: 3.7.11

## Timeline

- [x] Run the simulated Atlas model in `PyBullet`.
- [x] Wrap it into an OpenAI-Gym environment.
- [x] Run low pass filter or simulation with 100hz and action sampling with 30hz
- [x] Retarget motion files to AtlasEnv and see how it looks
- [x] Create a second backend for the OpenAI-Gym that connects to the real Atlas.
- [x] Train a very basic machine learning model with `stable_baselines` in the virtual Gym and run it in the real Gym. Fix most of the joints to zero, except, for example, the right arm.
- [ ] Add low pass filtering to joint motors. Low pass filter on the actual angles instead of the normalized action space
- [ ] See how the model behaves and decide on further steps.

## Setup

[Install Python 3.7](https://linuxize.com/post/how-to-install-python-3-7-on-ubuntu-18-04/). Create a virtual env (or something else).

```console
which python3.7
virtualenv -p /usr/bin/python3.7 env
source env/bin/activate
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Run the code

Run the following command to play around with the Atlas Robot in PyBullet.

```console
python3 atlasrl/simulator/standalone.py
```

![alt text](docs/AtlasInPyBullet.png)

For training the model, run

```console
python3 -m atlasrl
```

For testing the remote, run

```console
python3 -m atlasrl.robots.AtlasRemoteEnv_test
```

## State, Actions and Reward

- Observed state is not clear yet.
- Actions are the pd-controller targets for the 30 joints of the simulated Atlas.
- Rewards depend on the specific task.

## Data format

- 1st is dT in secs

## Reference motion player
python -m atlasrl.simulator.ShowReferenceMotion

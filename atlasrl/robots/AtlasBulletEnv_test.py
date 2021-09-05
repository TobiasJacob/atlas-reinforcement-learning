from atlasrl.robots.AtlasBulletEnv import AtlasBulletEnv

def test_AtlasBulletEnv():
    env = AtlasBulletEnv(render=True)

    for episode in range(10): 
        obs = env.reset()
        for step in range(5000000):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            env.render("human")
    env.close()
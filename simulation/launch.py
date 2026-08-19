 import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)


with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        
          # joint targets
        data.ctrl[0] = 0.0   # left hip pitch
        data.ctrl[1] = 0.0   # left hip roll
        data.ctrl[2] = 0.0   # left knee
        data.ctrl[3] = 0.0   # left ankle

        data.ctrl[4] = 0.0   # right hip pitch
        data.ctrl[5] = 0.0   # right hip roll
        data.ctrl[6] = 0.0   # right knee
        data.ctrl[7] = 0.0   # right ankle

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.002)

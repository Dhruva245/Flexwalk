import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("robot_with_collision.xml")
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
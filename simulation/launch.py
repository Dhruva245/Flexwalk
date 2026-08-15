import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("/home/dhruva/Downloads/robot/urdf/scene.xml")
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)

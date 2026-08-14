"""
Spawn the biped in MuJoCo with a free-floating base and joint control.

Run from inside your `robot/` folder:
    python3 launch.py

Layout expected:
    robot/
      urdf/robot.urdf
      meshes/*.stl
      launch.py   <- this file
"""

import mujoco
import mujoco.viewer
import time
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(BASE_DIR, "urdf", "robot.urdf")
MESH_DIR = os.path.join(BASE_DIR, "meshes")

print("Reading URDF...")
with open(URDF_PATH, "r") as f:
    urdf_text = f.read()

# --- Fix 1: your STL references use ROS-style "package://robot/meshes/Name.stl"
# URIs. MuJoCo doesn't understand the package:// scheme, so mesh loading was
# silently failing (that's why your last compiled robot.xml had zero geoms).
# Strip everything down to the bare filename, preserving the original case.
urdf_text = re.sub(r'filename="package://[^"]*/([^"/]+)"', r'filename="\1"', urdf_text)

# Cross-check every referenced filename against what's actually on disk and
# fix case mismatches individually, instead of blind-lowercasing everything
# (that was the risky part of your original script).
if os.path.isdir(MESH_DIR):
    disk_files = {name.lower(): name for name in os.listdir(MESH_DIR)}
    def _fix_case(m):
        wanted = m.group(1)
        actual = disk_files.get(wanted.lower())
        if actual is None:
            print(f"WARNING: no file on disk matches mesh '{wanted}' in {MESH_DIR}")
            return m.group(0)
        return f'filename="{actual}"'
    urdf_text = re.sub(r'filename="([^"/]+)"', _fix_case, urdf_text)
else:
    print(f"WARNING: mesh dir not found at {MESH_DIR}")

# --- Fix 2: the URDF's base link ("root") has no <inertial>, which can make
# it disappear during compilation instead of surviving as its own body.
urdf_text = urdf_text.replace(
    '<link name="root" />',
    '<link name="root">'
    '<inertial><mass value="0.05"/>'
    '<origin xyz="0 0 0" rpy="0 0 0"/>'
    '<inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>'
    '</inertial></link>',
)

spec = mujoco.MjSpec.from_string(urdf_text)

# --- Fix 3: point meshdir at the real folder (it was resolving relative to
# urdf/, i.e. looking for robot/urdf/meshes instead of robot/meshes).
spec.compiler.meshdir = MESH_DIR
spec.compiler.balanceinertia = True   # CAD-exported inertia tensors are often
                                       # slightly invalid; this repairs them
spec.compiler.boundmass = 0.001       # a couple of links (e.g. the foot cover)
                                       # have ~1e-9 kg mass in the URDF

# Floor + light
spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[2, 2, 0.1],
    rgba=[0.15, 0.2, 0.3, 1],  # dark blue-grey -- your robot meshes are light
                               # grey (0.6-0.9), so a light floor made it blend in
)
spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])

# --- Free-floating base: lift the robot above the floor and give the root
# body a free (6-DOF) joint so it can fall, balance, and walk under gravity.
root_body = spec.worldbody.bodies[0]  # 'root' — the only top-level URDF link
root_body.pos = [0, 0, 0.8]
root_body.add_freejoint()

# --- Actuators: without these there is nothing to command, and no sliders
# show up in the viewer's Control tab. Add a position actuator to every real
# (non-fixed) joint imported from the URDF -- fixed joints don't survive as
# MJCF <joint> elements so this only touches the 8 hinges you actually have.
KP = 15  # raise for stiffer/snappier tracking, lower if it oscillates
for joint in spec.joints:
    if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
        spec.add_actuator(
            name=joint.name + "_act",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=joint.name,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=[KP, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            biasprm=[0, -KP, 0, 0, 0, 0, 0, 0, 0, 0],
            ctrlrange=[-3.14, 3.14],
            ctrllimited=True,
        )

# Make sure the visual meshes (the only geoms in this URDF) also collide,
# so the robot actually lands on the floor instead of falling through it.
def _all_bodies(body):
    yield body
    for child in body.bodies:
        yield from _all_bodies(child)

n_fixed = 0
for body in _all_bodies(spec.worldbody):
    for geom in body.geoms:
        geom.contype = 1
        geom.conaffinity = 1
        n_fixed += 1
print(f"[pre-compile] geoms found while walking spec tree: {n_fixed}")

model = spec.compile()
data = mujoco.MjData(model)

# Ground truth check on the actual compiled model (bypasses any pre-compile
# spec-tree timing issues above).
print(f"[compiled model] total geoms: {model.ngeom}")
for i in range(model.ngeom):
    print(f"  geom {i}: name='{model.geom(i).name}' type={model.geom(i).type}")

# Belt-and-suspenders: force collision on directly in the compiled model's
# numpy arrays, which is unambiguous regardless of what happened above.
model.geom_contype[:] = 1
model.geom_conaffinity[:] = 1

# Save a clean, working copy for inspection/reuse
out_path = os.path.join(BASE_DIR, "urdf", "robot_fixed.xml")
with open(out_path, "w") as f:
    f.write(spec.to_xml())
print(f"\nSuccess! Wrote a working MJCF copy to {out_path}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("\nViewer running.")
    print("-> Press Tab to toggle the left UI panel, then open the 'Control'")
    print("   tab to see a slider for each of the 8 leg joints and drag them.")
    print("-> Click once in the 3D view, press 'A' to auto-align the camera.")

    last_print = time.time()
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        if time.time() - last_print > 1.0:
            x, y, z = data.qpos[0:3]
            print(f"root position: x={x:.3f} y={y:.3f} z={z:.3f}")
            last_print = time.time()
        time.sleep(0.002)

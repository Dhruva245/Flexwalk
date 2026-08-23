import mujoco
import mujoco.viewer
import numpy as np
import math
import time

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

STEP_LENGTH = -0.05
STEP_HEIGHT = 0.035
THIGH_LENGTH = 0.253
CALF_LENGTH = 0.255
SM = 0.5
T_SSP = 0.5
T_DSP = 0.1
T_s = T_SSP + T_DSP
dt = 0.01
x_start = 0.0
z_rest = 0.508
y_start = 0.0
ANKLE_REST_Z = 0.0
v_start = 0.0
v_end = 0.0
y_offset = 0.02
SWAY_AMPLITUDE = 0.02
NUM_STEPS = 8
PLAYBACK_DT = dt
ANKLE_PITCH_LIM = 0.785398  # 45 degrees

JOINT_NAMES = ["left_hip_roll", "left_hip_pitch", "left_knee_pitch", "left_ankle_pitch", "right_hip_roll", "right_hip_pitch", "right_knee_pitch", "right_ankle_pitch"]

QADR = {
    name: model.joint(name).qposadr[0]
    for name in JOINT_NAMES
}

WORLD_TO_HIP_QADR = model.joint(
    "world_to_hip_joint"
).qposadr[0]


def make_qpos(values):
    """Build a full-length qpos array (size model.nq) from a
    {joint_name: angle} dict, using the runtime-resolved addresses."""
    qpos = model.qpos0.copy()
    qpos[WORLD_TO_HIP_QADR + 1] =  -values.get(
        "world_to_hip_joint", 0.0
    )

    qpos[WORLD_TO_HIP_QADR + 2] = 0.508
    for name, val in values.items():
        if name in QADR:
            qpos[QADR[name]] = val
    return qpos

def compute_foot_swing_trajectory(dt, T_s, STEP_LENGTH, STEP_HEIGHT, x_start, z_rest):
    """
    Swing foot ankle trajectory during SSP.

    x(t): 3rd-order polynomial — smooth start/stop.
    z(t): 5th-order polynomial — lifts, peaks at SM*Ts, returns.

    Returns lists of (ankle_x, ankle_z) positions.
    """
    Ts = T_s
    Sl = STEP_LENGTH
    Hm = STEP_HEIGHT
    Sm_t = SM * Ts

    # x-trajectory coefficients
    a0 = x_start
    a1 = 0.0
    a2 = 3.0 * Sl / (Ts ** 2)
    a3 = -2.0 * Sl / (Ts ** 3)

    # z-trajectory coefficients (5th order, solve 4×4 system)
    K = np.array([
        [Ts**2,   Ts**3,     Ts**4,     Ts**5    ],
        [2*Ts,    3*Ts**2,   4*Ts**3,   5*Ts**4  ],
        [Sm_t**2, Sm_t**3,   Sm_t**4,   Sm_t**5  ],
        [2*Sm_t,  3*Sm_t**2, 4*Sm_t**3, 5*Sm_t**4],
    ])
    rhs = np.array([0.0, 0.0, Hm, 0.0])
    b2, b3, b4, b5 = np.linalg.solve(K, rhs)

    x_traj,z_traj = [], []
    t = 0.0
    while t <= Ts + 1e-9:
        x_traj.append(a0 + a1*t + a2*t**2 + a3*t**3)
        z_traj.append(z_rest + b2*t**2 + b3*t**3 + b4*t**4 + b5*t**5)
        t += dt
    return x_traj, z_traj



# ===========================================================================
# Section 2.3: Hip trajectory — 3rd-order polynomial
# ===========================================================================

def compute_smooth_trajectory(dt,T_s,x_start,x_end,v_start,v_end):
    """
    Smooth 3rd-order polynomial trajectory (hip motion per paper eq. 16/17).
    Drives pelvis_x continuously — this is what actually walks the robot
    forward now that the pelvis has a real translational DOF.
    """
    T = T_s
    c0 = x_start
    c1 = v_start
    A = np.array([[T**2, T**3], [2*T, 3*T**2]])
    b = np.array([x_end - c0 - c1*T, v_end - c1])
    c2, c3 = np.linalg.solve(A, b)

    traj = []
    t = 0.0
    while t <= T + 1e-9:
        traj.append(c0 + c1*t + c2*t**2 + c3*t**3)
        t += dt
    return traj


# ============================================================================
# Lateral sway
# ============================================================================

def compute_lateral_sway(dt,T_s,y_start,y_offset):
    """Half-cosine lateral weight shift."""
    traj = []
    t = 0.0
    while t <= T_s + 1e-9:
        s = 0.5 * (1.0 - math.cos(math.pi * t / T_s))
        traj.append(y_start + s * (y_offset - y_start))
        t += dt
    return traj



# ============================================================================
# 2-Link Inverse Kinematics
# ============================================================================

def ik_2link(x_start,z_rest,ankle_x,ankle_z,THIGH_LENGTH,CALF_LENGTH):
    """
    Solve 2-link planar IK for hip→knee→ankle.

    MuJoCo FK:
        knee_x  = hip_x  - l1 · sin(θh)
        knee_z  = hip_z  - l1 · cos(θh)
        ankle_x = knee_x - l2 · sin(θh + θk)
        ankle_z = knee_z - l2 · cos(θh + θk)

    Returns (θ_hip, θ_knee, θ_ankle) where ankle keeps foot flat on ground.
    """
    
    dx = x_start - ankle_x
    dz = z_rest - ankle_z
    D = math.sqrt(dx**2 + dz**2)

    # Clamp to feasible range
    D = max(abs(THIGH_LENGTH - CALF_LENGTH) + 1e-4, min(D, THIGH_LENGTH + CALF_LENGTH - 1e-4))

    # Knee angle (law of cosines)
    cos_k = (THIGH_LENGTH**2 + CALF_LENGTH**2 - D**2) / (2.0 * THIGH_LENGTH * CALF_LENGTH)
    cos_k = np.clip(cos_k, -1.0, 1.0)
    theta_knee = math.acos(cos_k) - math.pi   # ≤ 0

    # Hip angle
    gamma = math.atan2(dx, dz)
    cos_b = (THIGH_LENGTH**2 + D**2 - CALF_LENGTH**2) / (2.0 * THIGH_LENGTH * D)
    cos_b = np.clip(cos_b, -1.0, 1.0)
    beta = math.acos(cos_b)
    theta_hip = gamma + beta   # "knee-forward" solution
    # Ankle pitch to keep foot flat
    theta_ankle = -(theta_hip + theta_knee)

    # Clamp ankle pitch within limits; redistribute excess to hip
    if abs(theta_ankle) > ANKLE_PITCH_LIM:
        excess = theta_ankle - np.clip(theta_ankle, -ANKLE_PITCH_LIM, ANKLE_PITCH_LIM)
        theta_ankle = np.clip(theta_ankle, -ANKLE_PITCH_LIM, ANKLE_PITCH_LIM)
        theta_hip += excess  # absorb excess into hip
        # Q. why excess into hip here ? The knee primarily determines leg extension (how bent or straight the leg is). Changing the knee would change the distance between the hip and ankle, which would move the foot away from its desired position.
        #The hip, however, rotates the entire leg while keeping the same knee bend. A small hip adjustment mainly changes the leg's overall orientation, making it a better place to compensate for ankle saturation.

    return theta_hip, theta_knee, theta_ankle


def compute_rolls(y_offset,z_rest):
    """Hip roll and ankle roll for lateral weight shifting."""
    if abs(z_rest) < 0.01:
        return 0.0
    hip_roll = math.atan2(y_offset, z_rest)
    return hip_roll


def generate_gait_trajectory(num_steps,dt):
    """
    Generate a continuously-forward walking gait pattern.

    hip_x, left_foot_x, right_foot_x are persistent state carried across
    the whole gait (never reset). Each step:
      - the stance foot stays exactly where it is (SSP),
      - the swing foot starts from its own actual last touchdown position
        and advances step_length beyond the (fixed) stance foot,
      - the pelvis itself glides forward via compute_smooth_trajectory(),
        now physically driving the pelvis_x slide joint,
      - DSP uses the real post-touchdown foot positions, not ±SL/2.

    Returns list of np.ndarray(model.nq,) qpos frames.
    """
    frames = []

    # ---- Persistent state -------------------------------------------------

    hip_x = 0.0
    hip_z = z_rest
    left_foot_x = -STEP_LENGTH / 2.0
    right_foot_x = STEP_LENGTH / 2.0
    # -------------------------------------------------------------------

    swing_is_left = True

    for step_idx in range(num_steps):

        # Current actual foot positions (no artificial re-centring/reset)

        if swing_is_left:
            swing_start_x = left_foot_x
            stance_x = right_foot_x
        else:
            swing_start_x = right_foot_x
            stance_x = left_foot_x
        # Swing foot lands a step-length ahead of the current stance foot —
        # this is what makes the gait progress forward indefinitely instead
        # of oscillating around a fixed point.

        swing_end_x = stance_x + STEP_LENGTH

        # Pelvis advances smoothly toward the midpoint of the new support
        # polygon (average of stance foot and swing landing spot).
        hip_x_target = 0.5 * (stance_x + swing_end_x)

        # ---------------------------------------------------------------
        # Phase 1: SSP — swing foot lifts and advances, stance foot fixed
        # ---------------------------------------------------------------
        swing_x, swing_z = compute_foot_swing_trajectory(
            dt,T_SSP,STEP_LENGTH,STEP_HEIGHT,swing_start_x,ANKLE_REST_Z
        )

        hip_x_traj_ssp = compute_smooth_trajectory(
            dt,T_SSP,hip_x,hip_x_target,0.0,0.0
        )

        # Lateral sway: shift toward stance foot
        sway_target = -SWAY_AMPLITUDE if swing_is_left else SWAY_AMPLITUDE
        sway_start = 0.0
        y_sway_ssp = compute_lateral_sway(dt,T_SSP,sway_start,sway_target)

        n_ssp = min(len(swing_x),len(y_sway_ssp),len(hip_x_traj_ssp))

        for i in range(n_ssp):
            y_sway = y_sway_ssp[i]
            cur_hip_x = hip_x_traj_ssp[i]
            leg_h = hip_z- ANKLE_REST_Z

            # Swing leg IK (relative to the instantaneous, moving hip)
            sw_hip, sw_knee, sw_ankle = ik_2link(
                cur_hip_x,hip_z,swing_x[i],swing_z[i],THIGH_LENGTH,CALF_LENGTH
            )

            sw_hr = compute_rolls(y_sway,leg_h)
            # Stance leg IK — foot stays exactly planted at stance_x

            st_hip, st_knee, st_ankle = ik_2link(
                cur_hip_x,hip_z,stance_x,ANKLE_REST_Z,THIGH_LENGTH,CALF_LENGTH
            )

            st_hr = compute_rolls(y_sway,leg_h)

            values = {"world_to_hip_joint": cur_hip_x}
            if swing_is_left:
                values["left_hip_roll"],values["left_hip_pitch"],values["left_knee_pitch"],values["left_ankle_pitch"] = sw_hr,sw_hip,sw_knee,sw_ankle

                values["right_hip_roll"],values["right_hip_pitch"],values["right_knee_pitch"],values["right_ankle_pitch"] = -st_hr,st_hip,st_knee,st_ankle

            else:

                # Left = stance

                values["left_hip_roll"],values["left_hip_pitch"],values["left_knee_pitch"],values["left_ankle_pitch"] = st_hr,st_hip,st_knee,st_ankle

                # Right = swing

                values["right_hip_roll"],values["right_hip_pitch"],values["right_knee_pitch"],values["right_ankle_pitch"] = -sw_hr,sw_hip,sw_knee,sw_ankle

            frames.append(make_qpos(values))

        # Advance persistent hip_x to where SSP left it
        hip_x = hip_x_traj_ssp[-1] if n_ssp else hip_x

        # Touchdown: swing foot has now landed — update persistent state
        if swing_is_left:
            left_foot_x = swing_end_x
        else:
            right_foot_x = swing_end_x

        # ---------------------------------------------------------------
        # Phase 2: DSP — both feet on ground, weight shifts to centre
        # ---------------------------------------------------------------
        y_sway_dsp = compute_lateral_sway(dt, T_DSP, sway_target, 0.0)

        # Pelvis continues smoothly forward during DSP toward the midpoint
        # of the (now updated) actual foot positions.
        hip_x_dsp_target = 0.5 * (left_foot_x + right_foot_x)

        hip_x_traj_dsp = compute_smooth_trajectory(
            dt,T_DSP,hip_x,hip_x_dsp_target,0.0,0.0
        )

        n_dsp = min(len(y_sway_dsp), len(hip_x_traj_dsp))

        for i in range(n_dsp):
            y_sway = y_sway_dsp[i]
            cur_hip_x = hip_x_traj_dsp[i]
            leg_h = hip_z- ANKLE_REST_Z
            # Use the actual current foot positions (post touchdown)

            l_ankle_x = left_foot_x
            r_ankle_x = right_foot_x

            l_hip, l_knee, l_ankle = ik_2link(
                cur_hip_x,hip_z,l_ankle_x,ANKLE_REST_Z,THIGH_LENGTH,CALF_LENGTH
            )
            r_hip, r_knee, r_ankle = ik_2link(
                cur_hip_x,hip_z,r_ankle_x,ANKLE_REST_Z,THIGH_LENGTH,CALF_LENGTH
            )

            l_hr = compute_rolls(y_sway,leg_h)
            r_hr = compute_rolls(y_sway,leg_h)

            values = {
                "world_to_hip_joint": cur_hip_x,
                "left_hip_roll": l_hr,"left_hip_pitch": l_hip,
                "left_knee_pitch": l_knee,"left_ankle_pitch": l_ankle,
                "right_hip_roll": -r_hr,"right_hip_pitch": -r_hip,
                "right_knee_pitch": r_knee,"right_ankle_pitch": r_ankle,
            }

            frames.append(make_qpos(values))

        # Advance persistent hip_x to where DSP left it
        hip_x = hip_x_traj_dsp[-1] if n_dsp else hip_x
        # Swap swing leg for next step
        swing_is_left = not swing_is_left

    return frames



frames = generate_gait_trajectory(NUM_STEPS,dt)

print(f"Generated {len(frames)} qpos frames.")


#Initialize simulation

data.qpos[:] = model.qpos0
data.qvel[:] = 0

mujoco.mj_forward(
    model,
    data
)


with mujoco.viewer.launch_passive(model,data) as viewer:
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20
    viewer.cam.distance = 1.2
    viewer.cam.lookat[:] = [0.0,0.0,0.35]
    
    while viewer.is_running():
        for qpos in frames:
            if not viewer.is_running():
                break
            data.qpos[:] = qpos

            mujoco.mj_forward(model,data)
            viewer.sync()
            time.sleep(
                PLAYBACK_DT
            )

        if viewer.is_running():
            time.sleep(0.3)
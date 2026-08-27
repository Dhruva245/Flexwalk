#!/usr/bin/env python3

import time
import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# CONFIG
# ============================================================

XML_PATH = "flexwalk_v2.xml"

ENABLE_VIEWER = True
REALTIME = True

T_SSP = 0.50
T_DSP = 0.15
STEP_LENGTH = 0.30
FOOT_CLEARANCE = 0.10
NUM_STEPS = 16
DT = 0.01

N_PREVIEW = 150
GRAVITY = 9.81

IK_MAX_NFEV = 40
IK_POSITION_WEIGHT = 1.0
IK_POSTURE_WEIGHT = 0.02
IK_ANKLE_WEIGHT = 0.01


# ============================================================
# ROBOT NAMES
# ============================================================

LEFT_JOINTS = (
    "hip_pitch_l",
    "hip_roll_l",
    "knee_pitch_l",
    "ankle_pitch_l",
)

RIGHT_JOINTS = (
    "hip_pitch_r",
    "hip_roll_r",
    "knee_pitch_r",
    "ankle_pitch_r",
)

ALL_JOINTS = LEFT_JOINTS + RIGHT_JOINTS

LEFT_FOOT_BODY = "left_foot"
RIGHT_FOOT_BODY = "right_foot"
ROOT_BODY = "hips_connector"


# ============================================================
# SIMPLE ROBOT GEOMETRY CONTAINER
# ============================================================

class RobotGeometry:
    def __init__(self, root_z, left_foot, right_foot):
        self.root_z = root_z
        self.left_foot0 = left_foot
        self.right_foot0 = right_foot


# ============================================================
# DARE
# ============================================================

def solve_dare(A, B, Q, R, max_iter=5000, tol=1e-12):
    P = Q.copy()

    for _ in range(max_iter):
        BTP = B.T @ P
        S = R + BTP @ B
        K = np.linalg.solve(S, BTP @ A)

        P_next = A.T @ P @ A - A.T @ P @ B @ K + Q

        if np.max(np.abs(P_next - P)) < tol:
            P = P_next
            break

        P = P_next

    return P


# ============================================================
# NUMERICAL JACOBIAN
# ============================================================

def numerical_jacobian(fn, x, f0, eps=1e-6):
    J = np.zeros((len(f0), len(x)))

    for i in range(len(x)):
        dx = np.zeros(len(x))
        step = eps * max(1.0, abs(x[i]))
        dx[i] = step

        J[:, i] = (fn(x + dx) - f0) / step

    return J


# ============================================================
# BOUNDED LEVENBERG-MARQUARDT
# ============================================================

def bounded_least_squares(
    residual_fn,
    x0,
    lower,
    upper,
    max_iter=IK_MAX_NFEV,
    ftol=1e-9,
    xtol=1e-9,
):
    x = np.clip(np.asarray(x0, dtype=float).copy(), lower, upper)

    f = residual_fn(x)
    cost = 0.5 * float(np.dot(f, f))

    lam = 1e-3

    for _ in range(max_iter):
        J = numerical_jacobian(residual_fn, x, f)

        JTJ = J.T @ J
        JTf = J.T @ f

        improved = False
        delta = np.zeros_like(x)

        for _ in range(20):
            A = JTJ + lam * np.diag(np.diag(JTJ) + 1e-12)

            try:
                delta = np.linalg.solve(A, -JTf)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(A, -JTf, rcond=None)[0]

            x_new = np.clip(x + delta, lower, upper)

            f_new = residual_fn(x_new)
            new_cost = 0.5 * float(np.dot(f_new, f_new))

            if new_cost < cost:
                improvement = cost - new_cost

                x = x_new
                f = f_new
                cost = new_cost

                lam = max(lam * 0.7, 1e-12)
                improved = True
                break

            lam *= 2.0

            if lam > 1e12:
                break

        if not improved:
            break

        if np.linalg.norm(delta) < xtol:
            break

        if improvement < ftol:
            break

    return x


# ============================================================
# MUJOCO HELPERS
# ============================================================

def joint_limits(model, joint_name):
    jid = model.joint(joint_name).id

    # Prefer actuator position limits.
    for aid in range(model.nu):
        actuator_joint = model.actuator_trnid[aid, 0]

        if actuator_joint < 0:
            continue

        if model.joint(actuator_joint).name == joint_name:
            if model.actuator_ctrllimited[aid]:
                return (
                    float(model.actuator_ctrlrange[aid, 0]),
                    float(model.actuator_ctrlrange[aid, 1]),
                )
            break

    return (
        float(model.jnt_range[jid, 0]),
        float(model.jnt_range[jid, 1]),
    )


def clamp_leg(model, joint_names, q):
    q = np.asarray(q, dtype=float).copy()

    for i, name in enumerate(joint_names):
        low, high = joint_limits(model, name)
        q[i] = np.clip(q[i], low, high)

    return q


def body_position(model, data, body_name):
    return data.xpos[model.body(body_name).id].copy()


def freejoint_address(model):
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            return model.jnt_qposadr[jid]

    raise RuntimeError("No freejoint found")


def set_leg_q(model, qpos, joint_names, values):
    for name, value in zip(joint_names, values):
        jid = model.joint(name).id
        qpos[model.jnt_qposadr[jid]] = value
    

def get_leg_q(model, qpos, joint_names):
    return np.array(
        [
            qpos[model.jnt_qposadr[model.joint(name).id]]
            for name in joint_names
        ],
        dtype=float,
    )


# ============================================================
# ROBOT DISCOVERY
# ============================================================

def discover_robot(model, data):

    missing = [
        name
        for name in ALL_JOINTS
        if model.joint(name).id < 0
    ]

    if missing:
        raise RuntimeError(f"XML is missing joints: {missing}")

    for body_name in (ROOT_BODY, LEFT_FOOT_BODY, RIGHT_FOOT_BODY):
        try:
            model.body(body_name)
        except Exception as exc:
            raise RuntimeError(
                f"XML is missing required body '{body_name}'"
            ) from exc

    free_count = sum(
        model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        for j in range(model.njnt)
    )

    if free_count != 1:
        raise RuntimeError(
            f"Expected exactly one freejoint, found {free_count}"
        )

    mujoco.mj_forward(model, data)

    return RobotGeometry(
        root_z=float(
            data.xpos[model.body(ROOT_BODY).id, 2]
        ),
        left_foot=body_position(
            model, data, LEFT_FOOT_BODY
        ),
        right_foot=body_position(
            model, data, RIGHT_FOOT_BODY
        ),
    )


# ============================================================
# FOOTSTEP PLANNING
# ============================================================

def plan_footsteps(left0, right0, n_steps, step_length):
    left = left0.copy()
    right = right0.copy()

    steps = []

    for i in range(n_steps):

        if i % 2 == 0:
            left[0] = right[0] + step_length
            steps.append(("left", left.copy()))
        else:
            right[0] = left[0] + step_length
            steps.append(("right", right.copy()))

    return steps


# ============================================================
# ZMP REFERENCE
# ============================================================

def zmp_reference(left0, right0, footsteps):

    px = []
    py = []

    left = left0.copy()
    right = right0.copy()

    # Initial double support
    for _ in range(int(T_DSP / DT)):
        center = 0.5 * (left + right)

        px.append(center[0])
        py.append(center[1])

    # Walking
    for foot, landing in footsteps:

        stance = right if foot == "left" else left

        # Single support
        for _ in range(int(T_SSP / DT)):
            px.append(stance[0])
            py.append(stance[1])

        # Double support / weight transfer
        n = max(int(T_DSP / DT), 1)

        for k in range(n):
            a = k / max(n - 1, 1)

            px.append(
                (1 - a) * stance[0] +
                a * landing[0]
            )

            py.append(
                (1 - a) * stance[1] +
                a * landing[1]
            )

        if foot == "left":
            left = landing.copy()
        else:
            right = landing.copy()

    # Preview padding
    for _ in range(N_PREVIEW):
        px.append(px[-1])
        py.append(py[-1])

    return np.asarray(px), np.asarray(py)


# ============================================================
# ZMP PREVIEW CONTROLLER
# ============================================================

def preview_gains(dt, gravity, com_height, preview_steps):

    Qe = 1.0
    R = 1e-6

    A = np.array([
        [1.0, dt, dt**2 / 2.0],
        [0.0, 1.0, dt],
        [0.0, 0.0, 1.0],
    ])

    B = np.array([
        [dt**3 / 6.0],
        [dt**2 / 2.0],
        [dt],
    ])

    C = np.array([
        [1.0, 0.0, -com_height / gravity]
    ])

    Ahat = np.block([
        [np.eye(1), C @ A],
        [np.zeros((3, 1)), A],
    ])

    Bhat = np.vstack([
        C @ B,
        B,
    ])

    Qhat = np.zeros((4, 4))
    Qhat[0, 0] = Qe

    P = solve_dare(
        Ahat,
        Bhat,
        Qhat,
        np.array([[R]])
    )

    denominator = float(
        (Bhat.T @ P @ Bhat)[0, 0] + R
    )

    K = (
        Bhat.T @ P @ Ahat
    ) / denominator

    Gi = float(K[0, 0])
    Gx = K[0, 1:].copy()

    Acl = Ahat - Bhat @ K

    Gd = np.zeros(preview_steps)

    X = (
        -Acl.T
        @ P
        @ np.array([[1.0], [0.0], [0.0], [0.0]])
    )

    for i in range(preview_steps):
        Gd[i] = float(
            (Bhat.T @ X)[0, 0] / denominator
        )

        X = Acl.T @ X

    return A, B, C, Gx, Gi, Gd


def generate_com(
    ref_x,
    ref_y,
    A,
    B,
    C,
    Gx,
    Gi,
    Gd,
):

    preview = len(Gd)
    total = len(ref_x) - preview

    outputs = []

    for reference in (ref_x, ref_y):

        x = np.zeros((3, 1))
        error_sum = 0.0
        output = np.zeros(total)

        for k in range(total):

            zmp = float((C @ x)[0, 0])

            error_sum += zmp - reference[k]

            future = reference[
                k + 1:k + 1 + preview
            ]

            preview_term = float(
                np.dot(Gd, future)
            )

            u = (
                -Gi * error_sum
                - float((Gx @ x)[0])
                - preview_term
            )

            x = A @ x + B * u

            output[k] = x[0, 0]

        outputs.append(output)

    return outputs[0], outputs[1]


# ============================================================
# FREE-BASE POSE
# ============================================================

def make_freebase_pose(model, data, xyz):

    qpos = data.qpos.copy()

    adr = freejoint_address(model)

    qpos[adr:adr + 3] = xyz

    qpos[adr + 3:adr + 7] = np.array(
        [0.0, 1.0, 0.0, 0.0]
    )

    return qpos


# ============================================================
# LEG IK
# ============================================================

def solve_leg_ik(
    model,
    data,
    joint_names,
    foot_body,
    target,
    q_seed,
    nominal_ankle=0.0,
):

    q_base = data.qpos.copy()
    foot_id = model.body(foot_body).id

    lower = []
    upper = []

    for name in joint_names:
        lo, hi = joint_limits(model, name)
        lower.append(lo)
        upper.append(hi)

    lower = np.asarray(lower)
    upper = np.asarray(upper)

    q_seed = clamp_leg(
        model,
        joint_names,
        q_seed
    )

    def residual(q):

        q_trial = q_base.copy()

        set_leg_q(
            model,
            q_trial,
            joint_names,
            q
        )

        data.qpos[:] = q_trial
        mujoco.mj_forward(model, data)

        actual = data.xpos[foot_id]

        position_error = (
            actual - target
        ) * IK_POSITION_WEIGHT

        posture_error = np.array([
            IK_POSTURE_WEIGHT *
            (q[0] - q_seed[0]),

            IK_POSTURE_WEIGHT *
            (q[1] - q_seed[1]),

            IK_POSTURE_WEIGHT *
            (q[2] - q_seed[2]),

            IK_ANKLE_WEIGHT *
            (q[3] - nominal_ankle),
        ])

        return np.concatenate([
            position_error,
            posture_error,
        ])
    
    solution = bounded_least_squares(
        residual,
        q_seed,
        lower,
        upper,
        max_iter=IK_MAX_NFEV,
    )

    # Restore solved configuration
    set_leg_q(
        model,
        q_base,
        joint_names,
        solution
    )

    data.qpos[:] = q_base
    mujoco.mj_forward(model, data)

    error = np.linalg.norm(
        data.xpos[foot_id] - target
    )

    return solution.copy(), float(error)


# ============================================================
# BUILD COMPLETE FRAME
# ============================================================

def make_frame(model, base, left_q, right_q):

    qpos = base.copy()

    set_leg_q(
        model,
        qpos,
        LEFT_JOINTS,
        left_q
    )

    set_leg_q(
        model,
        qpos,
        RIGHT_JOINTS,
        right_q
    )

    return qpos


# ============================================================
# SWING FOOT TRAJECTORY
# ============================================================

def swing_profile(start, end, duration, clearance):

    n = max(int(round(duration / DT)), 1)

    trajectory = []

    for i in range(n):

        s = i / max(n - 1, 1)

        # Cubic smoothstep
        h = 3 * s**2 - 2 * s**3

        # Vertical foot clearance
        z_hump = 4 * s * (1 - s)

        p = (
            (1 - h) * start
            + h * end
        )

        p = p.copy()
        p[2] += clearance * z_hump

        trajectory.append(p)

    return trajectory


# ============================================================
# WALKING TRAJECTORY
# ============================================================

def generate_walking(model, data):

    geom = discover_robot(model, data)

    # --------------------------------------------------------
    # 1. Footstep plan
    # --------------------------------------------------------

    footsteps = plan_footsteps(
        geom.left_foot0,
        geom.right_foot0,
        NUM_STEPS,
        STEP_LENGTH
    )

    # --------------------------------------------------------
    # 2. ZMP reference
    # --------------------------------------------------------
    
    pref_x, pref_y = zmp_reference(
        geom.left_foot0,
        geom.right_foot0,
        footsteps
    )

    # --------------------------------------------------------
    # 3. Preview-controlled CoM
    # --------------------------------------------------------

    A, B, C, Gx, Gi, Gd = preview_gains(
        DT,
        GRAVITY,
        geom.root_z,
        N_PREVIEW
    )

    com_x, com_y = generate_com(
        pref_x,
        pref_y,
        A,
        B,
        C,
        Gx,
        Gi,
        Gd
    )

    # --------------------------------------------------------
    # 4. Initial robot configuration
    # --------------------------------------------------------

    q0 = data.qpos.copy()

    root_adr = freejoint_address(model)

    q0[root_adr + 3:root_adr + 7] = np.array(
        [0.0, 1.0, 0.0, 0.0]
    )

    q0[root_adr + 2] = geom.root_z

    left_q = get_leg_q(
        model,
        q0,
        LEFT_JOINTS
    )

    right_q = get_leg_q(
        model,
        q0,
        RIGHT_JOINTS
    )

    left_foot = geom.left_foot0.copy()
    right_foot = geom.right_foot0.copy()

    frames = []
    ik_errors = []

    # --------------------------------------------------------
    # 5. Initial double support
    # --------------------------------------------------------

    initial_frames = int(T_DSP / DT)

    for k in range(initial_frames):

        idx = min(k, len(com_x) - 1)

        base_xyz = np.array([
            com_x[idx],
            com_y[idx],
            geom.root_z,
        ])

        base = make_freebase_pose(
            model,
            data,
            base_xyz
        )

        data.qpos[:] = base
        mujoco.mj_forward(model, data)

        left_q, e1 = solve_leg_ik(
            model,
            data,
            LEFT_JOINTS,
            LEFT_FOOT_BODY,
            left_foot,
            left_q,
        )

        right_q, e2 = solve_leg_ik(
            model,
            data,
            RIGHT_JOINTS,
            RIGHT_FOOT_BODY,
            right_foot,
            right_q,
        )

        frames.append(
            make_frame(
                model,
                base,
                left_q,
                right_q,
            )
        )

        ik_errors.append(max(e1, e2))

    global_index = initial_frames

    # --------------------------------------------------------
    # 6. Walking steps
    # --------------------------------------------------------

    for foot_name, landing in footsteps:

        if foot_name == "left":

            stance = right_foot.copy()
            swing_start = left_foot.copy()
            swing_end = landing.copy()

        else:

            stance = left_foot.copy()
            swing_start = right_foot.copy()
            swing_end = landing.copy()

        # ----------------------------------------------------
        # SSP
        # ----------------------------------------------------

        swing = swing_profile(
            swing_start,
            swing_end,
            T_SSP,
            FOOT_CLEARANCE
        )

        for target in swing:

            if global_index >= len(com_x):
                break

            base_xyz = np.array([
                com_x[global_index],
                com_y[global_index],
                geom.root_z,
            ])

            base = make_freebase_pose(
                model,
                data,
                base_xyz
            )

            data.qpos[:] = base
            mujoco.mj_forward(model, data)

            if foot_name == "left":

                left_q, e1 = solve_leg_ik(
                    model,
                    data,
                    LEFT_JOINTS,
                    LEFT_FOOT_BODY,
                    target,
                    left_q,
                )

                right_q, e2 = solve_leg_ik(
                    model,
                    data,
                    RIGHT_JOINTS,
                    RIGHT_FOOT_BODY,
                    stance,
                    right_q,
                )

            else:

                left_q, e1 = solve_leg_ik(
                    model,
                    data,
                    LEFT_JOINTS,
                    LEFT_FOOT_BODY,
                    stance,
                    left_q,
                )

                right_q, e2 = solve_leg_ik(
                    model,
                    data,
                    RIGHT_JOINTS,
                    RIGHT_FOOT_BODY,
                    target,
                    right_q,
                )

            frames.append(
                make_frame(
                    model,
                    base,
                    left_q,
                    right_q,
                )
            )

            ik_errors.append(max(e1, e2))

            global_index += 1

        # ----------------------------------------------------
        # Landing
        # ----------------------------------------------------

        if foot_name == "left":
            left_foot = landing.copy()
        else:
            right_foot = landing.copy()

        # ----------------------------------------------------
        # DSP
        # ----------------------------------------------------

        for _ in range(int(T_DSP / DT)):

            if global_index >= len(com_x):
                break

            base_xyz = np.array([
                com_x[global_index],
                com_y[global_index],
                geom.root_z,
            ])

            base = make_freebase_pose(
                model,
                data,
                base_xyz
            )

            data.qpos[:] = base
            mujoco.mj_forward(model, data)

            left_q, e1 = solve_leg_ik(
                model,
                data,
                LEFT_JOINTS,
                LEFT_FOOT_BODY,
                left_foot,
                left_q,
            )

            right_q, e2 = solve_leg_ik(
                model,
                data,
                RIGHT_JOINTS,
                RIGHT_FOOT_BODY,
                right_foot,
                right_q,
            )

            frames.append(
                make_frame(
                    model,
                    base,
                    left_q,
                    right_q,
                )
            )

            ik_errors.append(max(e1, e2))

            global_index += 1

    # Restore first frame   
    if frames:
        data.qpos[:] = frames[0]
        mujoco.mj_forward(model, data)

    return (
        frames,
        com_x,
        com_y,
        pref_x,
        pref_y,
        np.asarray(ik_errors),
        geom,
    )


# ============================================================
# PRINT ROBOT INFORMATION
# ============================================================

def print_robot_summary(model, geom):

    print("\nXML robot summary")
    print("-----------------")
    print(f"root body       : {ROOT_BODY}")
    print(f"root height     : {geom.root_z:.4f} m")
    print(f"left foot start : {geom.left_foot0}")
    print(f"right foot start: {geom.right_foot0}")

    print("\nJoint limits / actuator ranges:")

    for name in ALL_JOINTS:

        jid = model.joint(name).id

        joint_limits_xml = (
            float(model.jnt_range[jid, 0]),
            float(model.jnt_range[jid, 1])
        )

        actuator_limits = None

        for aid in range(model.nu):

            actuator_joint = model.actuator_trnid[aid, 0]

            if actuator_joint < 0:
                continue

            if model.joint(actuator_joint).name == name:

                if model.actuator_ctrllimited[aid]:
                    actuator_limits = (
                        float(model.actuator_ctrlrange[aid, 0]),
                        float(model.actuator_ctrlrange[aid, 1]),
                    )

                break

        print(
            f"  {name:20s} "
            f"joint={joint_limits_xml} "
            f"actuator={actuator_limits}"
        )


# ============================================================
# VIEWER
# ============================================================

def playback(model, data, frames, realtime=True):

    print(
        "\nLaunching MuJoCo viewer. "
        "Close the viewer to exit."
    )

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15
        viewer.cam.distance = 1.2

        root_id = model.body(ROOT_BODY).id

        while viewer.is_running():

            for qpos in frames:

                if not viewer.is_running():
                    break

                data.qpos[:] = qpos
                data.qvel[:] = 0

                mujoco.mj_forward(model, data)

                viewer.cam.lookat[:] = data.xpos[root_id]

                viewer.sync()

                if realtime:
                    time.sleep(DT)

            if viewer.is_running():
                time.sleep(0.25)


# ============================================================
# MAIN
# ============================================================

def main():

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    geom = discover_robot(model, data)

    print_robot_summary(
        model,
        geom
    )

    print("\nGenerating walking trajectory...")

    (
        frames,
        com_x,
        com_y,
        pref_x,
        pref_y,
        ik_errors,
        _,
    ) = generate_walking(
        model,
        data
    )

    if not frames:
        raise RuntimeError(
            "No walking frames were generated"
        )

    print(f"Generated frames    : {len(frames)}")
    print(
        f"Trajectory duration : "
        f"{len(frames) * DT:.2f} s"
    )
    print(
        f"Step length         : "
        f"{STEP_LENGTH:.3f} m"
    )
    print(
        f"Maximum IK error    : "
        f"{ik_errors.max() * 1000.0:.2f} mm"
    )
    print(
        f"Mean IK error       : "
        f"{ik_errors.mean() * 1000.0:.2f} mm"
    )
    print(
        f"Forward travel      : "
        f"{com_x[-1] - com_x[0]:.3f} m"
    )

    if ENABLE_VIEWER:
        playback(
            model,
            data,
            frames,
            realtime=REALTIME
        )


if __name__ == "__main__":
    main()
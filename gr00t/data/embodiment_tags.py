from enum import Enum


"""
Embodiment tags are used to identify the robot embodiment in the data.

Naming convention:
<dataset>_<robot_name>

If using multiple datasets, e.g. sim GR1 and real GR1, we can drop the dataset name and use only the robot name.
"""


class EmbodimentTag(Enum):
    ##### Pretrain embodiment tags #####
    ROBOCASA_PANDA_OMRON = "robocasa_panda_omron"
    """
    The RoboCasa Panda robot with omron mobile base.
    """

    GR1 = "gr1"
    """
    The Fourier GR1 robot.
    """

    ##### Pre-registered posttrain embodiment tags #####
    UNITREE_G1 = "unitree_g1"
    """
    The Unitree G1 robot.
    """

    LIBERO_PANDA = "libero_panda"
    """
    The Libero panda robot.
    """

    OXE_GOOGLE = "oxe_google"
    """
    The Open-X-Embodiment Google robot.
    """

    OXE_WIDOWX = "oxe_widowx"
    """
    The Open-X-Embodiment WidowX robot.
    """

    OXE_DROID = "oxe_droid"
    """
    The Open-X-Embodiment DROID robot with relative joint position actions.
    """

    BEHAVIOR_R1_PRO = "behavior_r1_pro"
    """
    The Behavior R1 Pro robot.
    """

    ##### Open-H embodiment tags #####
    JHU_IMERSE_DVRK = "jhu_imerse_dvrk"
    """
    The da Vinci Research Kit (dVRK) surgical robot.
    Dual-arm (PSM1/PSM2) with REL_XYZ_ROT6D EEF control.
    """

    JHU_IMERSE_DVRK_MONO = "jhu_imerse_dvrk_mono"
    """
    Monocular JHU dVRK surgical robot variant.
    Uses only the left endoscope video stream with the same dual-arm state/action
    representation as the standard dVRK embodiment.
    """

    JHU_LSCR_DVRK_MIRACLE = "jhu_lscr_dvrk_miracle"
    """
    JHU LSCR MIRACLE datasets.
    Dual-arm joint-angle control with RELATIVE actions (14D action, 12D joints + grippers).
    Uses stereo endoscope cameras (15 Hz).
    """

    JHU_LSCR_DVRK_SMARTS = "jhu_lscr_dvrk_smarts"
    """
    JHU LSCR SMARTS offline datasets.
    Dual-arm joint-angle control with RELATIVE actions (PSM1/PSM2 joints + grippers).
    Uses endoscope left/right plus side-view camera (10 Hz).
    """

    JHU_IMERSE_DVRK_STAR_IL = "jhu_imerse_star_il"
    """
    JHU IMERSE star_IL dataset.
    Single-arm KUKA with 7D pose actions (xyz + quat) and 8D state (7 joints + endo360).
    Uses endoscope left + wrist left video.
    """

    CMR_VERSIUS = "cmr_versius"
    """
    The CMR Versius surgical robot.
    Dual-arm with REL_XYZ_ROT6D EEF control and energy buttons.
    """

    UCB_DVRK = "ucb_dvrk"
    """
    The UCBerkeley dVRK debridement dataset.
    Dual-arm (PSM1/PSM2) with cartesian EEF control (16D).
    Uses REL_XYZ_ROT6D for pose actions with quaternion inputs.
    """

    OBUDA_DVRK = "obuda_dvrk"
    """
    The Obuda University Open-H dVRK datasets.
    Dual-arm (PSM1/PSM2) with REL_XYZ_ROT6D EEF control (16D).
    Uses endoscope left + wrist left/right cameras; ECM is excluded in v1.
    """

    MOON_MAESTRO = "moon_maestro"
    """
    The Moon Surgical Maestro assistant dataset.
    Dual-arm robot with 18D joint state and 6D delta translation actions (xyz per arm).
    """

    UCSD_DVRK = "ucsd_dvrk"
    """
    The UCSD surgical learning dataset.
    Dual-arm (retraction + cutter) with delta EEF pose actions (16D).
    Uses REL_XYZ_ROT6D for EEF pose actions with wxyz quaternion ordering.
    """

    STANFORD_DVRK_REAL = "stanford_dvrk_real"
    """
    Stanford real-robot dVRK datasets (Needle Transfer, Tissue Retraction, Peg Transfer).
    Dual-arm (PSM1/PSM2) with absolute EEF pose actions in Euler RPY (camera/ECM frame).
    Uses REL_XYZ_ROT6D with Euler input and reference rotations.
    """

    SANOSCIENCE_SIM = "sanoscience_sim"
    """
    The SanoScience simulated surgical robot.
    4 instruments with REL_XYZ_ROT6D EEF control (32D state/action).
    Each instrument: xyz + quaternion (7D) + gripper (1D).
    """

    TUM_SONATA_FRANKA = "tum_sonata_franka"
    """
    The TUM SonATA ultrasound sonography dataset.
    Franka Panda robot with ultrasound probe end-effector.
    REL_XYZ_ROT6D EEF control with Euler angles (6D xyz + RPY -> 9D xyz_rel + rot6d).
    Includes force/torque sensor data and 3 camera views.
    """

    USTC_TORIN_TUODAO = "ustc_torin_tuodao"
    """
    The USTC Torin surgical dataset.
    Stereo endoscope video with 14D joint-angle state and 14D Cartesian delta actions
    (xyz + roll/pitch/yaw + gripper per arm). Actions are treated as pass-through deltas
    until a reliable Cartesian EEF state reference is available.
    """

    TUD_TUNDRA_UR5E = "tud_tundra_ur5e"
    """
    The TUD TUNDRA UR5e surgical assistance dataset.
    Stereo laparoscope video with grasping/retraction configured for REL_XYZ_ROT6D
    EEF actions using absolute pose targets sourced from observation.state.
    (Tundra Endoscope guidance requires an additional config.)
    """

    TURIN_MITIC_EX_VIVO = "turin_mitic_ex_vivo"
    """
    The Turin MITIC ex vivo surgical dataset.
    Dual-arm dVRK (PSM1/PSM2) with joint-angle state and absolute EEF pose actions
    (xyz + quaternion per arm). REL_XYZ_ROT6D uses action[t=0] as the pose reference.
    """

    ROB_SURGICAL_BITRACK = "rob_surgical_bitrack"
    """
    The Rob Surgical (bitrack) dataset.
    Single endoscope video with 4-arm Cartesian EEF state/action (24D).
    Uses REL_XYZ_ROT6D EEF control with Euler (RPY) rotation inputs.
    """

    HAMLYN_DVRK_15HZ = "hamlyn_dvrk_15hz"
    """
    Hamlyn Centre dVRK surgical robot dataset - 15Hz tasks.
    Tasks: knot_tying, needle_grasp_and_handover, peg_transfer, Suturing-1,
           Suturing-2, suturing_single_loop_2, tissue_lifting.
    Dual-arm with REL_XYZ_ROT6D EEF control (16D state/action).
    Uses wxyz quaternion ordering (scalar-first).
    """

    HAMLYN_DVRK_30HZ = "hamlyn_dvrk_30hz"
    """
    Hamlyn Centre dVRK surgical robot dataset - 30Hz tasks.
    Tasks: suturing_single_loop_1, tissue_retraction.
    Dual-arm with REL_XYZ_ROT6D EEF control (16D state/action).
    Uses wxyz quaternion ordering (scalar-first).
    """

    POLYU_SIM = "polyu_sim"
    """
    The PolyU OpenH_Dataset_full simulated surgical dataset.
    Single-arm surgical robot with joint-angle state (10D) and cartesian pose state (7D).
    Actions use REL_XYZ_ROT6D pose targets plus a gripper channel (1D).
    """

    # New embodiment during post-training
    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment.
    """

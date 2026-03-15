# Open-H Embodiment Overview

Summary of all surgical robot embodiments currently supported by GR00T-H.

## Embodiment Comparison

| Embodiment | Dataset Type | Raw State Format | Raw Action Format | Final Action Format | Arms Used | Cameras Used |
|---|---|---|---|---|---|---|
| **CMR Versius** (`cmr_versius`) | Clinical | Cartesian EEF pose + gripper per arm (26D) | Cartesian EEF pose + gripper per arm (26D) | 2×9D pose + 2×1D gripper = **20D** | 2 | 1 |
| **JHU IMERSE dVRK** (`jhu_imerse_dvrk`) | Surgical tabletop | Cartesian EEF pose + gripper per arm (16D) | Cartesian EEF setpoint + gripper per arm (16D) | 2×9D pose + 2×1D gripper = **20D** | 2 | 3 |
| **JHU LSCR dVRK** (`jhu_lscr_dvrk`) | Surgical tabletop | Cartesian EEF pose + gripper per arm (16D) | Cartesian EEF setpoint + gripper per arm (16D) | 2×9D pose + 2×1D gripper = **20D** | 2 | 2-3 |
| **UCB dVRK** (`ucb_dvrk`) | Surgical tabletop | Cartesian EEF pose + gripper (16D) + joints (14D) | Cartesian EEF setpoint + gripper per arm (16D) | 2×9D pose + 2×1D gripper = **20D** | 2 | 2 |
| **Obuda dVRK** (`obuda_dvrk`) | Surgical tabletop | Cartesian EEF pose + gripper per arm (16D) | Cartesian EEF setpoint + gripper per arm (16D) | 2×9D pose + 2×1D gripper = **20D** | 2 | 3 |
| **Stanford dVRK Real** (`stanford_dvrk_real`) | Surgical tabletop | Cartesian EEF pose (Euler) + gripper per arm (14D) | Cartesian EEF pose (Euler) + gripper per arm (14D) | 2×9D pose + 2×1D gripper = **20D** | 2 | 2 |
| **UCSD dVRK** (`ucsd_dvrk`) | Ex-vivo | Cartesian EEF pose + gripper per arm (16D) | Cartesian EEF pose + gripper per arm (16D) | 2×9D pose + 2×1D gripper = **20D** | 2 | 2 |
| **Hamlyn dVRK** (`hamlyn_dvrk`) | Ex-vivo | Cartesian EEF pose + gripper per arm (16D) | Cartesian EEF pose + gripper per arm (16D) | 2×9D pose + 2×1D gripper = **20D** | 2 | 3 |
| **Turin MITIC** (`turin_mitic_ex_vivo`) | Ex-vivo | Joint angles per arm (12D) + EEF pass-through (14D) | Cartesian EEF pose per arm (14D) | 2×9D pose = **18D** | 2 | 2 |
| **USTC Torin/Tuodao** (`ustc_torin_tuodao`) | Clinical + Ex-vivo | Joint angles per arm (14D) + EEF pass-through (16D) | Cartesian absolute pose + gripper per arm (16D) | 2×9D pose + 2×1D gripper = **20D** | 2 | 2 |
| **TUD TUNDRA UR5e** (`tud_tundra_ur5e`) | Clinical (porcine) | Joint position (6D) + EEF pass-through (7D) | Cartesian EEF pose (7D) + gripper (1D) | 1×9D pose + 1×1D gripper = **10D** | 1 | 2 |
| **TUM SonATA Franka** (`tum_sonata_franka`) | Ultrasound phantom | Joint angles (7D) + force/torque (6D) + EEF pass-through (6D) | Cartesian EEF pose, Euler (6D) | 1×9D pose = **9D** | 1 | 3 |
| **Moon Maestro** (`moon_maestro`) | Surgical tabletop | Joint angles per arm (18D) | Delta translation per arm (6D) | 2×3D delta xyz = **6D** | 2 | 2 |
| **Rob Surgical Bitrack** (`rob_surgical_bitrack`) | Surgical tabletop | Cartesian EEF pose (Euler) per arm (18D) | Cartesian EEF pose (Euler) per arm (18D) | 3×9D pose = **27D** | 3 | 1 |
| **SanoScience Sim** (`sanoscience_sim`) | Simulation | Cartesian EEF pose + gripper per instrument (32D) | Cartesian EEF pose + gripper per instrument (32D) | 4×9D pose + 4×1D gripper = **40D** | 4 | 1 |
| **PolyU Sim** (`polyu_sim`) | Simulation | Joint angles (10D) + EEF pass-through (7D) | Cartesian EEF pose (7D) + gripper (1D) | 1×9D pose + 1×1D gripper = **10D** | 1 | 1 |

## Column Definitions

- **Raw State Format**: Observation state representation as stored in the dataset, before any model preprocessing. "EEF pass-through" indicates the state is not embedded by the model but is used as a reference frame for relative action conversion.
- **Raw Action Format**: Action representation as stored in the dataset, before conversion to the model's internal format.
- **Final Action Format**: The model's output action dimension after REL_XYZ_ROT6D conversion. Each pose key becomes 9D (3D relative xyz + 6D rotation). Gripper and other scalar keys stay at their original dimension. The breakdown shows exactly how the total is computed.
- **Arms Used**: Number of independently controlled robot arms or instruments used by the model.
- **Cameras Used**: Number of camera views used by the model.

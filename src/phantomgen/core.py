import numpy as np

# ===============================================================
# ===  HELPER: Generate centered world coordinates for a volume ===
# ===============================================================
def _world_coords(shape, voxel_mm, center_mm):
    """
    Compute 3D world coordinates (mm) for voxel centers, relative to the
    volume's geometric center.

    Parameters
    ----------
    shape : tuple (Z, Y, X)
        Shape of the 3D array.
    voxel_mm : tuple (sz, sy, sx)
        Physical voxel sizes in mm.
    center_mm : tuple (cz, cy, cx)
        World-coordinate offset of the shape center in mm.

    Returns
    -------
    Zmm, Ymm, Xmm : np.ndarray
        3D coordinate grids with same broadcastable shapes.
    """
    Z, Y, X = shape
    sz, sy, sx = map(float, voxel_mm)
    cz, cy, cx = map(float, center_mm)
    z0 = (np.arange(Z) - (Z - 1) / 2.0) * sz
    y0 = (np.arange(Y) - (Y - 1) / 2.0) * sy
    x0 = (np.arange(X) - (X - 1) / 2.0) * sx
    return (z0[:, None, None] - cz,
            y0[None, :, None] - cy,
            x0[None, None, :] - cx)


# ===============================================================
# ===  BASIC GEOMETRIC PRIMITIVES (CYLINDER, BOX, SPHERE)       ===
# ===============================================================
def add_cylinder(volume, voxel_size_mm, radius_mm, height_mm, deg_range, center_mm, value=1):
    """
    Fill a cylindrical region (optionally sector-shaped) in-place.

    Assumptions
    -----------
    - Volume axes: (Z, Y, X)
    - Cylinder axis: Z
    - Angles measured CCW in XY plane (0° = +X).

    Parameters
    ----------
    volume : np.ndarray
        Target 3D array to modify in-place.
    voxel_size_mm : tuple (sz, sy, sx)
        Physical voxel size in mm.
    radius_mm : float
        Cylinder radius in mm.
    height_mm : float
        Cylinder height along Z in mm.
    deg_range : tuple(start_deg, end_deg) or None
        Angular span in degrees. None or full 360 for complete cylinder.
    center_mm : tuple (cz, cy, cx)
        Center of the cylinder in world coordinates (mm).
    value : scalar
        Value assigned inside the region.
    """
    assert volume.ndim == 3
    Zmm, Ymm, Xmm = _world_coords(volume.shape, voxel_size_mm, center_mm)
    mask = (np.abs(Zmm) <= height_mm / 2.0) & ((Xmm**2 + Ymm**2) <= radius_mm**2)

    if deg_range is not None:
        s, e = (deg_range[0] % 360.0, deg_range[1] % 360.0)
        span = (e - s) % 360.0
        if span and abs(span - 360.0) > 1e-9:
            theta = np.degrees(np.arctan2(Ymm, Xmm)) % 360.0
            ang_ok = (theta >= s) & (theta <= e) if s <= e else ((theta >= s) | (theta <= e))
            mask &= ang_ok

    volume[mask] = value
    return volume


def add_box(volume, voxel_size_mm, size_mm, center_mm=(0, 0, 0), rotation_deg=0.0, value=1):
    """
    Fill a rectangular box (cuboid) in-place, optionally rotated about Z.

    Parameters
    ----------
    volume : np.ndarray
        Target 3D array (Z,Y,X).
    voxel_size_mm : tuple (sz, sy, sx)
        Physical voxel size in mm.
    size_mm : tuple (height_z, size_y, size_x)
        Box dimensions in mm.
    center_mm : tuple (cz, cy, cx)
        Box center in world coordinates (mm).
    rotation_deg : float
        In-plane rotation about Z-axis in degrees (CCW).
    value : scalar
        Value assigned inside the box.
    """
    assert volume.ndim == 3
    Zmm, Ymm, Xmm = _world_coords(volume.shape, voxel_size_mm, center_mm)
    if rotation_deg % 360:
        th = np.deg2rad(rotation_deg)
        c, s = np.cos(th), np.sin(th)
        Xp, Yp = c * Xmm + s * Ymm, -s * Xmm + c * Ymm
    else:
        Xp, Yp = Xmm, Ymm
    hz, hy, hx = (size_mm[0]/2, size_mm[1]/2, size_mm[2]/2)
    inside = (np.abs(Xp) <= hx) & (np.abs(Yp) <= hy) & (np.abs(Zmm) <= hz)
    volume[inside] = value
    return volume


def add_sphere(volume, voxel_size_mm, radius_mm, center_mm=(0, 0, 0), value=1):
    """
    Fill a spherical region in-place.

    Parameters
    ----------
    volume : np.ndarray
        Target 3D array (Z,Y,X).
    voxel_size_mm : tuple (sz, sy, sx)
        Physical voxel size in mm.
    radius_mm : float
        Sphere radius in mm.
    center_mm : tuple (cz, cy, cx)
        Sphere center in world coordinates (mm).
    value : scalar
        Value assigned inside the sphere.
    """
    assert volume.ndim == 3
    Zmm, Ymm, Xmm = _world_coords(volume.shape, voxel_size_mm, center_mm)
    inside = (Xmm**2 + Ymm**2 + Zmm**2) <= radius_mm**2
    volume[inside] = value
    return volume


# ===============================================================
# ===  NEMA NU 2 / IEC BODY PHANTOM CREATOR                    ===
# ===============================================================
def create_nema(
    matrix_size=(256, 256, 256),              # (Z,Y,X)
    voxel_size_mm=(2.0, 2.0, 2.0),
    nema_dict=None
):
    """
    Create a 3D numerical phantom matching the IEC/NEMA IQ body phantom geometry.

    Generates both:
      • `act_vol` – activity map (MBq/voxel)
      • `ct_vol`  – attenuation map (cm⁻¹)

    Parameters
    ----------
    matrix_size : tuple (Z, Y, X)
        Number of voxels along each dimension.
    voxel_size_mm : tuple (sz, sy, sx)
        Physical voxel size in mm.
    nema_dict : dict
        Dictionary defining all phantom parameters:
        {
            "activity_concentration_background": 0.05,   # MBq/ml
            "fill_mu_value": 0.096,                      # cm^-1
            "perspex_mu_value": 0.12,                    # optional
            "lung_insert": {
                "include": True,
                "lung_mu_value": 0.029
            },
            "sphere_dict": {
                "ring_R": 57,                             # radius (mm)
                "ring_z": -37,                            # z-position (mm)
                "spheres": {
                    "diametre_mm": [10,13,17,22,28,37],
                    "angle_loc":   [30,90,150,210,270,330],
                    "act_conc_MBq_ml": [0.00,0.00,0.04,0.04,0.04,0.04]
                }
            }
        }

    Returns
    -------
    act_vol, ct_vol : np.ndarray
        Activity and CT μ-map volumes (same shape).
    """
    import numpy as np

    # --- defaults ---
    defaults = {
        "mu_values":{
            "perspex_mu_value": 0.1,
            "fill_mu_value": 0.096,
            "lung_mu_value": 0.029
        },
        "activity_concentration_background": 0.05,
        "include_lung_insert": True,
        "sphere_dict": {
            "ring_R": 57,
            "ring_z": -37,
            "spheres": {
                "diametre_mm": [10,13,17,22,28,37],
                "angle_loc":   [30,90,150,210,270,330],
                "act_conc_MBq_ml": [0.00,0.00,0.4,0.4,0.4,0.4],
            }
        }
    }

    # --- merge user input with defaults ---
    if nema_dict is None:
        nema_dict = defaults
    else:
        # recursively update nested dicts
        import copy
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d.setdefault(k, v)
            return d
        nema_dict = deep_update(copy.deepcopy(nema_dict), defaults)

    # --- unpack parameters ---
    act_conc_backgr = nema_dict["activity_concentration_background"]
    fill_mu_value = nema_dict["mu_values"]["fill_mu_value"]
    perspex_mu_value = nema_dict["mu_values"]["perspex_mu_value"]
    lung_insert = nema_dict["include_lung_insert"]
    lung_mu_value = nema_dict["mu_values"]["lung_mu_value"]

    sphere_info = nema_dict["sphere_dict"]
    ring_R = sphere_info["ring_R"]
    z_pos = sphere_info["ring_z"]

    # --- volume size check ---
    vol_dim = [m * v for m, v in zip(matrix_size, voxel_size_mm)]
    if any(v < lim for v, lim in zip(vol_dim, (220, 300, 230))):
        print("❌ Volume smaller than NEMA phantom dimensions!")
        return

    ctac_vol = np.zeros(matrix_size, np.float32)
    act_vol = np.zeros(matrix_size, np.float32)

    ml_per_vox = np.prod(voxel_size_mm) / 1000.0
    back_MBq_per_vox = act_conc_backgr * ml_per_vox

    # --- connecting box ---
    add_box(ctac_vol,  voxel_size_mm, (220, 75, 150), (0, 72.5, 0), value=perspex_mu_value)
    add_box(ctac_vol,  voxel_size_mm, (214, 72, 150), (0, 71, 0), value=fill_mu_value)
    add_box(act_vol, voxel_size_mm, (214, 72, 150), (0, 71, 0), value=back_MBq_per_vox)

    # --- tank structure ---
    tanks = [
        dict(r=150, h=220, deg=(180, 360), c=(0, 35, 0), mu="perspex"),
        dict(r=147, h=214, deg=(180, 360), c=(0, 35, 0), mu="fill"),
        dict(r=75,  h=220, deg=(90, 180),  c=(0, 35, -75), mu="perspex"),
        dict(r=72,  h=214, deg=(90, 180),  c=(0, 35, -75), mu="fill"),
        dict(r=75,  h=220, deg=(0, 90),    c=(0, 35, 75), mu="perspex"),
        dict(r=72,  h=214, deg=(0, 90),    c=(0, 35, 75), mu="fill"),
    ]
    if lung_insert:
        tanks.append(dict(r=25, h=214, deg=None, c=(0, 0, 0), mu="lung"))

    for t in tanks:
        if t["mu"] == "perspex":
            add_cylinder(ctac_vol, voxel_size_mm, t["r"], t["h"], t["deg"], t["c"], perspex_mu_value)
        elif t["mu"] == "fill":
            add_cylinder(ctac_vol, voxel_size_mm, t["r"], t["h"], t["deg"], t["c"], fill_mu_value)
            add_cylinder(act_vol, voxel_size_mm, t["r"], t["h"], t["deg"], t["c"], back_MBq_per_vox)
        else:
            add_cylinder(ctac_vol, voxel_size_mm, t["r"], t["h"], t["deg"], t["c"], lung_mu_value)
            add_cylinder(act_vol, voxel_size_mm, t["r"], t["h"], t["deg"], t["c"], 0)

    # --- add spheres ---
    for d, a, c in zip(
        sphere_info["spheres"]["diametre_mm"],
        sphere_info["spheres"]["angle_loc"],
        sphere_info["spheres"]["act_conc_MBq_ml"]
    ):
        r_shell = (d + 2) / 2.0
        r_interior = d / 2.0
        ang = np.deg2rad(a)
        cy, cx = -ring_R * np.sin(ang), ring_R * np.cos(ang)
        fill_act = c * ml_per_vox
        add_sphere(ctac_vol, voxel_size_mm, r_shell, (z_pos, cy, cx), value=perspex_mu_value)
        add_sphere(ctac_vol, voxel_size_mm, r_interior, (z_pos, cy, cx), value=fill_mu_value)
        add_sphere(act_vol, voxel_size_mm, r_interior, (z_pos, cy, cx), value=fill_act)

    return act_vol, ctac_vol


earl_nema_dict = {
    "mu_values":{
        "perspex_mu_value": 0.15,
        "fill_mu_value": 0.14,
        "lung_mu_value": 0.043
    },
    "activity_concentration_background": 0.05,
    "include_lung_insert": False,
    "sphere_dict": {
        "ring_R": 57,
        "ring_z": -37,
        "spheres": {
            "diametre_mm": [13,17,22,28,37,60],
            "angle_loc":   [270,150,30,90,330,210],
            "act_conc_MBq_ml": [2.0,2.0,2.0,2.0,2.0,2.0],
        }
    }
}


pet_nema_dict = {
    "mu_values":{
        "perspex_mu_value": 0.1,
        "fill_mu_value": 0.096,
        "lung_mu_value": 0.029
    },
    "activity_concentration_background": 0.05,
    "include_lung_insert": True,
    "sphere_dict": {
        "ring_R": 57,
        "ring_z": -37,
        "spheres": {
            "diametre_mm": [10,13,17,22,28,37],
            "angle_loc":   [30,90,150,210,270,330],
            "act_conc_MBq_ml": [0.00,0.00,0.4,0.4,0.4,0.4],
        }
    }
}

# ... your code exactly as provided above ...

def cli():
    """
    CLI to write activity and CT volumes to .npy files.
    Examples:
      phantomgen --preset pet --out-act act.npy --out-ct ct.npy
      phantomgen --preset earl --z 256 --y 256 --x 256 --voxel 2 2 2
    """
    import argparse, numpy as np

    p = argparse.ArgumentParser(prog="phantomgen", description="Generate IEC/NEMA phantoms")
    p.add_argument("--preset", choices=["pet", "earl"], default="pet", help="Parameter set")
    p.add_argument("--z", type=int, default=256)
    p.add_argument("--y", type=int, default=256)
    p.add_argument("--x", type=int, default=256)
    p.add_argument("--voxel", type=float, nargs=3, default=[2.0, 2.0, 2.0],
                   metavar=("sz","sy","sx"), help="Voxel size in mm")
    p.add_argument("--out-act", default="activity.npy", help="Output path for activity volume")
    p.add_argument("--out-ct",  default="ctmu.npy",     help="Output path for CT mu-map")
    args = p.parse_args()

    matrix_size = (args.z, args.y, args.x)
    voxel_mm = tuple(args.voxel)
    preset = pet_nema_dict if args.preset == "pet" else earl_nema_dict

    act, ct = create_nema(matrix_size=matrix_size, voxel_size_mm=voxel_mm, nema_dict=preset)
    np.save(args.out_act, act)
    np.save(args.out_ct, ct)
    print(f"Saved:\n  {args.out_act}  (shape {act.shape}, dtype {act.dtype})\n  {args.out_ct}   (shape {ct.shape}, dtype {ct.dtype})")

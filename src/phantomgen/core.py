import numpy as np

def _normalize_supersample(factors):
    """
    Ensure supersampling factors are a length-3 tuple of positive integers.
    """
    if isinstance(factors, int):
        normalized = (factors, factors, factors)
    elif isinstance(factors, (tuple, list)):
        if len(factors) != 3:
            raise ValueError("Supersample sequence must have exactly three elements.")
        normalized = tuple(int(f) for f in factors)
    else:
        raise TypeError("Supersample must be an int or a sequence of three ints.")

    if any(f < 1 for f in normalized):
        raise ValueError("Supersample factors must be >= 1.")

    return normalized


def _downsample_volume(volume, factors, reduce="mean"):
    """
    Downsample a 3D volume by integer factors along each axis.
    """
    if volume.ndim != 3:
        raise ValueError("Expected a 3D volume to downsample.")

    zf, yf, xf = factors
    Z, Y, X = volume.shape
    if (Z % zf) or (Y % yf) or (X % xf):
        raise ValueError(
            f"Volume shape {volume.shape} is not divisible by factors {factors}."
        )

    reshaped = volume.reshape(Z // zf, zf, Y // yf, yf, X // xf, xf)
    if reduce == "mean":
        reduced = reshaped.mean(axis=(1, 3, 5))
    elif reduce == "sum":
        reduced = reshaped.sum(axis=(1, 3, 5))
    else:
        raise ValueError("reduce must be either 'mean' or 'sum'.")
    return reduced.astype(volume.dtype, copy=False)

# ==================================================================
# ===  HELPER: Generate centered world coordinates for a volume  ===
# ==================================================================
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
# ===  BASIC GEOMETRIC PRIMITIVES (CYLINDER, BOX, SPHERE)     ===
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
    if volume.ndim != 3:
        raise ValueError("Expected a 3D volume.")
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
    if volume.ndim != 3:
        raise ValueError("Expected a 3D volume.")
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
    if volume.ndim != 3:
        raise ValueError("Expected a 3D volume.")
    Zmm, Ymm, Xmm = _world_coords(volume.shape, voxel_size_mm, center_mm)
    inside = (Xmm**2 + Ymm**2 + Zmm**2) <= radius_mm**2
    volume[inside] = value
    return volume


# ===============================================================
# ===  NEMA NU 2 / IEC BODY PHANTOM CREATOR                   ===
# ===============================================================
def create_nema(
    matrix_size=(256, 256, 256),              # (Z,Y,X)
    voxel_size_mm=(2.0, 2.0, 2.0),
    nema_dict=None,
    center_offset_mm=None,
    supersample=1,
):
    """
    Create a 3D numerical phantom matching the IEC/NEMA IQ body phantom geometry.

    Generates:
      • `act_vol`               – activity map (MBq/voxel)
      • `ctac_vol`              – attenuation map (cm⁻¹)
      • `mask_vols_binary_dict` – dictionary of binary mask volumes

    Parameters
    ----------
    matrix_size : tuple (Z, Y, X)
        Number of voxels along each dimension.
    voxel_size_mm : tuple (sz, sy, sx)
        Physical voxel size in mm.
    nema_dict : dict
        Dictionary defining all phantom parameters.
    center_offset_mm : tuple (cz, cy, cx) or None
        Optional explicit global offset applied to every primitive (mm).
        Overrides any value present in `nema_dict` when provided.
    supersample : int or tuple(int, int, int)
        Supersampling factor along (Z, Y, X). The phantom is generated on a higher
        resolution grid and downsampled back to `matrix_size`. Activity values are
        summed during downsampling to conserve total activity; attenuation values
        are averaged. Binary masks are downsampled conservatively, i.e. a coarse
        voxel is set to 1 if any supersampled subvoxel belongs to the region.

    Returns
    -------
    act_vol : np.ndarray
        Activity volume.
    ctac_vol : np.ndarray
        CT attenuation volume.
    mask_vols_binary_dict : dict[str, np.ndarray]
        Dictionary of binary mask volumes.
    """

    supersample = _normalize_supersample(supersample)
    use_supersample = supersample != (1, 1, 1)

    matrix_size = tuple(int(m) for m in matrix_size)
    voxel_size_mm = tuple(float(v) for v in voxel_size_mm)

    if use_supersample:
        working_matrix = tuple(int(m * f) for m, f in zip(matrix_size, supersample))
        working_voxel = tuple(float(v) / f for v, f in zip(voxel_size_mm, supersample))
    else:
        working_matrix = matrix_size
        working_voxel = voxel_size_mm

    # --- defaults ---
    defaults = {
        "mu_values": {
            "perspex_mu_value": 0.1,
            "fill_mu_value": 0.096,
            "lung_mu_value": 0.029
        },
        "activity_concentration_background": 0.05,
        "include_lung_insert": True,
        "center_offset_mm": (0.0, 0.0, 0.0),
        "sphere_dict": {
            "ring_R": 57,
            "ring_z": -37,
            "spheres": {
                "diametre_mm": [10, 13, 17, 22, 28, 37],
                "angle_loc":   [30, 90, 150, 210, 270, 330],
                "act_conc_MBq_ml": [0.00, 0.00, 0.4, 0.4, 0.4, 0.4],
            }
        }
    }

    # --- merge user input with defaults ---
    if nema_dict is None:
        nema_dict = defaults
    else:
        import copy

        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d.setdefault(k, v)
            return d

        nema_dict = deep_update(copy.deepcopy(nema_dict), defaults)

    # --- global offset ---
    default_offset = defaults.get("center_offset_mm", (0.0, 0.0, 0.0))
    dict_offset = nema_dict.get("center_offset_mm", default_offset)
    if center_offset_mm is None:
        center_offset_mm = dict_offset

    offset_z, offset_y, offset_x = map(float, center_offset_mm)

    def with_offset(center):
        cz, cy, cx = center
        return (cz + offset_z, cy + offset_y, cx + offset_x)

    # --- unpack parameters ---
    act_conc_backgr = nema_dict["activity_concentration_background"]
    fill_mu_value = nema_dict["mu_values"]["fill_mu_value"]
    perspex_mu_value = nema_dict["mu_values"]["perspex_mu_value"]
    lung_insert = nema_dict["include_lung_insert"]
    lung_mu_value = nema_dict["mu_values"]["lung_mu_value"]

    sphere_info = nema_dict["sphere_dict"]
    ring_R = sphere_info["ring_R"]
    z_pos = sphere_info["ring_z"]

    # --- sphere data validation ---
    sphere_params = sphere_info["spheres"]
    n_d = len(sphere_params["diametre_mm"])
    n_a = len(sphere_params["angle_loc"])
    n_c = len(sphere_params["act_conc_MBq_ml"])
    if not (n_d == n_a == n_c):
        raise ValueError(
            "sphere_dict['spheres'] entries 'diametre_mm', 'angle_loc', and "
            "'act_conc_MBq_ml' must have the same length."
        )

    # --- volume size check ---
    vol_dim = [m * v for m, v in zip(matrix_size, voxel_size_mm)]
    if any(v < lim for v, lim in zip(vol_dim, (220, 300, 230))):
        raise ValueError(
            f"Volume dimensions in mm {tuple(vol_dim)} are smaller than required "
            f"NEMA phantom dimensions {(220, 300, 230)}."
        )

    ctac_vol = np.zeros(working_matrix, np.float32)
    act_vol = np.zeros(working_matrix, np.float32)

    # high-resolution masks used to derive binary masks
    background_mask_hr = np.zeros(working_matrix, np.uint8)
    lung_mask_hr = np.zeros(working_matrix, np.uint8)
    sphere_masks_hr = {}

    ml_per_vox = np.prod(working_voxel) / 1000.0
    back_MBq_per_vox = act_conc_backgr * ml_per_vox

    # --- connecting box ---
    add_box(
        ctac_vol,
        working_voxel,
        (220, 75, 150),
        with_offset((0, 72.5, 0)),
        value=perspex_mu_value,
    )
    add_box(
        ctac_vol,
        working_voxel,
        (214, 72, 150),
        with_offset((0, 71, 0)),
        value=fill_mu_value,
    )
    add_box(
        act_vol,
        working_voxel,
        (214, 72, 150),
        with_offset((0, 71, 0)),
        value=back_MBq_per_vox,
    )
    add_box(
        background_mask_hr,
        working_voxel,
        (214, 72, 150),
        with_offset((0, 71, 0)),
        value=1,
    )

    # --- tank structure ---
    tanks = [
        dict(r=150, h=220, deg=(180, 360), c=(0, 35, 0),   mu="perspex"),
        dict(r=147, h=214, deg=(180, 360), c=(0, 35, 0),   mu="fill"),
        dict(r=75,  h=220, deg=(90, 180),  c=(0, 35, -75), mu="perspex"),
        dict(r=72,  h=214, deg=(90, 180),  c=(0, 35, -75), mu="fill"),
        dict(r=75,  h=220, deg=(0, 90),    c=(0, 35, 75),  mu="perspex"),
        dict(r=72,  h=214, deg=(0, 90),    c=(0, 35, 75),  mu="fill"),
    ]
    if lung_insert:
        tanks.append(dict(r=25, h=214, deg=None, c=(0, 0, 0), mu="lung"))

    for t in tanks:
        if t["mu"] == "perspex":
            add_cylinder(
                ctac_vol, working_voxel, t["r"], t["h"], t["deg"],
                with_offset(t["c"]), perspex_mu_value
            )
        elif t["mu"] == "fill":
            add_cylinder(
                ctac_vol, working_voxel, t["r"], t["h"], t["deg"],
                with_offset(t["c"]), fill_mu_value
            )
            add_cylinder(
                act_vol, working_voxel, t["r"], t["h"], t["deg"],
                with_offset(t["c"]), back_MBq_per_vox
            )
            add_cylinder(
                background_mask_hr, working_voxel, t["r"], t["h"], t["deg"],
                with_offset(t["c"]), 1
            )
        else:
            add_cylinder(
                ctac_vol, working_voxel, t["r"], t["h"], t["deg"],
                with_offset(t["c"]), lung_mu_value
            )
            add_cylinder(
                act_vol, working_voxel, t["r"], t["h"], t["deg"],
                with_offset(t["c"]), 0
            )
            add_cylinder(
                lung_mask_hr, working_voxel, t["r"], t["h"], t["deg"],
                with_offset(t["c"]), 1
            )

    # --- add spheres ---
    for i, (d, a, c) in enumerate(
        zip(
            sphere_info["spheres"]["diametre_mm"],
            sphere_info["spheres"]["angle_loc"],
            sphere_info["spheres"]["act_conc_MBq_ml"],
        ),
        start=1,
    ):
        r_shell = (d + 2) / 2.0
        r_interior = d / 2.0
        ang = np.deg2rad(a)
        cy, cx = -ring_R * np.sin(ang), ring_R * np.cos(ang)
        center = with_offset((z_pos, cy, cx))
        fill_act = c * ml_per_vox

        add_sphere(ctac_vol, working_voxel, r_shell, center, value=perspex_mu_value)
        add_sphere(ctac_vol, working_voxel, r_interior, center, value=fill_mu_value)
        add_sphere(act_vol, working_voxel, r_interior, center, value=fill_act)

        sphere_exclusion_mask = np.zeros(working_matrix, np.uint8)
        add_sphere(sphere_exclusion_mask, working_voxel, r_shell, center, value=1)
        
        sphere_mask = np.zeros(working_matrix, np.uint8)
        add_sphere(sphere_mask, working_voxel, r_interior, center, value=1)
        
        sphere_masks_hr[f"sphere_{i}"] = sphere_mask

        # full sphere assembly (shell + interior) is not background
        background_mask_hr[sphere_exclusion_mask > 0] = 0

    # lung insert is not background
    background_mask_hr[lung_mask_hr > 0] = 0

    # --- downsample if supersampled ---
    if use_supersample:
        act_vol = _downsample_volume(act_vol, supersample, reduce="sum")
        ctac_vol = _downsample_volume(ctac_vol, supersample, reduce="mean")

        background_mask = (
            _downsample_volume(background_mask_hr.astype(np.float32), supersample, reduce="mean") > 0.0
        ).astype(np.uint8)

        lung_mask = (
            _downsample_volume(lung_mask_hr.astype(np.float32), supersample, reduce="mean") > 0.0
        ).astype(np.uint8)

        sphere_masks = {
            k: (
                _downsample_volume(v.astype(np.float32), supersample, reduce="mean") > 0.0
            ).astype(np.uint8)
            for k, v in sphere_masks_hr.items()
        }
    else:
        background_mask = background_mask_hr
        lung_mask = lung_mask_hr
        sphere_masks = sphere_masks_hr

    mask_vols_binary_dict = {
        "background": background_mask,
        "lung_insert": lung_mask,
        **sphere_masks,
    }

    return act_vol, ctac_vol, mask_vols_binary_dict


earl_nema_dict = {
    "mu_values":{
        "perspex_mu_value": 0.15,
        "fill_mu_value": 0.14,
        "lung_mu_value": 0.043
    },
    "activity_concentration_background": 0.0,
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



def cli():
    """
    CLI to write activity, CTAC, and optionally mask volumes to .npy files.

    Examples:
      phantomgen --preset pet --out-act act.npy --out-ctac ctac.npy
      phantomgen --preset earl --z 256 --y 256 --x 256 --voxel 2 2 2
      phantomgen --preset pet --out-mask-dir masks/
    """
    import argparse
    import os

    p = argparse.ArgumentParser(prog="phantomgen", description="Generate IEC/NEMA phantoms")
    p.add_argument("--preset", choices=["pet", "earl"], default="pet", help="Parameter set")
    p.add_argument("--z", type=int, default=256)
    p.add_argument("--y", type=int, default=256)
    p.add_argument("--x", type=int, default=256)
    p.add_argument(
        "--voxel",
        type=float,
        nargs=3,
        default=[2.0, 2.0, 2.0],
        metavar=("sz", "sy", "sx"),
        help="Voxel size in mm",
    )
    p.add_argument("--out-act", default="activity.npy", help="Output path for activity volume")
    p.add_argument("--out-ctac", default="ctmu.npy", help="Output path for CT attenuation map")
    p.add_argument(
        "--out-mask-dir",
        default=None,
        help="Optional output directory for binary masks. If provided, all masks are saved as separate .npy files.",
    )
    p.add_argument(
        "--offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("cz", "cy", "cx"),
        help="Global offset (mm) applied to every primitive",
    )
    p.add_argument(
        "--supersample",
        type=int,
        nargs="+",
        default=[1],
        metavar="factor",
        help="Supersampling factor: provide one value for isotropic or three values for (z y x).",
    )
    args = p.parse_args()

    matrix_size = (args.z, args.y, args.x)
    voxel_mm = tuple(args.voxel)
    preset = pet_nema_dict if args.preset == "pet" else earl_nema_dict

    if len(args.supersample) == 1:
        supersample = args.supersample[0]
    elif len(args.supersample) == 3:
        supersample = tuple(args.supersample)
    else:
        p.error("--supersample expects one value or three values (z y x).")

    offset_mm = tuple(args.offset)
    act, ctac, mask_bin_dict = create_nema(
        matrix_size=matrix_size,
        voxel_size_mm=voxel_mm,
        nema_dict=preset,
        center_offset_mm=offset_mm,
        supersample=supersample,
    )

    np.save(args.out_act, act)
    np.save(args.out_ctac, ctac)

    saved_mask_paths = []
    if args.out_mask_dir is not None:
        os.makedirs(args.out_mask_dir, exist_ok=True)
        for mask_name, mask_vol in mask_bin_dict.items():
            mask_path = os.path.join(args.out_mask_dir, f"{mask_name}.npy")
            np.save(mask_path, mask_vol)
            saved_mask_paths.append(mask_path)

    print(
        f"Saved:\n"
        f"  {args.out_act}   (shape {act.shape}, dtype {act.dtype})\n"
        f"  {args.out_ctac}  (shape {ctac.shape}, dtype {ctac.dtype})"
    )

    if saved_mask_paths:
        print("  Masks:")
        for mask_path in saved_mask_paths:
            print(f"    {mask_path}")
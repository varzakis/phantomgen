# phantomgen 🧩  
**Virtual NEMA / IEC Nuclear Medicine Phantom Generator**

`phantomgen` is a lightweight Python package that generates realistic 3D numerical phantoms for Nuclear Medicine imaging simulations and dosimetry research.  
It implements the IEC Body Phantom (NEMA NU-2 / IQ) geometry and provides parameter sets matching **EARL** and **PET** standard configurations.

---

## ✨ Features
- Generate 3D **activity maps** and **CT μ-maps** with realistic geometry  
- Supports **spherical inserts**, and **lung inserts**
- Adjustable **voxel size** and **matrix dimensions**  
- Ready-made parameter dictionaries for **PET** and **EARL** configurations  
- Pure **NumPy** implementation (no heavy dependencies)

---

## 🧠 Background
The NEMA (National Electrical Manufacturers Association) IQ Body Phantom is a standardised test object used for quality control and performance evaluation in PET and SPECT systems.  
`phantomgen` provides a digital version of this phantom, suitable for:
- Monte Carlo simulations (e.g., GATE, SIMIND, STIR)
- Deep-learning–based dosimetry and image reconstruction
- Algorithm testing and reproducibility studies

---

## 🧩 Installation

```bash
# From source
git clone https://github.com/yourname/phantomgen.git
cd phantomgen
pip install -e .
```

Requires **Python ≥ 3.9** and **NumPy ≥ 1.23**.

---

## 🚀 Usage

### Python API
```python
import numpy as np
from phantomgen import create_nema, pet_nema_dict, earl_nema_dict

# Create a PET phantom (256³ voxels, 2 mm voxel size)
act_vol, ct_vol = create_nema(
    matrix_size=(256, 256, 256),
    voxel_size_mm=(2.0, 2.0, 2.0),
    nema_dict=pet_nema_dict
)

np.save("activity.npy", act_vol)
np.save("ctmu.npy", ct_vol)
```

### Command Line Interface
After installation, run from the terminal:

```bash
phantomgen --preset pet --out-act act.npy --out-ct ct.npy
```

Optional arguments:

| Argument | Default | Description |
|-----------|----------|-------------|
| `--preset` | `pet` | Choose between `pet` and `earl` presets |
| `--z --y --x` | `256 256 256` | Matrix size (voxels) |
| `--voxel` | `2 2 2` | Voxel size (mm) |
| `--out-act` | `activity.npy` | Output path for activity map |
| `--out-ct` | `ctmu.npy` | Output path for CT μ-map |

Example:
```bash
phantomgen --preset earl --z 192 --y 192 --x 192 --voxel 2.5 2.5 2.5
```

---

## 📦 Outputs

| Volume | Symbol | Unit | Description |
|---------|---------|------|-------------|
| Activity map | `act_vol` | MBq/voxel | Radiotracer activity distribution |
| CT μ-map | `ct_vol` | cm⁻¹ | Attenuation coefficients for reconstruction |

Both are returned as 3D NumPy arrays `(Z, Y, X)` in `float32` format.

---

## ⚙️ Parameters

Both default configurations (`pet_nema_dict`, `earl_nema_dict`) can be customized before generation.

Example:
```python
from phantomgen import create_nema, pet_nema_dict

custom = pet_nema_dict.copy()
custom["sphere_dict"]["spheres"]["act_conc_MBq_ml"] = [1.0]*6
act, ct = create_nema(nema_dict=custom)
```

---

## 🧪 Testing

A minimal test suite is included. Run it with:
```bash
pytest -q
```

---

## 🧰 License
MIT License © 2025 — Your Name  
This software is free and open source.

---

## 💡 Example Visualization
```python
import matplotlib.pyplot as plt
from phantomgen import create_nema, pet_nema_dict

act, ct = create_nema(nema_dict=pet_nema_dict)
z = act.shape[0] // 2

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(act[z], cmap="inferno")
ax[0].set_title("Activity (MBq/voxel)")
ax[1].imshow(ct[z], cmap="gray")
ax[1].set_title("CT μ-map (cm⁻¹)")
plt.show()
```

---

## 📁 Project structure
```
phantomgen/
├─ src/phantomgen/core.py       # Core implementation
├─ src/phantomgen/__init__.py   # Public API
├─ tests/test_basic.py          # Minimal tests
├─ pyproject.toml
└─ README.md
```

---

Enjoy generating your virtual NEMA phantoms 💫

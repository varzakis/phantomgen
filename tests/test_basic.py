import numpy as np
from phantomgen.core import create_nema

def test_create_nema_shapes():
    act, ct, masks = create_nema(
        matrix_size=(256,256,256),
        voxel_size_mm=(2,2,2)
    )
    assert act.shape == ct.shape
    assert isinstance(masks, dict)

def test_mask_keys():
    act, ct, masks = create_nema(
        matrix_size=(256,256,256),
        voxel_size_mm=(2,2,2)
    )
    assert "background" in masks
    assert any(k.startswith("sphere_") for k in masks)

def test_supersample():
    act, ct, masks = create_nema(
        matrix_size=(128,128,128),
        voxel_size_mm=(4,4,4),
        supersample=2
    )
    assert act.shape == (128,128,128)

def test_invalid_sphere_lengths():
    bad = {
        "sphere_dict":{
            "ring_R":57,
            "ring_z":-37,
            "spheres":{
                "diametre_mm":[10,13],
                "angle_loc":[30],
                "act_conc_MBq_ml":[1,1]
            }
        }
    }
    import pytest
    with pytest.raises(ValueError):
        create_nema(matrix_size=(256,256,256), voxel_size_mm=(2,2,2), nema_dict=bad)

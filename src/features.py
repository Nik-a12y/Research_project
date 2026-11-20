"""Step 3 & 4: baseline features and topology features."""
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, euler_number

K8 = np.ones((3, 3), int)

def baseline_features(img, mask):
    """Appearance-only features. NO connectivity / branching information."""
    f = [img.mean(), img.std(), mask.mean()]                       # intensity + vessel density
    f += list(np.histogram(img, bins=8, range=(0, 1))[0] / img.size)
    h, w = img.shape
    g = img[: h // 8 * 8, : w // 8 * 8].reshape(8, h // 8, 8, w // 8).mean((1, 3))
    f += list(g.ravel())                                           # coarse 8x8 grid
    return np.array(f, float)

BASE_NAMES = (["int_mean", "int_std", "vessel_density"] +
              [f"hist{i}" for i in range(8)] + [f"grid{i}" for i in range(64)])

def topology_features(mask, skel):
    """Network-structure features from the skeleton."""
    nb = ndi.convolve(skel.astype(int), K8, mode="constant") - 1
    nb = nb * skel
    endpoints = int(((nb == 1) & skel).sum())
    junction_px = (nb >= 3) & skel
    branchpoints = int(label(junction_px, connectivity=2).max())   # cluster junction pixels
    length = int(skel.sum())
    lab = label(mask, connectivity=2)
    n_comp = int(lab.max())
    sizes = np.bincount(lab.ravel())[1:] if n_comp else np.array([0])
    largest_frac = sizes.max() / max(sizes.sum(), 1)
    seg_count = max(branchpoints + endpoints / 2, 1)
    dist_bg = ndi.distance_transform_edt(~mask)
    return np.array([
        length / mask.size,                  # skeleton density
        endpoints / mask.size * 1e3,
        branchpoints / mask.size * 1e3,
        n_comp,                              # fragmentation (broken vessels)
        largest_frac,                        # connectedness
        sizes.mean(),
        euler_number(mask, connectivity=2),  # components minus holes
        length / seg_count,                  # mean segment length
        dist_bg.max(),                       # largest avascular gap
        (dist_bg > 5).mean(),                # fraction of area far from any vessel
    ], float)

TOPO_NAMES = ["skel_density", "endpoints_k", "branchpoints_k", "n_components",
              "largest_comp_frac", "mean_comp_size", "euler_number",
              "mean_segment_len", "max_avascular_gap", "frac_far_from_vessel"]

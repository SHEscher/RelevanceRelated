"""
Prune 3D image (reduce size): Remove black/zero surrounding of brain.

Effectively works on brain-masked T1 data.
Find the smallest cube (!) in the whole data set which can surround each brain in it.
That is, the cube of the 'biggest' brain.

Author: Simon M. Hofmann | 2020
"""

# %% Import
from __future__ import annotations

import contextlib
import sys

import numpy as np

from relevancerelated.utils import cprint, normalize

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def reverse_pruning(
    original_mri: np.ndarray,
    pruned_mri: np.ndarray,
    pruned_stats_map: np.ndarray | None = None,
) -> np.ndarray:
    """Reverse the pruning of a given pruned MRI."""
    cube2unprune = pruned_mri if pruned_stats_map is None else pruned_stats_map

    fill_cube = np.zeros(shape=original_mri.shape)
    fill_cube[...] = pruned_mri.min() if pruned_stats_map is None else 0.0  # set background

    # Find the edges of the brain (slice format)
    org_edge_slices = find_edges(x3d=original_mri, sl=True)
    prun_edge_slices = find_edges(x3d=pruned_mri, sl=True)

    # Use the edges to place the brain data at the right spot
    fill_cube[org_edge_slices] = cube2unprune[prun_edge_slices]

    return fill_cube


def prune_mri(
    x3d: np.ndarray, make_cube: bool = False, max_axis: list | np.ndarray | tuple | int | None = None, padding: int = 0
) -> tuple[str, np.ndarray] | np.ndarray:
    """
    Prune given 3D T1-image to (smaller) volume.

    With side-length(s) == max_axis [int OR 3D tuple].
    If max_axis is None, it finds the smallest cube/volume which covers the brain
    (i.e., removes zero-padding - 2-voxel-margin)
    Works very fast. [np.pad implementation possible, too]

    Compare to: *nilearn.image.crop_img() for NifTis:
        * This crops only exactly along the brain
        * which is the same as: mri[find_edges(mri, sl=True)]

    :param x3d: 3D T1 image
    :param max_axis: 1D-int: side-length of pruned cube; 3D-list[int]: side-length of pruned version
    :param make_cube: True: pruned MRI will be a cube;
    :param padding: int: number of zero-padding layers around brain
    :return: pruned volume
    """
    # Check argument:
    if (make_cube and max_axis is not None) and not isinstance(max_axis, int):
        msg = "If target volume suppose to be a cube, 'max_axis' must be int!"
        raise ValueError(msg)
    if (not make_cube) and (max_axis is not None) and len(max_axis) != 3:  # noqa: PLR2004
        msg = "If target volume suppose to be no cube, 'max_axis' 3D-shape tuple!"
        raise ValueError(msg)

    # Check x3d-format: (sic, brain) OR brain
    tp = isinstance(x3d, tuple)
    sic = x3d[0] if tp else None
    x3d = x3d[1] if tp else x3d

    if isinstance(x3d, np.ndarray):
        # Cut out
        x3d_minimal = x3d[find_edges(x3d, sl=True)]

        # Prune to smaller volume
        if max_axis is None:
            # find the longest axis for cubing [int] OR take the shape of the minimal volume [3D-tuple]
            max_axis = np.max(x3d_minimal.shape) if make_cube else np.array(x3d_minimal.shape)

        # Do padding at borders (if wanted) & make max_axis a 3D shape-tuple/list
        max_axis = [max_axis + padding] * 3 if make_cube else np.array(max_axis) + padding

        # Initialize an empty 3D target volume
        x3d_small_vol = np.zeros(max_axis, dtype=x3d.dtype)
        bg = 0.0
        if x3d.min() != bg:
            x3d_small_vol[x3d_small_vol == 0] = x3d.min()  # in case background is e.g. -1

        # Place brain in the middle of cube
        a1 = (max_axis[0] - x3d_minimal.shape[0]) // 2  # half of the rest
        a2 = (max_axis[1] - x3d_minimal.shape[1]) // 2
        a3 = (max_axis[2] - x3d_minimal.shape[2]) // 2

        x3d_small_vol[
            a1 : a1 + x3d_minimal.shape[0], a2 : a2 + x3d_minimal.shape[1], a3 : a3 + x3d_minimal.shape[2]
        ] = x3d_minimal

    else:
        x3d_small_vol = None

    if tp:
        return sic, x3d_small_vol
    return x3d_small_vol


def find_edges(x3d: np.ndarray, sl: bool = False) -> tuple[int]:
    """Find edges that encapsule the brain."""
    bg = x3d.min()  # usually: 0

    # # Find planes with first brain-data (i.e., being not black)
    # Find 'lower' planes (i.e., low, left, back, respectively)
    il, jl, kl = 0, 0, 0
    while np.all(x3d[il, :, :] == bg):  # sagittal slide
        il += 1
    while np.all(x3d[:, jl, :] == bg):  # transverse slide
        jl += 1
    while np.all(x3d[:, :, kl] == bg):  # coronal/posterior/frontal
        kl += 1

    # Find 'upper' planes (i.e., upper, right, front, respectively)
    iu, ju, ku = np.array(x3d.shape) - 1
    while np.all(x3d[iu, :, :] == bg):  # sagittal/longitudinal
        iu -= 1
    while np.all(x3d[:, ju, :] == bg):  # transverse/inferior/horizontal
        ju -= 1
    while np.all(x3d[:, :, ku] == bg):  # coronal/posterior/frontal
        ku -= 1

    if sl:  # return slices
        return slice(il, iu + 1), slice(jl, ju + 1), slice(kl, ku + 1)

    return il, iu, jl, ju, kl, ku


def get_brain_axes_length(x3d: np.ndarray | tuple[str, np.ndarray]) -> list[int] | tuple[str, list[int]]:
    """Get max length of each brain axis."""
    tp = False
    sic = None
    if isinstance(x3d, tuple):  # (sic, brain)
        tp = True
        sic = x3d[0]
        x3d = x3d[1]

    il, iu, jl, ju, kl, ku = find_edges(x3d)
    axes_lengths = [iu + 1 - il, ju + 1 - jl, ku + 1 - kl]

    if tp:
        return sic, axes_lengths
    return axes_lengths


def max_of_axes(x3d: np.ndarray | tuple[str, np.ndarray]) -> int | tuple[str, int]:
    """Get max of the image axes."""
    tp = False
    sic = None
    if isinstance(x3d, tuple):  # (sic, brain)
        tp = True
        sic = x3d[0]
        x3d = x3d[1]

    longest_axis = np.max(get_brain_axes_length(x3d)) if isinstance(x3d, np.ndarray) else np.nan

    if tp:
        return sic, longest_axis
    return longest_axis


def get_global_max_axis(
    space: str, mri_sequence: str | None = None, per_axis: bool = False, padding: bool = True
) -> int | np.ndarray[int]:
    """
    Find global max axis-length.

    The max axis-length, as spanned cube / box, surrounds each brain in LIFE MRI set,
    while removes unnecessary zero-padding.

    :param space: "native"/"raw" OR "t1"/"FreeSurfer" OR "mni" - space
    :param mri_sequence: "t1", "swi", "flair" [so far], None for non-native/raw space
    :param per_axis: False: max(max(axes)); True: return global-max(axes) per axis (3D)
    :param padding: True: add space-specific padding
    :return: global_max
    """
    space = space.lower()
    if space not in {"native", "raw", "t1", "freesurfer", "fs", "mni"}:
        msg = f"Given space '{space}' not known!"
        raise ValueError(msg)

    if space == "mni":
        global_max = np.array([72, 78, 90]) if per_axis else 90
        if padding:
            global_max += [8, 2, 8] if per_axis else 8

    elif space in {"t1", "freesurfer", "fs"}:
        global_max = np.array([148, 158, 197]) if per_axis else 197
        if padding:
            global_max += [2, 2, 1] if per_axis else 1

    else:
        if per_axis:
            global_maxes = {
                "t1": np.array([162, 170, 196]) + ([1, 2, 2] if padding else [0, 0, 0]),
                "flair": np.array([162, 347, 401]) + ([1, 1, 1] if padding else [0, 0, 0]),
                "swi": np.array([292, 72, 352]) + ([1, 2, 1] if padding else [0, 0, 0]),
            }
        else:
            # NOTE for both FLAIR & SWI pruning/cubing is not useful: raw.size < cube.size
            global_maxes = {
                "t1": 196 + (2 if padding else 0),
                "flair": 401 + (1 if padding else 0),
                "swi": 352 + (1 if padding else 0),
            }

        global_max = global_maxes[mri_sequence]

    return global_max


def permute_mri(xd: tuple | np.ndarray) -> np.ndarray:
    """
    Swap all entries (e.g., voxels) in given x-dimensional shape (e.g., 3D-MRI scan).

    Could be also 2D-brain-slice.

    :param xd: x-dimensional array
    :return: permuted array
    """
    tp = False
    _sic = None
    if isinstance(xd, tuple):  # (sic, brain)
        tp = True
        _sic = xd[0]
        xd = xd[1]

    _flat_mri = xd.flatten()
    np.random.shuffle(_flat_mri)
    xd_permuted = _flat_mri.reshape(xd.shape)

    if tp:
        return _sic, xd_permuted
    return xd_permuted


def compress_mri(x3d: np.ndarray, space: str, mri_sequence: str, verbose: bool = False) -> np.ndarray:
    """Compress an MRI."""
    # Set upper intensity bound
    abs_max = 255 + 2**7 if x3d.max() > 1.0 else 1.0
    # 383: max intensity ceiling [CHECKED for SWI & FLAIR]

    x3d_memory = sys.getsizeof(x3d)

    # Prune to global standard
    x3d = prune_mri(
        x3d=x3d, make_cube=False, max_axis=get_global_max_axis(space=space, mri_sequence=mri_sequence, per_axis=True)
    )

    x3d = np.round(  # round OR equal
        normalize(  # normalize [0-255]
            array=np.minimum(  # Set upper intensity boundary
                x3d, abs_max
            ),
            lower_bound=0.0,
            upper_bound=255.0,
            global_max=abs_max,
        )
    ).astype(np.uint8)  # Change data-type

    if verbose:
        cprint(
            string=f"The file size shrank to {sys.getsizeof(x3d) / 10 ** (3 * 2):.2f} MB "
            f"(from initial {x3d_memory / 10 ** (3 * 2):.2f} MB).",
            col="b",
        )

    return x3d


def get_mriset_memory_size(mri_set, in_="gb", approx=True):
    """Get the memory size of an MRI set."""
    # Infer which
    in_ = in_.lower()
    p = 3 if in_ == "gb" else 2 if in_ == "mb" else 1 if in_ == "kb" else 0

    mri_size = sys.getsizeof(mri_set)  # in bytes
    if approx:
        mri_size += len(mri_set) * sys.getsizeof(next(iter(mri_set.values())))
    else:
        for ksic in mri_set:
            mri_size += sys.getsizeof(mri_set[ksic])

    mri_size /= 10 ** (3 * p)

    cprint(f"The size of the given MRIset is{' approx.' if approx else ''}: {mri_size:.3f} {in_.upper()}", col="b")


def set_dtype_mriset(mri_set, dtype=np.uint8):
    """Set dtype for an MRI set."""
    if not isinstance(mri_set, dict):
        msg = "mri_set must be dictionary (SIC: mri)-pairs."
        raise TypeError(msg)
    ctn = 0
    while True:
        ctn += 1
        sic = np.random.choice(list(mri_set.keys()))  # take random SIC (key)
        if mri_set[sic] is not None:
            break

        if ctn > len(mri_set):
            cprint(string="No data for nearly every SIC in given MRI set found! Won't change dtype!", col="r")
            return mri_set

    swap = False
    if mri_set[sic].dtype != dtype:
        swap = True

        fct = np.round if "int" in dtype.__name__ else lambda x: x  # round for ints else identity funct

        if (mri_set[sic].dtype in {float, np.float16}) and "int" in dtype.__name__:
            cprint(
                f"Data will be compressed from float -> {dtype.__name__}. This might cause information loss",
                col="y",
            )

        # Check data normalization:
        #   if [0, 1] then re-norm to [0, 255] before saving (=suffic. resol)
        #   if max-intensity values too big (>>255) then apply the ceiling to the upper bound at 383
        max_set = np.nanmax(list(mri_set.values()))
        abs_max = 255 + 2**7  # 383: max intensity ceiling [CHECKED for SWI (max: 998.079) & FLAIR]

        if max_set < 255 * 4:
            # Do ceil the range of dataset
            cprint(
                string=f"Max intensity value of given MRI-set is {max_set:.3f}. "
                f"Global max boundary is set to {np.minimum(max_set, abs_max)}.",
                col="y",
            )

        else:
            msg = (
                "Given MRI dataset has unanticipated intensity distribution. Revisit this function to solve the issue."
            )
            raise ValueError(msg)

        # Get memory size
        cprint(f"Before converting MRIset to dtype '{dtype.__name__}' ...", "b")
        get_mriset_memory_size(mri_set=mri_set)

        # Adapt SIC by SIC
        for sic in mri_set:
            with contextlib.suppress(AttributeError):
                mri_set[sic] = fct(  # round OR equal
                    normalize(  # normalize [0-255]
                        array=np.minimum(  # Set upper intensity boundary
                            mri_set[sic], abs_max
                        ),
                        lower_bound=0.0,
                        upper_bound=255.0,
                        global_max=np.minimum(max_set, abs_max),
                    )
                ).astype(dtype)  # Change data-type
    cprint(string=f"dtype of given MRI set {'is now' if swap else 'was already'}: {mri_set[sic].dtype}", col="y")

    # cprint(f"After converting MRIset to dtype '{dtype.__name__}' ...", 'b')  # noqa: ERA001
    get_mriset_memory_size(mri_set=mri_set)

    return mri_set


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END

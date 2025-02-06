"""
Adaptation from apply_heatmap.py by Sebastian L. & Leander Weber (FH HHI).

Author: Simon M. Hofmann | 2021
"""

# %% Import
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

from relevancerelated.utils import cprint, normalize

# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<


def produce_supported_maps():
    """Generate a list of supported maps."""
    # return a list of names and extreme color values.
    print(*(list(custom_maps.keys()) + matplotlib_maps), sep="\n")
    return list(custom_maps.keys()) + matplotlib_maps


def colorize_matplotlib(R: np.ndarray, cmapname: str):  # noqa: N803, ANN201
    """Colorize matplotlib."""
    # fetch color mapping function by string
    cmap = cm.__dict__[cmapname]

    # bring data to [-1 1]
    R /= np.max(np.abs(R))  # noqa: N806

    # push data to [0 1] to avoid automatic color map normalization
    R = (R + 1) / 2  # noqa: N806

    sh = R.shape

    return cmap(R.flatten())[:, 0:3].reshape([*sh, 3])


# %% Functions to create colored heatmap << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<


def gregoire_black_firered(R):  # noqa: N803
    """Gregoire black firered."""
    # normalize to [-1, 1] for Real numbers, or [0, 1] for R+, where zero remains zero:
    R /= np.max(np.abs(R))  # noqa: N806
    x = R

    hrp = np.clip(x - 0.00, 0, 0.25) / 0.25  # all pos. values(+) above 0 get red, above .25 full red(=1.)
    hgp = np.clip(x - 0.25, 0, 0.25) / 0.25  # all above .25 get green, above .50 full green
    hbp = np.clip(x - 0.50, 0, 0.50) / 0.50  # all above .50 get blue until full blue at 1. (mix 2 white)

    hbn = np.clip(-x - 0.00, 0, 0.25) / 0.25  # all neg. values(-) below 0 get blue ...
    hgn = np.clip(-x - 0.25, 0, 0.25) / 0.25  # ... green ....
    hrn = np.clip(-x - 0.50, 0, 0.50) / 0.50  # ... red ... mixes to white (1.,1.,1.)

    return np.concatenate([(hrp + hrn)[..., None], (hgp + hgn)[..., None], (hbp + hbn)[..., None]], axis=x.ndim)


def create_cmap(color_fct, res=4999):
    """
    Create cmap for given color-function.

    Create cmap similar to gregoire_black_firered, which can be used for color-bars and other purposes.

    The function creates a color-dict in the following form and
    feeds it to matplotlib.colors.LinearSegmentedColormap

    cdict_gregoire_black_firered = {
        "red": [
            [0., 1., 1.],
            [.25, 0., 0.],
            [.5, 0., 0.],
            [.625, 1., 1.],
            [1., 1., 1.]
        ],
        "green": [
            [0., 1., 1.],
            [.25, 1., 1.],
            [.375, .0, .0],
            [.5, 0., 0.],
            [.625, .0, .0],
            [.75, 1., 1.],
            [1., 1., 1.]
        ],
        "blue": [
            [0., 1., 1.],
            [.375, 1., 1.],
            [.5, 0., 0.],
            [.75, 0., 0.],
            [1., 1., 1.]
        ]
    }
    """
    # Prep resolution (res):
    if not float(res).is_integer():
        msg = "'res' must be a positive natural number."
        raise AssertionError(msg)
    if res < 10:  # noqa: PLR2004
        cprint(f"res={res} is too small in order to create a detailed cmap. res was set to 999, instead.", col="y")
        res = 999
    if res % 2 == 0:
        res += 1
        print("res was incremented by 1 to zero center the cmap.")

    linear_space = np.linspace(-1, 1, res)
    linear_space_norm = normalize(linear_space, 0.0, 1.0)

    colored_linear_space = color_fct(linear_space)
    red = colored_linear_space[:, 0]
    green = colored_linear_space[:, 1]
    blue = colored_linear_space[:, 2]

    cdict = {
        "red": [[linear_space_norm[i_], col_, col_] for i_, col_ in enumerate(red)],
        "green": [[linear_space_norm[i_], col_, col_] for i_, col_ in enumerate(green)],
        "blue": [[linear_space_norm[i_], col_, col_] for i_, col_ in enumerate(blue)],
    }

    return LinearSegmentedColormap(name=color_fct.__name__, segmentdata=cdict)


# %% List of supported color map << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# the maps need to be implemented above this line
custom_maps = {
    "black-firered": gregoire_black_firered,
}

matplotlib_maps = ["afmhot", "jet", "seismic", "bwr", "cool"]


# %% << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<


def symmetric_clip(analyzer_obj, percentile=1 - 1e-2, min_sym_clip=True):
    """
    Clip relevance object symmetrically around zero.

    :param analyzer_obj: LRP analyzer object
    :param percentile: default: keep very small values at the border of range out. percentile=1: no change
    :param min_sym_clip: True: finds the min(abs(R.min), R.max) to clip symmetrically
    """
    assert 0.5 <= percentile <= 1, "percentile must be in range (.5, 1)!"  # noqa: PLR2004, S101

    if not (analyzer_obj.min() < 0.0 < analyzer_obj.max()) and min_sym_clip:
        cprint(
            string="Relevance object has only values larger OR smaller than 0., thus 'min_sym_clip' is switched off!",
            col="y",
        )
        min_sym_clip = False

    # Get cut-off values for lower and upper percentile
    if min_sym_clip:
        # min_clip example: [-7, 10] => clip(-7, 7) | [-10, 7] => clip(-7, 7)
        max_min_q = min(abs(analyzer_obj.min()), analyzer_obj.max())  # > 0
        min_min_q = -max_min_q  # < 0

    if percentile < 1:
        max_q = -np.percentile(a=-analyzer_obj, q=1 - percentile)
        min_q = np.percentile(a=analyzer_obj, q=1 - percentile)

        # Clip-off at max-abs percentile value
        max_q = max(abs(min_q), abs(max_q))  # > 0
        # Does opposite of min_clip, example: [-7, 10] => clip(-10, 10) | [-10, 7] => clip(-10, 10)
        if min_sym_clip:
            # However, when both options are active, 'percentile' is prioritized
            max_q = min(max_q, max_min_q)
        min_q = -max_q  # < 0

        return np.clip(a=analyzer_obj, a_min=min_q, a_max=max_q)

    if percentile == 1.0 and min_sym_clip:
        return np.clip(a=analyzer_obj, a_min=min_min_q, a_max=max_min_q)

    return analyzer_obj


def apply_colormap(
    R,  # noqa: N803
    inputimage=None,
    cmapname="black-firered",
    cintensifier=1.0,
    clipq=1e-2,
    min_sym_clip=False,
    gamma=0.2,
    gblur=0,
    true_scale=False,
):
    """
    Merge relevance tensor with input image to receive a heatmap over the input space.

    :param R: relevance map/tensor
    :param inputimage: input image
    :param cmapname: name of color-map (cmap) to be applied
    :param cintensifier: [1, ...[ increases the color strength by multiplying + clipping [DEPRECATED]
    :param clipq: clips off given percentile of relevance symmetrically around zero. range: [0, .5]
    :param min_sym_clip: True: finds the min(abs(R.min), R.max) to clip symmetrically around zero
    :param gamma: the smaller the gamma (< 1.) the brighter, for gamma > 1., the image gets darker
    :param gblur: ignore for now
    :param true_scale: True: return min/max R value (after clipping) for true col scaling in e.g. cbar
    :return: heatmap merged with input
    """
    # # Prep Input Image
    img = inputimage.copy()
    _R = R.copy()  # since mutable  # noqa: N806
    # Check whether image has RGB(A) channels
    max_ndims = 4
    if img.shape[-1] <= max_ndims:
        img = np.mean(img, axis=-1)  # removes rgb channels
    # Following creates a grayscale image (for the MRI case, no difference)
    img = np.concatenate([img[..., None]] * 3, axis=-1)  # (X,Y,Z, [r,g,b]), where r=g=b (i.e., grayscale)
    # normalize image (0, 1)
    bg = 0.0
    if img.min() < bg:  # for img range (-1, 1)
        img += np.abs(img.min())
    img /= np.max(np.abs(img))

    # Symmetrically clip relevance values around zero
    assert bg <= clipq <= 0.5, "clipq must be in range (0, .5)!"  # noqa: S101, PLR2004
    _R = symmetric_clip(analyzer_obj=_R, percentile=1 - clipq, min_sym_clip=min_sym_clip)  # noqa: N806
    rmax = np.abs(_R).max()  # symmetric: rmin = -rmax

    # # Normalize relevance tensor
    _R /= np.max(np.abs(_R))  # noqa: N806
    # norm to [-1, 1] for real numbers, or [0, 1] for R+, where zero remains zero

    # # Apply chosen cmap
    if cmapname in custom_maps:
        r_rgb = custom_maps[cmapname](_R)
    elif cmapname in matplotlib_maps:
        r_rgb = colorize_matplotlib(_R, cmapname)
    else:
        msg = (
            f"You have managed to smuggle in the unsupported colormap {cmapname} into method "
            f"apply_colormap. Supported mappings are:\n\t{produce_supported_maps()}"
        )
        raise Exception(msg)  # noqa: TRY002

    # Increase col-intensity
    min_intens = 1.0
    if cintensifier != min_intens:
        if cintensifier < min_intens:
            msg = "cintensifier must be 1 (i.e. no change) OR greater (intensify color)!"
            raise ValueError(msg)
        r_rgb *= cintensifier
        r_rgb = r_rgb.clip(bg, min_intens)

    # Merge input image with heatmap via inverse alpha channels
    alpha = np.abs(_R[..., None])  # as alpha channel, use (absolute) relevance map amplitude.
    alpha = np.concatenate([alpha] * 3, axis=-1) ** gamma  # (X,Y,Z, 3)
    heat_img = (1 - alpha) * img + alpha * r_rgb

    # Apply Gaussian blur
    if gblur > 0:  # there is a bug in opencv, which causes an error with this command
        msg = "'gaussblur' currently not activated, keep 'gaussblur=0' for now!"
        raise ValueError(msg)
        # hm = cv2.GaussianBlur(HM, (gaussblur, gaussblur), 0)  # noqa: ERA001

    if true_scale:
        return heat_img, r_rgb, rmax
    return heat_img, r_rgb


# ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >> END

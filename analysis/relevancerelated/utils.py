"""
Collection of utility functions.

Author: Simon M. Hofmann | <[firstname].[lastname][at]cbs.mpg.de> | 2022
"""

# %% Imports
from __future__ import annotations

import difflib
import gzip
import math
import pickle
import platform
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path, PosixPath
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


# %% Timer << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def chop_microseconds(delta: timedelta) -> timedelta:
    """Chop microseconds from given time delta."""
    return delta - timedelta(microseconds=delta.microseconds)


def function_timed(dry_funct: Callable[..., Any] | None = None, ms: bool | None = None):  # noqa: ANN201
    """
    Time the processing duration of wrapped function.

    Way to use:

    Following returns duration without micro-seconds:

    @function_timed
    def abc():
        return 2+2

    The following returns micro-seconds also:

    @function_timed(ms=True)
    def abcd():
        return 2+2

    :param dry_funct: parameter can be ignored. Results in output without micro-seconds
    :param ms: if micro-seconds should be printed, set to True
    :return:
    """

    def _function_timed(funct):
        @wraps(funct)
        def wrapper(*args, **kwargs):
            start_timer = datetime.now()

            # whether to suppress wrapper: use functimer=False in main funct
            w = kwargs.pop("functimer", True)

            output = funct(*args, **kwargs)

            duration = datetime.now() - start_timer

            if w:
                if ms:
                    print(f"\nProcessing time of {funct.__name__}: {duration} [h:m:s:ms]")

                else:
                    print(f"\nProcessing time of {funct.__name__}: {chop_microseconds(duration)} [h:m:s]")

            return output

        return wrapper

    if dry_funct:
        return _function_timed(dry_funct)

    return _function_timed


def loop_timer(
    start_time: datetime, loop_length: int, loop_idx: int, loop_name: str | None = None, add_daytime: bool = False
) -> None:
    """
    Estimate the remaining time to run through given loop.

    Function must be placed at the end of the loop inside.
    Before the loop, take start time by start_time=datetime.now()
    Provide position within in the loop via enumerate()
    In the form:
        '
        start = datetime.now()
        for idx, ... in enumerate(iterable):
            ... operations ...

            loop_timer(start_time=start, loop_length=len(iterable), loop_idx=idx)
        '
    :param start_time: time at the start of the loop
    :param loop_length: total length of loop-object
    :param loop_idx: position within loop
    :param loop_name: provide name of loop for print
    :param add_daytime: add leading day time to print-out
    """
    _idx = loop_idx
    ll = loop_length

    duration = datetime.now() - start_time
    rest_duration = chop_microseconds(duration / (_idx + 1) * (ll - _idx - 1))

    loop_name = "" if loop_name is None else " of " + loop_name

    now_time = f"{datetime.now().replace(microsecond=0)} | " if add_daytime else ""
    string = (
        f"{now_time}Estimated time to loop over rest{loop_name}: {rest_duration} [hh:mm:ss]\t "
        f"[ {'*' * int((_idx + 1) / ll * 30)}{'.' * (30 - int((_idx + 1) / ll * 30))} ] "
        f"{(_idx + 1) / ll * 100:.2f} %"
    )

    print(string, "\r" if (_idx + 1) != ll else "\n", end="")

    if (_idx + 1) == ll:
        cprint(
            string=f"{now_time}Total duration of loop{loop_name.split(' of')[-1]}: "
            f"{chop_microseconds(duration)} [hh:mm:ss]\n",
            col="b",
        )


# %% Normalizer & numerics  o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def normalize(
    array: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    global_min: float | None = None,
    global_max: float | None = None,
) -> np.ndarray:
    """
    Min-Max-Scaling: Normalizes Input Array to lower and upper bound.

    :param array: To be transformed array
    :param lower_bound: lower Bound a
    :param upper_bound: upper Bound b
    :param global_min: if the array is part of a larger tensor, normalize w.r.t. global min and ...
    :param global_max: ... global max (i.e., tensor min/max)
    :return: normalized array
    """
    assert lower_bound < upper_bound, "lower_bound must be < upper_bound"  # noqa: S101

    array = np.array(array)
    a, b = lower_bound, upper_bound

    if global_min is not None:
        assert global_min <= np.nanmin(array), "global_min must be <=  np.nanmin(array)"  # noqa: S101
        mini = global_min
    else:
        mini = np.nanmin(array)

    if global_max is not None:
        assert global_max >= np.nanmax(array), "global_max must be >= np.nanmax(array)"  # noqa: S101
        maxi = global_max
    else:
        maxi = np.nanmax(array)

    return (b - a) * ((array - mini) / (maxi - mini)) + a  # normed array


def denormalize(array: np.ndarray, denorm_minmax: tuple[float, float], norm_minmax: tuple[float, float]) -> np.ndarray:
    """
    Undo normalization of given array back to previous scaling.

    :param array: array to be denormalized
    :param denorm_minmax: tuple of (min, max) of denormalized (target) vector
    :param norm_minmax: tuple of (min, max) of normalized vector
    :return: denormalized vector
    """
    array = np.array(array)

    dn_min, dn_max = denorm_minmax
    n_min, n_max = norm_minmax

    assert n_min < n_max, "norm_minmax must be tuple (min, max), where min < max"  # noqa: S101
    assert dn_min < dn_max, "denorm_minmax must be tuple (min, max), where min < max"  # noqa: S101

    de_normed_array = (array - n_min) / (n_max - n_min) * (dn_max - dn_min) + dn_min

    return np.array(de_normed_array)


def z_score(array: np.ndarray) -> np.ndarray:
    """
    Create z-score of the given array.

    :return: z-score array
    """
    sub_mean = np.nanmean(array)
    sub_std = np.nanstd(array)
    z_array = (array - sub_mean) / sub_std

    return np.array(z_array)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between two vectors.

    :param vec1: vector 1
    :param vec2: vector 2
    :return: cosine similarity of two vectors
    """
    return np.inner(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_factors(n: int) -> list[int]:
    """Get factors of given integer."""
    # Create an empty list for factors
    factors = []

    # Loop over all factors
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)  # noqa: PERF401

    # Return the list of
    return factors


def oom(number: float) -> float:
    """Return order of magnitude of given number."""
    return math.floor(math.log10(number))


# %% Sorter < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def split_in_n_bins(a: list[Any] | tuple[Any] | np.ndarray, n: int = 5, attribute_remainder: bool = True) -> list[Any]:
    """Split in three bins and attribute the remainder equally: [1,2,3,4,5,6,7,8] => [1,2,7], [3,4,8], [5,6]."""
    size = len(a) // n
    split = np.split(a, np.arange(size, len(a), size))

    if attribute_remainder and (len(split) != n):
        att_i = 0
        remainder = list(split.pop(-1))
        while len(remainder) > 0:
            split[att_i] = np.append(split[att_i], remainder.pop(0))
            att_i += 1  # can't overflow
    elif len(split) != n:
        cprint(
            string=f"{len(split[-1])} remainder were put in extra bin. Return {len(split)} bins instead of {n}.",
            col="y",
        )

    return split


def get_string_overlap(s1: str, s2: str) -> str:
    """
    Find the longest overlap between two strings, starting from the left.

    :param s1: first string
    :param s2: second string
    :return overlap between two strings [str]
    """
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2))

    return s1[pos_a : pos_a + size]


# %% Color prints & I/O ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class Bcolors:
    r"""
    Use for colored print-commands in console.

    Usage:
    print(Bcolors.HEADER + "Warning: No active frommets remain. Continue?" + Bcolors.ENDC)
    print(Bcolors.OKBLUE + "Warning: No active frommets remain. Continue?" + Bcolors.ENDC)

    For more:

        CSELECTED = '\33[7m'

        CBLACK  = '\33[30m'
        CRED    = '\33[31m'
        CGREEN  = '\33[32m'
        CYELLOW = '\33[33m'
        CBLUE   = '\33[34m'
        CVIOLET = '\33[35m'
        CBEIGE  = '\33[36m'
        CWHITE  = '\33[37m'

        CBLACKBG  = '\33[40m'
        CREDBG    = '\33[41m'
        CGREENBG  = '\33[42m'
        CYELLOWBG = '\33[43m'
        CBLUEBG   = '\33[44m'
        CVIOLETBG = '\33[45m'
        CBEIGEBG  = '\33[46m'
        CWHITEBG  = '\33[47m'

        CGREY    = '\33[90m'
        CBEIGE2  = '\33[96m'
        CWHITE2  = '\33[97m'

        CGREYBG    = '\33[100m'
        CREDBG2    = '\33[101m'
        CGREENBG2  = '\33[102m'

        CYELLOWBG2 = '\33[103m'
        CBLUEBG2   = '\33[104m'
        CVIOLETBG2 = '\33[105m'
        CBEIGEBG2  = '\33[106m'
        CWHITEBG2  = '\33[107m'

    # For preview type:
    for i in [1, 4, 7] + list(range(30, 38)) + list(range(40, 48)) + list(range(90, 98)) + list(
            range(100, 108)):  # range(107+1)
        print(i, '\33[{}m'.format(i) + "ABC & abc" + '\33[0m')
    """

    HEADERPINK = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    UNDERLINE = "\033[4m"
    BOLD = "\033[1m"
    ENDC = "\033[0m"  # this is necessary in the end to reset to default print

    DICT = {  # noqa: RUF012
        "p": HEADERPINK,
        "b": OKBLUE,
        "g": OKGREEN,
        "y": WARNING,
        "r": FAIL,
        "ul": UNDERLINE,
        "bo": BOLD,
    }


def cprint(string: str, col: str | None = None, fm: str | None = None, ts: bool = False) -> None:
    """
    Colorize and format print-out. Add leading time-stamp (fs) if required.

    :param string: print message
    :param col: color:'p'(ink), 'b'(lue), 'g'(reen), 'y'(ellow), OR 'r'(ed)
    :param fm: format: 'ul'(:underline) OR 'bo'(:bold)
    :param ts: add leading time-stamp
    """
    if col:
        col = col[0].lower()
        if col not in {"p", "b", "g", "y", "r"}:
            msg = "col must be 'p'(ink), 'b'(lue), 'g'(reen), 'y'(ellow), 'r'(ed)"
            raise ValueError(msg)
        col = Bcolors.DICT[col]

    if fm:
        fm = fm[0:2].lower()
        if fm not in {"ul", "bo"}:
            msg = "fm must be 'ul'(:underline), 'bo'(:bold)"
            raise ValueError(msg)
        fm = Bcolors.DICT[fm]

    if ts:
        pfx = ""  # collecting leading indent or new line
        while string.startswith(("\n", "\t")):
            pfx += string[:1]
            string = string[1:]
        string = f"{pfx}{datetime.now():%Y-%m-%d %H:%M:%S} | " + string

    # print given string with formatting
    print(f"{col or ''}{fm or ''}{string}{Bcolors.ENDC}")
    # print("{}{}".format(col if col else "",
    #                     fm if fm else "") + string + Bcolors.ENDC)


def true_false_request(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap print function with true false request."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrap function with true-false request."""
        func(*args, **kwargs)  # should be only a print command

        tof = input("(T)rue or (F)alse: ").lower()
        if tof not in {"true", "false", "t", "f"}:
            msg = "Must be 'T', 't' or 'T/true', or 'F', 'f', 'F/false'"
            raise ValueError(msg)
        return tof in "true"

    return wrapper


@true_false_request
def ask_true_false(question: str, col: str = "b") -> None:
    """
    Ask user for input for given True-or-False question.

    :param question: str
    :param col: print-color of question
    :return: answer
    """
    cprint(question, col)


def check_system() -> str:
    """Check on which computer or server the code is currently running."""
    if platform.system().lower() == "linux":  # == sys.platform
        current_system = "MPI"

    elif platform.system().lower() == "darwin":
        current_system = "mymac"

    else:
        msg = "Unknown compute system."
        raise SystemError(msg)

    return current_system


def only_mpi(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap function that should only be executable on MPI server."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if check_system() == "MPI":
            return func(*args, **kwargs)  # should be only a print command
        msg = f"Function '{func.__name__}()' can only be executed on MPI server!"
        raise OSError(msg)

    return wrapper


# %% OS  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def browse_files(initialdir: str | None = None, filetypes: str | None = None) -> str:
    """
    Browse and choose a file from the finder.

    :param initialdir: Where to start the search (ARG MUST BE NAMED 'initialdir')
    :param filetypes: what type of file-ending (suffix, e.g., '*.jpg')
    :return: path to chosen file
    """
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename

    root = Tk()
    root.withdraw()

    kwargs = {}
    if initialdir:
        kwargs.update({"initialdir": initialdir})
    if filetypes:
        kwargs.update({"filetypes": [(filetypes + " File", "*." + filetypes.lower())]})

    return askopenfilename(parent=root, title="Choose the file", **kwargs)


# %% Save objects externally & load them  o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


@function_timed
def save_obj(obj: Any, name: str, folder: str, hp: bool = True, as_zip: bool = False, save_as: str = "pkl"):
    """
    Save object as pickle or numpy file.

    :param obj: object to be saved
    :param name: name of pickle/numpy file
    :param folder: target folder
    :param hp: True:
    :param as_zip: True: zip file
    :param save_as: default is pickle, can be "npy" for numpy arrays
    """
    # Remove suffix here, if there is e.g. "*.gz.pkl":
    if name.endswith(".gz"):
        name = name[:-3]
        as_zip = True
    if name.endswith((".pkl", ".npy", ".npz")):
        save_as = "pkl" if name.endswith(".pkl") else "npy"
        name = name[:-4]

    p2save = Path(folder, name)

    # Create parent folder if not available
    p2save.parent.mkdir(parents=True, exist_ok=True)

    save_as = save_as.lower()
    if save_as == "pkl":
        open_it, suffix = (gzip.open, ".pkl.gz") if as_zip else (open, ".pkl")
        with open_it(p2save.with_suffix(suffix), "wb") as f:
            if hp:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
            else:
                p = pickle.Pickler(f)
                p.fast = True
                p.dump(obj)
    elif save_as == "npy":
        if not isinstance(obj, np.ndarray):
            raise ValueError
        if as_zip:
            np.savez_compressed(file=p2save, arr=obj)
        else:
            np.save(arr=obj, file=p2save)
    else:
        msg = f"Format save_as='{save_as}' unknown!"
        raise ValueError(msg)


@function_timed
def load_obj(name: str, folder: str | PosixPath) -> Any:
    """
    Load the pickle or numpy object into workspace.

    :param name: name of the dataset
    :param folder: target folder
    :return: object
    """
    # Check whether the zipped version is available as well: "*.pkl.gz"
    possible_fm = [".pkl", ".pkl.gz", ".npy", ".npz"]

    def _raise_name_issue():
        msg = f"'{folder}' contains too many files which could fit name='{name}'.\nSpecify full name including suffix!"
        raise ValueError(msg)

    # Check all files in the folder which find the name + *suffix
    found_files = [str(pa) for pa in Path(folder).glob(name + "*")]

    if not any(name.endswith(fm) for fm in possible_fm):
        # No file-format found, check folder for files
        if len(found_files) == 0:
            msg = f"In '{folder}' no file with given name='{name}' was found!"
            raise FileNotFoundError(msg)

        if len(found_files) == 2:  # len(found_files) == 2  # noqa: PLR2004
            # There can be a zipped & unzipped version, take the unzipped version if applicable
            file_name_overlap = get_string_overlap(found_files[0], found_files[1])
            if file_name_overlap.endswith(".pkl"):  # .pkl and .pkl.gz found
                name = Path(file_name_overlap).name
            if file_name_overlap.endswith(".np"):  # .npy and .npz found
                name = Path(file_name_overlap).name + "y"  # .npy
            else:  # if the two found files are not of the same file-type
                _raise_name_issue()

        elif len(found_files) > 2:  # noqa: PLR2004
            _raise_name_issue()

        else:  # len(found_files) == 1
            name = Path(found_files[0]).name  # un-list

    path_to_file = Path(folder, name)

    # Load and return
    if str(path_to_file).endswith((".pkl", ".pkl.gz")):  # pickle case
        open_it = gzip.open if str(path_to_file).endswith(".gz") else open
        with open_it(path_to_file, "rb") as f:
            return pickle.load(f)
    else:  # numpy case
        file = np.load(path_to_file)
        if isinstance(file, np.lib.npyio.NpzFile):  # name.endswith(".npz"):
            # If numpy zip (.npz)
            file = file["arr"]
            # This asserts that object was saved this way: np.savez_compressed(file=..., arr=obj), as
            # in save_obj():
        return file


# <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> END

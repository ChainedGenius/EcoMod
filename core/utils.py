import re
from itertools import chain
from functools import wraps
from time import perf_counter
from logging import Logger


def timeit(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        st = perf_counter()
        func(*args, **kwargs)
        et = perf_counter()
        #logger.info(msg=f'Elapsed time: {et - st}')
        print(f'Elapsed time: {et - st}')

    return wrapper


def unpack(yaml_dict):
    ret = {}
    for k, v in yaml_dict.items():
        if isinstance(v, dict):
            header = ' -- ' + k
            ret.update({var: desc + header for var, desc in v.items()})
        if isinstance(v, str):
            ret.update({k: v})

    return ret


def trim(s: str, i=1, j=-1):
    """
    :param s: input string
    :param i: start
    :param j: end
    :return: substring s[i:j]. by default trimming from start and end 1 symbol
    """
    return s[i:j]


def extract_dim_desc(raw_desc):
    try:
        dim = "".join("".join(raw_desc.split('[')[1:]).split(']')[:-1])
    except IndexError:
        dim = "no dim"
    desc = raw_desc.split('[')[0]
    return dim, desc


def set_equality(set1, set2):
    if len(set1 - set2) + len(set2 - set1) == 0:
        return True
    else:
        return False


def embrace(s: str, brackets='()'):
    return brackets[0] + s + brackets[1]


def find_objective_markers(s: str):
    ARROW_MARKERS = ["to", "Rightarrow", "rightarrow", "->", "-->"]
    OBJECTIVE_MARKERS = ["max", "min", "extr"]
    MARKER_REGEXP = "{a}[ ]*{o}".format(
        a="*".join([embrace(a) for a in ARROW_MARKERS]) + "*",
        o="*".join([embrace(o) for o in OBJECTIVE_MARKERS]) + "*"
    )
    matches = re.findall(MARKER_REGEXP, s)
    matches = set(chain(*[i for i in matches]))
    matches.remove('')
    if matches:
        return True, re.sub(MARKER_REGEXP, '', s)
    else:
        return False, s


def iterable_substract(a, b):
    """
    :param a:
    :param b:
    :return: a - b
    """
    return a.__class__([i for i in a if i not in b])


if __name__ == "__main__":
    s = "$J = \int_0^T (\exp^{-\delta*t} \ln(c(x(t))))dt ->        max$"
    a = find_objective_markers(s)
    print(a)

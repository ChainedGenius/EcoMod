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

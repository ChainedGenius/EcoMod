

def collector(yaml_dict):
    ret = {}
    for k, v in yaml_dict.items():
        if isinstance(v, dict):
            header = ' -- ' + k
            ret.update({var: desc + header for var, desc in v.items()})
        if isinstance(v, str):
            ret.update({k: v})

    return ret
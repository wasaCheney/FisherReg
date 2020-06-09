"""parse cofig file"""


def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and
    returns module definitions.
    Return:
        module_defs, a list with dicts as elements
    """
    with open(path, 'r') as fmodel:
        lines = fmodel.read().split('\n')
    lines = [x.strip() for x in lines if x.strip() and not x.startswith('#')]
    module_defs = []
    for line in lines:
        # This marks the start of a new block
        if line.startswith('['):
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].strip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.strip()] = value.strip()

    return module_defs


def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '4'
    with open(path, 'r') as fdata:
        lines = fdata.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

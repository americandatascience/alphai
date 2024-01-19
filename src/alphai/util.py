import re


def is_package_installed(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def extract_param_value(model_name):
    match = re.search(r"-(\d+\.?\d*)(x(\d+\.?\d*))?", model_name)
    if match:
        if match.group(3):  # If 'x' is present
            value = float(match.group(1)) * float(match.group(3))
        else:
            value = float(match.group(1))
    return value

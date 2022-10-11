import numpy as np

def _error_format(value, error):
    precision = np.floor(np.log10(error)).astype(int)
    if(precision >= 0):
        scale = precision + 1
        value = value / 10.**scale
        error = error / 10.**scale
        precision = np.floor(np.log10(error)).astype(int)
        print(precision)
    else: 
        scale = None

    aprec = np.abs(precision)
    
    digits_before_decimal = np.log10(np.abs(value))

    if(digits_before_decimal < 0):
        leading_precision = 0
    else:
        leading_precision = (1 + np.floor(digits_before_decimal)).astype(int)
    
    normalized_to_precision_value = value * 10**(aprec + 1)
    normalized_to_precision_error = error * 10**(aprec + 1)
    
    rounded_at_precision_value = np.round(normalized_to_precision_value)
    rounded_at_precision_error = np.round(normalized_to_precision_error).astype(int)
    
    rounded_value = rounded_at_precision_value * 10.**(-aprec - 1)
    
    if scale is None:
        f_string = "{rounded_value:" f".{np.abs(precision) + 1}f" "}({rounded_at_precision_error})"
        return f_string.format(**locals())
    else:
        f_string = "({rounded_value:" f".{np.abs(precision) + 1}f" "}({rounded_at_precision_error}))"
        exp_string = r"\times 10^{" f"{scale}" "}"
        return f_string.format(**locals()) + exp_string

def error_format(value, error):
    if(not isinstance(value, np.ndarray)):
        return _error_format(value, error)
    vef = np.vectorize(_error_format, otypes=[str])
    return vef(value, error)

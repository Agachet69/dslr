def my_sum(list):
    total = 0
    for item in list:
        if item:
            total += item
    return item

def my_min(list):
    min = None
    for item in list:
        if item:
            if min == None or item < min:
                min = item
    return min

def my_max(list):
    max = None
    for item in list:
        if item:
            if max == None or item > max:
                max = item
    return max

def my_len(list):
    index = 0
    for _ in list:
        index+=1
    return index

def my_floor(nb):
    if nb == int(nb):
        return int(nb)
    if nb > 0:
        return int(nb)
    return int(nb) - 1

def my_round(x):
    int_part = int(x)
    frac_part = x - int_part
    if frac_part < 0.5:
        return int_part
    elif frac_part > 0.5:
        return int_part + 1
    else:
        if int_part % 2 == 0:
            return int_part
        else:
            return int_part + 1
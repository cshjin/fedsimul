import pickle


def save_obj(obj, name):
    """ Save object to pickle file

    Parameters
    ----------
    obj : object
    name : string
        Name of pickle file
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """ Load object from pickle file

    Parameters
    ----------
    name : string
        Name of pickle file

    Returns
    -------
    object
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def iid_divide(lst, g):
    ''' Divide list lst among g groups
    each group has either int(len(l)/g) or int(len(l)/g)+1 elements
    returns a list of groups

    Parameters
    ----------
    lst : list
        List of indices
    g: int
        Number of groups

    Returns
    -------
    glist : list of list
        List of groups
    '''
    num_elems = len(lst)
    group_size = int(len(lst) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(lst[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(lst[bi + group_size * i: bi + group_size * (i + 1)])
    return glist

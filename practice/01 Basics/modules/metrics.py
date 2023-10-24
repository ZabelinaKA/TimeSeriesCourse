import numpy as np
from modules.utils import z_normalize

def ED_distance(ts1, ts2):
    """
    Calculate the Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    ed_dist : float
        Euclidean distance between ts1 and ts2.
    """
    
    ed_dist = 0

    # Проверяем, что длины временных рядов одинаковы
    if len(ts1) != len(ts2):
        raise ValueError("Длины временных рядов должны быть одинаковыми.")

    # Вычисляем евклидову метрику
    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))
    
    return ed_dist


def norm_ED_distance(ts1, ts2):
    """
    Calculate the normalized Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    norm_ed_dist : float
        The normalized Euclidean distance between ts1 and ts2.
    """

    norm_ed_dist = 0

    norm_series1 = z_normalize(ts1)
    norm_series2 = z_normalize(ts2)
    norm_ed_dist = np.sqrt(np.sum(np.square(norm_series1 - norm_series2))) 

    return norm_ed_dist



def DTW_distance(ts1, ts2, r=None):
    """
    Calculate DTW distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    r : float
        Warping window size.
    
    Returns
    -------
    dtw_dist : float
        DTW distance between ts1 and ts2.
    """

    dtw_dist = 0   
    assert len(ts1) == len(ts2) 

    dtw_matrix = np.full((len(ts1), len(ts2)), np.inf)
    
    dtw_matrix[0, 0] = 0

    for i in range(1, len(ts1)):
        for j in range(1, len(ts2)):
            cost = (ts1[i] - ts2[j])** 2
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                          dtw_matrix[i, j-1],
                                          dtw_matrix[i-1, j-1])
    dtw_dist = dtw_matrix[-1, -1]
    
    return dtw_dist

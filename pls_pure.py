import numpy as np
from numpy import linalg as nl
from src.qf import qf
# 自己定义的计算二次型的函数


def pls_pure(xtx, xty, n, p):
    xtx_scale = xtx 
    xty_scale = xty 
    x_std = None
    ete = np.copy(xtx_scale)
    # E0TE0
    etf = np.copy(xty_scale)
    # E0TF0
    wp_mat = np.eye(p)
    # 对角阵
    b_ps = []
    for i in np.arange(p):
        # 迭代
        w = etf
        etf_norm = nl.norm(etf)
        w = w / etf_norm
        t_norm = qf(ete, w)
        r = etf_norm / t_norm
        ws = r * np.dot(wp_mat, w)
        b_ps.append(np.copy(ws))
        pp = np.dot(ete, w) / t_norm

        if i < (p - 1):
            wp = np.eye(p) - np.outer(w, pp)
            wpt = wp.T
            # 迭代更新
            ete = np.dot(wpt, np.dot(ete, wp))
            etf = np.dot(wpt, etf)
            wp_mat = np.dot(wp_mat, wp)
    b_ps = np.array(b_ps)
    b_ps = np.cumsum(b_ps, axis=0)
    return b_ps

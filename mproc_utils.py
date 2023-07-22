"""Helper functions for multiprocessing."""

import os
import inspect
from tqdm import tqdm
from multiprocessing import Pool

from sys_eval import construct_and_evaluate_system



def multi_proc_constr_and_eval_system(args_list):
    return construct_and_evaluate_system(*args_list)


def parallel_task(func, iterable, n_procs):
    """Schedule multiprocessing of function evaluations.

    Adapted from solution to https://stackoverflow.com/questions/47313732/jupyter-notebook-never-finishes-processing-using-multiprocessing-python-3/47374811

    NOTE: the implementation needed for use in a notebook
    is slightly different.

    Args:
        func: Function to evualte.
        iterable: Iterable of arugments for evaluations.
        n_procs (int, optional): Number of processes to use in pool.
            Defaults to os.cpu_count().

    Returns:
        List of function evaluations (returns).
    """

    pool = Pool(processes=n_procs)
    res = list(tqdm(pool.imap(func, iterable), total=len(iterable)))
    pool.close()
    return res
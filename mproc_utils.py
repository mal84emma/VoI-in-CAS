"""Helper functions for multiprocessing."""

import os
import inspect
from tqdm import tqdm
from multiprocessing import Pool

def parallel_task(func, iterable, n_procs=os.cpu_count()):
    """Schedule multiprocessing of function evaluations.

    Adapted from solution to https://stackoverflow.com/questions/47313732/jupyter-notebook-never-finishes-processing-using-multiprocessing-python-3/47374811

    NOTE: this function must be in a notebook for ipynb multipproc
    NOTE: in this workaround, the function passed to `parallel_task`
    must do all of the importing it needs, and unwrap the arguments
    (only a single argument can be passed)

    Args:
        func: Function to evualte.
        iterable: Iterable of arugments for evaluations
        n_procs (int, optional): Number of processes to use in pool.
            Defaults to os.cpu_count().

    Returns:
        List of function evaluations (returns).
    """

    temp_path = f'./tmp_func.py'
    with open(temp_path, 'w') as file:
        file.write(inspect.getsource(func).replace(func.__name__, "task"))

    from tmp_func import task

    if __name__ == '__main__':
        pool = Pool(processes=n_procs)
        res = list(tqdm(pool.imap(task, iterable), total=len(iterable)))
        pool.close()
        os.remove(temp_path)
        return res
    else:
        raise "Not in Jupyter Notebook"
import datetime
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import requests
import urllib.request
from contextlib import closing

__all__ = ['DependencyError',
           'resource',
           'dependency',
           'asserted_dependency',
           'split_df',
           'data_path',
           'date_to_decimal_year']


class DependencyError(AssertionError):
    """Exception raised in case of an erroneous dependency."""

def resource(target, url, post=False, **kw_args):
    """Specify a dependency on an online resource.

    Further takes in keyword arguments that are passed to the appropriate method
    from :mod:`requests` or :mod:`urllib`.

    Args:
        target (str): Target file.
        url (str): Source URL.
        post (bool, optional): Make a POST request instead of a GET request.
            Only applicable if the URL starts with "http" or "https". Defaults
            to `False`.
    """
    if not os.path.exists(target):
        # Ensure that all directories in the path exist.
        make_dirs(target)

        # If the URL starts with "ftp", use the :mod:`urllib` library.
        # Otherwise, use the :mod:`requests` library.
        if url.startswith('ftp'):
            with closing(urllib.request.urlopen(url, **kw_args)) as r:
                with open(target, 'wb') as f:
                    shutil.copyfileobj(r, f)
        else:
            request = requests.post if post else requests.get
            with request(url, stream=True, **kw_args) as r:
                with open(target, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)


def dependency(target, source, commands):
    """Specify a dependency that is generated from an existing file.

    Args:
        target (str): Target file.
        source (str): Source file.
        commands (list[str]): List of commands to generate target file.
    """
    if not os.path.exists(target):
        # Check that the source exists.
        if not os.path.exists(source):
            raise DependencyError(f'Source "{source}" asserted to exist, '
                                  f'but it does not.')

        # Save current working directory.
        current_wd = os.getcwd()

        # Ensure that all directories in the path exist.
        make_dirs(target)

        # Perform commands.
        for command in commands:
            # Change working directory to directory of target file, run
            # command, and restore working directory afterwards.
            os.chdir(os.path.dirname(target))
            subprocess.call(command, shell=True)
            os.chdir(current_wd)


def asserted_dependency(target):
    """Specify a dependency that cannot be fetched.

    Args:
        target (str): Target file.
    """
    if not os.path.exists(target):
        raise DependencyError(f'Dependency "{target}" is asserted to exist, '
                              f'but it does not, and it cannot be '
                              f'automatically fetched. Please put the file '
                              f'into place manually.')


def make_dirs(path):
    """Make the directories in the path of a file.

    Args:
        path (url): Path of a file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)


def data_path(*xs):
    """Get the path of a data file.

    Args:
        *xs (str): Parts of the path.

    Returns:
        str: Absolute path.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        os.pardir,
                                        os.pardir,
                                        'data',
                                        *xs))


def split_df(df, index_range, columns, iloc=False):
    """Split a data frame by selecting from columns a particular range.

    Args:
        df (:class:`pd.DataFrame`): Data frame to split.
        index_range (tuple): Tuple containing lower and upper limit of the
            range to split the index by. If `index_range = (a, b)`, then
            `[a, b)` is taken.
        columns (list[object]): Columns to select.
        iloc (bool, optional): The index range is the integer location instead
            of the index value. Defaults to `False`.

    Returns:
        tuple[:class:`pd.DataFrame`]: Selected rows from selected columns
            and the remainder.
    """
    if iloc:
        inds = np.arange(df.shape[0])
        rows = (inds >= index_range[0]) & (inds < index_range[1])
    else:
        rows = (df.index >= index_range[0]) & (df.index < index_range[1])
    selected = pd.DataFrame([df[name][rows] for name in columns]).T
    remainder = pd.DataFrame([df[name][~rows] for name in columns] +
                             [df[name] for name in
                              set(df.columns) - set(columns)]).T

    # Fix order of columns.
    selected_inds = [i for i, c in enumerate(df.columns) if c in columns]
    selected = selected.reindex(df.columns[np.array(selected_inds)], axis=1)
    remainder = remainder.reindex(df.columns, axis=1)

    return selected, remainder


def date_to_decimal_year(date, format):
    """Convert a date to decimal year.

    Args:
        date (str): Date as a string.
        format (str): Format of the dat.

    Returns:
        float: Decimal year corresponding to the date.
    """
    date = datetime.datetime.strptime(date, format)
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year + 1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length

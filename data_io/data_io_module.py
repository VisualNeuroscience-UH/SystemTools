# Builtin
from pathlib import Path
import os
import zlib
import pickle
import logging
import pdb

# io tools from common packages
import scipy.io as sio
import scipy.sparse as scprs
import pandas as pd

# io tools from cxsystem
from cxsystem2.core.tools import write_to_file, load_from_file

# This package
from data_io.data_io_base_module import DataIOBase
from context.context_module import Context


class DataIO(DataIOBase):

    # self.context. attributes
    _properties_list = ["path", "input_folder", "output_folder"]

    def __init__(self, context, map_ana_names=None) -> None:

        self.context = context.set_context(self._properties_list)
        self.map_ana_names = map_ana_names

        # Attach cxsystem2 methods
        self.write_to_file = write_to_file
        self.load_from_file = load_from_file

        # Attach other methods/packages
        self.savemat = sio.savemat
        self.csr_matrix = scprs.csr_matrix

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, value):
        if isinstance(value, Context):
            self._context = value
        else:
            raise AttributeError(
                "Trying to set improper context. Context must be a context object."
            )

    def _check_candidate_file(self, path, filename):
        candidate_data_fullpath_filename = Path.joinpath(path, filename)
        if candidate_data_fullpath_filename.is_file():
            data_fullpath_filename = candidate_data_fullpath_filename
            return data_fullpath_filename
        else:
            return None

    def listdir_loop(self, path, data_type=None, exclude=None):
        """
        Parameters
        ----------
        path : pathlib.Path object or str
            The path to search for files and folders.
        data_type : str, optional
            A substring to search for in the names of the files and folders. Only files and folders with names
            containing this substring will be returned. If not provided, all files and folders will be returned.
        exclude : str, optional
            A substring to exclude from the search. Any files and folders with names containing this substring
            will be skipped. If not provided, no files or folders will be excluded.

        Returns
        -------
        List[str]
            A list of the full path names of the files and folders in `path` that match the criteria specified by
            `data_type` and do not contain `exclude`.
        """
        
        if isinstance(path, str):
            path = Path(path)

        files = []
        for f in Path.iterdir(path):
            if data_type is not None and exclude is not None:
                if (
                    data_type.lower() in f.as_posix().lower()
                    and exclude.lower() not in f.as_posix().lower()
                ):
                    files.append(f)
            elif data_type is not None and exclude is None:
                if data_type.lower() in f.as_posix().lower():
                    files.append(f)
            elif data_type is None and exclude is not None:
                if exclude.lower() not in f.as_posix().lower():
                    files.append(f)
            else:
                files.append(f)

        paths = [Path.joinpath(path, basename) for basename in files]

        return paths

    def most_recent(self, path, data_type=None, exclude=None):
        """
        Parameters
        ----------
        path : str
            The path to search for the most recently modified file or folder.
        data_type : str, optional
            A substring to search for in the names of the files and folders. Only files and folders with names
            containing this substring will be considered. If not provided, all files and folders will be considered.
        exclude : str, optional
            A substring to exclude from the search. Any files and folders with names containing this substring
            will be skipped. If not provided, no files or folders will be excluded.

        Returns
        -------
        str
            The name of the most recently modified file or folder in `path` that matches the criteria specified by
            `data_type` and does not contain `exclude`.
        """
        paths = self.listdir_loop(path, data_type, exclude)

        if not paths:
            return None
        else:
            data_fullpath_filename = max(paths, key=os.path.getmtime)
            return data_fullpath_filename

    def parse_path(self, filename, data_type=None, exclude=None):
        """
        The search for the file is performed in the following order:
            1. The current directory (e.g., for direct IPython testing).
            2. The project's output directory.
            3. The project's input directory.
            4. The project's root directory.

        If `filename` is provided, the function returns the full path to that file if it exists in one of the above
        directories. If `filename` is not provided, the function searches for the most recently modified file in the
        above directories that has a name containing `data_type`. If no such file is found, an assertion error is raised.

        Note that `data_type` can also be a timestamp.

        Parameters
        ----------
        filename : str, optional
            The name of the file to search for.
        data_type : str, optional
            A substring to search for in the names of the files. If `filename` is not provided, only files with names
            containing this substring will be considered. If not provided, all files will be considered.
        exclude : str, optional
            A substring to exclude from the search. Any files with names containing this substring will be skipped. If
            not provided, no files will be excluded.

        Returns
        -------
        str
            The full path to the file that was found.

        """

        data_fullpath_filename = None
        path = self.context.path
        input_folder = self.context.input_folder
        output_folder = self.context.output_folder

        if output_folder is not None:
            output_path = Path.joinpath(path, output_folder)
        else:
            # Set current path if run separately from project
            output_path = Path.joinpath(path, "./")
        if input_folder is not None:
            input_path = Path.joinpath(path, input_folder)
        else:
            input_path = Path.joinpath(path, "./")

        # Check first for direct load in current directory. E.g. for direct ipython testing
        if filename:
            data_fullpath_filename = self._check_candidate_file(Path("./"), filename)

            # Next check direct load in output path, input path and project path in this order
            if not data_fullpath_filename:
                data_fullpath_filename = self._check_candidate_file(
                    output_path, filename
                )
            if not data_fullpath_filename:
                data_fullpath_filename = self._check_candidate_file(
                    input_path, filename
                )
            if not data_fullpath_filename:
                data_fullpath_filename = self._check_candidate_file(path, filename)

        # Parse data_type next in project/input and project paths
        elif data_type is not None:
            # Parse output folder for given data_type
            data_fullpath_filename = self.most_recent(
                output_path, data_type=data_type, exclude=exclude
            )
            if not data_fullpath_filename:
                # Check for data_type first in input folder
                data_fullpath_filename = self.most_recent(
                    input_path, data_type=data_type, exclude=exclude
                )
            if not data_fullpath_filename:
                # Check for data_type next in project folder
                data_fullpath_filename = self.most_recent(
                    path, data_type=data_type, exclude=exclude
                )

        assert (
            data_fullpath_filename is not None
        ), f"I Could not find file {filename}, aborting..."

        return data_fullpath_filename

    def get_data(
        self,
        filename=None,
        data_type=None,
        exclude=None,
        return_filename=False,
        full_path=None,
    ):
        """
        Open the requested file and get the data. If `filename` is not provided, the most recent file in the path with the specified `data_type` will be used. If `data_type` is not provided, the file with the specified `filename` will be used. Either `filename`, `data_type` or `full_path` must be provided.
        Allowed extensions are .csv, .pkl, .gz, .mat, and .txt.


        Parameters
        ----------
        filename : str, optional
            The name of the file to be opened.
        data_type : str, optional
            A keyword in the filename.
        exclude : str, optional
            An exclusion keyword. If the filename contains this keyword, it will be excluded even if it also contains the `data_type` keyword.
        return_filename : bool, optional
            If True, returns the full path to the filename as well.
        full_path : str, optional
            The full path to the filename.

        Returns
        -------
        data : array_like
            The data from the file.
        filename : str, optional
            The full path to the file (only returned if `return_filename` is True).
        """

        if full_path is None:
            if data_type is not None:
                data_type = data_type.lower()
            # Explore which is the most recent file in path of data_type and add full path to filename
            data_fullpath_filename = self.parse_path(
                filename, data_type=data_type, exclude=exclude
            )
        else:
            if isinstance(full_path, str):
                full_path = Path(full_path)
            assert (
                full_path.is_file()
            ), f"Full path: {full_path} given, but such file does not exist, aborting..."
            data_fullpath_filename = full_path

        # Open file by extension type
        filename_extension = data_fullpath_filename.suffix
        if "gz" in filename_extension or "pkl" in filename_extension:
            try:
                fi = open(data_fullpath_filename, "rb")
                data_pickle = zlib.decompress(fi.read())
                data = pickle.loads(data_pickle)
            except:
                with open(data_fullpath_filename, "rb") as data_pickle:
                    data = pickle.load(data_pickle)
        elif "mat" in filename_extension:
            data = {}
            sio.loadmat(data_fullpath_filename, data)
        elif "csv" in filename_extension:
            data = pd.read_csv(data_fullpath_filename)
            if "Unnamed: 0" in data.columns:
                data = data.drop(["Unnamed: 0"], axis=1)
        elif "txt" in filename_extension:
            # Read plain text to list line by line
            with open(data_fullpath_filename, "r") as f:
                data = f.readlines()
        else:
            raise TypeError("U r trying to input unknown filetype, aborting...")

        # print(f"Loaded file {data_fullpath_filename}")
        # Check for existing loggers (python builtin, called from other modules, such as the run_script.py)
        if logging.getLogger().hasHandlers():
            logging.info(f"Loaded file {data_fullpath_filename}")

        if return_filename is True:
            return data, data_fullpath_filename
        else:
            return data

    def get_csv_as_df(self, folder_name=None, csv_path=None, include_only=None):
        """The function searches for CSV files that contain the string `include_only` in their names, or, if `include_only` is not provided, CSV files with either a time stamp or the string 'mean' in their names. If no such files are found, an assertion error is raised.

        The search for the files is performed in the following order:
            1. If `csv_path` is provided, it is used as the root directory for the search.
            2. If `folder_name` is provided, the function searches for CSV files in the `path`/specified directory and its subdirectories.
            3. If neither `csv_path` nor `folder_name` is provided, the function searches for CSV files in the project's output directory and its subdirectories.

        Parameters
        ----------
        folder_name : str, optional
            The name of the root directory to search for CSV files.
        csv_path : str, optional
            The path to the root directory to search for CSV files.
        include_only : str or list of str, optional
            A substring or list of substrings to search for in the names of the CSV files. If provided, only CSV files with names
            containing one of these substrings will be considered. If not provided, CSV files with either a time stamp or the
            string 'mean' in their names will be considered.

        Returns
        -------
        Pandas DataFrame
            A DataFrame containing the data from the first CSV file that were found.
        Pandas DataFrame
            A DataFrame containing the data compiled from all the CSV files that were found.
        List[str]
            A list of the independent variable column names from the DataFrame.
        List[str]
            A list of the dependent variable column names from the DataFrame.
        str
            The time stamp of the CSV files that were used to create the DataFrame, or None if no time stamp was found.
        """

        if csv_path is None:
            if folder_name is None:
                csv_path = Path.joinpath(self.context.path, self.context.output_folder)
            else:
                csv_path = Path.joinpath(self.context.path, folder_name)

        assert Path.is_dir(
            csv_path
        ), f"Path {str(csv_path)} does not exists, aborting..."

        root_subs_files_list = [tp for tp in os.walk(csv_path)]
        csv_file_list = []
        time_stamp = None
        for this_tuple in root_subs_files_list:
            this_root_path = Path(this_tuple[0])
            if include_only is not None:
                include_only_list = list(include_only)
                if isinstance(include_only, str):
                    include_only_list = [include_only]
                elif isinstance(include_only, list):
                    include_only_list = include_only
                else:
                    raise TypeError(
                        "include_only (csv_substring) parameter must be string or list of strings"
                    )
                this_csv_file_list = [
                    fn
                    for fn in this_tuple[2]
                    for sub in include_only_list
                    if fn.endswith(".csv") and sub in fn
                ]
            else:
                # csv files are either with timestamp (single array runs) or they are averaged over multiple iterations
                # Try timestamp. Assuming '_20' is a time stamp.
                this_csv_file_list = [
                    p for p in this_tuple[2] if p.endswith(".csv") and "_20" in p
                ]
                if this_csv_file_list:
                    time_stamp_set = set(
                        [p[p.find("_20") : p.find(".csv")] for p in this_csv_file_list]
                    )
                    if len(time_stamp_set) > 1:
                        # Get not updated time stamps
                        time_stamp_set_not_updated = {
                            s for s in time_stamp_set if "updated" not in s
                        }
                        assert (
                            len(time_stamp_set_not_updated) < 2
                        ), "Multiple updated csv time stamps found in same folder, aborting..."
                        this_time_stamp = next(iter(time_stamp_set_not_updated))
                        # Check that this one time stamp is included all strings, i.e. that the updated time stamps are the same
                        err_list = [
                            err for err in time_stamp_set if this_time_stamp not in err
                        ]
                        assert (
                            not err_list
                        ), f"{time_stamp_set} found in same folder, aborting..."
                    time_stamp = time_stamp_set.pop()
                else:  # Rescue
                    this_csv_file_list = [
                        p for p in this_tuple[2] if p.endswith(".csv") and "mean" in p
                    ]
            assert (
                len(this_csv_file_list) > 0
            ), "No csv files with either time stamp or substring 'mean' found, aborting..."

            for this_file in this_csv_file_list:
                first_underscore_idx = this_file.find("_")
                # allowed types should start with {self.map_ana_names.values()}
                if this_file[:first_underscore_idx] not in self.map_ana_names.values():
                    continue
                this_csv_file_path = Path.joinpath(this_root_path, this_file)
                csv_file_list.append(this_csv_file_path)

        ## Get data from csv files ##
        data0_df = self.get_data(filename=csv_file_list[0], data_type=None)
        if "Dimension-2 Parameter" in data0_df.columns:
            independent_var_col_list = [
                "Dimension-1 Parameter",
                "Dimension-1 Value",
                "Dimension-2 Parameter",
                "Dimension-2 Value",
            ]
        else:
            independent_var_col_list = [
                "Dimension-1 Parameter",
                "Dimension-1 Value",
            ]
        # if "Unnamed: 0" in data0_df.columns:
        #     data_df = data0_df.drop(["Unnamed: 0"], axis=1)

        data_df_compiled = data0_df
        dependent_var_col_list = data0_df[
            data0_df.columns.difference(independent_var_col_list)
        ].columns.to_list()
        for csv_filename in csv_file_list[1:]:
            data_df = self.get_data(filename=csv_filename, data_type=None)
            # Drop 'Unnamed: 0'
            if "Unnamed: 0" in data_df.columns:
                data_df = data_df.drop(["Unnamed: 0"], axis=1)
            # If independent dimensions match, add column values
            if data_df[independent_var_col_list].equals(
                data_df_compiled[independent_var_col_list]
            ):
                # Get list of dependent variable columns
                this_dependent_variable_columns_list = [
                    col
                    for col in data_df.columns
                    if col not in independent_var_col_list
                ]
                data_df_compiled[this_dependent_variable_columns_list] = data_df[
                    this_dependent_variable_columns_list
                ]
                dependent_var_col_list.extend(this_dependent_variable_columns_list)

        # drop repetitions
        dependent_var_col_list = sorted(list(set(dependent_var_col_list)))

        return (
            data0_df,
            data_df_compiled,
            independent_var_col_list,
            dependent_var_col_list,
            time_stamp,
        )

    def read_input_matfile(self, filename=None, variable="stimulus"):
        """
        Parameters
        ----------
        filename : str
            The name of the Matlab file to read.
        variable : str
            The name of the variable to extract from the Matlab file.

        Returns
        -------
        numpy.ndarray
            A 2D NumPy array containing the `variable` data from the Matlab file.
        """

        assert (
            filename is not None
        ), "Filename not defined for read_input_matfile(), aborting..."

        analog_input = self.get_data(filename, data_type=None)
        analog_signal = analog_input[variable].T
        assert analog_signal.ndim == 2, "input is not a 2-dim vector, aborting..."
        # analog_timestep = analog_input['frameduration']

        return analog_signal

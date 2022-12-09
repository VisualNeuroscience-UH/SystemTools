# Builtin
from pathlib import Path
from argparse import ArgumentError
import copy
import sys
import shutil
import pdb
from math import nan

# Analysis
import numpy as np
import pandas as pd

from cxsystem2.core.tools import change_anat_file_header_value, read_config_file


class ProjectUtilities:
    """
    Utilities for ProjectManager class. This class is not instantiated. It serves as a container for project independent helper functions.
    """

    def transfer_precalculated_results(
        self, input_folder=None, output_folder=None, verbose=False
    ):
        """
        iput_folder: Path to folder containing precalculated results. If no input_folder is provided, use self.context.input_folder.
        If no output_folder is provided, transfer precalculated results from input_folder to self.context.output_folder.
        """
        if input_folder is None:
            input_folder = self.context.input_folder
        if output_folder is None:
            output_folder = self.context.output_folder

        # Check that input_folder and output_folder are both pathlib Path object. If not, convert them to Path objects.
        if not isinstance(input_folder, Path):
            input_folder = Path(input_folder)
        if not isinstance(output_folder, Path):
            output_folder = Path(output_folder)

        # Prepend input_folder project path
        input_folder = self.context.path.parent / input_folder
        # Prepend input_folder experiment path
        output_folder = self.context.path / output_folder

        # Check that input_folder exists
        assert input_folder.exists(), f"Input folder {input_folder} does not exist"
        # Create output_folder with parents if it does not exist
        output_folder.mkdir(parents=True, exist_ok=True)

        # Find all files in input_folder
        files = list(input_folder.glob("*"))
        # Copy files to output_folder
        for file in files:
            shutil.copy(file, output_folder)
            if verbose:
                print(f"Transferred {file.name} to {output_folder.name}")

    def phys_updater(self, physio_config_df, param_list):
        """
        Replace values in physio_config_df with values from param_list
        The param_list contains five items which are either strings or nan (from built-in math module),
        [0] = Variable name
        [1] = Key name
        [2] = Value
        [3] = Unit
        [4] = "variable" or "key", indicating whether the value is to be assigned to the row by variable or by key

        Examples:
        Replace value of variable "base_ci_path" with in_folder_full value
        ["base_ci_path", nan, f"r'{in_folder_full}'", "", "variable"]
        Replace value of variable "L4_CI_BC" key "C" with "{ 30.0 | 270.0 | 10.0 } * pF" value
        ["L4_CI_BC", "C", "{ 30.0 | 270.0 | 10.0 }", " * pF", "key"]
        """

        # Some of the param list items might be pathlib objects, convert them to strings
        param_list = [
            str(item) if isinstance(item, Path) else item for item in param_list
        ]

        # Find row index for correct Variable
        index = physio_config_df.index
        condition = physio_config_df["Variable"] == param_list[0]
        variable_index = index[condition].values
        assert (
            len(variable_index) == 1
        ), "Zero or non unique variable name found, aborting..."

        if param_list[4] == "variable":
            physio_config_df.loc[variable_index[0], "Value"] = (
                param_list[2] + param_list[3]
            )
        elif param_list[4] == "key":
            condition_keys = physio_config_df["Key"] == param_list[1]
            key_indices = index[condition_keys].values
            # Find first correct Key after correct Variable. This is dangerous, because it does not check for missing Key
            key_index = key_indices[key_indices >= variable_index][0]
            physio_config_df.loc[key_index, "Value"] = (
                param_list[2] + param_list[3]
            )  # concatenate value and unit
        else:
            raise NotImplementedError(
                'Unknown row_by value, should be "key" or "variable"'
            )
        return physio_config_df

    def prepare_csvs_for_simulation(
        self,
        anat_name=None,
        phys_name=None,
        anat_param_to_change: dict = None,
        phys_param_to_change: dict = None,
    ):
        """
        This function prepares the csv files for simulation.

        For each startpoint, it creates a new folder in path_to_csvs_out, and copies the csv files from path_to_csvs_in to the new folder.
        Then it changes the anatomical and physiological parameters in the csv files according to the input dictionaries. Note that
        only anat header values can be changed, not neuron group or synapse parameters. For these, the csv files must be edited manually
        and the manually edited csv files must be copied to path_to_csvs_in. Provide the manually edited file name as the anat_file.
        """

        output_folder = self.context.output_folder
        if anat_param_to_change is None:
            anat_param_to_change = {
                "workspace_path": self.context.path,
                "simulation_title": output_folder.name,
            }
        if phys_param_to_change is None:
            phys_param_to_change = {
                "base_ci_path": f"r'{str(self.context.input_folder)}'",
                "ci_filename": f"r'{str(self.context.input_filename)[:-4]}_ci.mat'",
            }

        startpoint_string_end_idx = str(output_folder.name).find("_")
        startpoint = str(output_folder.name)[0:startpoint_string_end_idx]
        parameter = str(output_folder.name)[startpoint_string_end_idx + 1 :]

        startpoint_csv_folder = self.context.startpoint_csv_folder
        path_to_csvs_in = self.context.path.parent.joinpath(startpoint_csv_folder)

        # From path_to_csvs_in find all csv files
        csv_files = list(path_to_csvs_in.glob("*.csv"))

        if anat_name is not None:
            anat_file_list = [
                file
                for file in csv_files
                if anat_name == file.stem or anat_name == file.name
            ]
        else:
            anat_file_list = [
                file
                for file in csv_files
                if f"Anat_{startpoint}_" in file.stem and "startpoint" in file.stem
            ]

        # From theses csv files check that one and only one is the anatomical file
        assert len(anat_file_list) == 1, "Anatomical file not found or not unique"
        anat_file_fullpath = anat_file_list[0]
        # Get timestamp from the anat file
        timestamp_anat = anat_file_fullpath.name.split("_")[-2]

        anat_file_out = f"Anat_{startpoint}_{timestamp_anat}_{parameter}.csv"
        anat_file_fullpath_out = Path.joinpath(self.context.path, anat_file_out)

        # Create the path_to_csvs_out folder if it does not exist
        if not self.context.path.exists():
            self.context.path.mkdir(parents=False)

        # Copy original files to new folder
        shutil.copyfile(anat_file_fullpath, anat_file_fullpath_out)

        if anat_param_to_change is not None:
            # Update the anat csv parameters
            for this_key in anat_param_to_change:
                change_anat_file_header_value(
                    anat_file_fullpath_out,
                    anat_file_fullpath_out,
                    this_key,
                    anat_param_to_change[this_key],
                )

        if phys_name is not None:
            phys_file_list = [
                file
                for file in csv_files
                if phys_name == file.stem or phys_name == file.name
            ]
        else:
            phys_file_list = [
                file
                for file in csv_files
                if f"Phys_{startpoint}_" in file.stem and "startpoint" in file.stem
            ]
        assert len(phys_file_list) == 1, "Physiological file not found or not unique"
        phys_file_fullpath = phys_file_list[0]
        timestamp_phys = phys_file_fullpath.name.split("_")[-2]

        phys_file_out = f"Phys_{startpoint}_{timestamp_phys}_{parameter}.csv"
        phys_file_fullpath_out = Path.joinpath(self.context.path, phys_file_out)

        # Read the phys csv file into dataframe
        phys_df = pd.read_csv(phys_file_fullpath, header=0)
        if phys_param_to_change is not None:
            # Convert dict to list of lists
            phys_params_to_change = [
                #  [Variable (str), Key (str/nan), Value (str), Unit (str), "variable" / "key"]
                [Variable, nan, phys_param_to_change[Variable], "", "variable"]
                for Variable in phys_param_to_change
            ]

            # Update the phys csv parameters
            for param_list in phys_params_to_change:
                phys_df = self.phys_updater(phys_df, param_list)

        # Write the updated phys_df to file
        phys_df.to_csv(phys_file_fullpath_out, index=False, header=True)

        return anat_file_fullpath_out, phys_file_fullpath_out

    def multiple_cluster_metadata_compiler_and_data_transfer(self):
        # This searches the cluster_metadata_XXX_.pkl files from the current path/cluster_run_XXX folders
        # The list of fullpath/cluster_metadata_XXX_.pkl strings are input to cluster_metadata_compiler_and_data_transfer
        # one-by-one

        # get list of cluster_run paths in temporal order, earliest first
        path_list = sorted(self.listdir_loop(self.path, "cluster_run_20", None))
        for this_path in path_list:
            # Calculate expected number of files
            job_file_list = self.listdir_loop(
                Path.joinpath(self.path, this_path), "_tmp_slurm_20", None
            )
            zero_file_dict = {
                f: int(f[f.find("_part") + 5 : f.find(".job")]) for f in job_file_list
            }
            latest_file = max(zip(zero_file_dict.values(), zero_file_dict.keys()))[1]
            with open(latest_file, "r") as file:
                all_lines_list = file.readlines()
            last_line = all_lines_list[-1]
            start_str = "cluster_run_start_idx="
            step_str = "cluster_run_step="
            cluster_run_start_idx = int(
                last_line[
                    last_line.find(start_str)
                    + len(start_str) : last_line.find(step_str)
                    - 1
                ]
            )
            cluster_run_step = int(
                last_line[
                    last_line.find(step_str)
                    + len(step_str) : last_line.find("); cx.run()")
                    - 1
                ]
            )
            expected_number_of_files = cluster_run_start_idx + cluster_run_step

            # Check if this many data files exist in downloads folder
            datafile_list = self.listdir_loop(
                Path.joinpath(self.path, this_path, "downloads"), ".gz", "metadata"
            )
            print(f"expected_number_of_files = {expected_number_of_files}")
            print(f"len(datafile_list) = {len(datafile_list)}")
            if len(datafile_list) < expected_number_of_files:
                print(
                    f"WARNING: possible socket.timeout() or missing data in:{this_path}"
                )
                continue

    def cluster_metadata_compiler_and_data_transfer(
        self, metapath_pkl_download_fullfile
    ):
        """
        Cluster metadata comes in partial files metadata_part_1... etc.
        This method combines these parts to a single metadata file for analysis and visualization

        It also adds folder with name corresponding to "simulation title" parameter in the anatomy csv
        metapath_pkl_download_fullfile is full path to .pkl file containing global metadata about the cluster run.
        It should be in the downloads folder after downloading the results from cluster.
        """

        metafile_master_dict = self.get_data(metapath_pkl_download_fullfile)
        local_cluster_run_download_folder = metafile_master_dict[
            "local_cluster_run_download_folder"
        ]
        metapathfiles = Path.iterdir(local_cluster_run_download_folder)
        metafiles_list = [f for f in metapathfiles if "metadata_part" in f]
        metafiles_fullfile_list = []
        for this_file in metafiles_list:
            metafiles_fullfile_list.append(
                Path.joinpath(local_cluster_run_download_folder, this_file)
            )

        # Read first metadata file to df
        metafile_cluster_paths_df = self.get_data(metafiles_fullfile_list[0])

        # Go through all files in the list
        for this_file in metafiles_fullfile_list[1:]:
            this_df = self.get_data(this_file)
            # Put not nan values from ['Full path'] column to metafile_cluster_paths_df
            notna_idx = this_df["Full path"].notna().values
            metafile_cluster_paths_df["Full path"][notna_idx] = this_df["Full path"][
                notna_idx
            ]

        # Get simulation folder name, a.k.a. condition, from metafile_master_dict['cluster_simulation_folder']
        cluster_simulation_folder = metafile_master_dict["cluster_simulation_folder"]
        simulation_folder_name = cluster_simulation_folder.name

        # Change cluster_simulation_folder root to final local_workspace
        path_to_remove = cluster_simulation_folder.parent
        path_to_paste = Path(metafile_master_dict["local_workspace"])
        metafile_local_paths_df = copy.deepcopy(metafile_cluster_paths_df)
        # Change paths in-place
        metafile_local_paths_df["Full path"] = metafile_local_paths_df[
            "Full path"
        ].str.replace(
            pat=path_to_remove.as_posix(), repl=path_to_paste.as_posix(), regex=True
        )

        # Create local repo
        local_simulation_folder = path_to_paste.joinpath(simulation_folder_name)
        local_simulation_folder.mkdir(exist_ok=True)

        # Save compiled metadata file to final local repo
        meta_file_name = f'metadata_{metafile_master_dict["suffix"]}.gz'
        meta_fullpath_out = Path.joinpath(
            path_to_paste, simulation_folder_name, meta_file_name
        )
        self.data_io.write_to_file(meta_fullpath_out, metafile_local_paths_df)

        # Move relevant files to final local repo
        local_download_folder = metafile_master_dict[
            "local_cluster_run_download_folder"
        ]
        path_to_paste_download = Path(local_download_folder)
        local_download_paths_S = copy.deepcopy(metafile_cluster_paths_df["Full path"])
        local_download_paths_S = local_download_paths_S.str.replace(
            pat=path_to_remove.joinpath(simulation_folder_name).as_posix(),
            repl=path_to_paste_download.as_posix(),
            regex=True,
        )
        local_final_paths_S = copy.deepcopy(metafile_local_paths_df["Full path"])

        # Loop filenames and replace addresses with Path(). This will move the files to final position without copying them
        try:
            [
                Path(dp).replace(fp)
                for dp, fp in zip(local_download_paths_S.values, local_final_paths_S)
            ]
        except FileNotFoundError:
            print(
                "Did not find data files, maybe they were already moved. Continuing..."
            )

        # Update master metadata file, write new metadatafile to project folder
        metafile_master_dict["project_data_folder"] = local_simulation_folder.as_posix()
        metadata_pkl_filename = metapath_pkl_download_fullfile.name
        metapath_pkl_final_fullfile = local_simulation_folder.joinpath(
            metadata_pkl_filename
        )
        self.data_io.write_to_file(metapath_pkl_final_fullfile, metafile_master_dict)

        # Loop anat and phys and replace addresses with Path(). This will move the files to final position without copying them
        allfiles = Path.iterdir(local_cluster_run_download_folder)
        anat_phys_filename_list = [f for f in allfiles if "anat" in f or "phys" in f]
        anat_phys_download_fullfile_list = []
        anat_phys_final_fullfile_list = []
        for this_file in anat_phys_filename_list:
            anat_phys_download_fullfile_list.append(
                Path.joinpath(local_cluster_run_download_folder, this_file)
            )
            anat_phys_final_fullfile_list.append(
                Path.joinpath(local_simulation_folder.as_posix(), this_file)
            )

        try:
            [
                Path(dp).replace(fp)
                for dp, fp in zip(
                    anat_phys_download_fullfile_list, anat_phys_final_fullfile_list
                )
            ]
        except FileNotFoundError:
            print(
                "Did not find anat & phys files, maybe they were already moved. Continuing..."
            )

    def cluster_to_ws_metadata_mapper(self, metapath_pkl_download_fullfile):
        """
        This checks if cluster_metadata pkl file contains local folders in the current system. If not, it assumes
        the .pkl file folder's (named cluster_run...) parent folder is the local workspace folder.

        """

        cluster_metafile_dict = self.get_data(metapath_pkl_download_fullfile)

        # Check whether current folder match cluster_metafile_dict['local workspace']
        pkl_path = Path(metapath_pkl_download_fullfile)
        local_workspace = pkl_path.parents[1].as_posix()
        if local_workspace == cluster_metafile_dict["local_workspace"]:
            print(
                "local_workspace matches cluster_metadata.pkl local_workspace, no need to update"
            )
        else:  # If not, backup cluster_metadata.pkl local workspace folder and check and set local workspace
            cluster_metafile_dict["local_workspace_backup"] = cluster_metafile_dict[
                "local_workspace"
            ]
            cluster_metafile_dict["local_workspace"] = local_workspace
            cluster_metafile_dict[
                "local_workspace_unexpanded_backup"
            ] = cluster_metafile_dict["local_workspace"]
            cluster_metafile_dict["local_workspace_unexpanded"] = local_workspace

        local_cluster_run_download_folder = (
            pkl_path.parents[0].joinpath("downloads").as_posix()
        )
        local_cluster_run_folder = pkl_path.parents[0].as_posix()
        if (
            local_cluster_run_download_folder
            == cluster_metafile_dict["local_cluster_run_download_folder"]
        ):
            print(
                "local_cluster_run_download_folder matches cluster_metadata.pkl local_cluster_run_download_folder, no need to update"
            )
        else:  # If not, backup and set local
            cluster_metafile_dict[
                "local_cluster_run_download_folder_backup"
            ] = cluster_metafile_dict["local_cluster_run_download_folder"]
            cluster_metafile_dict[
                "local_cluster_run_download_folder"
            ] = local_cluster_run_download_folder
            cluster_metafile_dict[
                "local_cluster_run_folder_backup"
            ] = cluster_metafile_dict["local_cluster_run_folder"]
            cluster_metafile_dict["local_cluster_run_folder"] = local_cluster_run_folder

        backup_project_data_folder = cluster_metafile_dict["project_data_folder"]
        project_folder_name = Path(backup_project_data_folder).parts[-1]
        local_project_data_folder = (
            Path(local_workspace).joinpath(project_folder_name).as_posix()
        )
        cluster_metafile_dict["project_data_folder_backup"] = cluster_metafile_dict[
            "project_data_folder"
        ]
        cluster_metafile_dict["project_data_folder"] = local_project_data_folder

        # save tranformed cluster_metadata file
        self.data_io.write_to_file(
            metapath_pkl_download_fullfile, cluster_metafile_dict
        )

    def round_to_n_significant(self, value_in, significant_digits=3):

        boolean_test = value_in != 0

        if boolean_test and not np.isnan(value_in):
            int_to_subtract = significant_digits - 1
            value_out = round(
                value_in, -int(np.floor(np.log10(np.abs(value_in))) - int_to_subtract)
            )
        else:
            value_out = value_in

        return value_out

    def meta_substr_replace(self, meta_fname, old, new="", new_meta_fname=None):
        """
        Manual Full path manipulation for metadata.gz. Without new str set, will remove the old substring.

        For not manipulating accidentally metadata in other folders (from path context), this method works only either at the metadata folder or with full path.
        """
        if not isinstance(old, str) or not isinstance(new, str):
            raise ArgumentError(
                "The second and optional third arguments should be strings, aborting..."
            )

        if Path(meta_fname).is_file() and "metadata" in str(meta_fname):
            data_df = self.data_io.get_data(meta_fname)
        else:
            raise FileNotFoundError(
                "The first argument must be valid metadata file name in current folder, or full path to metadata file"
            )

        if not "Full path" in data_df.columns:
            raise KeyError('Cannot find "Full path" column, nothing to do, aborting...')

        before_str = str(data_df.loc[0]["Full path"])
        print(f"Before (example): {before_str}")

        format = lambda x: str(x).replace(old, new)
        data_df["Full path"].map(format)

        after_str = str(data_df.loc[0]["Full path"])
        print(f"After (example): {after_str}")

        if before_str == after_str:
            print("Nothing changed...")

        if new_meta_fname is None:
            new_meta_fname = meta_fname

        self.data_io.write_to_file(new_meta_fname, data_df)

    def update_metadata(self, meta_full):
        """
        For folder path changes or transferring of data between systems, we need to update metadata paths
        """

        # Get metadata file as df
        data_df = self.data_io.get_data(meta_full)

        # Check whether metadata full path data (.gz) files exist in their expected places
        full_data_filepaths_list = data_df["Full path"].values.tolist()
        data_file_exist_list_of_bool = [
            Path(f).is_file() for f in full_data_filepaths_list
        ]
        if all(data_file_exist_list_of_bool) is True:
            print(f"All datafiles in metadatafile found")
            return meta_full, data_df

        # We assume that the data files are in the same folder as the metadata
        # Change full path roots to metadataroot folder
        metadataroot_folder_path = meta_full.parent
        if metadataroot_folder_path == ".":
            metadataroot_folder_path = Path.joinpath(self.context.output_folder)
        # updated_fílepaths_list = []
        for this_file_name in full_data_filepaths_list:
            data_filename = Path(this_file_name).name
            updated_this_file_name = Path.joinpath(
                metadataroot_folder_path, data_filename
            )
            # Check the updated path
            if updated_this_file_name.is_file():
                data_df.replace(
                    to_replace=this_file_name,
                    value=updated_this_file_name,
                    inplace=True,
                )
            else:
                raise FileNotFoundError(
                    f"Data file {data_filename} not found in the metadata folder, aborting..."
                )

        new_metadata_filename = self._write_updated_metadata_to_file(meta_full, data_df)

        return new_metadata_filename, data_df

    def _write_updated_metadata_to_file(self, meta_full, data_df):
        if "_updated" in meta_full.stem:
            new_metadata_filename = meta_full
        else:
            # Write updated metadata to file
            metadataroot = Path.joinpath(meta_full.parent, meta_full.stem)
            metadataextension = meta_full.suffix
            new_metadata_filename = f"{metadataroot}_updated{metadataextension}"
        self.data_io.write_to_file(new_metadata_filename, data_df)
        return new_metadata_filename

    def destroy_from_folders(self, path=None, dict_key_list=None):
        """
        Run destroy_data from root folder, deleting selected variables from data files one level towards leafs.
        """

        if path is None:
            p = Path(".")
        elif isinstance(path, Path):
            p = path
        elif isinstance(path, str):
            p = Path(path)
            if not p.is_dir():
                raise ArgumentError(f"path argument is not valid path, aborting...")

        folders = [x for x in p.iterdir() if x.is_dir()]
        metadata_full = []
        for this_folder in folders:
            for this_file in list(this_folder.iterdir()):
                if "metadata" in str(this_file):
                    metadata_full.append(this_file.resolve())

        for this_metadata in metadata_full:
            try:
                print(f"Updating {this_metadata}")
                updated_meta_full, foo_df = self.update_metadata(this_metadata)
                self.destroy_data(updated_meta_full, dict_key_list=dict_key_list)
            except FileNotFoundError:
                print(f"No files for {this_metadata}, nothing changed...")

    def destroy_data(self, meta_fname, dict_key_list=None):
        """
        Sometimes you have recorded too much and you want to reduce the filesize by removing some data.

        For not manipulating accidentally data in other folders (from path context), this method works only either at the metadata folder or with full path.

        :param meta_fname: str or pathlib object, metadata file name or full path
        :param dict_key_list: list, list of dict keys to remove from the file.
        example dict_key_list={'vm_all' : ['NG1_L4_CI_SS_L4', 'NG2_L4_CI_BC_L4']}

        Currently specific to destroying the second level of keys, as in above example.
        """
        if dict_key_list is None:
            raise ArgumentError(
                dict_key_list, "dict_key_list is None - nothing to do, aborting..."
            )

        if Path(meta_fname).is_file() and "metadata" in str(meta_fname):
            meta_df = self.data_io.get_data(meta_fname)
        else:
            raise FileNotFoundError(
                "The first argument must be valid metadata file name in current folder, or full path to metadata file"
            )

        def format(filename, dict_key_list):
            # This will destroy the selected data
            data_dict = self.data_io.get_data(filename)
            for key in dict_key_list.keys():
                for key2 in dict_key_list[key]:
                    try:
                        del data_dict[key][key2]
                    except KeyError:
                        print(
                            f"Key {key2} not found, assuming removed, nothing changed..."
                        )
                        return
            self.data_io.write_to_file(filename, data_dict)

        for filename in meta_df["Full path"]:
            if Path(filename).is_file():
                format(filename, dict_key_list)

    def end2idx(self, t_idx_end, n_samples):

        if t_idx_end is None:
            t_idx_end = n_samples
        elif t_idx_end < 0:
            t_idx_end = n_samples + t_idx_end
        return t_idx_end

    def metadata_manipulator(
        self, meta_full=None, filename=None, multiply_rows=1, replace_dict={}
    ):
        """
        Replace strings in a metadata file.
        :param path: str or pathlib object
        :param filename: str, metadata filename, if empty, search most recent in path
        :param replace_dict: dict,
            keys: 'columns', 'find' and 'replace'
            values: lists of same length
            key: 'rows'
            values: list of row index values (as in df.loc) for the changes to apply
        """

        if meta_full is None:
            raise ArgumentError("Need full path to metadatafile, aborting...")

        if not replace_dict:
            raise ArgumentError("Missing replace dict, aborting...")

        data_df = self.data_io.load_from_file(meta_full)

        # multiply rows by factor multiply_rows
        multiply_rows = 2
        new_df = pd.DataFrame(
            np.repeat(data_df.values, multiply_rows, axis=0), columns=data_df.columns
        )

        for this_row in replace_dict["rows"]:
            for col_idx, this_column in enumerate(replace_dict["columns"]):
                f = replace_dict["find"][col_idx]
                r = replace_dict["replace"][col_idx]
                print(f"Replacing {f=} for {r=}, for {this_row=}, {this_column=}")
                new_df.loc[this_row][this_column] = str(
                    new_df.loc[this_row][this_column]
                ).replace(
                    f, r
                )  # str method

        self.pp_df_full(new_df)
        new_meta_full = self._write_updated_metadata_to_file(meta_full, new_df)
        print(f"Created {new_meta_full}")

    # Debugging
    def pp_df_full(self, df):
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.max_colwidth",
            -1,
        ):
            print(df)

    def pp_df_memory(self, df):
        BYTES_TO_MB_DIV = 0.000001
        mem = round(df.memory_usage().sum() * BYTES_TO_MB_DIV, 3)
        print("Memory usage is " + str(mem) + " MB")

    def pp_obj_size(self, obj):
        from IPython.lib.pretty import pprint

        pprint(obj)
        print(f"\nObject size is {sys.getsizeof(obj)} bytes")

    def get_added_attributes(self, obj1, obj2):

        XOR_attributes = set(dir(obj1)).symmetric_difference(dir(obj2))
        unique_attributes_list = [n for n in XOR_attributes if not n.startswith("_")]
        return unique_attributes_list

    def pp_attribute_types(self, obj, attribute_list=[]):

        if not attribute_list:
            attribute_list = dir(obj)

        for this_attribute in attribute_list:
            attribute_type = eval(f"type(obj.{this_attribute})")
            print(f"{this_attribute}: {attribute_type}")

    def countlines(self, start, lines=0, header=True, begin_start=None):
        # Counts lines in folder .py files.
        # From https://stackoverflow.com/questions/38543709/count-lines-of-code-in-directory-using-python
        if header:
            print("{:>10} |{:>10} | {:<20}".format("ADDED", "TOTAL", "FILE"))
            print("{:->11}|{:->11}|{:->20}".format("", "", ""))

        for thing in Path.iterdir(start):
            thing = Path.joinpath(start, thing)
            if thing.is_file():
                if thing.endswith(".py"):
                    with open(thing, "r") as f:
                        newlines = f.readlines()
                        newlines = len(newlines)
                        lines += newlines

                        if begin_start is not None:
                            reldir_of_thing = "." + thing.replace(begin_start, "")
                        else:
                            reldir_of_thing = "." + thing.replace(start, "")

                        print(
                            "{:>10} |{:>10} | {:<20}".format(
                                newlines, lines, reldir_of_thing
                            )
                        )

        for thing in Path.iterdir(start):
            thing = Path.joinpath(start, thing)
            if Path.is_dir(thing):
                lines = self.countlines(thing, lines, header=False, begin_start=start)

        return lines

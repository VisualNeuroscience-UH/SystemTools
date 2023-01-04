from abc import ABCMeta, abstractmethod


class DataIOBase(metaclass=ABCMeta):
    """
    This class defines the interface for reading and writing data and provides several abstract methods that must be implemented by derived classes.

    Properties
    ----------
    context : object
        An object containing context information that is used when reading and writing data.

    Methods
    -------
    get_data()
        Reads data from a file or other source.
    listdir_loop()
        Finds files and folders in a specified directory with a specified key substring and exclusion substring.
    most_recent()
        Finds the most recently modified file or folder in a specified directory with a specified key substring and exclusion substring.
    parse_path()
        Returns the full path to either a specified file or to the most recently updated file with a specified key substring.
    get_csv_as_df()
        Returns a Pandas DataFrame containing the data from one or more CSV files found in a specified directory and its subdirectories.
    """

    @property
    @abstractmethod
    def context():
        pass

    @abstractmethod
    def get_data():
        pass

    @abstractmethod
    def listdir_loop():
        pass

    @abstractmethod
    def most_recent():
        pass

    @abstractmethod
    def parse_path():
        pass

    @abstractmethod
    def get_csv_as_df():
        pass

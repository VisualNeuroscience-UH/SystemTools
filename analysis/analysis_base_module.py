from abc import ABCMeta, abstractmethod

class AnalysisBase(metaclass = ABCMeta):

    @property
    @abstractmethod
    def context():
        pass

    @property
    @abstractmethod
    def data_io():
        pass

    @property
    @abstractmethod
    def cxparser():
        pass

    @abstractmethod
    def get_PCA():
        pass

    @abstractmethod
    def get_MSE():
        pass

    @abstractmethod
    def analyze_arrayrun():
        pass

    @abstractmethod
    def analyze_IxO_array():
        pass

    @abstractmethod
    def compile_analyzes_over_iterations():
        pass

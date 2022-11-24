from abc import ABCMeta, abstractmethod


class VizBase(metaclass = ABCMeta):

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

    @property
    @abstractmethod
    def ana():
        pass


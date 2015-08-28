from abc import ABCMeta, abstractmethod


class BaseConnector(object):
    """
    Abstract Base class for all database / file database connections
    Will not allow creating instances, unless methods decorated with @abstractmethod is reeimplemented
    All connections in RunConfiguration is checked if they are a subclass of this.
    """

    __metaclass__ = ABCMeta

    def __enter__(self):
        return self._open_resource()

    def __exit__(self, type, value, traceback):
        self._cleanup()

    @abstractmethod
    def _cleanup(self):
        '''Release resources used for reading'''
        pass

    @abstractmethod
    def _open_resource(self):
        '''Open any resources needed in read_data '''
        pass

    @abstractmethod
    def read_data(self):
        '''Read the data'''
        pass

    @abstractmethod
    def write_results(self):
        '''Write the results to the defined connector'''
        pass

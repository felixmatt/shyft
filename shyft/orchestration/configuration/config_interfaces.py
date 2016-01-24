from abc import ABCMeta, abstractmethod

class RegionConfigError(Exception):
    pass


class RegionConfig(object):
    __metaclass__ = ABCMeta
    """
    Provides region model catchment level parameters and domain bounding box
    """
    @abstractmethod
    def parameter_overrides(self):
        """
        TODO: Comments from SiH:
          Intent is clear, but returned result is unclear
          regarding definition of valid parameter_overrides.
          The practical use is like
           {    1234: {
                    kirchner:{c1:-2.41,c2:-0.93},
                    gamma_snow:{tx:0.7}
                },
                5678: {
                    kirchner: {c3:0.03},
                    gamma_snow:{cx:-0.8}
                }
            }

            What if we instead of overrides, just specified the complete valid parameters ?
            Then we got type-safe checked parameters instead.
            Then any errors (in name/mapping from storage to shyft) is detected early,
             - at the spot where it needs to be fixed - ?


        Returns
        -------
        overrides: dict
            Dictionary with parameter overrides for catchments:
            {catchment_id: parameter_overrides}
        """
        pass

    @abstractmethod
    def domain(self):
        """
        TODO: Comments from SiH;
          This specific method should provide a BoundingBoxRegion(interfaces.BoundingRegion)
          instead of a dictionary.
        Returns
        -------
        domain: dict
            Dictionary with region specification, or None.
        """
        pass

    @abstractmethod
    def repository(self):
        """
        Returns
        -------
        repository: dict
            dict with key "class" and value subclass of
            shyft.repository.interfaces.RegionModelRepository
            Additional key/value pairs can be found, and are typically used
            as arguments to the repository constructor.
        """
        pass


class ModelConfig(object):
    __metaclass__ = ABCMeta
    """
    Provide
        interpolation parameters, for projection of region level environment
            like precipitation temperature, radiation, wind-speed.

        model parameters, specific for the method stacks to be used.

    """
    @abstractmethod
    def interpolation_parameters(self):
        """
        TODO: Comments from SiH;
            Use api-type for interpolation parameters, and return that.
            Ensures that the config provides something well known and verified.

        Returns
        -------
        parameters: dict
            Parameters for the various interpolation routines
        """
        pass

    @abstractmethod
    def model_parameters(self):
        """
        TODO: Comments from SiH;
            Similar for the above comments, - use well defined and specific types
            instead of dictionaries. Strong model parameters types are provided by shyft api.

        Returns
        -------
        parameters: dict
            Parameters for the method stack
        """
        pass
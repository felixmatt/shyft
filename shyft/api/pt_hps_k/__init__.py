from ._pt_hps_k import *
from .. import ByteVector
# Fix up types that we need attached to the model
PTHPSKStateVector.push_back = lambda self, x: self.append(x)
PTHPSKStateVector.size = lambda self: len(self)

PTHPSKModel.cell_t = PTHPSKCellAll
PTHPSKParameter.map_t = PTHPSKParameterMap
PTHPSKModel.parameter_t = PTHPSKParameter
PTHPSKModel.state_t = PTHPSKState
PTHPSKModel.state_with_id_t = PTHPSKStateWithId
PTHPSKModel.state = property(lambda self:PTHPSKCellAllStateHandler(self.get_cells()))
PTHPSKModel.statistics = property(lambda self: PTHPSKCellAllStatistics(self.get_cells()))

PTHPSKModel.hbv_physical_snow_state = property(lambda self: PTHPSKCellHBVPhysicalSnowStateStatistics(self.get_cells()))
PTHPSKModel.hbv_physical_snow_response = property(lambda self: PTHPSKCellHBVPhysicalSnowResponseStatistics(self.get_cells()))
PTHPSKModel.priestley_taylor_response = property(lambda self: PTHPSKCellPriestleyTaylorResponseStatistics(self.get_cells()))
PTHPSKModel.actual_evaptranspiration_response=property(lambda self: PTHPSKCellActualEvapotranspirationResponseStatistics(self.get_cells()))
PTHPSKModel.kirchner_state = property(lambda self: PTHPSKCellKirchnerStateStatistics(self.get_cells()))

PTHPSKOptModel.cell_t = PTHPSKCellOpt
PTHPSKOptModel.parameter_t = PTHPSKParameter
PTHPSKOptModel.state_t = PTHPSKState
PTHPSKOptModel.state_with_id_t = PTHPSKStateWithId
PTHPSKOptModel.state = property(lambda self:PTHPSKCellOptStateHandler(self.get_cells()))
PTHPSKOptModel.statistics = property(lambda self:PTHPSKCellOptStatistics(self.get_cells()))

PTHPSKOptModel.optimizer_t = PTHPSKOptimizer

PTHPSKOptModel.full_model_t =PTHPSKModel
PTHPSKModel.opt_model_t =PTHPSKOptModel
PTHPSKModel.create_opt_model_clone = lambda self: create_opt_model_clone(self)
PTHPSKModel.create_opt_model_clone.__doc__ = create_opt_model_clone.__doc__
PTHPSKOptModel.create_full_model_clone = lambda self: create_full_model_clone(self)
PTHPSKOptModel.create_full_model_clone.__doc__ = create_full_model_clone.__doc__

PTHPSKCellAll.vector_t = PTHPSKCellAllVector
PTHPSKCellOpt.vector_t = PTHPSKCellOptVector
PTHPSKState.vector_t = PTHPSKStateVector
#PTHPSKState.serializer_t= PTHPSKStateIo

#decorate StateWithId for serialization support
def serialize_to_bytes(state_with_id_vector:PTHPSKStateWithIdVector)->ByteVector:
    if not isinstance(state_with_id_vector,PTHPSKStateWithIdVector):
        raise RuntimeError("supplied argument must be of type PTHPSKStateWithIdVector")
    return serialize(state_with_id_vector)

def __serialize_to_str(state_with_id_vector:PTHPSKStateWithIdVector)->str:
    return str(serialize_to_bytes(state_with_id_vector))  # returns hex-string formatted vector

def __deserialize_from_str(s:str)->PTHPSKStateWithIdVector:
    return deserialize_from_bytes(ByteVector.from_str(s))

PTHPSKStateWithIdVector.serialize_to_bytes = lambda self: serialize_to_bytes(self)
PTHPSKStateWithIdVector.serialize_to_str = lambda self: __serialize_to_str(self)
PTHPSKStateWithIdVector.deserialize_from_str = __deserialize_from_str
PTHPSKStateWithIdVector.state_vector = property(lambda self: extract_state_vector(self),doc=extract_state_vector.__doc__)
PTHPSKStateWithId.vector_t = PTHPSKStateWithIdVector

def deserialize_from_bytes(bytes: ByteVector)->PTHPSKStateWithIdVector:
    if not isinstance(bytes,ByteVector):
        raise RuntimeError("Supplied type must be a ByteVector, as created from serialize_to_bytes")
    states=PTHPSKStateWithIdVector()
    deserialize(bytes,states)
    return states
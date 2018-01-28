from ._hbv_stack import *
from .. import ByteVector
# Fix up types that we need attached to the model
HbvStateVector.push_back = lambda self, x: self.append(x)
HbvStateVector.size = lambda self: len(self)

HbvModel.cell_t = HbvCellAll
HbvParameter.map_t = HbvParameterMap
HbvModel.parameter_t = HbvParameter
HbvModel.state_t = HbvState
HbvModel.state_with_id_t = HbvStateWithId
HbvModel.state = property(lambda self:HbvCellAllStateHandler(self.get_cells()))
HbvModel.statistics = property(lambda self: HbvCellAllStatistics(self.get_cells()))
HbvModel.hbv_snow_state = property(lambda self: HbvCellHBVSnowStateStatistics(self.get_cells()))
HbvModel.hbv_snow_response = property(lambda self: HbvCellHBVSnowResponseStatistics(self.get_cells()))
HbvModel.priestley_taylor_response = property(lambda self: HbvCellPriestleyTaylorResponseStatistics(self.get_cells()))
HbvModel.hbv_actual_evaptranspiration_response=property(lambda self: HbvCellHbvActualEvapotranspirationResponseStatistics(self.get_cells()))
HbvModel.soil_state = property(lambda self: HbvCellSoilStateStatistics(self.get_cells()))
HbvModel.tank_state = property(lambda self: HbvCellTankStateStatistics(self.get_cells()))
HbvModel.create_opt_model_clone = lambda self: create_opt_model_clone(self)
HbvModel.create_opt_model_clone.__doc__ = create_opt_model_clone.__doc__
HbvOptModel.create_full_model_clone = lambda self: create_full_model_clone(self)
HbvOptModel.create_full_model_clone.__doc__ = create_full_model_clone.__doc__

HbvOptModel.cell_t = HbvCellOpt
HbvOptModel.parameter_t = HbvParameter
HbvOptModel.state_t = HbvState
HbvOptModel.state_with_id_t=HbvStateWithId
HbvOptModel.state = property(lambda self:HbvCellOptStateHandler(self.get_cells()))
HbvOptModel.statistics = property(lambda self:HbvCellOptStatistics(self.get_cells()))
HbvOptModel.optimizer_t = HbvOptimizer
HbvOptModel.full_model_t =HbvModel
HbvModel.opt_model_t =HbvOptModel

HbvCellAll.vector_t = HbvCellAllVector
HbvCellOpt.vector_t = HbvCellOptVector
HbvState.vector_t = HbvStateVector

#decorate StateWithId for serialization support
def serialize_to_bytes(state_with_id_vector:HbvStateWithIdVector)->ByteVector:
    if not isinstance(state_with_id_vector,HbvStateWithIdVector):
        raise RuntimeError("supplied argument must be of type HbvStateWithIdVector")
    return serialize(state_with_id_vector)

def __serialize_to_str(state_with_id_vector:HbvStateWithIdVector)->str:
    return str(serialize_to_bytes(state_with_id_vector))  # returns hex-string formatted vector

def __deserialize_from_str(s:str)->HbvStateWithIdVector:
    return deserialize_from_bytes(ByteVector.from_str(s))

HbvStateWithIdVector.serialize_to_bytes = lambda self: serialize_to_bytes(self)
HbvStateWithIdVector.serialize_to_str = lambda self: __serialize_to_str(self)
HbvStateWithIdVector.state_vector = property(lambda self: extract_state_vector(self),doc=extract_state_vector.__doc__)
HbvStateWithIdVector.deserialize_from_str = __deserialize_from_str
HbvStateWithId.vector_t = HbvStateWithIdVector

def deserialize_from_bytes(bytes: ByteVector)->HbvStateWithIdVector:
    if not isinstance(bytes,ByteVector):
        raise RuntimeError("Supplied type must be a ByteVector, as created from serialize_to_bytes")
    states=HbvStateWithIdVector()
    deserialize(bytes,states)
    return states
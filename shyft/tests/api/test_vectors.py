from builtins import range

from shyft import api
import numpy as np
from numpy.testing import assert_array_almost_equal
import unittest

class Vectors(unittest.TestCase):
    """
    Some basic test to ensure that the exposure of c++ vector<T> is working as expected
    """

    def test_double_vector(self):
        dv_from_list= api.DoubleVector([x for x in range(10)])
        dv_np = np.arange(10.0)
        dv_from_np = api.DoubleVector.from_numpy(dv_np)
        self.assertEqual(len(dv_from_list), 10)
        assert_array_almost_equal(dv_from_list.to_numpy(), dv_np)
        assert_array_almost_equal(dv_from_np.to_numpy(), dv_np)
        dv_from_np[5] = 8
        dv_from_np.append(11)
        dv_from_np.push_back(12)
        dv_np[5] = 8
        dv_np.resize(12)
        dv_np[10] = 11
        dv_np[11] = 12
        assert_array_almost_equal(dv_from_np.to_numpy(), dv_np)
        # this does not work yet
        # nv= api.DoubleVector(dv_np).. would be very nice!

    def test_int_vector(self):
        dv_from_list= api.IntVector([x for x in range(10)])
        dv_np = np.arange(10,dtype=np.int32) #notice, default is int64, which does not convert automatically to int32
        dv_from_np = api.IntVector.from_numpy(dv_np)
        self.assertEqual(len(dv_from_list), 10)
        assert_array_almost_equal(dv_from_list.to_numpy(), dv_np)
        assert_array_almost_equal(dv_from_np.to_numpy(), dv_np)
        dv_from_np[5] = 8
        dv_from_np.append(11)
        dv_from_np.push_back(12)
        dv_np[5] = 8
        dv_np.resize(12)
        dv_np[10] = 11
        dv_np[11] = 12
        assert_array_almost_equal(dv_from_np.to_numpy(), dv_np)

    def test_utctime_vector(self):
        dv_from_list= api.UtcTimeVector([x for x in range(10)])
        dv_np = np.arange(10, dtype=np.int64)
        dv_from_np = api.UtcTimeVector.from_numpy(dv_np)
        self.assertEqual(len(dv_from_list), 10)
        assert_array_almost_equal(dv_from_list.to_numpy(), dv_np)
        assert_array_almost_equal(dv_from_np.to_numpy(), dv_np)
        dv_from_np[5] = 8
        dv_from_np.append(11)
        dv_from_np.push_back(12)
        dv_np[5] = 8
        dv_np.resize(12)
        dv_np[10] = 11
        dv_np[11] = 12
        assert_array_almost_equal(dv_from_np.to_numpy(), dv_np)

    def test_string_vector(self):
        #NOTE: support for string vector is very limited, e.g. numpy does not work, only lists
        #     but for now this is sufficient in SHyFT
        s_list = ['abc', 'def']
        dv_from_list = api.StringVector(s_list)
        self.assertEqual(len(dv_from_list), 2)
        for i in range(len(dv_from_list)):
            self.assertEqual(s_list[i],dv_from_list[i])

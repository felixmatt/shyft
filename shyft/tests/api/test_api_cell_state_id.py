from shyft.api import CellStateId
import unittest


class VerifyCellStateId(unittest.TestCase):

    def test_cell_state_id(self):
        a=CellStateId()
        self.assertEqual(a.cid,0)
        self.assertEqual(a.x,0)
        self.assertEqual(a.y,0)
        self.assertEqual(a.area,0)

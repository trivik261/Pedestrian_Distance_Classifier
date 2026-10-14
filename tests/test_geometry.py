import unittest
from pedestrian import distances
class Geometry(unittest.TestCase):
 def test_vertical_centers(self):self.assertEqual(distances([[0,0,10,10],[0,30,10,10]])[0]['distance'],30)
 def test_pairs_not_permutations(self):self.assertEqual(len(distances([[0,0,2,2],[3,0,2,2],[0,4,2,2]])),3)
 def test_meters_require_scale(self):
  self.assertEqual(distances([[0,0,2,2],[0,10,2,2]],2)[0]['distance'],5)
  with self.assertRaises(ValueError):distances([],0)
 def test_empty_single(self):self.assertEqual(distances([]),[]);self.assertEqual(distances([[0,0,1,1]]),[])
if __name__=='__main__':unittest.main()

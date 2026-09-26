import math
import unittest

from scripts.quality_ablation.score import hit, percentile, ray, score_values
from scripts.quality_ablation.run import solve3, triangulate


class GeometryTests(unittest.TestCase):
    def test_point_to_plane_ray_and_radial_depth(self):
        calibration={"width":4,"height":4,"fx":2,"fy":2,"cx":1.5,"cy":1.5}
        camera={"rotation":[1,0,0,0,1,0,0,0,1],"translationMm":[0,0,0]}
        planes=[{"x":[-10,10],"y":[-10,10],"z":10}]
        target=hit(camera,3,2,4,4,calibration,planes)
        origin,direction,norm=ray(camera,3,2,4,4,4,4,calibration)
        self.assertEqual(origin,(0,0,0))
        self.assertAlmostEqual(direction[0],.75)
        self.assertAlmostEqual(direction[1],.25)
        self.assertEqual(target[0],10)
        self.assertAlmostEqual(10*norm*direction[2]/norm,10)
        self.assertGreater(10*norm,10)  # radial depth differs from camera Z

    def test_missing_inclusive_score(self):
        result=score_values([1,6],4,3,2,1)
        self.assertEqual(result["mae_mm"],3.5)
        self.assertEqual(result["p50_mm"],3.5)
        self.assertEqual(result["bad5_matched"],.5)
        self.assertEqual(result["bad5_missing_inclusive"],.75)
        self.assertEqual(result["edge_coverage"],.5)

    def test_triangulation_observations_only(self):
        c={"fx":1,"fy":1,"cx":0,"cy":0}
        views={"a":{"rotation":[1,0,0,0,1,0,0,0,1],"translationMm":[0,0,0]},
               "b":{"rotation":[1,0,0,0,1,0,0,0,1],"translationMm":[-2,0,0]}}
        point={"observations":[{"imageId":"a","pixel":[.1,0]},
                                {"imageId":"b","pixel":[-.1,0]}]}
        result=triangulate(point,views,c)
        for got,want in zip(result,[1,0,10]): self.assertAlmostEqual(got,want,places=9)


if __name__=="__main__": unittest.main()

import unittest
import numpy as np
from detail_profiles import ray_depth, relief_metrics

class ReliefTests(unittest.TestCase):
    def test_perspective_depth_of_tilted_plane(self):
        # A plane z=2+x: inverse depth is affine in screen x.
        tri=np.array([[[-1,-2,1],[1,-2,3],[1,2,3]],[[-1,-2,1],[1,2,3],[-1,2,1]]],float)
        row={'rotation':np.eye(3).tolist(),'translation':[0,0,0],'k':[10,10,5.5,5.5]}
        depth=ray_depth(tri,row,[3,3,5,5]);x=np.arange(3,8)
        expected=2/(1-(x+.5-5.5)/10)
        np.testing.assert_allclose(depth,np.broadcast_to(expected,(5,5)),rtol=1e-12)
    def test_flattened_relief_is_detected(self):
        y,x=np.mgrid[:20,:20];inverse=1+.001*x+.002*y+.02*np.exp(-((x-10)**2+(y-10)**2)/10)
        ref=1/inverse
        same,_=relief_metrics(ref,ref);self.assertAlmostEqual(same['inverse_depth_relief_gain'],1)
        flat=1/(1+.001*x+.002*y)
        lost,_=relief_metrics(ref,flat);self.assertAlmostEqual(lost['inverse_depth_relief_gain'],0,places=10)
        self.assertGreater(lost['ray_depth_rmse'],0)
    def test_missing_pixels_do_not_count_as_recovered(self):
        ref=np.ones((10,10));test=ref.copy();test[:5]=np.nan
        result,_=relief_metrics(ref,test)
        self.assertEqual(result['common_pixels'],50);self.assertEqual(result['coverage'],.5)

if __name__=='__main__':unittest.main()

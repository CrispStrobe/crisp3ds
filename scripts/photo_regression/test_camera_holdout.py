import unittest
from camera_holdout import partition, evaluate
import numpy as np
class HoldoutTests(unittest.TestCase):
    def test_transitive_track_edges_cannot_leak_into_training(self):
        matches={'keypoints':[[]]*4,'pairs':[{'first':0,'second':1,'matches':[[i,i] for i in range(20)]},{'first':1,'second':2,'matches':[[i,i] for i in range(20)]},{'first':2,'second':3,'matches':[[i,i] for i in range(20)]}]}
        train,held=partition(matches)
        self.assertEqual(len(held['tracks']),2)
        reserved={tuple(x) for t in held['tracks'] for x in t}
        for p in train['pairs']:
            for a,b in p['matches']:
                self.assertNotIn((p['first'],a),reserved);self.assertNotIn((p['second'],b),reserved)
        self.assertEqual(sum(len(p['matches']) for p in train['pairs']),54)
    def test_held_out_pose_error_is_measured_without_fitting_to_it(self):
        lens=dict(fx=100,fy=100,cx=0,cy=0,k1=0,k2=0,k3=0)
        rows=[];points=[]
        for i in range(8):
            translation=[(i-3.5)*.1,0,0]
            rows.append({'name':str(i),'rotation':np.eye(3).tolist(),'translation':translation})
            q=np.array([.1,.2,2])+translation;points.append([(q[:2]/q[2]*100).tolist()])
        held={'tracks':[[[i,0] for i in range(8)]]};matches={'keypoints':points}
        clean=evaluate(matches,held,rows,lens)
        self.assertLess(clean['held_out_reprojection_pixels']['p95'],1e-8)
        rows[1]['translation'][0]+=.1
        perturbed=evaluate(matches,held,rows,lens)
        self.assertGreater(perturbed['per_view'][1]['median'],4.9)
        self.assertEqual(perturbed['fit_verified_tracks'],1)
if __name__=='__main__':unittest.main()

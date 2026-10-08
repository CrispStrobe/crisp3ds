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
    def test_mask_groups_use_recovered_intrinsics_without_changing_scores(self):
        lens=dict(fx=100,fy=100,cx=0,cy=0,k1=0,k2=0,k3=0)
        rows=[];points=[];masks=[]
        for i in range(8):
            translation=[(i-3.5)*.1,0,0]
            rows.append({'name':str(i),'rotation':np.eye(3).tolist(),'translation':translation,'k':[100,100,50.5,50.5],'width':100,'height':100})
            observations=[]
            for y in [.2,-.2]:
                q=np.array([.1,y,2])+translation;observations.append((q[:2]/q[2]*100).tolist())
            points.append(observations)
            mask=np.zeros((100,100),np.uint8);mask[50:]=255;masks.append(mask)
        held={'tracks':[[[i,f] for i in range(8)] for f in range(2)]};matches={'keypoints':points}
        rows[1]['translation'][0]+=.1
        original=evaluate(matches,held,rows,lens)
        grouped=evaluate(matches,held,rows,lens,masks)
        self.assertEqual(original['held_out_reprojection_pixels'],grouped['held_out_reprojection_pixels'])
        self.assertEqual(original['fit_verified_tracks'],grouped['fit_verified_tracks'])
        for name in ['foreground','background']:
            self.assertEqual(grouped['photo_mask_groups'][name]['held_out_reprojection_pixels']['count'],4)
            self.assertEqual(grouped['photo_mask_groups'][name]['fit_verified_held_out_pixels']['count'],4)
        self.assertEqual(grouped['photo_mask_groups']['outside_mask']['held_out_reprojection_pixels']['count'],0)
        # A one-pixel mask tests the engine's half-pixel centre convention.
        narrow=[]
        for observations in points:
            mask=np.zeros((100,100),np.uint8)
            x,y=np.floor(np.array(observations[0])+[50.5,50.5]).astype(int)
            mask[y,x]=255;narrow.append(mask)
        narrow_result=evaluate(matches,held,rows,lens,narrow)
        self.assertEqual(narrow_result['photo_mask_groups']['foreground']['held_out_reprojection_pixels']['count'],4)
        with self.assertRaisesRegex(ValueError,'dimensions'):
            evaluate(matches,held,rows,lens,[m[:90] for m in masks])

if __name__=='__main__':unittest.main()

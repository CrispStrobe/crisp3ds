"""Synthetic-only tests; no staged TRAIN files are opened."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.classical_backend import openmvg_mustard_mask_qa as qa


class MaskQATest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'images').mkdir()
        (self.root / 'masks').mkdir()
        self.labels = self.root / 'labels'
        self.labels.mkdir()
        self.names = ('NP3_000.jpg', 'NP3_006.jpg')
        self.size = (8, 6)
        self.stage = {'schema': 'mustard_train_only_stage_v1', 'status': 'complete',
                      'mask_role': qa.ROLE, 'train_names': list(self.names),
                      'train_photo_sha256': {}, 'cleaned_masks': {}}
        self.manifest = {'schema': 'mustard_photo_semantic_labels_v1',
                         'status': 'validated_photo_only', 'source': 'manual_photo_review',
                         'reviewer': 'synthetic tester', 'reviewed_at': '2026-01-01', 'labels': {}}
        for name in self.names:
            photo = self.root / 'images' / name
            Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8), 'RGB').save(photo, format='JPEG')
            mask = np.zeros((6, 8), dtype=np.uint8)
            mask[1:4, 2:5] = 255
            mask_path = self.root / 'masks' / (name + '.png')
            Image.fromarray(mask, 'L').save(mask_path)
            label = np.zeros((6, 8), dtype=np.uint8)
            label[1:4, 2:5] = 1
            label[1:3, 5:7] = 2
            label[4:6, 0:2] = 3
            label_path = self.labels / (name + '.png')
            Image.fromarray(label, 'L').save(label_path)
            self.stage['train_photo_sha256'][name] = qa.sha256(photo)
            self.stage['cleaned_masks'][name] = {'sha256': qa.sha256(mask_path),
                                                  'kept_pixels': 9, 'ignored_pixels': 39}
            self.manifest['labels'][name] = {'sha256': qa.sha256(label_path),
                                              'photo_sha256': qa.sha256(photo)}

    def test_abstains_without_validated_labels(self):
        result = qa.audit(self.root, self.stage, size=self.size, names=self.names)
        self.assertEqual(result['semantic_status'], 'unknown_abstain')
        self.assertEqual(result['total_support_pixels'], 18)
        self.assertEqual(result['per_view'][0]['support']['components_8'], 1)
        self.assertIsNone(result['per_view'][0]['semantics']['bottle_object'])

    def test_reviewed_labels_report_only_labeled_coverage(self):
        result = qa.audit(self.root, self.stage, self.labels, self.manifest,
                          size=self.size, names=self.names)
        row = result['per_view'][0]['semantics']
        self.assertEqual(row['bottle_object']['support_coverage_of_labeled_class'], 1)
        self.assertEqual(row['board_support']['support_coverage_of_labeled_class'], 0)
        self.assertEqual(row['background']['inside_support'], 0)
        self.assertEqual(row['unknown_pixels'], 31)

    def test_invalid_provenance_fails_closed(self):
        self.manifest['source'] = 'model_inference'
        with self.assertRaisesRegex(ValueError, 'reviewed'):
            qa.audit(self.root, self.stage, self.labels, self.manifest,
                     size=self.size, names=self.names)

    def test_changed_mask_fails_closed(self):
        path = self.root / 'masks' / (self.names[0] + '.png')
        Image.fromarray(np.zeros((6, 8), dtype=np.uint8), 'L').save(path)
        with self.assertRaisesRegex(ValueError, 'changed photo or support'):
            qa.audit(self.root, self.stage, size=self.size, names=self.names)

    def test_wrong_label_geometry_fails_closed(self):
        path = self.labels / (self.names[0] + '.png')
        Image.fromarray(np.zeros((5, 8), dtype=np.uint8), 'L').save(path)
        self.manifest['labels'][self.names[0]]['sha256'] = qa.sha256(path)
        with self.assertRaisesRegex(ValueError, 'format, mode, or size'):
            qa.audit(self.root, self.stage, self.labels, self.manifest,
                     size=self.size, names=self.names)

    def test_missing_name_and_bad_codes_fail_closed(self):
        del self.manifest['labels'][self.names[0]]
        with self.assertRaisesRegex(ValueError, 'reviewed'):
            qa.audit(self.root, self.stage, self.labels, self.manifest,
                     size=self.size, names=self.names)
        label = np.zeros((6, 8), dtype=np.uint8)
        label[0, 0] = 4
        with self.assertRaisesRegex(ValueError, 'codes'):
            qa.semantic_metrics(np.zeros((6, 8), dtype=np.uint8), label)


if __name__ == '__main__':
    unittest.main()

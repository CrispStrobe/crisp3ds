import unittest
from unittest import mock

from scripts.classical_backend import mustard_mask_pair as pair


class TestMustardMaskPair(unittest.TestCase):
    def test_commands_change_only_mask_and_output(self):
        sam = pair.command(pair.OUTPUTS[0], pair.STAGE / 'masks', pair.ARM_WALL_MINUTES)
        v2 = pair.command(pair.OUTPUTS[1], pair.V2 / 'masks', pair.ARM_WALL_MINUTES)
        differing_values = {'--pose-mask-dir', '--output'}
        # Stop-after-SfM is a flag, so compare the command positionally.
        self.assertEqual(sam[:3], v2[:3])
        for flag in differing_values:
            self.assertIn(flag, sam)
            self.assertIn(flag, v2)
        for i, (left, right) in enumerate(zip(sam, v2)):
            if left != right:
                self.assertIn(sam[i - 1], differing_values)
        self.assertEqual(len(sam), len(v2))
        self.assertNotIn('--init-image-pair', sam)
        self.assertNotIn('--sparse-model', sam)
        self.assertIn('--stop-after-sfm', sam)
        self.assertEqual(sam[sam.index('--seed') + 1], '20260927')
        self.assertEqual(sam[sam.index('--sfm-intrinsics-policy') + 1], 'fixed-initial')

    def test_frozen_train_split(self):
        self.assertEqual(len(pair.mustard_stage.TRAIN_NAMES), 48)
        self.assertNotIn('NP3_024.jpg', pair.mustard_stage.TRAIN_NAMES)
        self.assertNotIn('NP3_354.jpg', pair.mustard_stage.TRAIN_NAMES)

    def test_post_receipt_preflight_failure_seals_partial_status(self):
        saved = []
        with (mock.patch.object(pair, 'preflight', side_effect=[{'status': 'ready_for_review'},
                                                                ValueError('source changed')]),
              mock.patch.object(pair, 'source_hashes', return_value={'input': 'same'}),
              mock.patch.object(pair, 'save_receipt', side_effect=lambda report: saved.append(report.copy()))):
            result = pair.run_pair()
        self.assertEqual(result['status'], 'partial_or_failed')
        self.assertIn('source changed', result['failure'])
        self.assertEqual(saved[-1]['status'], 'partial_or_failed')
        self.assertEqual(result['arms'], [])

    def test_zero_exit_without_sparse_complete_cannot_complete_pair(self):
        report = {'arms': [{'exit_code': 0, 'producer_status': 'sparse_complete'},
                           {'exit_code': 0, 'producer_status': 'failed'}],
                  'sources_unchanged': True,
                  'internal_free_bytes_after': pair.MIN_FREE,
                  'external_free_bytes_after': pair.MIN_FREE,
                  'total_output_bytes': 100}
        self.assertFalse(pair.completed(report))
        report['arms'][1]['producer_status'] = 'sparse_complete'
        self.assertTrue(pair.completed(report))


if __name__ == '__main__':
    unittest.main()

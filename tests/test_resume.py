import unittest
from unittest.mock import Mock

from lakonlab.runner.dynamic_iter_based_runner import (
    DynamicIterBasedRunnerMod, strip_initial_lr)


class ResumeTest(unittest.TestCase):

    def test_resume_requires_optimizer_state_by_default(self):
        runner = object.__new__(DynamicIterBasedRunnerMod)
        runner.load_checkpoint = Mock(return_value={
            'meta': {'epoch': 0, 'iter': 1},
            'state_dict': {},
        })

        with self.assertRaisesRegex(KeyError, 'optimizer state'):
            runner.resume('weights_only.pth', map_location='cpu')

    def test_resume_can_intentionally_skip_optimizer(self):
        runner = object.__new__(DynamicIterBasedRunnerMod)
        runner.load_checkpoint = Mock(return_value={
            'meta': {'epoch': 0, 'iter': 1},
            'state_dict': {},
        })
        runner.logger = Mock()

        runner.resume(
            'weights_only.pth', resume_optimizer=False, map_location='cpu')

        self.assertEqual(runner.iter, 1)
        runner.logger.info.assert_called_once()

    def test_initial_lr_is_removed_from_optimizer_groups(self):
        state = {
            'state': {},
            'param_groups': [{'lr': 1e-4, 'initial_lr': 1e-3}],
        }

        strip_initial_lr(state)

        self.assertNotIn('initial_lr', state['param_groups'][0])
        self.assertEqual(state['param_groups'][0]['lr'], 1e-4)


if __name__ == '__main__':
    unittest.main()

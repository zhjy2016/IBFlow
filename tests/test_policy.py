import unittest

import torch

from lakonlab.models.diffusions.policies.ibflow import IBFlowPolicy


class IBFlowPolicyTest(unittest.TestCase):

    def make_policy(self):
        x_t = torch.zeros(1, 2, 2, 2)
        denoising_output = {
            'means': torch.ones(1, 2, 2, 2, 2),
            'logweights': torch.tensor([0.0, 2.0]).reshape(1, 2, 1, 1, 1).expand(
                1, 2, 1, 2, 2).clone(),
            'loggammas': torch.zeros(1, 1, 2, 2, 2),
        }
        return IBFlowPolicy(
            denoising_output,
            x_t,
            torch.ones(1))

    def test_temperature_one_is_identity(self):
        policy = self.make_policy()
        original = policy.denoising_output_x_0['logweights'].clone()
        self.assertIs(policy.temperature_(1.0), policy)
        torch.testing.assert_close(
            policy.denoising_output_x_0['logweights'], original)

    def test_temperature_normalizes_logits(self):
        policy = self.make_policy().temperature_(0.5)
        probabilities = policy.denoising_output_x_0['logweights'].exp()
        torch.testing.assert_close(
            probabilities.sum(dim=1),
            torch.ones_like(probabilities.sum(dim=1)))

    def test_temperature_copy_does_not_mutate_source(self):
        policy = self.make_policy()
        original = policy.denoising_output_x_0['logweights'].clone()
        transformed = policy.temperature(0.5)
        torch.testing.assert_close(
            policy.denoising_output_x_0['logweights'], original)
        self.assertFalse(torch.equal(
            transformed.denoising_output_x_0['logweights'], original))

    def test_invalid_temperature(self):
        policy = self.make_policy()
        for value in (0, -1, float('inf')):
            with self.assertRaises(ValueError):
                policy.temperature_(value)
        with self.assertRaises(TypeError):
            policy.temperature_('auto')


if __name__ == '__main__':
    unittest.main()

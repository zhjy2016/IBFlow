import unittest

import torch
from mmcv import Config
from mmgen.models.builder import MODULES

from lakonlab import models  # noqa: F401
from export_ibflow_to_diffusers import (
    REQUIRED_OUTPUT_KEYS, config_signature, validate_export_tensors)


class ReleaseConfigTest(unittest.TestCase):

    def test_release_recipe_is_registered(self):
        cfg = Config.fromfile(
            'configs/qwen/ibflow_qwen_release_2nfe_k16.py')
        self.assertEqual(
            cfg.model.diffusion.type,
            'IBFlowImitationDataFreeLinearDynamicCFG')
        self.assertIsNotNone(MODULES.get(cfg.model.diffusion.type))
        self.assertEqual(cfg.train_cfg.nfe, 2)
        self.assertEqual(cfg.model.diffusion.scale_alpha_eq, 0.0)
        self.assertEqual(cfg.model.diffusion.scale_alpha_gt, 1.0)
        self.assertEqual(cfg.model.diffusion.lambda_cfg_loss, 1.0)

    def test_release_and_ibcfg_signatures_are_distinct(self):
        release_cfg = Config.fromfile(
            'configs/qwen/ibflow_qwen_release_2nfe_k16.py')
        ibcfg = Config.fromfile('configs/qwen/ibflow_qwen_2nfe_k16.py')
        self.assertNotEqual(
            config_signature(release_cfg), config_signature(ibcfg))

    def test_export_tensor_validation_accepts_complete_adapter(self):
        tensors = {
            key: torch.empty(1) for key in REQUIRED_OUTPUT_KEYS
        }
        tensors.update({
            'block.lora_A.weight': torch.empty(2, 3),
            'block.lora_B.weight': torch.empty(4, 2),
        })

        validate_export_tensors(tensors)

    def test_export_tensor_validation_rejects_missing_head(self):
        tensors = {
            key: torch.empty(1) for key in REQUIRED_OUTPUT_KEYS
            if key != 'proj_out_means.weight'
        }
        tensors.update({
            'block.lora_A.weight': torch.empty(2, 3),
            'block.lora_B.weight': torch.empty(4, 2),
        })

        with self.assertRaisesRegex(ValueError, 'missing output tensors'):
            validate_export_tensors(tensors)

    def test_export_tensor_validation_rejects_unpaired_lora(self):
        tensors = {
            key: torch.empty(1) for key in REQUIRED_OUTPUT_KEYS
        }
        tensors['block.lora_A.weight'] = torch.empty(2, 3)

        with self.assertRaisesRegex(ValueError, 'missing or unpaired'):
            validate_export_tensors(tensors)


if __name__ == '__main__':
    unittest.main()

"""Serve the fixed Bridge checkpoint using the unmodified official OpenPI model."""
import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np

from .logging_utils import digest, file_digest, git_commit, write_json
from .simpler_protocol import load_config


class BridgeInputs:
    def __call__(self, data):
        image = np.asarray(data['image'], dtype=np.uint8)
        return {'state': np.zeros(8, dtype=np.float32),
                'image': {'base_0_rgb': image, 'left_wrist_0_rgb': np.zeros_like(image),
                          'right_wrist_0_rgb': np.zeros_like(image)},
                'image_mask': {'base_0_rgb': np.True_, 'left_wrist_0_rgb': np.False_,
                               'right_wrist_0_rgb': np.False_}, 'prompt': data['prompt']}


class PairedNoisePolicy:
    def __init__(self, policy, metadata):
        self.policy, self.metadata = policy, metadata

    def infer(self, request):
        noise = np.asarray(request['noise'], dtype=np.float32)
        if noise.shape != (5, 32) or not np.isfinite(noise).all():
            raise ValueError('Explicit paired noise of shape (5, 32) is required')
        result = self.policy.infer({'image': request['image'], 'prompt': request['prompt']}, noise=noise)
        result['actions'] = np.asarray(result['actions'][:, :7], dtype=np.float32)
        if result['actions'].shape != (5, 7) or not np.isfinite(result['actions']).all():
            raise RuntimeError('The fixed native checkpoint must return five finite 7D actions')
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cfg = load_config()
    checkpoint = Path(os.environ['SV_CHECKPOINT'])
    release = json.loads((checkpoint/'provenance.json').read_text())
    assert release['model_sha256'] == cfg['checkpoint_sha256']
    assert file_digest(checkpoint/'model.safetensors') == cfg['checkpoint_sha256']
    assert file_digest(checkpoint/'norm_stats.json') == release['supporting_source_sha256']['norm_stats.json']
    assert git_commit(os.environ['SV_OPENPI']) == '215abfb217dbac7d5f1273282331b9b1866c0479'

    import torch
    from safetensors.torch import load_model
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.policies.policy import Policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.shared.normalize import NormStats

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('One allocated CUDA GPU is required')
    torch.manual_seed(cfg['seed'])
    model_config = Pi0Config(**release['saved_model_config'], pytorch_compile_mode=None)
    model = PI0Pytorch(model_config)
    # Strict loading: never use partially initialized or architecture-mismatched weights.
    load_model(model, str(checkpoint/'model.safetensors'), strict=True)
    model.paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16')
    stats_json = json.loads((checkpoint/'norm_stats.json').read_text())['norm_stats']
    stats = {k: NormStats(**{n: np.array(v) for n, v in s.items()}) for k, s in stats_json.items()}
    metadata = {'checkpoint': str(checkpoint), 'release': release, 'model': dataclasses.asdict(model_config),
                'config_name': 'pi05_bridge_saved_metadata', 'openpi_commit': git_commit(os.environ['SV_OPENPI']),
                'flow_steps': cfg['flow_steps'], 'normalization': 'quantiles',
                'sampling': 'official_OpenPI_ODE_explicit_standard_normal_noise',
                'image_preprocessing': 'BridgeInputs; official ResizeImages(224,224); no state tokens',
                'pytorch_compile_mode': None, 'native_prediction_horizon': 5}
    metadata['inference_fingerprint'] = digest(metadata)
    policy = Policy(model, is_pytorch=True, pytorch_device='cuda:0',
        transforms=[BridgeInputs(), transforms.Normalize(stats, use_quantiles=True),
                    transforms.ResizeImages(224, 224),
                    transforms.TokenizePrompt(PaligemmaTokenizer(200), discrete_state_input=False),
                    transforms.PadStatesAndActions(32)],
        output_transforms=[transforms.Unnormalize(stats, use_quantiles=True)],
        sample_kwargs={'num_steps': cfg['flow_steps']}, metadata=metadata)
    write_json(args.output, metadata)
    print('Strict checkpoint load complete; serving fixed P=5 pi0.5', flush=True)
    WebsocketPolicyServer(PairedNoisePolicy(policy, metadata), host='127.0.0.1', port=args.port,
                          metadata=metadata).serve_forever()


if __name__ == '__main__':
    main()

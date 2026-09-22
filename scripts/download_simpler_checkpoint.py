"""CPU-only preparation; download weights and provenance, never optimizer data."""
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import urllib.request
import zipfile

from huggingface_hub import hf_hub_download
import yaml

from frequency_vla.logging_utils import file_digest, write_json


class NothingPlaceholder:
    pass


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        # The saved metadata contains this marker, not weights. Do not execute
        # arbitrary globals from a remote pickle.
        if (module, name) == ('flax.nnx.filterlib', 'Nothing'):
            return NothingPlaceholder
        raise pickle.UnpicklingError(f'Unexpected metadata class: {module}.{name}')


def main():
    cfg = yaml.safe_load(Path(os.environ['SV_CONFIG']).read_text())
    dst = Path(os.environ['SV_CHECKPOINT'])
    dst.mkdir(parents=True, exist_ok=True)
    for filename in ['README.md', 'metadata.pt', 'model.safetensors']:
        hf_hub_download(cfg['checkpoint_repo'], filename, revision=cfg['checkpoint_revision'], local_dir=dst)
    assert file_digest(dst/'model.safetensors') == cfg['checkpoint_sha256'], 'Checkpoint checksum mismatch'
    with zipfile.ZipFile(dst/'metadata.pt') as archive:
        names = [n for n in archive.namelist() if n.endswith('/data.pkl')]
        assert len(names) == 1
        metadata = MetadataReader(io.BytesIO(archive.read(names[0]))).load()
    model = metadata['config']['model']
    assert model['pi05'] and model['action_horizon'] == cfg['native_prediction_horizon'] == 5
    rev = '6deaf18631fc5db2d4029acec8837d502f343e4a'
    base = f'https://raw.githubusercontent.com/LogosRoboticsGroup/ProphRL/{rev}/'
    sources = {
        'norm_stats.json': 'rl/data/bridge/norm_stats.json',
        'ProphRL_README.md': 'rl/README.md',
        'ProphRL_bridge_policy.py': 'rl/verl_vla/utils/vla_utils/pi05/openpi/policies/bridge_policy.py',
        'ProphRL_config.py': 'rl/verl_vla/utils/vla_utils/pi05/openpi/training/config.py',
    }
    hashes = {}
    for name, remote in sources.items():
        with urllib.request.urlopen(base+remote, timeout=60) as response:
            data = response.read()
        (dst/name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    write_json(dst/'provenance.json', {
        'repo': cfg['checkpoint_repo'], 'revision': cfg['checkpoint_revision'],
        'model_sha256': cfg['checkpoint_sha256'], 'metadata_sha256': file_digest(dst/'metadata.pt'),
        'saved_model_config': model, 'saved_global_step': metadata['global_step'],
        'prophrl_commit': rev, 'supporting_source_sha256': hashes,
        'training_stage': 'unresolved_SFT_vs_RL', 'note': cfg['provenance_note'],
    })
    # Populate the tokenizer cache before requesting GPU time.
    from openpi.models.tokenizer import PaligemmaTokenizer
    PaligemmaTokenizer(model['max_token_len'])
    print(json.dumps({'checkpoint': str(dst), 'P': model['action_horizon'], 'download_verified': True}))


if __name__ == '__main__':
    main()

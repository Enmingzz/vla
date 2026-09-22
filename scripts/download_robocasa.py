"""Download only the official model's inference files and simulator assets (CPU)."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import zipfile

from frequency_vla.logging_utils import file_digest, write_json
from frequency_vla.robocasa_protocol import load_config


def checkpoint(config, work):
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.hf_api import RepoFile
    repo, revision, subdir = (config[k] for k in
        ('checkpoint_repo', 'checkpoint_revision', 'checkpoint_subdir'))
    entries = [f for f in HfApi().list_repo_tree(repo, path_in_repo=subdir,
        revision=revision, recursive=True) if isinstance(f, RepoFile)
        and (f.path.startswith(subdir+'/params/') or f.path.startswith(subdir+'/assets/')
             or f.path == subdir+'/_CHECKPOINT_METADATA')]
    if not any('/params/' in f.path for f in entries) or not any('/assets/' in f.path for f in entries):
        raise RuntimeError('Incomplete checkpoint listing')
    def fetch(entry):
        print('Checkpoint:', entry.path, entry.size, flush=True)
        path = Path(hf_hub_download(repo, entry.path, revision=revision,
                                   local_dir=work/'checkpoints'))
        checksum = file_digest(path)
        if path.stat().st_size != entry.size:
            raise RuntimeError('Size mismatch: ' + entry.path)
        lfs = entry.lfs
        expected = getattr(lfs, 'sha256', None) if lfs else None
        if isinstance(lfs, dict):
            expected = lfs.get('sha256')
        if expected and checksum != expected:
            raise RuntimeError('Remote SHA256 mismatch: ' + entry.path)
        return {'name': entry.path[len(subdir)+1:], 'size': entry.size,
                'sha256': checksum, 'remote_lfs_sha256': expected}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        objects = list(pool.map(fetch, entries))
    write_json(work/'checkpoints'/subdir/'download_manifest.json', {
        'source': 'hf://'+repo+'/'+subdir, 'revision': revision,
        'objects': sorted(objects, key=lambda x:x['name'])})


def assets(work):
    root = Path(os.environ['RC_CASA'])/'robocasa/models/assets'
    links = json.loads((root/'box_links/box_links_assets.json').read_text())
    destinations = {'textures':'textures', 'generative_textures':'generative_textures',
        'fixtures_lightwheel':'fixtures', 'objaverse':'objects/objaverse',
        'aigen_objs':'objects/aigen_objs', 'objects_lightwheel':'objects/lightwheel'}
    cache = work/'downloads'
    cache.mkdir(parents=True, exist_ok=True)
    def fetch(item):
        name, destination = item
        marker = work/'provenance'/('asset_'+name+'.json')
        if marker.exists():
            return
        url = links[name].replace('/s/', '/shared/static/')+'.zip'
        archive = cache/(name+'.zip')
        print('Asset:', name, url, flush=True)
        subprocess.run(['curl','--fail','--location','--retry','5','--retry-delay','3',
            '--connect-timeout','30','--speed-limit','1024','--speed-time','120',
            '--continue-at','-','--output',str(archive),url],check=True)
        checksum = file_digest(archive)
        destination = root/destination
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            files = z.infolist()
            for item in files:
                if not (parent/item.filename).resolve().is_relative_to(parent.resolve()):
                    raise ValueError('Unsafe archive path')
            z.extractall(parent)
        if not destination.is_dir() or not any(destination.iterdir()):
            raise RuntimeError('Asset extraction missing: '+str(destination))
        write_json(marker, {'url':url, 'zip_sha256':checksum, 'archive_files':len(files),
                            'destination':str(destination)})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch, destinations.items()))


if __name__ == '__main__':
    config, work = load_config(), Path(os.environ['RC_WORK'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(checkpoint, config, work), pool.submit(assets, work)]
        for future in futures:
            future.result()

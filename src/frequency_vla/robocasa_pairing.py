"""Narrow serialization equivalence; never relax physical-state/image checks."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .logging_utils import digest


def xml_comparison_hash(xml):
    """Ignore only the redundant inferred OBJ MIME type in serialized meshes.

    MuJoCo's asset cache can omit content_type on an already loaded .obj file.
    Keep every filename, scale, pose, numeric string and other attribute intact.
    Do not reload exported XML: its rounded poses change rendered observations.
    """
    tree = ET.fromstring(xml)
    for mesh in tree.iter('mesh'):
        if mesh.get('file', '').endswith('.obj') and mesh.get('content_type') == 'model/obj':
            del mesh.attrib['content_type']
    return hashlib.sha256(ET.tostring(tree)).hexdigest()


def same_inference_spec(a, b):
    """A saved snapshot's phase alias may change; all inference details must match."""
    import copy
    aa, bb = copy.deepcopy(a), copy.deepcopy(b)
    for spec in (aa, bb):
        temporal = spec['temporal_opsd']
        step = temporal['optimizer_step']
        phase = temporal['phase']
        if step == 0:
            if phase != 'baseline':
                return False
        elif phase not in ('student', 'step_' + str(step)):
            return False
        temporal['phase'] = 'baseline' if step == 0 else 'step_' + str(step)
    return aa == bb


def reuse_task_records(config,task_id,horizon,catalog,condition,source,metadata):
    """Validate completed rows against the immutable catalog and loaded weights."""
    task = config['tasks'][task_id]
    path = Path(source)/'raw'/condition/(task+'.jsonl')
    if not path.exists():
        return {}, None
    origin = json.loads((Path(source)/'provenance'/condition/(task+'_server.json')).read_text())
    ancestors = {}
    def collect(item):
        if item is None:
            return
        ancestors[item['inference_fingerprint']] = item
        for parent in item.get('validated_source_servers',[]):
            collect(parent)
    collect(origin)
    prior = Path(source)/'provenance'/condition/(task+'_reuse.json')
    if prior.exists():
        collect(json.loads(prior.read_text())['source_server'])
    for item in [metadata,*ancestors.values()]:
        if item['inference_fingerprint'] != digest(item['experiment_spec']):
            raise ValueError('Invalid inference fingerprint')
        if not same_inference_spec(item['experiment_spec'],metadata['experiment_spec']):
            raise ValueError('Cached episodes used different inference settings or weights')
    records = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        r = json.loads(line)
        index = r['episode_index']
        if index in records or not 0 <= index < config['evaluation_episodes_per_task']:
            raise ValueError('Duplicate or out-of-range cached episode')
        folder = Path(catalog)/task/('episode_%03d'%index)
        expected = json.loads((folder/'identity.json').read_text())
        xml = (folder/'model.xml').read_text()
        if hashlib.sha256(xml.encode()).hexdigest() != expected['initial_xml_sha256']:
            raise ValueError('Recorded XML was changed')
        comparison_hash = xml_comparison_hash(xml)
        exact_xml = r['initial_xml_sha256'] == expected['initial_xml_sha256']
        checked_xml = (r.get('initial_xml_comparison_sha256') == comparison_hash
            and r.get('paired_catalog_identity_sha256') == digest(expected))
        if not (exact_xml or checked_xml) or any(r.get(k) != v for k,v in expected.items()
                                               if k != 'initial_xml_sha256'):
            raise ValueError('Cached episode differs from the recorded initial condition')
        if not (r['condition'] == condition and r['H'] == horizon and r['task_id'] == task_id
                and r['config_sha256'] == digest(config)
                and r['inference_fingerprint'] in ancestors
                and r['renderer'] == 'egl' and r['prediction_horizon'] == 50 and r['flow_steps'] == 10
                and type(r['success']) is bool and 0 < r['env_steps'] <= expected['task_horizon']
                and r['policy_calls'] == math.ceil(r['env_steps']/horizon)):
            raise ValueError('Cached episode protocol differs')
        if not Path(r['video']).is_file() or Path(r['video']).stat().st_size == 0:
            raise FileNotFoundError('Cached episode video is missing: '+r['video'])
        records[index] = dict(r,initial_xml_comparison_sha256=comparison_hash,
            paired_catalog_identity_sha256=digest(expected),reused_record_source=str(path.resolve()))
    return records,dict(origin,validated_source_servers=list(ancestors.values()))

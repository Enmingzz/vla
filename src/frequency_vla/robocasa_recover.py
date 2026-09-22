"""Finish frozen evaluations using the existing checkpoint and verified rows."""
import argparse
import json
from pathlib import Path

from .logging_utils import write_json
from .robocasa_analysis import analyze
from .robocasa_eval import evaluate
from .robocasa_protocol import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--results-dir',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--catalog',help='Original immutable catalog, even for a later recovery attempt')
    args = parser.parse_args()
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    config = load_config()
    root,source = Path(args.results_dir),Path(args.source)
    catalog = Path(args.catalog) if args.catalog else source/'paired_episodes'
    if not catalog.is_dir():
        raise FileNotFoundError('Episode catalog is missing: '+str(catalog))
    if root.resolve() == source.resolve():
        raise ValueError('Preserve the interrupted archive; use a fresh recovery directory')
    client = WebsocketClientPolicy('127.0.0.1',args.port)
    command = lambda **kw:client.infer({'_flow_opsd':kw})
    try:
        status = command(operation='status')
        if status['step'] != 0 or status['phase'] != 'baseline':
            raise ValueError('Recovery requires a fresh server, with no training updates')
        loaded = command(operation='load_snapshot',checkpoint=args.checkpoint,
                         manifest_sha256=args.manifest_sha256,step=500)
        archived_counts = {condition:sum(sum(bool(line) for line in path.read_text().splitlines())
            for path in (source/folder/'raw'/condition).glob('*.jsonl')) for folder,condition in
            [('h5','original_h5'),('h20','original_h20'),
             ('train500','step500_h20'),('train500','step500_h5')]}
        write_json(root/'provenance/recovery.json',{'source':str(source.resolve()),
            'snapshot':loaded,'additional_optimizer_updates':0,'xml_replay_used':False,
            'paired_catalog':str(catalog.resolve()),'recorded_counter_region_labels':True,
            'pairing':'Exact physical state, observation and metadata; infer redundant OBJ MIME type',
            'reset_attempt_limit':3,'reset_retry_seed_changes':False,
            'archived_counts':archived_counts})
        for folder,condition,horizon,phase in [
            ('h20','original_h20',20,'baseline'),
            ('h5','original_h5',5,'baseline'),
            ('train500','step500_h20',20,'step_500'),
            ('train500','step500_h5',5,'step_500')]:
            command(operation='set_phase',phase=phase)
            evaluate(config,horizon,args.port,root/folder,catalog,condition,
                     reuse_root=str(source/folder),allow_obj_mime_equivalence=True,reset_attempts=3)
            analyze(root)
        status = command(operation='status')
        if status['step'] != 0:
            raise RuntimeError('Recovery must never perform optimizer updates')
        write_json(root/'provenance/recovery_complete.json',status)
        print(json.dumps(status),flush=True)
    finally:
        client._ws.close()
        analyze(root)


if __name__ == '__main__':
    main()

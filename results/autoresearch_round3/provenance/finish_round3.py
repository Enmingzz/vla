import json,subprocess
from pathlib import Path
from frequency_vla.logging_utils import file_digest,write_json,digest
root=Path('results/autoresearch_round3')
validation=json.loads((root/'aggregated/validation.json').read_text())
assert validation['complete'] and validation['optimizer_updates_added']==500 and validation['evaluation_episodes']==200
videos=[]
for shard in sorted(root.glob('evaluations/confirmation/step_*/raw/smoke/libero_10/seed_27/H_20/*.jsonl')):
 for line in shard.read_text().splitlines():
  row=json.loads(line); path=shard.parents[5]/row['video_path']
  assert path.is_file() and path.stat().st_size>0, path
  info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate','-of','json',str(path)],text=True))['streams'][0]
  assert (info['width'],info['height'],info['r_frame_rate'])==(224,224,'10/1'),path
  assert int(info['nb_frames'])==row['controlled_environment_steps'],path
  videos.append(dict(video=str(path.relative_to(root)),frames=int(info['nb_frames'])))
assert len(videos)==len({x['video'] for x in videos})==200
write_json(root/'provenance/video_checks.json',dict(passed=True,videos=200,all_frame_counts_match=True,checks=videos))
step=json.loads((root/'provenance/step_1000.json').read_text())
manifest=json.loads((Path(step['path'])/'training_manifest.json').read_text())
assert digest(manifest)==step['manifest_sha256'] and manifest['step']==1000
assert manifest['reloaded_native_inference_max_abs_difference']==0
assert manifest['resumed_from']['step']==500
write_json(root/'provenance/exported_training_manifest_step_1000.json',manifest)
job=json.loads((root/'provenance/submission.json').read_text())['job_id']
accounting=subprocess.check_output(['sacct','-j',job,'-nP','-o','JobID,State,ElapsedRaw,AllocTRES,ExitCode,NodeList'],text=True)
(root/'provenance/slurm_accounting.psv').write_text(accounting)
main=next(line.split('|') for line in accounting.splitlines() if line.split('|')[0]==job)
assert main[1]=='COMPLETED' and main[4]=='0:0', main
write_json(root/'provenance/resource_accounting.json',dict(job_id=job,state=main[1],gpu_seconds=int(main[2]),gpu_hours=int(main[2])/3600,maximum_concurrent_gpus=1,allocation_released=True,node=main[5],alloc_tres=main[3],cpu_preflight_jobs=[60146110,60146702]))
artifacts=list((root/'aggregated').glob('*'))+list((root/'figures').glob('*'))+[root/'FINDINGS.md',root/'training.jsonl',root/'rollouts.jsonl']
write_json(root/'provenance/final_checks.json',dict(complete=True,new_optimizer_updates=500,paired_evaluation_episodes=200,validated_videos=200,checkpoint_manifest_sha256=step['manifest_sha256'],native_reload_max_abs_difference=0,gpu_allocation_released=True,artifacts={str(p.relative_to(root)):file_digest(p) for p in artifacts if p.is_file()}))
print(json.dumps(dict(complete=True,primary=validation['primary'],checkpoint=step,gpu_seconds=int(main[2])),indent=2))

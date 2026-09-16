import json, subprocess
from pathlib import Path
root=Path('results/libero90_first10')
complete=json.loads((root/'provenance/study_complete.json').read_text())
assert complete['complete']
rows=[]
for shard in sorted(root.glob('evaluations/step_*/raw/smoke/libero_90/seed_37/H_*/*.jsonl')):
    for line in shard.read_text().splitlines():
        row=json.loads(line)
        path=shard.parents[5]/row['video_path']
        assert path.is_file() and path.stat().st_size>0, path
        rows.append((shard.parents[5].name,row,path))
assert len(rows)==90 and len({str(p) for _,_,p in rows})==90
checks=[]
for step,r,p in rows:
    # The reduced screen is small enough to inspect metadata for every video.
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(p)],text=True))
    v=info['streams'][0]
    assert (v['width'],v['height'],v['r_frame_rate'])==(224,224,'10/1'), p
    assert int(v['nb_frames'])==r['controlled_environment_steps'], p
    assert abs(float(info['format']['duration'])-r['controlled_environment_steps']/10)<.02, p
    checks.append({'snapshot':step,'task_id':r['task_id'],'horizon':r['replan_steps'],'video':str(p.relative_to(root)),'expected_controlled_steps':r['controlled_environment_steps'],'probe':info})
result={'complete':True,'formal_episode_videos':len(rows),'all_paths_unique':True,'all_files_nonempty':True,'metadata_spot_checks':len(checks),'checks':checks}
(root/'provenance/video_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))

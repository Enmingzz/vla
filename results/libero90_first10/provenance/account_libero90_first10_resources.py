import csv, datetime, io, json, subprocess
from pathlib import Path
root=Path('results/libero90_first10')
notes={
'60096722':'EGL single pilot native abort; no formal episodes',
'60097833':'Repeated EGL abort with faulthandler; no formal episodes',
'60098934':'400 random-action GPU render steps passed',
'60099057':'800 random-action steps passed with CUDA memory allocation',
'60099096':'GDB inferior caught NVIDIA native abort despite COMPLETED batch status',
'60099709':'Exact action replay passed; GDB backtrace-after-exit caused nonzero wrapper status',
'60099822':'Separate CUDA-memory replay passed; busy-CUDA variant timed out',
'60100127':'Captured EGL frames for comparison',
'60100455':'Disabling JAX preallocation did not fix EGL abort',
'60102194':'GPU failed NVML startup check; no model load, zero-second allocation',
'60102294':'OSMesa pilot exceeded 180s walltime at 385 controlled steps; no native crash, no formal episodes',
'60104287':'User interrupted full-suite run after 68 episodes; 30 H=5 prefix episodes reused, plus nine excluded pilots',
'60098927':'CPU diagnostic: shared filesystem library-read failure',
'60099402':'CPU diagnostic: incompatible CVMFS OSMesa glibc',
'60100107':'CPU diagnostic: compatible user-local OSMesa passed 400 random and 261 replay steps',
}
final_job=json.loads((root/'provenance/resource_plan.json').read_text())['job_id']
notes[final_job]='Completed 60 H=20 episodes for the user-requested first-ten comparison; 30 H=5 episodes reused'
fields='JobID,JobName,State,Start,End,ElapsedRaw,AllocTRES,NodeList,ExitCode'
raw=subprocess.check_output(['sacct','-j',','.join(notes),'--allocations','--format='+fields,'-P'],text=True)
rows=list(csv.DictReader(io.StringIO(raw),delimiter='|'))
assert {r['JobID'] for r in rows}==set(notes)
assert all(r['State'] not in ['RUNNING','PENDING','COMPLETING'] for r in rows)
for r in rows:
    r['elapsed_seconds']=int(r['ElapsedRaw']); r['interpretation']=notes[r['JobID']]
    tres=dict(x.split('=',1) for x in r['AllocTRES'].split(','))
    r['gpus']=int(tres.get('gres/gpu',0))
gpu=[r for r in rows if r['gpus']]
assert all(r['gpus']==1 for r in gpu)
nonempty=sorted((r for r in gpu if r['elapsed_seconds']),key=lambda r:r['Start'])
assert all(a['End']<=b['Start'] for a,b in zip(nonempty,nonempty[1:])), 'GPU allocations overlapped'
complete=json.loads((root/'provenance/study_complete.json').read_text())
assert complete['complete']
total=sum(r['elapsed_seconds']*r['gpus'] for r in rows)
result=dict(complete=True,formal_job_id=final_job,formal_job_allocation_seconds=next(r['elapsed_seconds'] for r in rows if r['JobID']==final_job),prior_full_suite_allocation_seconds=1189,diagnostic_gpu_seconds=sum(r['elapsed_seconds'] for r in gpu if r['JobID'] not in ['60104287',final_job]),total_allocated_gpu_seconds=total,total_allocated_gpu_hours=total/3600,maximum_concurrent_gpus=1,all_allocations_released=True,includes_all_failed_gpu_attempts=True,cpu_only_diagnostics_count=len(rows)-len(gpu),jobs=rows)
(root/'provenance/resource_accounting.json').write_text(json.dumps(result,indent=2)+'\n')
(root/'provenance/slurm_accounting.psv').write_text(raw)
try:
    seff=subprocess.check_output(['seff',final_job],text=True,stderr=subprocess.STDOUT)
except (OSError,subprocess.CalledProcessError) as e:
    seff=str(e)
(root/'provenance/slurm_efficiency.txt').write_text(seff)
print(json.dumps({k:v for k,v in result.items() if k!='jobs'},indent=2))

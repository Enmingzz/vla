"""Report the fixed-ten-task pilot, retaining incomplete/negative outcomes."""
import argparse
import csv
import fcntl
import json
from pathlib import Path

import numpy as np

from .logging_utils import write_json


CONDITIONS = [('h5','original_h5'),('h20','original_h20'),
              ('train500','step500_h20'),('train500','step500_h5')]


def paired(rows_a,rows_b):
    key = lambda r:(r['task_id'],r['episode_index'],r['episode_seed'])
    a,b = {key(r):r for r in rows_a},{key(r):r for r in rows_b}
    if set(a) != set(b):
        raise ValueError('Conditions have different evaluation episodes')
    for k in a:
        for field in ['initial_state_sha256','initial_xml_sha256','initial_observation_sha256',
                      'environment_metadata_sha256','config_sha256','prediction_horizon','flow_steps','renderer']:
            if a[k][field] != b[k][field]:
                raise ValueError('Paired comparison differs in '+field)
    differences = np.array([int(b[k]['success'])-int(a[k]['success']) for k in sorted(a)])
    rng = np.random.default_rng(20260921)
    samples = differences[rng.integers(0,len(differences),(10000,len(differences)))].mean(axis=1)
    recovered,regressed = int(np.sum(differences==1)),int(np.sum(differences==-1))
    from scipy.stats import binomtest
    p = float(binomtest(recovered,recovered+regressed,0.5).pvalue) if recovered+regressed else 1.0
    return {'episodes':len(a),'difference_b_minus_a':float(differences.mean()),
        'paired_bootstrap_ci95':np.quantile(samples,[.025,.975]).tolist(),
        'mcnemar_exact_p_unadjusted':p,'recoveries':recovered,'regressions':regressed}


def analyze(root):
    root = Path(root)
    with (root/'analysis.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        summaries,rows = {},{}
        for folder,condition in CONDITIONS:
            path = root/folder/'aggregated'/(condition+'.json')
            if path.exists():
                summaries[condition] = json.loads(path.read_text())
                rows[condition] = [json.loads(line) for file in sorted((root/folder/'raw'/condition).glob('*.jsonl'))
                                  for line in file.read_text().splitlines() if line]
                if len(rows[condition]) != summaries[condition]['episodes']:
                    raise ValueError('Summary/raw row count differs')
        comparisons = {}
        for name,a,b in [('replanning','original_h5','original_h20'),
                         ('opsd_recovery','original_h20','step500_h20'),
                         ('h5_retention','original_h5','step500_h5')]:
            if a in rows and b in rows:
                comparisons[name] = paired(rows[a],rows[b])
        out = root/'aggregated'
        out.mkdir(exist_ok=True)
        write_json(out/'comparisons.json',comparisons)
        columns = ['condition','H','episodes','successes','success_rate','ci95_low','ci95_high',
                   'mean_policy_calls','mean_actions','mean_seconds','successful_mean_actions','successful_mean_seconds']
        with (out/'summary.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=columns)
            writer.writeheader()
            for item in summaries.values():
                flat = {k:v for k,v in item.items() if k in columns}
                flat.update(ci95_low=item['wilson_ci95'][0],ci95_high=item['wilson_ci95'][1])
                writer.writerow(flat)
        with (out/'per_task.csv').open('w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['condition','task','success_rate'])
            for condition,item in summaries.items():
                writer.writerows((condition,t,rate) for t,rate in item['per_task'].items())
        lines = ['# RoboCasa365 fixed-ten-task pilot','',
            'This is a selected ten-task pilot on the official pretrain scene/object split, not the full RoboCasa365 benchmark.',
            'The ten tasks were fixed before observing results. Training and evaluation use distinct seed namespaces.',
            'Each complete evaluation condition contains 10 tasks × 10 episodes = 100 episodes. OPSD uses 500 optimizer updates, batch size 4.',
            '','| Condition | H | Success | 95% CI | Calls/episode | Actions/episode | Seconds/episode |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for _,condition in CONDITIONS:
            if condition not in summaries:
                lines.append('| '+condition+' | — | Pending | — | — | — | — |')
                continue
            r = summaries[condition]
            lines.append('| {} | {} | {}/{} ({:.1%}) | [{:.1%}, {:.1%}] | {:.1f} | {:.1f} | {:.2f} |'.format(
                condition,r['H'],r['successes'],r['episodes'],r['success_rate'],*r['wilson_ci95'],
                r['mean_policy_calls'],r['mean_actions'],r['mean_seconds']))
        lines += ['', 'All environments use EGL. Episode seconds exclude reset/startup and include rollout video writing.',
            'Success uses the official `info["success"]`; episode limits come directly from the native task registry.',
            'Intervals below are paired episode bootstrap intervals, conditional on these ten tasks; p-values are unadjusted and exploratory.']
        for name,c in comparisons.items():
            lines += ['', '{}: change = {:+.1%}; paired 95% CI [{:+.1%}, {:+.1%}]; exact McNemar p = {:.4g}.'.format(
                name,c['difference_b_minus_a'],*c['paired_bootstrap_ci95'],c['mcnemar_exact_p_unadjusted'])]
        if 'replanning' in comparisons:
            a,b = summaries['original_h5'],summaries['original_h20']
            saved = 1-b['mean_policy_calls']/a['mean_policy_calls']
            gaps = sorted(((t,a['per_task'][t]-b['per_task'][t]) for t in a['per_task']),key=lambda x:-x[1])
            lines += ['', 'H=20 saves {:.1%} of policy calls per episode relative to original H=5.'.format(saved),
                'Largest measured task gaps (H=5 minus H=20): '+', '.join('{} {:+.0%}'.format(t,g) for t,g in gaps[:3])+'.']
            gap = comparisons['replanning']
            convincing = gap['paired_bootstrap_ci95'][1]<0 and gap['mcnemar_exact_p_unadjusted']<.05
            lines += ['The H=5 advantage is '+('statistically supported in this pilot.' if convincing else
                'not statistically established in this pilot; do not assume sparse replanning caused a material accuracy loss.')]
        if 'opsd_recovery' in comparisons:
            c = comparisons['opsd_recovery']
            lines += ['', 'The measured 500-step H=20 change is {:+.1%}; positive changes indicate recovery, negative changes indicate deterioration.'.format(c['difference_b_minus_a'])]
        lines += ['', 'The method ports the existing temporal Gaussian velocity-matching OPSD implementation with an EMA teacher (decay 0.9999). It is not a reproduction of an image-generation Flow-OPD policy-gradient objective.',
            'No task was selected by baseline success. Shared task identities mean this measures adaptation to new seeded episodes, not cross-task generalization.']
        if len(summaries)<4:
            lines += ['', 'Experiment incomplete: pending conditions are not scored as failures.']
        (root/'FINDINGS.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    analyze(parser.parse_args().root)

"""Use the existing paired statistics and figures for Simpler episode records."""
import argparse
import json
from pathlib import Path

from .analysis import build_tables, plots, write_csv
from .logging_utils import write_json


def aggregate(root, mode):
    root = Path(root)
    cfg = json.loads((root/'submission_sources.json').read_text())['config']
    output = root/mode
    records = [json.loads(line) for line in (output/'raw/episodes.jsonl').read_text().splitlines()]
    expected = len(cfg['tasks']) * cfg['episodes_per_task'][mode] * len(cfg['horizons'])
    if len(records) != expected or {r['replan_steps'] for r in records} != set(cfg['horizons']):
        raise ValueError('The fixed paired sweep is incomplete; do not report a full comparison')
    tables = build_tables(records, cfg)
    names = ['frequency_sweep','per_task','per_seed','replanning_gaps','per_task_gaps']
    for name, rows in zip(names, tables):
        write_csv(output/'aggregated'/f'{name}.csv', rows)
    write_csv(output/'raw/episodes.csv', records)
    summaries, _, _, gaps, _ = tables
    plots(summaries, gaps, output/'figures', cfg['benchmark'], mode, 5, cfg)
    primary = next(g for g in gaps if g['student_H'] == 5)
    convincing = primary['gap_ci95_low'] > 0 and primary.get('mcnemar_p_holm', 1) < 0.05
    sentences = [
        '# SimplerEnv π0.5 replanning experiment', '',
        f'Mode: {mode}. Same fixed third-party Bridge checkpoint; native P=5; flow steps=10.',
        'All four official WidowX tasks, 5 Hz control, official prepackaged visual matching.',
        'Primary endpoint: final simulator success at the official time limit (60/120 actions).',
        'The upstream evaluator continues after transient success; any-hit success is recorded separately.', '',
        '| H | Replanning Hz | Successes / episodes | Success rate (95% Wilson CI) | Mean policy calls | Rollout seconds |',
        '|---|---|---|---|---|---|',
    ]
    for s in summaries:
        sentences.append(f"| {s['H']} | {5/s['H']:.2f} | {s['total_successes']}/{s['total_episodes']} | "
                         f"{s['success_rate']:.1%} ({s['ci95_low']:.1%}–{s['ci95_high']:.1%}) | "
                         f"{s['mean_policy_calls']:.2f} | {s['mean_wall_clock_seconds']:.2f} |")
    sentences += ['', f"Primary H=1 minus H=5 gap: {primary['replanning_gap']*100:+.2f} percentage points; "
        f"paired 95% bootstrap CI [{primary['gap_ci95_low']*100:+.2f}, {primary['gap_ci95_high']*100:+.2f}]. "
        f"McNemar p={primary['mcnemar_p_two_sided']:.4g}; Holm-adjusted p={primary['mcnemar_p_holm']:.4g}.",
        f"H=5 saves {primary['relative_policy_calls_saved']:.1%} of policy calls per episode.", '',
        ('Measured data support a positive frequency gap in this setting.' if convincing else
         'These measurements do not establish a statistically convincing positive frequency gap.'),
        'Probe/smoke results are functionality checks, not confirmatory hypothesis tests.' if mode != 'main' else
        'This is one seed and one checkpoint on the fixed prepackaged visual-matching layouts.', '',
        'Per-task sensitivities and candidate H=2/H=5 gaps are in the aggregated CSVs.',
        'No teacher/student pair is selected automatically and no OPSD training is started.', '',
        '## Interpretation limits', '', cfg['provenance_note'],
        'This checks H=1/2/5 at native P=5. It is not the P=50, H=5/20 experiment.',
        'Rollout time excludes reset, checkpoint startup and video encoding, and can include first-call warmup.',
        'The fixed task/time-limit protocol makes mean episode length constant across H.',
        'Statistical inference concerns these tasks/layouts under this inference configuration; it does not establish a universal law.', '',
    ]
    (output/'FINDINGS.md').write_text('\n'.join(sentences))
    write_json(output/'aggregated/completion.json', {'complete': True, 'episodes': len(records),
        'conditions': summaries, 'gaps': gaps, 'hypothesis_convincing': convincing})
    print(json.dumps({'mode': mode, 'episodes': len(records), 'successes': {
        str(s['H']): s['total_successes'] for s in summaries}}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--mode', choices=['probe','smoke','main'], required=True)
    a = p.parse_args()
    aggregate(a.root, a.mode)


if __name__ == '__main__':
    main()

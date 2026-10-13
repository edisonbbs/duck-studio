"""Export selected public metadata only; never copies models, paths or credentials."""
import json, math, time
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ROOT = Path(__file__).resolve().parent

def export():
    versions, runs = [], []
    for path in sorted((ROOT / 'versions').glob('v*/recipe.json')):
        v = json.loads(path.read_text())
        versions.append({k: v[k] for k in ('id', 'name', 'parent', 'params')})
        base = ROOT.parent / 'xgoduck_rl/logs/rsl_rl' if v['kind'] == 'imported' else path.parent / 'workspace/logs/rsl_rl'
        for run in sorted(base.glob('*/*')):
            if not run.is_dir() or not list(run.glob('events.out.tfevents*')): continue
            ea = EventAccumulator(str(run), size_guidance={'scalars': 300})
            ea.Reload()
            series = {}
            for tag in ea.Tags().get('scalars', []):
                series[tag] = [[x.step, x.value] for x in ea.Scalars(tag) if math.isfinite(x.value)]
            checkpoints = sorted((p.name for p in run.glob('model_*.pt')), key=lambda x: int(x[6:-3]))
            runs.append({'version': v['id'], 'experiment': run.parent.name, 'name': run.name, 'series': series, 'checkpoints': checkpoints})
    result = {'schema': 1, 'exported_at': time.time(), 'versions': versions, 'runs': runs}
    dest = ROOT / 'docs/data.json'
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
    tmp.replace(dest)
    print(f'Exported {len(versions)} versions and {len(runs)} runs; {dest.stat().st_size} bytes. No upload performed.')
if __name__ == '__main__': export()

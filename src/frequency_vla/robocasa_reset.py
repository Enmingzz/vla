"""Restore recorded counter-region labels before the native seeded sampler.

Upstream Counter.get_reset_regions uses list(set(XML elements)), so the mapping
from geom_N labels to left/right regions depends on object identity. Restore
only this mapping from the saved episode metadata, preserving geometry, RNG
draws, placement sampling, controllers and simulation. Final state/image checks
remain exact. This wrapper is never used for training.
"""
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
from unittest.mock import patch


def relabel_regions(regions, recorded):
    """Use recorded labels only when they uniquely identify native geometry."""
    assignments = {}
    for saved in recorded:
        name = saved['name']
        if name not in regions:
            return regions
        geometry = {k:v for k,v in saved.items() if k != 'name'}
        # Lists, tuples and numpy scalars arise in native region definitions.
        normalize = lambda value:json.dumps(value,sort_keys=True,
            default=lambda x:x.tolist() if hasattr(x,'tolist') else x.item())
        matches = [key for key,value in regions.items() if normalize(value) == normalize(geometry)]
        if len(matches) != 1:
            return regions  # Constructor can temporarily load another scene.
        if name in assignments and assignments[name] != matches[0]:
            raise ValueError('Conflicting recorded region identities')
        assignments[name] = matches[0]
    if not assignments or len(set(assignments.values())) != len(assignments):
        return regions
    remaining = [k for k in regions if k not in assignments]
    unused = [k for k in regions if k not in assignments.values()]
    if len(remaining) > 1:
        return regions  # Never guess an unrecorded order.
    assignments.update(zip(remaining,unused))
    return {k:regions[assignments[k]] for k in regions}


@contextmanager
def recorded_counter_regions(catalog,task,index,events):
    if catalog is None:
        yield
        return
    from robocasa.models.fixtures.counter import Counter
    folder = Path(catalog)/task/('episode_%03d'%index)
    meta = json.loads((folder/'environment_metadata.json').read_text())
    groups = {}
    for cfg in meta['object_cfgs']:
        placement = cfg.get('placement',{})
        kwargs = placement.get('sample_region_kwargs',{})
        region = cfg.get('reset_region')
        if region and region.get('name','').startswith('geom_'):
            key = (placement.get('fixture'),kwargs.get('ref'),kwargs.get('loc','nn'))
            groups.setdefault(key,[]).append(region)
    original = Counter.get_reset_regions
    signature = inspect.signature(original)

    def restore(counter,*args,**kwargs):
        regions = original(counter,*args,**kwargs)
        bound = signature.bind(counter,*args,**kwargs)
        bound.apply_defaults()
        ref = bound.arguments['ref']
        key = (counter.name,getattr(ref,'name',ref),bound.arguments['loc'])
        updated = relabel_regions(regions,groups.get(key,[]))
        if any(updated[k] is not regions[k] for k in regions):
            events.append({'fixture':key[0],'reference':key[1],'location':key[2],
                           'restored_labels':list(updated)})
        return updated

    with patch.object(Counter,'get_reset_regions',restore):
        yield

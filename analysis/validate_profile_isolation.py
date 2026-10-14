"""Check the two-cell workflow without running full research training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import copy
import json
import time
import nbformat

root=Path(__file__).resolve().parent.parent
nb=nbformat.read(root/'Notebook_Revisi_KomtekInfo_v2_1.ipynb',as_version=4)
output=root/'analysis/iso_out'/str(int(time.time()))
ns={'__name__':'profile_validation'}
checks=[]
def passed(name,detail):
    checks.append(dict(test=name,status='passed',detail=detail))
    print('PASS',name,flush=True)

for cell in nb.cells:
    if cell.cell_type!='code':continue
    tag=cell.metadata.get('tags',[''])[0]
    if tag.startswith('execute_'):continue
    code=cell.source
    if tag=='config':
        code=code.replace("DATA_ROOT=Path('Pothole-dataset')",f"DATA_ROOT=Path({str(root/'analysis/synthetic_smoke_dataset')!r})")
        code=code.replace("OUT_ROOT=Path('outputs')",f'OUT_ROOT=Path({str(output)!r})')
        code=code.replace('PRETRAINED=True','PRETRAINED=False')
    exec(compile(code,'profile_validation_'+tag,'exec'),ns)
ns['torch'].set_num_threads(2)
assert not output.exists()
execute=[c for c in nb.cells if c.cell_type=='code' and c.metadata.get('tags',[''])[0].startswith('execute_')]
assert [c.metadata['tags'][0] for c in execute]==['execute_smoke','execute_research']
passed('setup_is_passive_and_run_cells_ordered','Shared setup does not train, audit data, or create output folders; smoke precedes research.')

base=copy.deepcopy(ns['BASE_CFG'])
smoke=ns['get_profile_config']('smoke');research=ns['get_profile_config']('research')
assert len(smoke['folds'])*2*3*len(smoke['seeds'])==6
assert len(research['folds'])*2*3*len(research['seeds'])==90
assert len(research['folds'])*2*len(research['lr_grid'])==20
assert research['epochs']==20 and research['bootstrap']==10000 and research==base
smoke['seeds'].append(999);smoke['lr_grid'].clear();research['folds'].clear()
assert ns['BASE_CFG']==base and ns['get_profile_config']('research')==base
assert ns['get_profile_config']('smoke')['seeds']==[42]
passed('fresh_nested_configuration_per_profile','Smoke has 6 main runs; research restores all 90 main and 20 tuning runs. Mutable nested settings do not leak.')

ns['activate_profile']('smoke');smoke_dir=ns['OUT']
assert len(ns['CONDITIONS'])==16
marker=smoke_dir/'metadata'/'isolation_sentinel.txt';marker.write_text('smoke remains intact')
ns['CFG']['seeds'].append(999);ns['CFG']['folds'].clear();ns['CFG']['replicas']=99
ns['PROTOCOL_SIGNATURE']='stale-smoke-signature'
ns['activate_profile']('research');research_dir=ns['OUT']
assert ns['CFG']==base and len(ns['CONDITIONS'])==44 and ns['PROTOCOL_SIGNATURE'] is None
assert research_dir!=smoke_dir and research_dir.name=='research'
assert marker.read_text()=='smoke remains intact'
ns['activate_profile']('smoke')
assert ns['CFG']['seeds']==[42] and ns['CFG']['folds']==[0] and len(ns['CONDITIONS'])==16
passed('bidirectional_switch_and_separate_outputs','Smoke/research/smoke rebuilds configuration, corruption grid and signature; smoke files are preserved.')

before=ns['OUT'];before_config=copy.deepcopy(ns['CFG'])
try:ns['run_experiment']('typo')
except ValueError:pass
else:raise AssertionError('Invalid profile accepted')
assert ns['OUT']==before and ns['CFG']==before_config and not ns['_EXPERIMENT_RUNNING']
ns['_EXPERIMENT_RUNNING']=True
try:
    try:ns['run_experiment']('research')
    except RuntimeError:pass
    else:raise AssertionError('Overlapping run accepted')
finally:ns['_EXPERIMENT_RUNNING']=False
passed('invalid_profile_and_busy_guard','Invalid and overlapping calls are rejected without changing the active output/configuration.')

class StopAfterResearchAudit(Exception):pass
def stop_before_eda(frame):
    assert ns['PROFILE']=='research' and ns['CFG']==base
    assert len(frame)==80 and len(ns['CONDITIONS'])==44
    raise StopAfterResearchAudit('Intentional stop before expensive research stages')
original_eda=ns['plot_eda'];ns['plot_eda']=stop_before_eda
try:
    try:exec(compile(execute[1].source,'execute_research_cell','exec'),ns)
    except StopAfterResearchAudit:pass
    else:raise AssertionError('Research stop not reached')
finally:ns['plot_eda']=original_eda
assert not ns['_EXPERIMENT_RUNNING'] and 'research_results' not in ns
lock=json.loads((research_dir/'metadata'/'protocol_lock.json').read_text())
assert lock['protocol']['profile']=='research' and lock['protocol']['config']['seeds']==[42,43,44]
assert not list(research_dir.glob('*.zip'))
assert marker.read_text()=='smoke remains intact'
passed('research_cell_routes_correctly_and_cleans_up_on_interruption','The actual research cell reaches real dataset audit with full settings without requiring a smoke result; a planned interruption releases the runner and creates no completed-research ZIP.')

result=dict(success=True,n_passed=len(checks),checks=checks,full_research_training_executed=False,
            test_data='80 synthetic unique images plus one exact duplicate',device='cpu')
(root/'analysis/profile_isolation_validation_result.json').write_text(json.dumps(result,indent=2))
print('ALL PROFILE CHECKS PASSED',len(checks),flush=True)

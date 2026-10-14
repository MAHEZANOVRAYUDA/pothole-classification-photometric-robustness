import os
import pathlib,json,nbformat,numpy as np,copy,time
root=pathlib.Path(__file__).resolve().parent.parent
unit_root=pathlib.Path('\\\\?\\' + str((root/'analysis/u_out'/str(int(time.time()))).resolve()))
nb=nbformat.read(root/'Notebook_Revisi_KomtekInfo_v2_1.ipynb',as_version=4);ns={'__name__':'validation'}
for c in nb.cells:
    if c.cell_type!='code':continue
    tag=c.metadata.get('tags',[''])[0]
    if tag.startswith('execute_'):continue
    code=c.source
    if tag=='config':
        code=code.replace("DATA_ROOT=Path('Pothole-dataset')",f'DATA_ROOT=Path({str(root/"analysis/synthetic_smoke_dataset")!r})')
        code=code.replace("OUT_ROOT=Path('outputs')",f'OUT_ROOT=Path({str(unit_root)!r})')
        code=code.replace('PRETRAINED=True','PRETRAINED=False')
    exec(compile(code,'validation_'+tag,'exec'),ns)
ns['torch'].set_num_threads(2);ns['activate_profile']('smoke');checks=[]
def passed(name,detail):
    checks.append(dict(test=name,status='passed',detail=detail));print('PASS',name,flush=True)
frame,summary=ns['audit_dataset'](ns['DATA_ROOT'])
assert summary['unique_images']==80 and summary['exact_duplicates_removed']==1
test_ids=[]
for f in range(5):
    tr,va,te=ns['split_frames'](frame,f)
    assert not (set(tr.group_id)&set(va.group_id) or set(tr.group_id)&set(te.group_id) or set(va.group_id)&set(te.group_id))
    test_ids.extend(te.image_id)
assert len(test_ids)==len(set(test_ids))==80
passed('five_fold_disjoint_and_oof_coverage','Each unique image tests once; train, validation, test groups are disjoint in all five folds.')

conflict=ns['DATA_ROOT']/'potholes'/'temporary_conflicting_label.png'
conflict.write_bytes((ns['DATA_ROOT']/'normal'/'synthetic_000.png').read_bytes())
try:
    try:ns['audit_dataset'](ns['DATA_ROOT'])
    except ValueError as e:assert 'label berbeda' in str(e)
    else:raise AssertionError('Duplicate conflict not caught')
finally:conflict.unlink()
passed('conflicting_duplicate_rejected','An identical pixel image with the opposite label is rejected.')

p=root/'analysis/source_groups_fixture.csv'
p.write_text('relative_path,source_group\nnormal/synthetic_000.png,scene_a\npotholes/synthetic_000.png,scene_a\n')
ns['SOURCE_GROUPS_CSV']=p;grouped,_=ns['audit_dataset'](ns['DATA_ROOT'])
normal=frame.loc[frame.relative_path=='normal/duplicate_same_label.png','image_id'].iloc[0]
pothole=frame.loc[frame.relative_path=='potholes/synthetic_000.png','image_id'].iloc[0]
assert grouped.set_index('image_id').loc[[normal,pothole],'group_id'].nunique()==1
passed('source_metadata_survives_deduplication','Source groups attached to a removed duplicate propagate to its retained canonical image.')
ns['SOURCE_GROUPS_CSV']=None;frame,_=ns['audit_dataset'](ns['DATA_ROOT'])

rng=np.random.default_rng(77);im=ns['Image'].fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8))
for k in ns['KINDS']:
    assert np.array_equal(np.asarray(ns['perturb_image'](im,k,3,91)),np.asarray(ns['perturb_image'](im,k,3,91)))
assert np.array_equal(np.asarray(ns['perturb_image'](im,'low_light',1,1)),np.asarray(ns['perturb_image'](im,'low_light',1,2)))
old=ns['CFG']['replicas'];ns['CFG']['replicas']=3;conditions=ns['make_conditions']();assert len(conditions)==44
w=ns['scope_weights'](conditions,['low_light','shadow'])
for k in ['low_light','shadow']:
    for s in range(1,6):assert np.isclose(sum(v for c,v in zip(conditions,w) if c[:2]==(k,s)),.1)
ns['CFG']['replicas']=old
passed('corruption_determinism_and_equal_weights','44 research conditions; each primary family/severity has weight 0.1 despite different replica counts.')

y=np.array([0,1]*15);groups=np.repeat(np.arange(10),3)
pred=rng.integers(0,2,(3,len(conditions),len(y)))
counts=ns['confusion_counts_by_group'](y,pred,groups);f1=ns['f1_from_counts'](counts.sum(0))
for s in range(3):
    for c in range(len(conditions)):
        assert np.isclose(f1[s,c],ns['f1_score'](y,pred[s,c],labels=[0,1],average='macro',zero_division=0))
names,point,draws,per_seed=ns['bootstrap_all_scopes'](y,pred,pred,groups,conditions,128,123)
assert np.all(point==0) and np.all(draws==0) and np.all(per_seed==0)
other=rng.integers(0,2,pred.shape)
_,point,draws,per_seed=ns['bootstrap_all_scopes'](y,pred,other,groups,conditions,128,123)
for j,name in enumerate(names):
    weights=ns['scope_weights'](conditions,ns['SCOPE_KINDS'][name]);expected=[]
    for s in range(3):
        differences=[ns['f1_score'](y,other[s,c],labels=[0,1],average='macro',zero_division=0)-f1[s,c] for c in range(len(conditions))]
        expected.append(np.dot(differences,weights))
    assert np.isclose(point[j],np.mean(expected))
passed('bootstrap_matches_independent_metric_calculation','F1 agrees with sklearn on 132 arrays; paired identical predictions have exactly zero effects/draws; weighted effects match separate calculation.')

rho,status=ns['safe_spearman'](np.zeros((10,10)),np.ones((10,10)))
assert np.isnan(rho) and status=='constant_map'
passed('constant_cam_is_undefined','Constant maps produce NaN with an explicit reason.')

ns['CFG']['keep_last_completed']=True;ns['CFG']['patience']=20
ns['PROTOCOL_SIGNATURE']=ns['freeze_protocol'](frame)
lock=ns['OUT']/'metadata'/'protocol_lock.json';before=lock.read_bytes();ns['CFG']['batch_size']+=1
try:
    try:ns['freeze_protocol'](frame)
    except RuntimeError:pass
    else:raise AssertionError('Changed configuration accepted')
    assert lock.read_bytes()==before
finally:ns['CFG']['batch_size']-=1
passed('protocol_lock_rejects_changed_configuration','Batch-size change is rejected without modifying the old lock.')

tr,va,te=ns['split_frames'](frame,0);args=('mobilenet_v3_large','illum',909,3e-4,2)
reference=ns['fit_run'](*args,'resume_reference',0,tr,va)
atomic=ns['atomic_torch_save'];interrupted=[False]
class PlannedInterruption(Exception):pass
def interrupt_after_checkpoint(obj,path):
    atomic(obj,path)
    if pathlib.Path(path).name=='last.pt' and not interrupted[0]:
        interrupted[0]=True;raise PlannedInterruption('Intentional software test')
ns['atomic_torch_save']=interrupt_after_checkpoint
try:
    try:ns['fit_run'](*args,'resume_interrupted',0,tr,va)
    except PlannedInterruption:pass
    else:raise AssertionError('Interruption not injected')
finally:ns['atomic_torch_save']=atomic
resumed=ns['fit_run'](*args,'resume_interrupted',0,tr,va)
a=ns['safe_load'](pathlib.Path(reference['checkpoint']).with_name('last.pt'))
b=ns['safe_load'](pathlib.Path(resumed['checkpoint']).with_name('last.pt'))
assert a['next_epoch']==b['next_epoch']==2
for k,v in a['model'].items():assert ns['torch'].equal(v,b['model'][k]),k
for x,yh in zip(a['history'],b['history']):
    for k in ['epoch','train_loss','train_f1','val_loss','val_f1','lr']:assert x[k]==yh[k],k
passed('resume_matches_uninterrupted_training','All final model tensors and non-timing history fields are exactly equal after a planned epoch-1 interruption and resume on CPU.')

builder=ns['build_model']
def reject_build(*args,**kwargs):raise AssertionError('Completed run should not instantiate a model')
ns['build_model']=reject_build
try:skipped=ns['fit_run'](*args,'resume_reference',0,tr,va)
finally:ns['build_model']=builder
assert skipped['checkpoint_sha256']==reference['checkpoint_sha256']
passed('completed_run_skips_retraining','Completed run is returned without model creation.')

model=ns['load_eval_model'](reference);signature=ns['state_signature'](model);randomized=copy.deepcopy(model)
ns['reset_randomized_model'](randomized,'head_randomized',33)
assert ns['state_signature'](model)==signature and ns['state_signature'](randomized)!=signature
with ns['GradCAM'](model) as cam:result=cam(ns['tensor_image'](ns['resize_image'](im)).unsqueeze(0),0)
assert result['target']==0 and 0<=result['p_pothole']<=1 and ns['state_signature'](model)==signature
assert len(model.features[-1]._forward_hooks)==0
passed('gradcam_model_integrity_and_hook_cleanup','Class-0 CAM works; trained state is unchanged; cloned random head differs; hooks are removed.')
threads=ns['torch'].get_num_threads();bench,times=ns['benchmark_latency'](model,ns['torch'].device('cpu'))
assert bench['cpu_threads_actual']==1 and ns['torch'].get_num_threads()==threads
passed('cpu_thread_application_and_restoration','CPU benchmark records one thread and restores previous setting.')

result=dict(success=True,n_passed=len(checks),checks=checks,environment=ns['environment_info'](),
            fixture='80 unique synthetic random images, untrained initialization, CPU only')
(root/'analysis/dual_profile_unit_validation_result.json').write_text(json.dumps(result,indent=2))
print('ALL CHECKS PASSED',len(checks),flush=True)

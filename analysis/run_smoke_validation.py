import os
os.environ['CUDA_VISIBLE_DEVICES']=''  # Software validation is deliberately CPU-only.
import pathlib,json,time,nbformat,numpy as np
from PIL import Image
root=pathlib.Path(__file__).resolve().parent.parent
data=root/'analysis/synthetic_smoke_dataset';rng=np.random.default_rng(891)
for name in ['normal','potholes']:
    folder=data/name;folder.mkdir(parents=True,exist_ok=True)
    for i in range(40):Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8)).save(folder/f'synthetic_{i:03d}.png')
(data/'normal'/'duplicate_same_label.png').write_bytes((data/'normal'/'synthetic_000.png').read_bytes())
nb=nbformat.read(root/'Notebook_Revisi_KomtekInfo_v2_1.ipynb',as_version=4)
for c in nb.cells:
    if c.cell_type=='code' and 'config' in c.metadata.get('tags',[]):
        c.source=c.source.replace("DATA_ROOT=Path('Pothole-dataset')",f'DATA_ROOT=Path({str(data)!r})')
        c.source=c.source.replace("OUT_ROOT=Path('outputs')",f'OUT_ROOT=Path({str(root/"analysis/dual_profile_smoke_outputs")!r})')
        c.source=c.source.replace('PRETRAINED=True','PRETRAINED=False')
        c.source+='\ntorch.set_num_threads(2)\n'
namespace={'__name__':'__main__'};executed=0;t=time.time()
try:
    for i,c in enumerate(nb.cells):
        if c.cell_type!='code' or 'execute_research' in c.metadata.get('tags',[]):continue
        print('EXECUTING',i,c.metadata.get('tags'),flush=True)
        exec(compile(c.source,f'notebook_cell_{i}','exec'),namespace)
        executed+=1;c.execution_count=executed
finally:
    nb.metadata['validation_execution_mode']='Sequential Python cell execution'
    nbformat.write(nb,root/'analysis/dual_profile_smoke_executed.ipynb')
result=dict(success=True,seconds=time.time()-t,executed_code_cells=executed,fixture_unique_images=80,
            pretrained=False,device='cpu',execution_mode='sequential Python cell execution')
(root/'analysis/dual_profile_smoke_validation_result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

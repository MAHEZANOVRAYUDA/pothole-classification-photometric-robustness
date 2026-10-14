import nbformat
from nbclient import NotebookClient
from pathlib import Path
import time

root = Path(__file__).resolve().parent.parent
nb_path = root / 'Notebook_Revisi_KomtekInfo_v2_1.ipynb'

print("Loading notebook:", nb_path)
nb = nbformat.read(nb_path, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name='python3')
start_time = time.time()

with client.setup_kernel():
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            tags = cell.metadata.get('tags', [])
            if 'execute_research' in tags:
                print(f"Skipping cell {index} ({tags}) - reserved for full research mode.")
                continue
            print(f"Executing cell {index} ({tags})...", flush=True)
            client.execute_cell(cell, index)

elapsed = time.time() - start_time
print(f"Execution completed in {elapsed:.2f} seconds! Saving notebook with all outputs...")
nbformat.write(nb, nb_path)
print("Notebook saved successfully with all visual outputs and tables embedded!")

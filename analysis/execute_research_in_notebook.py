import os
import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

os.environ['MPLBACKEND'] = 'Agg'
os.environ['PYTHONUNBUFFERED'] = '1'

class StreamingNotebookClient(NotebookClient):
    """Client yang menyalurkan output stream kernel secara langsung ke stdout terminal."""
    def output(self, outs, msg, display_id, cell_index):
        out = super().output(outs, msg, display_id, cell_index)
        if out and out.output_type == 'stream':
            sys.stdout.write(out.text)
            sys.stdout.flush()
        return out

def main():
    root = Path(__file__).resolve().parent.parent
    nb_path = root / 'Notebook_Revisi_KomtekInfo_v2_1.ipynb'

    print("==================================================================", flush=True)
    print("      MEMULAI PELATIHAN PENUH (FULL RESEARCH RUN - PROTOKOL v2.1)", flush=True)
    print("==================================================================", flush=True)
    print(f"Target notebook: {nb_path}", flush=True)

    nb = nbformat.read(nb_path, as_version=4)
    client = StreamingNotebookClient(nb, timeout=None, kernel_name='python3')
    start_total = time.time()

    with client.setup_kernel():
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                tags = cell.metadata.get('tags', [])
                
                if 'execute_smoke' in tags:
                    print(f"\n[INFO] Melewati Sel {index} (tags: {tags}) - output profil SMOKE dipertahankan.", flush=True)
                    continue

                if 'execute_research' in tags:
                    print(f"\n==================================================================", flush=True)
                    print(f"[EKSEKUSI] Menjalankan Sel {index} - PROFIL RESEARCH (90 RUNS)...", flush=True)
                    print(f"==================================================================", flush=True)
                else:
                    print(f"[SETUP] Menjalankan Sel {index} ({cell.source.strip().splitlines()[0][:50]}...)", flush=True)

                cell_t0 = time.time()
                client.execute_cell(cell, index)
                cell_dur = time.time() - cell_t0

                if 'execute_research' in tags:
                    print(f"\n[SUKSES] Sel RESEARCH selesai dalam {cell_dur/60:.2f} menit!", flush=True)
                    print("Menyimpan notebook dengan seluruh output visual dan tabel RESEARCH...", flush=True)
                    nbformat.write(nb, nb_path)
                    print("Notebook berhasil disimpan dan diperbarui!", flush=True)

    total_elapsed = time.time() - start_total
    print(f"\n==================================================================", flush=True)
    print(f"SEMUA PROSES SELESAI TUNTAS DALAM {total_elapsed/60:.2f} MENIT!", flush=True)
    print(f"==================================================================", flush=True)
    nbformat.write(nb, nb_path)
    print("Notebook final tersimpan sempurna.")

if __name__ == '__main__':
    main()

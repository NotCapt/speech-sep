import json
import base64
import os
import re

notebooks = [
    'sepformer-and-convtasnet.ipynb',
    'mossformer2.ipynb',
    'svoice.ipynb'
]

output_dir = 'audio_samples'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for nb_file in notebooks:
    if not os.path.exists(nb_file):
        continue
        
    nb_name = nb_file.replace('.ipynb', '')
    print(f'Processing {nb_file}...')
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    audio_count = 0
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            for output in cell.get('outputs', []):
                if output.get('output_type') == 'display_data':
                    data = output.get('data', {})
                    html = data.get('text/html', '')
                    if isinstance(html, list):
                        html = ''.join(html)
                    
                    matches = re.findall(r'src="data:audio/(.*?);base64,([^"]+)"', html)
                    for match in matches:
                        ext = match[0]
                        if ext == 'x-wav':
                            ext = 'wav'
                        b64_data = match[1]
                        
                        audio_bytes = base64.b64decode(b64_data)
                        
                        out_path = os.path.join(output_dir, f'{nb_name}_sample_{audio_count}.{ext}')
                        with open(out_path, 'wb') as audio_f:
                            audio_f.write(audio_bytes)
                        
                        print(f'  Saved {out_path}')
                        audio_count += 1
    
    if audio_count == 0:
        print(f'  No audio samples found in {nb_file}.')

import os
import shutil

base_dir = r"d:\cocktail\Benchmarking Existing Models\audio_samples"

# Definitions of how to map the numerical sample ID to meaningful names
mapping = {
    'sepformer-and-convtasnet': {
        0: 'input_mixture.wav',
        1: 'sepformer_estimated_speaker_1.wav',
        2: 'sepformer_estimated_speaker_2.wav',
        3: 'sepformer_estimated_speaker_3.wav',
        4: 'convtasnet_estimated_speaker_1.wav',
        5: 'convtasnet_estimated_speaker_2.wav',
        6: 'convtasnet_estimated_speaker_3.wav'
    },
    'mossformer2': {
        0: 'input_mixture.wav',
        1: 'estimated_speaker_1.wav',
        2: 'ground_truth_speaker_1.wav',
        3: 'estimated_speaker_2.wav',
        4: 'ground_truth_speaker_2.wav',
        5: 'estimated_speaker_3.wav',
        6: 'ground_truth_speaker_3.wav',
        7: 'estimated_speaker_4.wav',
        8: 'ground_truth_speaker_4.wav',
        9: 'estimated_speaker_5.wav',
        10: 'ground_truth_speaker_5.wav'
    },
    'svoice': {
        0: 'input_mixture.wav',
        1: 'estimated_speaker_1.wav',
        2: 'ground_truth_speaker_1.wav',
        3: 'estimated_speaker_2.wav',
        4: 'ground_truth_speaker_2.wav',
        5: 'estimated_speaker_3.wav',
        6: 'ground_truth_speaker_3.wav',
        7: 'estimated_speaker_4.wav',
        8: 'ground_truth_speaker_4.wav',
        9: 'estimated_speaker_5.wav',
        10: 'ground_truth_speaker_5.wav'
    }
}

for model_name, files_map in mapping.items():
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    for sample_id, new_name in files_map.items():
        old_name = f"{model_name}_sample_{sample_id}.wav"
        old_path = os.path.join(base_dir, old_name)
        new_path = os.path.join(model_dir, new_name)
        
        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
            print(f"Moved {old_name} -> {model_name}/{new_name}")

import torchaudio
import torch
import os
import json

def generate_fbank_features(wav_path, save_dir):
    waveform, sample_rate = torchaudio.load(wav_path)
    fbank_features = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=23, sample_frequency=sample_rate)

    # Construct the path to save the fbank features
    print(wav_path)
    some_id = wav_path.split('/')[-3]  # Assuming the speaker ID is three levels up in the path
    spk_id = wav_path.split('/')[-2].split('/')[-1]
    file_id = wav_path.split('/')[-1].split('.')[0]  # Extract file ID from wav path
    print(some_id)
    print(spk_id)
    print(file_id)
    fbank_path = os.path.join(save_dir, f"{some_id+spk_id+file_id}.pt")
    print(fbank_path)

    # Ensure the directory exists
    #os.makedirs(os.path.dirname(fbank_path), exist_ok=True)

    # Save the fbank features
    print(fbank_path)
    torch.save(fbank_features, fbank_path)

    return fbank_path

def create_fbank_json(original_json_path, save_dir, new_json_path):
    with open(original_json_path, 'r') as file:
        data = json.load(file)

    new_data = {}
    for key, value in data.items():
        wav_path = value['wav']
        fbank_path = generate_fbank_features(wav_path, save_dir)
        new_data[key] = {
            "fbank": fbank_path,
            "duration": value['duration'],
            "spk_id": value['spk_id'],
            "phn": value['phn']
        }

    with open(new_json_path, 'w') as file:
        json.dump(new_data, file, indent=4)


original_train_json = '/rds/user/yo279/hpc-work/MLMI2/exp/train.json'
original_dev_json = '/rds/user/yo279/hpc-work/MLMI2/exp/dev.json'
original_test_json = '/rds/user/yo279/hpc-work/MLMI2/exp/test.json'

fbank_save_dir = '/rds/user/yo279/hpc-work/MLMI2/fbanks'  # Directory to save FBank features
new_train_json = '/rds/user/yo279/hpc-work/MLMI2/fbanks/train_fbank.json'
new_dev_json = '/rds/user/yo279/hpc-work/MLMI2/fbanks/dev_fbank.json'
new_test_json = '/rds/user/yo279/hpc-work/MLMI2/fbanks/test_fbank.json'

create_fbank_json(original_train_json, fbank_save_dir, new_train_json)
create_fbank_json(original_dev_json, fbank_save_dir, new_dev_json)
create_fbank_json(original_test_json, fbank_save_dir, new_test_json)



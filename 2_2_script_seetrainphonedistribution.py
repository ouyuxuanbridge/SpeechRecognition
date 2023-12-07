import json

def count_mapped_phones(train_file, mapping_file, vocab_file):
    # Reading the mapping file
    phone_mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            original, mapped = line.strip().split(':')
            phone_mapping[original.strip()] = mapped.strip()

    # Read the vocab file to get the list of phones
    with open(vocab_file, 'r') as f:
        vocab_phones = set(f.read().splitlines())

    # Initialize a dictionary to count the phones
    phone_counts = {phone: 0 for phone in vocab_phones}

    # Read and parse the train.json file
    with open(train_file, 'r') as f:
        train_data = json.load(f)

    # Iterate through each entry and count the mapped phones
    for entry in train_data.values():
        phn_list = entry['phn'].split()
        for phn in phn_list:
            mapped_phn = phone_mapping.get(phn, phn)
            if mapped_phn in vocab_phones:
                phone_counts[mapped_phn] += 1

    return phone_counts

# Example usage
train_file = '/rds/user/yo279/hpc-work/MLMI2/exp/train.json'  # Replace with your file path
vocab_file = '/rds/user/yo279/hpc-work/MLMI2/exp/vocab_39.txt' # Replace with your file path
mapping_file='/rds/user/yo279/hpc-work/MLMI2/exp/phone_map'
phone_distribution = count_mapped_phones(train_file, mapping_file, vocab_file)
print(phone_distribution)
print(len(phone_distribution))


# Example usage


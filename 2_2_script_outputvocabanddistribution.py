def create_phone_map_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    # Extracting unique mappings
    unique_phones = set()
    for line in lines:
        parts = line.strip().split(':')
        if len(parts) == 2:
            unique_phones.add(parts[1].strip())

    # Writing to the output file
    with open(output_file_path, 'w') as file:
        # Writing the blank token
        file.write('_\n')

        # Writing unique phones
        for phone in sorted(unique_phones):
            file.write(f'{phone}\n')

# Example usage
input_file_path = '/rds/user/yo279/hpc-work/MLMI2/exp/phone_map'  # Replace with your file path
output_file_path = '/rds/user/yo279/hpc-work/MLMI2/exp/vocab_39.txt'    # Replace with your desired output file path

create_phone_map_file(input_file_path, output_file_path)

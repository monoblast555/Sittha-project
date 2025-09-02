# Trim m/z ratio from 1000 - 18000

import os


def trim(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, "t_" + filename)

            with open(input_path, 'r') as infile:
                lines = infile.readlines()

                # First line is the header
                header = lines[0]
                data_lines = lines[1:]

                # Filter data lines based on first column
                trimmed_lines = []
                for line in data_lines:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue  # Skip malformed lines
                    try:
                        value = float(parts[0])
                        if 1000 <= value <= 18000:
                            trimmed_lines.append(line)
                    except ValueError:
                        continue  # Skip lines with non-numeric values

            # Write result to new file in output folder
            with open(output_path, 'w') as outfile:
                outfile.write(header)
                outfile.writelines(trimmed_lines)

    print(f"Trimming completed. Files saved in: {output_folder}")



input_folder = "D://Research project//MALDI data//Lipids//4th bio GRE 11-7-2014"
output_folder = "D://Research project//MALDI data//Lipids_trimmed//4bio"
trim(input_folder, output_folder)

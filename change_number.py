import os

# Set the path to the folder containing the .txt files
folder_path = 'C:/Users/Ueli/Documents/GitHub/Bildverarbeitung_1/training_pepper_yoghurt/dataset_pepper_yoghurt/labels/val'  # <-- Change this to your folder

# Choose whether to overwrite the original files or create new ones
overwrite = True  # Set to False if you want to save with a new name

# Process each .txt file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)

        # Read all lines
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Modify each line
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts and parts[0] == '0':
                parts[0] = '1'
            new_line = ' '.join(parts) + '\n'
            new_lines.append(new_line)

        # Save changes
        if overwrite:
            save_path = file_path
        else:
            save_path = os.path.join(folder_path, f"modified_{filename}")

        with open(save_path, 'w') as file:
            file.writelines(new_lines)

print("Processing completed!")

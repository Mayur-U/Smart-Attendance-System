import os

def rename_files(folder_path, base_name):
    files = os.listdir(folder_path)
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

    for index, filename in enumerate(files, start=1):
        file_extension = os.path.splitext(filename)[1]
        new_name = f"{base_name}_{index}{file_extension}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)
        print(f"Renamed '{filename}' -> '{new_name}'")

if __name__ == "__main__":
    folder = input("Enter the folder path: ").strip()
    base_name = input("Enter the base name for files: ").strip()
    rename_files(folder, base_name)
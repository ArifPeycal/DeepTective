import os
import shutil

def combine_image_files(folder1,output_folder):
    """Combine image files from three folders into one output folder."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    all_files = set()  # To keep track of already processed filenames
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists

    def copy_images(source_folder):
        """Copy image files from the source folder to the output folder."""
        nonlocal all_files
        for root, _, files in os.walk(source_folder):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    original_path = os.path.join(root, file)
                    filename = file
                    # Resolve duplicate filenames by appending an index
                    counter = 1
                    while filename in all_files:
                        name, ext = os.path.splitext(file)
                        filename = f"{name}_{counter}{ext}"
                        counter += 1

                    destination_path = os.path.join(output_folder, filename)
                    shutil.copy2(original_path, destination_path)  # Copy with metadata
                    all_files.add(filename)  # Mark the filename as processed

    # Process all three folders
    copy_images(folder1)
    # copy_images(folder2)
    # copy_images(folder3)

    print(f"All image files have been combined into '{output_folder}'.")

if __name__ == "__main__":
    folder1 = r"C:\Users\ariff\Desktop\CelebDF Dataset\train\Real"
    # folder2 = r"C:\Users\ariff\Desktop\CelebDF Dataset(2)\val\fake"
    # folder3 = r"C:\Users\ariff\Desktop\prepared_dataset\fake"
    output_folder = r"C:\Users\ariff\Desktop\Combined_Images\Real"

    combine_image_files(folder1,output_folder)

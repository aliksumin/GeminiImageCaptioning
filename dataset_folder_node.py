import os
import torch
import numpy as np
from PIL import Image

class DatasetFolder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "PATH": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "FILENAME")
    FUNCTION = "load_image"
    CATEGORY = "Gemini"
    
    # We want this node to run every time to iterate through the folder
    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    # Class-level state to track index across batch runs
    _current_index = 0
    _current_folder = ""
    _image_cache = []

    def load_image(self, PATH):
        # Reset state if folder changes
        if PATH != DatasetFolder._current_folder:
            DatasetFolder._current_folder = PATH
            DatasetFolder._current_index = 0
            DatasetFolder._image_cache = []
            
            if os.path.exists(PATH) and os.path.isdir(PATH):
                valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
                file_list = []
                for f in os.listdir(PATH):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in valid_extensions:
                        file_list.append(f)
                
                # Sort to ensure consistent order
                DatasetFolder._image_cache = sorted(file_list)
        
        if not DatasetFolder._image_cache:
            # Return dummy image if empty or invalid
             return (torch.zeros(1, 64, 64, 3), "None")

        # Get current image
        filename = DatasetFolder._image_cache[DatasetFolder._current_index % len(DatasetFolder._image_cache)]
        full_path = os.path.join(DatasetFolder._current_folder, filename)
        
        # Load Image
        try:
            i = Image.open(full_path)
            i = i.convert("RGB") # Ensure RGB
            i = np.array(i).astype(np.float32) / 255.0
            image = torch.from_numpy(i)[None,]
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            return (torch.zeros(1, 64, 64, 3), "Error")
            
        # Return filename without extension
        filename_only = os.path.splitext(filename)[0]

        # Increment index for next run
        DatasetFolder._current_index += 1
        
        return (image, filename_only)

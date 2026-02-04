from .gemini_image_captioning_node import GeminiImageCaptioning
from .dataset_folder_node import DatasetFolder

NODE_CLASS_MAPPINGS = {
    "GeminiImageCaptioning": GeminiImageCaptioning,
    "DatasetFolder": DatasetFolder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageCaptioning": "Gemini Image Captioning",
    "DatasetFolder": "Dataset Folder"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

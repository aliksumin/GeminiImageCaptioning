from .gemini_image_captioning_node import GeminiImageCaptioning

NODE_CLASS_MAPPINGS = {
    "GeminiImageCaptioning": GeminiImageCaptioning
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageCaptioning": "Gemini Image Captioning"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

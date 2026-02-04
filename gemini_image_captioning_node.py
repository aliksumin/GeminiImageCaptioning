import torch
import numpy as np
from PIL import Image
import requests
import json
import base64
import io
import os

class GeminiImageCaptioning:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "PROMPT TYPE": (["SD1.5 – SDXL", "FLUX"], {"default": "SD1.5 – SDXL"}),
                "GEMINI MODEL": (["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-2.0-flash-exp"], {"default": "gemini-1.5-pro"}),
                "API KEY PATH": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "PROMPT LENGTH": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "defaultInput": True}),
                "PROMPT STRUCTURE": ("STRING", {"default": "", "multiline": True, "defaultInput": True}),
                "IGNORE": ("STRING", {"default": "", "multiline": True, "defaultInput": True}),
                "EMPHASIS": ("STRING", {"default": "", "multiline": True, "defaultInput": True}),
                "DICTIONARY": ("STRING", {"default": "", "multiline": True, "defaultInput": True}),
                "SAVE TO PATH": ("STRING", {"default": "", "multiline": False, "defaultInput": True}),
                "TXT NAME": ("STRING", {"default": "", "multiline": False, "defaultInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("CHECK_RESULT PROMPT", "LOG", "CAPTION")
    FUNCTION = "gen_caption"
    CATEGORY = "Gemini"

    def gen_caption(self, IMAGE, **kwargs):
        # Extract inputs
        prompt_type = kwargs.get("PROMPT TYPE")
        gemini_model = kwargs.get("GEMINI MODEL")
        api_key_path = kwargs.get("API KEY PATH")
        
        # Optional inputs (handle missing as empty/default)
        prompt_length = kwargs.get("PROMPT LENGTH", 0)
        prompt_structure = kwargs.get("PROMPT STRUCTURE", "")
        ignore = kwargs.get("IGNORE", "")
        emphasis = kwargs.get("EMPHASIS", "")
        dictionary = kwargs.get("DICTIONARY", "")
        save_to_path = kwargs.get("SAVE TO PATH", "")
        txt_name = kwargs.get("TXT NAME", "")

        log = []
        log.append("Starting Gemini Image Captioning...")

        # 1. Read API Key
        api_key = ""
        if api_key_path and os.path.exists(api_key_path):
            try:
                with open(api_key_path, 'r', encoding='utf-8') as f:
                    api_key = f.read().strip()
                log.append(f"API Key loaded from {api_key_path}")
            except Exception as e:
                log.append(f"Error reading API key file: {e}")
                return ("", "\n".join(log), "")
        else:
            log.append(f"API Key path invalid or not found: {api_key_path}")
            return ("", "\n".join(log), "")

        if not api_key:
            log.append("No API Key found.")
            return ("", "\n".join(log), "")

        # 2. Process Image
        try:
            # Convert tensor to PIL
            # ComfyUI images are [B, H, W, C] tensors in range 0-1
            i = 255. * IMAGE[0].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            log.append("Image processed and converted to base64.")
        except Exception as e:
            log.append(f"Error processing image: {e}")
            return ("", "\n".join(log), "")

        # 3. Construct Prompt
        prompt_parts = []

        # Block 1: Base Instruction
        prompt_parts.append("Give me a description of this image in the format of a text prompt for AI generative model. It should be only the descriptive text according to the provided template, without any additional comments from you. The text should be continuous, without headings, lists, or any other formatting.")

        # Block 2: Style Reference
        prompt_parts.append("Use the following reference as an example of the prompt format and structure, showing how the text should look. Use it only as a reference, do not use its content for the current request unless it is present in the attached image.")
        
        if prompt_type == "SD1.5 – SDXL":
            prompt_parts.append('"It should be in CLIP-L comma-separated keywords SDXL prompt style. This is the sample, don’t use it directly only like a style reference: "Architecture, high-end modernist residential complex, minimalist design, open balconies, subtle architectural details, concrete and glass façades, elegant geometric volumes, tiered rooftop terraces, panoramic floor-to-ceiling windows, neutral-toned stone panels, tinted glass curtain walls, brushed metal railings, integrated with lush landscaping, manicured hedges, ornamental grasses, sculptural trees, wooden pathway leading to a reflective metal sphere, secluded urban oasis, tranquil environment, free from city noise, surrounded by curated greenery, creating a serene and balanced atmosphere, soft diffused lighting, overcast sky, early morning mist, gentle atmospheric glow, cinematic wide-angle perspective, symmetrical framing, high dynamic range, RAW photo, hyper-detailed, photorealistic""')
        elif prompt_type == "FLUX":
            prompt_parts.append('"It should be in CLIP-G natural language FLUX prompt style. This is the sample, don’t use it directly only like a style reference: "Architecture, high-end modernist residential complex surrounded by lush greenery, designed with a minimalist and elegant aesthetic. The buildings feature a combination of natural stone and glass façades, with subtle architectural details and open balconies. A linear yet dynamic composition with clean geometric volumes, softened by carefully curated landscaping, including hedges, ornamental grasses, and small trees. The façade combines smooth concrete panels with floor-to-ceiling tinted glass windows, creating a refined balance of opacity and transparency. The outdoor space is defined by a wooden pathway meandering through a meticulously designed garden, leading towards a focal point—a polished metal sphere sculpture. Strategic lighting elements subtly highlight the landscape, while the gentle play of reflections on the glass surfaces enhances the depth of the environment. Set in a tranquil urban enclave, free from visual noise, framed by an overcast sky that casts a soft, diffused glow over the buildings. Early morning atmosphere with slight fog in the distance, lending an ethereal and cinematic quality to the scene. RAW photo, slightly elevated wide-angle viewpoint, cinematic framing, balanced symmetry, moderate depth of field, high dynamic range, hyper-detailed, photorealistic rendering.""')

        # Block 3: Prompt Structure
        prompt_parts.append("The structure of the prompt should be as follows (do not create headings or comments, only follow the order of information in the description):")
        if prompt_structure and prompt_structure.strip():
            prompt_parts.append(prompt_structure)
        else:
            default_structure = "1) Type of the building, \n2) Shape of the building, \n3) Building materials, \n4) Location and surroundings, \n5) Season, weather, daytime, lighting, \n6) Camera position and angle, composition, camera parameters"
            prompt_parts.append(default_structure)

        # Block 4: Ignore
        if ignore and ignore.strip():
            prompt_parts.append(f'In the prompt, be sure to ignore any mention of anything related to: {ignore}')

        # Block 5: Emphasis
        if emphasis and emphasis.strip():
            prompt_parts.append(f'In the prompt, emphasize additional attention on: {emphasis}')

        # Block 6: Dictionary (Added best effort based on input existence)
        if dictionary and dictionary.strip():
             prompt_parts.append(f'Consider using these words from the dictionary if strictly appropriate for the image: {dictionary}')

        # Block 7: Length
        if prompt_length and int(prompt_length) > 0:
            prompt_parts.append(f'The number of words in the prompt should be no more than {prompt_length}')

        final_prompt = "\n\n".join(prompt_parts)
        log.append("Prompt constructed.")
        
        # 4. Call Gemini API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": final_prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": img_base64
                            }
                        }
                    ]
                }
            ]
        }

        generated_text = ""
        try:
            log.append(f"Sending request to {url}...")
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                try:
                    generated_text = result['candidates'][0]['content']['parts'][0]['text']
                    log.append("Received successful response from Gemini.")
                except (KeyError, IndexError) as e:
                    log.append(f"Error parsing JSON response: {e}")
                    log.append(f"Full response: {result}")
            else:
                log.append(f"API Error: {response.status_code} - {response.text}")

        except Exception as e:
            log.append(f"Request Exception: {e}")

        # 5. Save to File
        if save_to_path and save_to_path.strip():
            try:
                if not os.path.exists(save_to_path):
                    os.makedirs(save_to_path)
                
                filename = txt_name.strip() if txt_name and txt_name.strip() else "caption"
                if not filename.lower().endswith(".txt"):
                    filename += ".txt"
                
                file_path = os.path.join(save_to_path, filename)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(generated_text)
                log.append(f"Saved caption to {file_path}")
            except Exception as e:
                log.append(f"Error saving to file: {e}")

        return (final_prompt, "\n".join(log), generated_text)

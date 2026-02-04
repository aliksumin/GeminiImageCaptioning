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
                "GEMINI MODEL": (["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash-preview", "gemini-3-pro-preview"], {"default": "gemini-3-flash-preview"}),
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

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("CHECK_RESULT PROMPT", "LOG", "CAPTION", "COST INFO")
    FUNCTION = "gen_caption"
    CATEGORY = "Gemini"

    # Pricing per 1M tokens (Input, Output)
    PRICING = {
        "gemini-2.5-flash": (0.30, 2.50),
        "gemini-2.5-pro": (1.25, 10.00),
        "gemini-3-flash-preview": (0.50, 3.00), # Preview pricing
        "gemini-3-pro-preview": (2.00, 12.00),   # Preview pricing
    }

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
                return ("", "\n".join(log), "", "Error loading API Key")
        else:
            log.append(f"API Key path invalid or not found: {api_key_path}")
            return ("", "\n".join(log), "", "API Key Missing")

        if not api_key:
            log.append("No API Key found.")
            return ("", "\n".join(log), "", "API Key Empty")

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
            return ("", "\n".join(log), "", "Image Processing Error")

        # 3. Construct Prompt
        prompt_parts = []
        
        # Base Instruction
        if prompt_type == "SD1.5 – SDXL":
            prompt_parts.append("Give me a description of this image in the format of a text prompt for AI generative model. It should be ONE string of CLIP-L comma-separated keywords or short phrases. Do not use full sentences.")
        else: # FLUX
            prompt_parts.append("Give me a detailed description of this image in the format of a text prompt for AI generative model. Use natural language sentences.")

        # Structure
        if prompt_structure and prompt_structure.strip():
            prompt_parts.append(f"Follow this structure strictly:\n{prompt_structure}")
        
        # Dictionary
        if dictionary and dictionary.strip():
            prompt_parts.append(f"Use these words if they apply to the image (do not force them if they don't fit):\n{dictionary}")
        
        # Ignore
        if ignore and ignore.strip():
            prompt_parts.append(f"Do NOT mention or include the following in the description:\n{ignore}")
            
        # Emphasis
        if emphasis and emphasis.strip():
            prompt_parts.append(f"Pay special attention to and emphasize these details:\n{emphasis}")
            
        # Length
        if prompt_length > 0:
            prompt_parts.append(f"The description should be no more than {prompt_length} words.")

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
        cost_info = "Cost calculation unavailable"
        
        try:
            log.append(f"Sending request to {url}...")
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                try:
                    generated_text = result['candidates'][0]['content']['parts'][0]['text']
                    log.append("Received successful response from Gemini.")
                    
                    # Calculate Cost
                    usage = result.get('usageMetadata', {})
                    prompt_tokens = usage.get('promptTokenCount', 0)
                    # candidatesTokenCount might be missing if response is purely safety blocked, but here we have text
                    candidates_tokens = usage.get('candidatesTokenCount', 0) 
                    
                    price_in, price_out = self.PRICING.get(gemini_model, (0, 0))
                    
                    # Cost = (Input Tokens / 1M) * Price In + (Output Tokens / 1M) * Price Out
                    cost = (prompt_tokens / 1_000_000) * price_in + (candidates_tokens / 1_000_000) * price_out
                    
                    cost_info = f"I: {prompt_tokens} toks | O: {candidates_tokens} toks | Cost: ${cost:.6f}"
                    log.append(f"Cost calculated: {cost_info}")
                    
                except (KeyError, IndexError) as e:
                    log.append(f"Error parsing JSON response: {e}")
                    log.append(f"Full response: {result}")
                    cost_info = "Error parsing usage metadata"
            else:
                log.append(f"API Error: {response.status_code} - {response.text}")
                cost_info = f"API Error: {response.status_code}"
                
                # Debug: if 404, list available models
                if response.status_code == 404:
                    try:
                        list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
                        log.append(f"Attempting to list available models from {list_url}...")
                        list_resp = requests.get(list_url)
                        if list_resp.status_code == 200:
                            models = list_resp.json().get('models', [])
                            model_names = [m.get('name') for m in models]
                            log.append(f"Available models for this key: {', '.join(model_names)}")
                        else:
                            log.append(f"Failed to list models: {list_resp.status_code}")
                    except Exception as list_e:
                        log.append(f"Error listing models: {list_e}")

        except Exception as e:
            log.append(f"Request Exception: {e}")
            cost_info = f"Request Exception: {e}"

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

        return (final_prompt, "\n".join(log), generated_text, cost_info)

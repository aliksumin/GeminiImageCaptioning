# Gemini Image Captioning Node for ComfyUI

This custom node for ComfyUI allows you to generate detailed image descriptions using Google's Gemini models. It is designed to help create captions for training LoRA models or other image-to-text tasks.

## Description
The node takes an image and various optional parameters to construct a prompt, sends it to the Google Gemini API, and returns the generated description. It can also save the description to a text file.

## Inputs

### Required
- **IMAGE**: The image you want to describe.
- **PROMPT TYPE**: Choose the style of the prompt.
    - `SD1.5 – SDXL`: Produces comma-separated keywords suitable for Stable Diffusion 1.5 and SDXL.
    - `FLUX`: Produces a natural language DESCRIPTION suitable for models like FLUX.
- **GEMINI MODEL**: Select the Google Gemini model to use for image understanding (e.g., `gemini-1.5-flash`, `gemini-1.5-pro`).
- **API KEY PATH**: The absolute path to a `.txt` file containing your Google Gemini API key.

### Optional
- **PROMPT LENGTH**: The target maximum number of words for the description.
- **PROMPT STRUCTURE**: A custom structure for the description. If left empty, a default structure will be used.
- **IGNORE**: Specify objects or details to ignore in the description.
- **EMPHASIS**: Specify objects or details to emphasize in the description.
- **DICTIONARY**: A list of words that the model should consider using if appropriate.
- **SAVE TO PATH**: The folder path where the result should be saved as a `.txt` file.
- **TXT NAME**: The filename for the saved `.txt` file (without extension).

## Outputs
- **CHECK_RESULT PROMPT**: The final text prompt that was sent to Gemini.
- **LOG**: A log of the process, useful for debugging.
- **CAPTION**: The generated image description from Gemini.

## Installation

To install the required libraries, run the following command found in your ComfyUI python environment:

```sh
pip install Pillow requests
```

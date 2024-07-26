import os
import easyocr
import numpy as np
import ocr
import torch
from diffusers import StableDiffusionPipeline


# if not os.path.exists('stable_diffusion_pytorch'):
#     os.system('git clone https://github.com/kjsman/stable-diffusion-pytorch.git')
#     os.rename('stable-diffusion-pytorch', 'stable_diffusion_pytorch')
#     os.system("wget https://huggingface.co/jinseokim/stable-diffusion-pytorch-data/resolve/main/data.v20221029.tar")
#     # decompres the data in the stable_diffusion_pytorch folder
#     os.system("tar -xvf data.v20221029.tar -C stable_diffusion_pytorch")
#     # delete the tar file
#     os.system("rm data.v20221029.tar")
    
# from stable_diffusion_pytorch.stable_diffusion_pytorch import model_loader, pipeline
    
# models = model_loader.preload_models(device="cpu")

model_id = "runwayml/stable-diffusion-v1-5"
prompt = "a man holding a sign that says 'Me hago caca'"
replace = ["Me hago caca"]



# encapsulate diffusion params
def diffusion(prompt):
    
    prompts = [prompt]

    uncond_prompt = ""  #@param { type: "string" }
    uncond_prompts = [uncond_prompt] if uncond_prompt else None

    # input_images = None

    # strength = 0.8  #@param { type:"slider", min: 0, max: 1, step: 0.01 }

    # do_cfg = True  #@param { type: "boolean" }
    cfg_scale = 7.5  #@param { type:"slider", min: 1, max: 14, step: 0.5 }
    height = 512  #@param { type: "integer" }
    width = 512  #@param { type: "integer" }
    # sampler = "k_lms"  #@param ["k_lms", "k_euler", "k_euler_ancestral"]
    n_inference_steps = 35  #@param { type: "integer" }

    use_seed = False  #@param { type: "boolean" }
    if use_seed:
        seed = 42  #@param { type: "integer" }
    else:
        seed = None
        
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    image = pipe(prompt=prompt,
                 height = height,
                 width = width,
                 num_inference_steps=n_inference_steps,
                 guidance_scale=cfg_scale,
                 negative_prompt=uncond_prompts,
                 generator=torch.Generator().manual_seed(seed) if seed is not None else None,)
    
    
    return image.images[0]

def simple_inpaint(image, bounds, word):
    from td_inpaint import inpaint
    from inpaint_functions import parse_bounds
    
    global_dict = {}
    global_dict["stack"] = parse_bounds(bounds, word)
    print(global_dict["stack"])
    #image = "./hat.jpg"
    prompt = ""
    keywords = ""
    positive_prompt = ""
    radio = 8
    slider_step = 25
    slider_guidance= 7
    slider_batch= 4
    slider_natural= False
    return inpaint(image, prompt,keywords,positive_prompt,radio,slider_step,slider_guidance,slider_batch,slider_natural, global_dict)


# models = model_loader.preload_models(device="cpu")

# Basic Stable Diffusion
image = diffusion(prompt)
ocr.display(image)
# model_loader.delete_models(models)

image = image.resize((512,512))
images = [image]

image_arr = np.array(image)

# OCR Mask Gen
reader = easyocr.Reader(['en'])
bounds = reader.readtext(image_arr)

# TD2 Inpaint
result = simple_inpaint(image, bounds, replace)
stitched = ocr.stitch(result[0])
ocr.display(stitched)
from flask import Flask, render_template, request, send_file
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
import torch
import os
import io
import base64

app = Flask(__name__)

# 모델 로드
model_name = "CompVis/stable-diffusion-v1-4"
text2img_pipe = StableDiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    variant="fp16",
    low_cpu_mem_usage=True
).to("mps")

img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    variant="fp16",
    low_cpu_mem_usage=True
).to("mps")

def enhance_prompt(user_input):
    enhancements = {
        "baby": "infant, small, cute, chubby cheeks",
        "old": "elderly, wrinkled, grey hair",
        "young": "youthful, energetic, smooth skin",
        "tall": "long legs, towering, statuesque",
        "short": "petite, compact, small stature",
        "fat": "overweight, plump, round belly",
        "thin": "skinny, slender, lean figure",
        "muscular": "strong, well-built, defined muscles",
        "beautiful": "attractive, gorgeous, stunning features",
        "ugly": "unattractive, grotesque, unpleasant features",
    }
    enhanced = enhancements.get(user_input.lower(), user_input)
    return f"({enhanced})1.5"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['prompt']
        num_images = int(request.form['num_images'])
        
        sketch_file = request.files.get('sketch')
        canvas_image = request.form.get('canvasImage')
        
        enhanced_prompt = enhance_prompt(user_input)
        base_prompt = "A 2.5D digital cartoon character, bright and colorful, well-lit, "
        prompt = f"{enhanced_prompt}, {base_prompt} centered composition, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, cinematic lighting, 3D render style, vibrant colors"
        negative_prompt = "dark, gloomy, shadowy, blurry, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, extra background, partial figure, close-up, cut off, no face"

        if sketch_file:
            init_image = Image.open(sketch_file.stream).convert("RGB")
        elif canvas_image:
            canvas_data = canvas_image.split(',')[1]
            canvas_data_decoded = base64.b64decode(canvas_data)
            init_image = Image.open(io.BytesIO(canvas_data_decoded)).convert("RGB")
        else:
            init_image = None

        width, height = 512, 768

        if init_image:
            init_image = init_image.resize((width, height))
            images = img2img_pipe(
                prompt=prompt,
                image=init_image,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=50,
                guidance_scale=8.0,
                strength=0.75
            ).images
        else:
            images = text2img_pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=50,
                width=width,
                height=height,
                guidance_scale=8.0,
                generator=torch.manual_seed(0)
            ).images

        os.makedirs('static', exist_ok=True)
        image_data = []
        for i, img in enumerate(images):
            path = f"static/output_{i}.png"
            img.save(path)
            image_data.append({"id": i, "path": path})

        return render_template('result.html', image_data=image_data)

    return render_template('index.html')

@app.route('/download/<int:image_id>')
def download(image_id):
    file_path = f'static/output_{image_id}.png'
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

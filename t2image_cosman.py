import torch
from diffusers import StableDiffusionImg2ImgPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from PIL import Image

# 1. 本地路径配置
base_path = r"F:\Media_Department\head_avatar\text2avatar\baseline\HeadStudio_lib\realistic-vision-51"
unet_path = r"H:\work2\diffusion_lib\cosmicman"

# 2. 加载模型 (注意这里改成了 StableDiffusionImg2ImgPipeline)
unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16, local_files_only=True)

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    base_path,
    unet=unet,
    torch_dtype=torch.float16,
    local_files_only=True
).to("cuda")

pipe.safety_checker = None

pipe.scheduler = EulerDiscreteScheduler.from_pretrained(
    base_path, subfolder="scheduler", torch_dtype=torch.float16, local_files_only=True
)

#
init_image_path = r"F:\Media_Department\head_avatar\text2avatar\baseline\T2Head_new_teeth_v3\outputs\reconstruction\00000001\driven\frames\000058.png"
init_image = Image.open(init_image_path).convert("RGB").resize((512, 512))

#
positive_prompt = "a DSLR portrait of a handsome young brown lightly wavy Pompadour with fade, muscled sportsman in navy satin large pinstripe double-breasted suit, pompadour haircut"
negative_prompt = "worst quality, low quality, lowres, blurry, out of focus, jpeg artifacts, noise, grain, watermark, text, logo, signature, cropped, out of frame, duplicate, deformed, disfigured, bad anatomy, bad proportions, asymmetry, extra face, extra eyes, extra mouth, deformed iris, deformed pupils, cross-eyed, dead eyes, glasses, eyeglasses, strong shadow, harsh shadow, underexposed, dark face, overexposed, blown highlights, high contrast, HDR, neon, glow, bloom, lens flare, light streaks, god rays, volumetric lighting, chromatic aberration, color fringing, oversaturated, weird colors, over-sharpened, halos, ringing, banding, cgi, 3d render, cartoon, anime, sketch"

# 5. SDEdit 生成图像
image = pipe(
    prompt=positive_prompt,
    image=init_image,             # 传入输入图像
    strength=0.3,                 # SDEdit 的核心参数 (0.0~1.0)
    num_inference_steps=30,
    guidance_scale=7.5,
    negative_prompt=negative_prompt,
    output_type="pil"
).images[0]

# 6. 保存输出结果
image.save("sdedit_output.png")
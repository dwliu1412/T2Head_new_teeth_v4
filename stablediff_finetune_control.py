from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline,AutoencoderTiny
import numpy as np
from pathlib import Path
import glob
import os
# suppress partial model loading warning
import threestudio

logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import yaml
import tqdm
from threestudio.utils.perpneg_utils import weighted_perpendicular_aggregator
from threestudio.utils.config import ExperimentConfig, load_config
from threestudio.utils.typing_ import Optional

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, t_range=[0.02, 0.98],):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        #self.vae = AutoencoderKL.from_pretrained('F:/high_quality_3DPortraitGAN/exp/stable-dreamfusion/pretrained/vae-ft-mse-840000-ema-pruned', torch_dtype=self.precision_t).to(self.device)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]
        with torch.no_grad():
            inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
            embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None):

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1

            # feature_image + (1 - weights_samples) * bcg_image
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        # import kiui
        # latents_tmp = torch.randn((1, 4, 64, 64), device=self.device)
        # latents_tmp = latents_tmp.detach()
        # kiui.lo(latents_tmp)
        # self.scheduler.set_timesteps(30)
        # for i, t in enumerate(self.scheduler.timesteps):
        #     latent_model_input = torch.cat([latents_tmp] * 3)
        #     noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
        #     noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + 10 * (noise_pred_pos - noise_pred_uncond)
        #     latents_tmp = self.scheduler.step(noise_pred, t, latents_tmp)['prev_sample']
        # imgs = self.decode_latents(latents_tmp)
        # kiui.vis.plot_image(imgs)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        if save_guidance_path:
            with torch.no_grad():
                if as_latent:
                    pred_rgb_512 = self.decode_latents(latents)
                #
                # # visualize predicted denoised image
                # # The following block of code is equivalent to `predict_start_from_noise`...
                # # see zero123_utils.py's version for a simpler implementation.
                # alphas = self.scheduler.alphas.to(latents)
                # total_timesteps = self.max_step - self.min_step + 1
                # index = total_timesteps - t.to(latents.device) - 1
                # b = len(noise_pred)
                # a_t = alphas[index].reshape(b, 1, 1, 1).to(self.device)
                # sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
                # sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b, 1, 1, 1)).to(self.device)
                # pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt()  # current prediction for x_0
                # result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))
                #
                # # visualize noisier image
                # result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))
                #
                # # TODO: also denoise all-the-way
                # # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                # # print(F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False).shape, pred_rgb_512.shape)
                # viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image], dim=0)
                # save_image(viz_images, save_guidance_path)

                guidance_eval_utils = {
                    "use_perp_neg": False,
                    "neg_guidance_weights": None,
                    "text_embeddings": text_embeddings,
                    "t_orig": t,
                    "latents_noisy": latents_noisy,
                    "noise_pred": noise_pred,
                    "guidance_scale": guidance_scale,
                    "return_imgs_final": False,
                }

                guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
                # decode_latents(latents_1step).permute(0, 2, 3, 1)
                # "imgs_noisy": imgs_noisy,
                # "imgs_1step": imgs_1step,
                # "imgs_1orig": imgs_1orig,
                # "imgs_final": imgs_final,
                viz_images = [pred_rgb_512]
                for k in guidance_eval_out:
                    if k.startswith("imgs_"):
                        viz_images.append(guidance_eval_out[k])
                viz_images = torch.cat(viz_images, dim=0)

                save_image(viz_images, save_guidance_path)

        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        return loss

    @torch.no_grad()
    def get_noise_pred(
            self,
            latents_noisy,
            t,
            text_embeddings,
            use_perp_neg=False,
            neg_guidance_weights=None,
            guidance_scale=100.0,
    ):
        batch_size = latents_noisy.shape[0]

        if use_perp_neg:
            raise NotImplementedError
        else:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred =  self.unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            ).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_text + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )

        return noise_pred

    @torch.no_grad()
    def guidance_eval(
            self,
            t_orig,
            text_embeddings,
            latents_noisy,
            noise_pred,
            use_perp_neg=False,
            neg_guidance_weights=None,
            guidance_scale=100.0,
            return_imgs_final=False,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        max_items_eval = 4
        bs = (
            min(max_items_eval, latents_noisy.shape[0])
            if max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[:bs].unsqueeze(
            -1)  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs])

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b: b + 1], t[b], latents_noisy[b: b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step)
        imgs_1orig = self.decode_latents(pred_1orig)

        res = {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,

        }
        if return_imgs_final:
            latents_final = []
            for b, i in enumerate(idxs):
                latents = latents_1step[b: b + 1]
                text_emb = (
                    text_embeddings[
                        [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                    ]
                    if use_perp_neg
                    else text_embeddings[[b, b + len(idxs)], ...]
                )
                neg_guid = neg_guidance_weights[b: b + 1] if use_perp_neg else None
                for t in self.scheduler.timesteps[i + 1:]:
                    # pred noise
                    # noise_pred = self.get_noise_pred(
                    #     latents, t, text_emb, use_perp_neg, neg_guid,guidance_scale = guidance_scale
                    # )

                    latent_model_input = torch.cat([latents] * 2, dim=0)
                    noise_pred = self.unet(
                        latent_model_input,
                        torch.cat([t.reshape(1)] * 2).to(self.device),
                        encoder_hidden_states=text_emb,
                    ).sample

                    # perform guidance (high scale from paper!)
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_text + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                    )


                    # get prev latent
                    latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                        "prev_sample"
                    ]
                latents_final.append(latents)

            latents_final = torch.cat(latents_final)
            imgs_final = self.decode_latents(latents_final)

            res["imgs_final"] = imgs_final

        return res

    def train_step_perpneg(self, text_embeddings, weights, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None):

        B = pred_rgb.shape[0]
        K = (text_embeddings.shape[0] // B) - 1 # maximum number of prompts

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * (1 + K))
            tt = torch.cat([t] * (1 + K))
            unet_output = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = unet_output[:B], unet_output[B:]
            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
            noise_pred = noise_pred_uncond + guidance_scale * weighted_perpendicular_aggregator(delta_noise_preds, weights, B)

        # import kiui
        # latents_tmp = torch.randn((1, 4, 64, 64), device=self.device)
        # latents_tmp = latents_tmp.detach()
        # kiui.lo(latents_tmp)
        # self.scheduler.set_timesteps(30)
        # for i, t in enumerate(self.scheduler.timesteps):
        #     latent_model_input = torch.cat([latents_tmp] * 3)
        #     noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
        #     noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + 10 * (noise_pred_pos - noise_pred_uncond)
        #     latents_tmp = self.scheduler.step(noise_pred, t, latents_tmp)['prev_sample']
        # imgs = self.decode_latents(latents_tmp)
        # kiui.vis.plot_image(imgs)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        if save_guidance_path:
            with torch.no_grad():
                if as_latent:
                    pred_rgb_512 = self.decode_latents(latents)

                # visualize predicted denoised image
                # The following block of code is equivalent to `predict_start_from_noise`...
                # see zero123_utils.py's version for a simpler implementation.
                alphas = self.alphas.to(latents)
                total_timesteps = self.max_step - self.min_step + 1
                index = total_timesteps - t.to(latents.device) - 1
                b = len(noise_pred)
                a_t = alphas[index].reshape(b,1,1,1).to(self.device)
                sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
                sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(self.device)
                pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0
                result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))

                # visualize noisier image
                result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))



                # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image],dim=0)
                save_image(viz_images, save_guidance_path)

        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        return loss

    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts) # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs

    def denoise_latents(self, text_embeddings, start_t, num_inference_steps=50, guidance_scale=7.5, latents=None):

        self.scheduler.set_timesteps(num_inference_steps)
        for t in tqdm.tqdm(self.scheduler.timesteps):
            if t > start_t:
                continue
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def denoise_latents_control(self, text_embeddings, start_t, num_inference_steps=50, guidance_scale=7.5,
                                latents=None, guidance=None, image_cond=None, controlnet_scale=1.5,):
        device = latents.device
        self.scheduler.set_timesteps(num_inference_steps)
        for t in tqdm.tqdm(self.scheduler.timesteps):
            if t > start_t:
                continue
            latent_model_input = torch.cat([latents] * 2)
            image_cond_input = torch.cat([image_cond] * 2)
            (
                down_block_res_samples,
                mid_block_res_sample,
            ) = guidance.forward_controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                image_cond=image_cond_input,
                condition_scale=controlnet_scale,
            )

            noise_pred = guidance.forward_control_unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )
            # perform classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )
            # get previous sample, continue loop
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def sdedit(self, guidance, data_dir, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, start=0, end=100000):

        noise_level = 200
        res_dir = data_dir
        imgs_dir = os.path.join(res_dir, 'nersamble', 'imgs')
        flame_dir = os.path.join(res_dir, 'nersamble', 'flame')
        if not (os.path.exists(imgs_dir) or os.path.exists(flame_dir)):
            print('no data dir')
            return

        update_data_dir = os.path.join(res_dir, 'finetune_data')
        os.makedirs(update_data_dir, exist_ok=True)

        if len(glob.glob(imgs_dir + '/*.png')) == len(glob.glob(update_data_dir + '/*.png')):
            print('already done')
            return
        print('gen data for ', res_dir)

        name = os.path.basename(res_dir)

        # 读取prompt
        with open(os.path.join(res_dir, 'configs/raw.yaml'), "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # print(cfg)
        prompts = cfg['system']['prompt_processor']['prompt']
        neg_prompts = cfg['system']['prompt_processor']['negative_prompt']
        # prompt_path = os.path.join(test_data_dir, f'{name}/prompt.txt')
        pmts = []
        pmts.append(neg_prompts)
        pmts.append(prompts)
        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(pmts)  # [2, 77, 768]

        imgs_list = sorted(glob.glob(os.path.join(imgs_dir, '*.png')))

        for idx, image_path in enumerate(tqdm.tqdm(imgs_list)):
            if not (start <= idx <= end):
                continue
            base = os.path.basename(image_path)  # '000000.png'
            flame_path = os.path.join(flame_dir, 'flame_' + base)  # 'flame_000000.png'

            image = PIL.Image.open(image_path).convert('RGB')
            image = np.array(image)

            origin_img = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)  # 1,3,1024,1024
            origin_img = origin_img / 255.0

            # flame
            flame = PIL.Image.open(flame_path).convert('RGB')
            flame = np.array(flame)
            flame_img = torch.from_numpy(flame).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0 # --> 0,1
            flame_img = F.interpolate(
                flame_img, (1024, 1024), mode="bilinear", align_corners=False
            )

            latents = self.encode_imgs(origin_img)

            t = torch.tensor([noise_level], dtype=torch.long, device=self.device)

            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latents = self.denoise_latents_control(
                    text_embeddings=text_embeds,
                    start_t=noise_level,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    latents=latents_noisy,
                    guidance=guidance,
                    image_cond=flame_img,
                    controlnet_scale=1.5
                )

            # Img latents -> imgs
            img = self.decode_latents(latents)  # [1, 3, 1024, 1024]
            # Img to Numpy
            img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
            img = (img * 255).round().astype('uint8')[0]

            PIL.Image.fromarray(img).save(os.path.join(update_data_dir, os.path.basename(image_path)))

    def denoise_img(self, guidance, img, flame_img, config_path, noise, elevation, azimuth, camera_distances, num_inference_steps=50, guidance_scale=7.5):

        noise_level = 200

        # 读取prompt
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        front_threshold = (45, 135)
        back_threshold = (-135, -45)
        overhead_threshold = 60.0
        azimuth_shifted = (azimuth + 180) % 360 - 180
        if elevation > overhead_threshold:
            view_idx = 3  # overhead
        elif (azimuth_shifted > front_threshold[0]) and (azimuth_shifted < front_threshold[1]):
            view_idx = 1  # front
        elif (azimuth_shifted > back_threshold[0]) and (azimuth_shifted < back_threshold[1]):
            view_idx = 2  # back
        else:
            view_idx = 0  # side
        prompt_raw = cfg['system']['prompt_processor']['prompt']
        neg_prompts = cfg['system']['prompt_processor']['negative_prompt']
        view_names = ["side", "front", "back", "overhead"]
        view_name = view_names[view_idx]
        current_prompt = cfg['system']['prompt_processor'].get(f'prompt_{view_name}', None)
        if current_prompt is None:
            if view_name == "side":
                current_prompt = f"{prompt_raw}, side view"
            elif view_name == "front":
                current_prompt = f"{prompt_raw}, front view"
            elif view_name == "back":
                current_prompt = f"{prompt_raw}, back view"
            elif view_name == "overhead":
                current_prompt = f"{prompt_raw}, overhead view"

        pmts = [neg_prompts, current_prompt]
        # pmts.append(neg_prompts)
        # pmts.append(prompts)
        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(pmts)  # [2, 77, 768]

        latents = self.encode_imgs(img)
        flame_img = F.interpolate(flame_img, (1024, 1024), mode="bilinear", align_corners=False)

        t = torch.tensor([noise_level], dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latents = self.denoise_latents_control(
                text_embeddings=text_embeds,
                start_t=noise_level,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                latents=latents_noisy,
                guidance=guidance,
                image_cond=flame_img,
                controlnet_scale=1.5
            )

        # Img latents -> imgs
        fine_img = self.decode_latents(latents)  # [1, 3, 1024, 1024]
        # Img to Numpy
        fine_img = fine_img.detach().cpu().permute(0, 2, 3, 1).numpy()
        fine_img = (fine_img * 255).round().astype('uint8')[0]

        return fine_img

    def _prepare_mask(self, mouth_mask, img, latents):
        # mouth_mask: [1,H,W] or [1,1,H,W]
        if mouth_mask.ndim == 3:
            mouth_mask = mouth_mask.unsqueeze(1)
        mouth_mask = mouth_mask.to(img.device, dtype=img.dtype)

        # 二值化，避免软边太弱
        mouth_mask = (mouth_mask > 0.5).float()

        # latent 尺度的 mask
        mask_latent = F.interpolate(
            mouth_mask, size=latents.shape[-2:], mode="nearest"
        )
        return mouth_mask, mask_latent

    @torch.no_grad()
    def denoise_latents_control_masked(
            self,
            text_embeddings,
            start_t,
            num_inference_steps=50,
            guidance_scale=7.5,
            latents=None,
            guidance=None,
            image_cond=None,
            controlnet_scale=1.5,
            latents_orig=None,
            mask_latent=None,
            base_noise=None,
    ):
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        started = False
        for i, t in enumerate(timesteps):
            if t > start_t:
                continue

            started = True

            latent_model_input = torch.cat([latents] * 2)
            image_cond_input = torch.cat([image_cond] * 2)

            down_block_res_samples, mid_block_res_sample = guidance.forward_controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                image_cond=image_cond_input,
                condition_scale=controlnet_scale,
            )

            noise_pred = guidance.forward_control_unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # 非编辑区重新锁回原图轨迹
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                if t_next <= start_t:
                    orig_next = self.scheduler.add_noise(
                        latents_orig, base_noise, t_next.reshape(1).to(latents.device)
                    )
                else:
                    orig_next = latents_orig
            else:
                orig_next = latents_orig

            latents = mask_latent * latents + (1.0 - mask_latent) * orig_next

        # 防御：如果一步都没跑，直接返回初始化结果
        if not started:
            latents = mask_latent * latents + (1.0 - mask_latent) * latents_orig

        return latents

    @torch.no_grad()
    def denoise_img_masked(
            self,
            guidance,
            img,
            mouth_mask,
            flame_img,
            config_path,
            noise,
            elevation,
            azimuth,
            camera_distances,
            num_inference_steps=50,
            guidance_scale=7.5,
    ):
        noise_level = 200

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        front_threshold = (45, 135)
        back_threshold = (-135, -45)
        overhead_threshold = 60.0
        azimuth_shifted = (azimuth + 180) % 360 - 180

        if elevation > overhead_threshold:
            view_idx = 3
        elif (azimuth_shifted > front_threshold[0]) and (azimuth_shifted < front_threshold[1]):
            view_idx = 1
        elif (azimuth_shifted > back_threshold[0]) and (azimuth_shifted < back_threshold[1]):
            view_idx = 2
        else:
            view_idx = 0

        prompt_raw = cfg['system']['prompt_processor']['prompt']
        neg_prompts = cfg['system']['prompt_processor']['negative_prompt']
        view_names = ["side", "front", "back", "overhead"]
        view_name = view_names[view_idx]
        current_prompt = cfg['system']['prompt_processor'].get(f'prompt_{view_name}', None)

        if current_prompt is None:
            if view_name == "side":
                current_prompt = f"{prompt_raw}, side view"
            elif view_name == "front":
                current_prompt = f"{prompt_raw}, front view"
            elif view_name == "back":
                current_prompt = f"{prompt_raw}, back view"
            else:
                current_prompt = f"{prompt_raw}, overhead view"

        # 可选：嘴部更强提示
        current_prompt = current_prompt + ", open mouth, visible teeth, clear mouth interior, detailed lips"

        pmts = [neg_prompts, current_prompt]
        text_embeds = self.get_text_embeds(pmts)

        # 原图编码
        latents_orig = self.encode_imgs(img)

        # 准备 mask
        mouth_mask, mask_latent = self._prepare_mask(mouth_mask, img, latents_orig)

        flame_img = F.interpolate(flame_img, (1024, 1024), mode="bilinear", align_corners=False)
        t = torch.tensor([noise_level], dtype=torch.long, device=self.device)

        # 原图 noisy latent
        latents_orig_noisy = self.scheduler.add_noise(latents_orig, noise, t)

        # 初始化：非 mouth 区保持原图 noisy，mouth 区改成随机噪声
        latents_init = (1.0 - mask_latent) * latents_orig_noisy + mask_latent * noise

        latents = self.denoise_latents_control_masked(
            text_embeddings=text_embeds,
            start_t=noise_level,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            latents=latents_init,
            guidance=guidance,
            image_cond=flame_img,
            controlnet_scale=1.5,
            latents_orig=latents_orig,
            mask_latent=mask_latent,
            base_noise=noise,
        )

        fine_img = self.decode_latents(latents).clamp(0, 1)

        # 最终只替换嘴部，外部区域强制保留原图
        fine_img = mouth_mask * fine_img + (1.0 - mouth_mask) * img

        fine_img = fine_img.detach().cpu().permute(0, 2, 3, 1).numpy()
        fine_img = (fine_img * 255).round().astype("uint8")[0]
        return fine_img
    
    @torch.no_grad()
    def denoise_img_residual(
            self,
            guidance,
            img,
            flame_img,
            mouth_mask,
            config_path,
            noise,
            elevation,
            azimuth,
            camera_distances,
            num_inference_steps=50,
            guidance_scale=7.5,
    ):
        # 比原来的 200 更保守
        noise_level = 160

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        front_threshold = (45, 135)
        back_threshold = (-135, -45)
        overhead_threshold = 60.0
        azimuth_shifted = (azimuth + 180) % 360 - 180

        if elevation > overhead_threshold:
            view_idx = 3
        elif (azimuth_shifted > front_threshold[0]) and (azimuth_shifted < front_threshold[1]):
            view_idx = 1
        elif (azimuth_shifted > back_threshold[0]) and (azimuth_shifted < back_threshold[1]):
            view_idx = 2
        else:
            view_idx = 0

        prompt_raw = cfg['system']['prompt_processor']['prompt']
        neg_prompts = cfg['system']['prompt_processor']['negative_prompt']
        view_names = ["side", "front", "back", "overhead"]
        view_name = view_names[view_idx]
        current_prompt = cfg['system']['prompt_processor'].get(f'prompt_{view_name}', None)

        if current_prompt is None:
            if view_name == "side":
                current_prompt = f"{prompt_raw}, side view"
            elif view_name == "front":
                current_prompt = f"{prompt_raw}, front view"
            elif view_name == "back":
                current_prompt = f"{prompt_raw}, back view"
            else:
                current_prompt = f"{prompt_raw}, overhead view"

        # 不要太激进，只做轻微口腔修正
        current_prompt = current_prompt + ", natural mouth details, clear oral cavity, consistent face identity"

        text_embeds = self.get_text_embeds([neg_prompts, current_prompt])

        latents = self.encode_imgs(img)
        flame_img = F.interpolate(flame_img, (1024, 1024), mode="bilinear", align_corners=False)

        t = torch.tensor([noise_level], dtype=torch.long, device=self.device)

        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        latents_edit = self.denoise_latents_control(
            text_embeddings=text_embeds,
            start_t=noise_level,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            latents=latents_noisy,
            guidance=guidance,
            image_cond=flame_img,
            controlnet_scale=1.5,
        )

        fine_img = self.decode_latents(latents_edit).clamp(0, 1)

        # 只替换更小的 inner-mouth core，外部全部保留原图
        if mouth_mask.ndim == 3:
            mouth_mask = mouth_mask.unsqueeze(1)
        mouth_mask = F.interpolate(
            mouth_mask.to(fine_img.device, fine_img.dtype),
            size=fine_img.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

        blended = fine_img * mouth_mask + img * (1.0 - mouth_mask)

        blended = blended.detach().cpu().permute(0, 2, 3, 1).numpy()
        blended = (blended * 255).round().astype("uint8")[0]
        return blended


if __name__ == '__main__':

    import argparse
    import PIL

    parser = argparse.ArgumentParser()
    parser.add_argument('--sd_version', type=str, default='1.5', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default='../HeadStudio_lib/realistic-vision-51', help="hugging face Stable diffusion model key")

    parser.add_argument('--data_dir', type=str, default='./outputs/headstudio/20250807-070243Messi', help='Network pickle filename')
    # parser.add_argument('--test_data_dir', type=str, help='test_data_dir', required=True)

    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=1024)
    parser.add_argument('-W', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=30000)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)
    # init guidance
    cfg = load_config("configs/t2head.yaml")
    guidance = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)

    imgs = sd.sdedit(guidance, opt.data_dir, opt.H, opt.W, opt.steps, start=opt.start, end=opt.end)

    print('finished!')

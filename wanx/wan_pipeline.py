import os
import math
import random
import torch
import numpy as np
import pickle as pkl
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision import transforms
from einops import rearrange, repeat
from transformers import AutoTokenizer
from wanx.wan_transformers_pose import WanTransformer3DModel
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

import PIL
from PIL import Image
from diffusers.utils import PIL_INTERPOLATION

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

PRECISION_TO_TYPE = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

class WanImageToVideoPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for image-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        image_encoder ([`CLIPVisionModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel), specifically
            the
            [clip-vit-huge-patch14](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md#vit-h14-xlm-roberta-large)
            variant.
        transformer ([`WanTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        transformer: WanTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        saved_prompt_embeds: str = None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.prompt_embeds, self.n_prompt_embeds = torch.zeros((1, 6, 4096)), torch.zeros((1, 6, 4096))
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1)
        self.latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1)

    @torch.no_grad()
    def get_latent(self, img_list, width, height):
        latents_mean = self.latents_mean.to(self.vae.device, self.vae.dtype)
        latents_std = self.latents_std.to(self.vae.device, self.vae.dtype)

        dtype = self.vae.dtype
        transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5]),
                        ]
                    )
        img_tensor = torch.stack([transform(img.resize((width, height))) for img in img_list], dim=1).unsqueeze(0)
        img_tensor = img_tensor.to(dtype=dtype, device=self.vae.device) # [1, 3, 2, 1024, 768]
        img_tensor = retrieve_latents(self.vae.encode(img_tensor), sample_mode="argmax")
        img_tensor = (img_tensor - latents_mean) * latents_std
        return img_tensor

    def get_saved_prompt(self, saved_prompt_embeds):
        prompt_embeds = None
        n_prompt_embeds = None
        if os.path.exists(saved_prompt_embeds):
            with open(saved_prompt_embeds, 'rb') as f:
                data = pkl.load(f)
                print("==> load saved_prompt_embeds success: {}".format(saved_prompt_embeds))

            prompt_embeds = torch.from_numpy(data['context'][0])
            n_prompt_embeds = torch.from_numpy(data['context_null'][0])
            prompt = data['prompt']
            n_prompt = data['n_prompt']
            print("--> prompt: ", prompt)
            print("--> n_prompt: ", n_prompt)
        return prompt_embeds, n_prompt_embeds

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
        pose_imgs=None,
        ref_img=None,
        ref_pose=None,
        dummy_ref_img=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # vae编码
        pose_latents = pose_imgs
        ref_img_1_latents = ref_img[0]
        ref_img_2_latents = ref_img[-1]
        ref_pose_1_latents = ref_pose[0]
        ref_pose_2_latents = ref_pose[-1]
        if dummy_ref_img is None:
            dummpy_reference_input = torch.zeros_like(latents)
        else:
            padding_shape = latents[:, :, 1:, :, :].shape
            padding_latents = torch.zeros(padding_shape, device=device, dtype=dtype)
            dummpy_reference_input = torch.cat([dummy_ref_img, padding_latents], dim=2)
        # 在通道维度上复制一遍
        ref_img_1_latents = ref_img_1_latents.repeat(1,2,1,1,1)
        ref_img_2_latents = ref_img_2_latents.repeat(1,2,1,1,1)
        
        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        if self.do_classifier_free_guidance:
            pose_latents = torch.cat([torch.zeros_like(pose_latents), pose_latents], dim=0)
            dummpy_reference_input = torch.cat([torch.zeros_like(dummpy_reference_input), dummpy_reference_input]) if self.do_classifier_free_guidance else dummpy_reference_input
            ref_img_1_latents = torch.cat([torch.zeros_like(ref_img_1_latents), ref_img_1_latents], dim=0) if self.do_classifier_free_guidance else ref_img_1_latents
            ref_img_2_latents = torch.cat([torch.zeros_like(ref_img_2_latents), ref_img_2_latents], dim=0) if self.do_classifier_free_guidance else ref_img_2_latents
            ref_pose_1_latents = torch.cat([torch.zeros_like(ref_pose_1_latents), ref_pose_1_latents], dim=0) if self.do_classifier_free_guidance else ref_pose_1_latents
            ref_pose_2_latents = torch.cat([torch.zeros_like(ref_pose_2_latents), ref_pose_2_latents], dim=0) if self.do_classifier_free_guidance else ref_pose_2_latents

        # 旋转c和f
        latent_model_input = rearrange(latent_model_input, "b c f h w -> b f c h w")
        dummpy_reference_input = rearrange(dummpy_reference_input, "b c f h w -> b f c h w")
        ref_img_1_latents = rearrange(ref_img_1_latents, "b c f h w -> b f c h w")
        ref_img_2_latents = rearrange(ref_img_2_latents, "b c f h w -> b f c h w")
        pose_latents = rearrange(pose_latents, "b c f h w -> b f c h w")
        ref_pose_1_latents = rearrange(ref_pose_1_latents, "b c f h w -> b f c h w")
        ref_pose_2_latents = rearrange(ref_pose_2_latents, "b c f h w -> b f c h w")
        # 通道拼接噪声和空序列
        latent_model_input = torch.cat([latent_model_input, dummpy_reference_input], dim=2)
        # 多帧参考在时序上拼接
        latent_model_input = torch.cat([ref_img_1_latents, ref_img_2_latents, latent_model_input], dim=1)
        pose_latents = torch.cat([ref_pose_1_latents, ref_pose_2_latents, pose_latents], dim=1)
        # 拼接pose
        latent_model_input = torch.cat([latent_model_input, pose_latents], dim=2)
        latent_model_input = rearrange(latent_model_input, "b f c h w -> b c f h w")

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latent_model_input = latent_model_input * self.scheduler.init_noise_sigma
        return latent_model_input, latents

    def prepare_mask_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
        condition_latents=None,
        ref_img=None,
        dummy_ref_img=None,
        pose_latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # vae编码
        ref_img_1_latents = ref_img[0]
        ref_img_2_latents = ref_img[-1]
        if dummy_ref_img is None:
            dummpy_reference_input = torch.zeros_like(latents)
        else:
            padding_shape = latents[:, :, 1:, :, :].shape
            padding_latents = torch.zeros(padding_shape, device=device, dtype=dtype)
            dummpy_reference_input = torch.cat([dummy_ref_img, padding_latents], dim=2)
        # 在通道维度上复制一遍
        ref_img_1_latents = ref_img_1_latents.repeat(1,2,1,1,1)
        ref_img_2_latents = ref_img_2_latents.repeat(1,2,1,1,1)
        
        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        # 没有pose的condition是不参与cfg的
        condition_latents = torch.cat([condition_latents] * 2) if self.do_classifier_free_guidance else condition_latents
        if self.do_classifier_free_guidance:
            # condition_latents = torch.cat([torch.zeros_like(condition_latents), condition_latents], dim=0)
            dummpy_reference_input = torch.cat([dummpy_reference_input, dummpy_reference_input], dim=0) if self.do_classifier_free_guidance else dummpy_reference_input
            ref_img_1_latents = torch.cat([torch.zeros_like(ref_img_1_latents), ref_img_1_latents], dim=0) if self.do_classifier_free_guidance else ref_img_1_latents
            ref_img_2_latents = torch.cat([torch.zeros_like(ref_img_2_latents), ref_img_2_latents], dim=0) if self.do_classifier_free_guidance else ref_img_2_latents
            if pose_latents is not None:
                pose_latents = torch.cat([torch.zeros_like(pose_latents), pose_latents], dim=0) if self.do_classifier_free_guidance else pose_latents
                # pose_latents = torch.cat([pose_latents, pose_latents], dim=0) if self.do_classifier_free_guidance else pose_latents
        if pose_latents is not None:
            condition_latents = torch.cat([condition_latents, pose_latents], dim=2)
        # 旋转c和f
        latent_model_input = rearrange(latent_model_input, "b c f h w -> b f c h w")
        dummpy_reference_input = rearrange(dummpy_reference_input, "b c f h w -> b f c h w")
        ref_img_1_latents = rearrange(ref_img_1_latents, "b c f h w -> b f c h w")
        ref_img_2_latents = rearrange(ref_img_2_latents, "b c f h w -> b f c h w")
        # 通道拼接噪声和空序列
        latent_model_input = torch.cat([latent_model_input, dummpy_reference_input], dim=2)
        # 多帧参考在时序上拼接
        latent_model_input = torch.cat([ref_img_1_latents, ref_img_2_latents, latent_model_input], dim=1)
        # 拼接condition
        latent_model_input = torch.cat([latent_model_input, condition_latents], dim=2)
        latent_model_input = rearrange(latent_model_input, "b f c h w -> b c f h w")

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latent_model_input = latent_model_input * self.scheduler.init_noise_sigma
        return latent_model_input, latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    def prepare_mask_image(
        self, mask, width, height, batch_size, device, dtype,
    ):
        if not isinstance(mask, torch.Tensor):
            if isinstance(mask, PIL.Image.Image):
                mask = [mask]

            if isinstance(mask[0], PIL.Image.Image):
                masks = []

                for mask_ in mask:
                    mask_ = mask_.convert("L")
                    mask_ = mask_.resize((width, height), resample=PIL_INTERPOLATION["nearest"])
                    mask_ = np.array(mask_)
                    mask_ = (mask_ > 128).astype(np.float32)
                    mask_ = mask_[None, None, :]
                    masks.append(mask_)

                mask = np.concatenate(masks, axis=0)
                mask = torch.from_numpy(mask)
            elif isinstance(mask[0], torch.Tensor):
                mask = torch.cat(mask, dim=0)

        mask_batch_size = mask.shape[0]

        if mask_batch_size == 1:
            repeat_by = batch_size
        else:
            repeat_by = 1

        mask = mask.repeat_interleave(repeat_by, dim=0)

        mask = mask.to(device=device, dtype=dtype)

        return mask
    
    def prepare_pose_image(
        self, image, width, height, batch_size, device, dtype,
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = (image - 0.5) / 0.5
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            repeat_by = 1

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        return image

    @torch.no_grad()
    def __mask_call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        video_length: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        overlap_frames = 0,
        seed = None,
        ref_img = None,
        mask_image = None,
        agnostic_image = None,
        first_inpainted_frame = None,
        pose_image = None,
        device = None
    ):

        reference_cnt = 2

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [
                random.randint(0, 1_000_000)
                for _ in range(batch_size * num_videos_per_prompt)
            ]
        elif isinstance(seed, int):
            seeds = [
                seed + i
                for _ in range(batch_size)
                for i in range(num_videos_per_prompt)
            ]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [
                    int(seed[i]) + j
                    for i in range(batch_size)
                    for j in range(num_videos_per_prompt)
                ]
            elif len(seed) == batch_size * num_videos_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt(batch_size) or "
                    f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got {seed}."
                )
        else:
            raise ValueError(
                f"Seed must be an integer, a list of integers, or None, got {seed}."
            )
        generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = device

        prompt_embeds = self.prompt_embeds.unsqueeze(0).to(device)
        negative_prompt_embeds = self.n_prompt_embeds.unsqueeze(0).to(device)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds])
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, negative_prompt_embeds])

        # Encode image embedding
        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        stride = video_length - overlap_frames
        num_tasks = math.ceil((len(mask_image) - video_length) / stride) + 1
        full_frames = []
        dummpy_ref_image = first_inpainted_frame
        padding_length = 0
        print(f'mask image len = {len(mask_image)}')
        ref_img = [self.get_latent([img], width, height) for img in ref_img]
        if dummpy_ref_image is not None:
            dummpy_ref_image = self.get_latent([dummpy_ref_image], width, height)
        mask_image = self.prepare_mask_image(
            mask_image,
            width,
            height,
            batch_size,
            device,
            self.vae.dtype
        )

        '''agnostic_image = self.prepare_pose_image(
            agnostic_image,
            width,
            height,
            batch_size,
            device,
            self.vae.dtype
        )'''
        '''mask_image = [self.get_latent([img], width, height) for img in mask_image]
        agnostic_image = [self.get_latent([img], width, height) for img in agnostic_image]'''
        # ref_pose = [self.get_latent([img], width, height) for img in ref_pose]
        latent_length = (video_length - 1) // 4 + 1

        target_dtype = torch.bfloat16
        autocast_enabled = (
            target_dtype != torch.float32
        ) and True
        vae_dtype = torch.float16
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and True

        for task_id in range(num_tasks):
            # process self regression
            print("====> task_id: ", task_id)
            mask_image_i = mask_image[task_id*stride:(task_id+1)*stride+overlap_frames,...]
            agnostic_image_i = agnostic_image[task_id*stride:(task_id+1)*stride+overlap_frames]
            if pose_image is not None:
                pose_image_i = pose_image[task_id*stride:(task_id+1)*stride+overlap_frames]
            print(f'start={task_id*stride},end={(task_id+1)*stride+overlap_frames-1},raw_pose_len={len(mask_image_i)}')
            if len(mask_image_i) == overlap_frames:
                continue
            if mask_image_i.shape[0] < video_length:
                padding_length = video_length - mask_image_i.shape[0]
                mask_image_i = torch.cat([mask_image_i, mask_image_i[-1:].repeat(padding_length,1,1,1)], dim=0)
                agnostic_image_i = torch.cat([agnostic_image_i, agnostic_image_i[-1:].repeat(padding_length,1,1,1)], dim=0)
                if pose_image is not None:
                    pose_image_i = torch.cat([pose_image_i, pose_image_i[-1:].repeat(padding_length,1,1,1)], dim=0)
            mask_image_i = rearrange(mask_image_i, "b c h w -> 1 c b h w")
            # agnostic_image_i = rearrange(agnostic_image_i, "b c h w -> 1 c b h w")

            if mask_image_i.shape[2] > 1:
                mask_latents = torch.cat([mask_image_i[:,:,:1,:,:].repeat(1,4,1,1,1), 
                        mask_image_i[:,:,1:,:,:].view(*mask_image_i.shape[:2], -1, 4, *mask_image_i.shape[3:]).view(mask_image_i.size(0), -1, mask_image_i.size(2)//4, *mask_image_i.shape[3:])]
                        , dim=2)
            else:
                # fallback 或 raise 错误，或只用前一部分
                mask_latents = mask_image_i[:,:,:1,:,:].repeat(1,4,1,1,1)
            mask_latents = torch.nn.functional.interpolate(mask_latents.view(-1, 1, mask_latents.shape[-2], mask_latents.shape[-1]), size=(height//self.vae_scale_factor_spatial, width//self.vae_scale_factor_spatial), mode='nearest')
            mask_latents = mask_latents.view(batch_size, 4, (video_length-1)//4+1, height//self.vae_scale_factor_spatial, width//self.vae_scale_factor_spatial).bool()
            mask_latents = rearrange(mask_latents, "b c f h w -> b f c h w")
            mask_latents = torch.cat([
                    mask_latents.new_full((batch_size, reference_cnt, 4, height//self.vae_scale_factor_spatial, width//self.vae_scale_factor_spatial), 
                                        fill_value=0.),
                    mask_latents], dim=1)

            # agnostic_latents = self.vae.encode(agnostic_image_i.to(dtype=self.vae.dtype)).latent_dist.sample()
            '''agnostic_latents = torch.cat([self.vae.encode(tmp).latent_dist.sample()
                                        for tmp in agnostic_image_i.to(self.vae.dtype).chunk(agnostic_image_i.shape[0], dim=0)],  dim=0)
            agnostic_latents = agnostic_latents * self.vae.config.scaling_factor'''
            agnostic_latents = self.get_latent(agnostic_image_i, width, height)
            agnostic_latents = rearrange(agnostic_latents, "b c f h w -> b f c h w")
            agnostic_latents = torch.cat([
                    agnostic_latents.new_full((batch_size, reference_cnt, 16, height//self.vae_scale_factor_spatial, width//self.vae_scale_factor_spatial), 
                                          fill_value=0.),
                    agnostic_latents], dim=1)

            condition_latents = torch.cat([mask_latents, agnostic_latents], dim=2)

            if pose_image is not None:
                pose_latents = self.get_latent(pose_image_i, width, height)
                pose_latents = rearrange(pose_latents, "b c f h w -> b f c h w")
                pose_latents = torch.cat([
                        pose_latents.new_full((batch_size, reference_cnt, 16, height//self.vae_scale_factor_spatial, width//self.vae_scale_factor_spatial),
                                            fill_value=0.),
                        pose_latents], dim=1)
                # condition_latents = torch.cat([condition_latents, pose_latents], dim=2)
            else:
                pose_latents = None

            prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_prompt_embeds = torch.zeros_like(negative_prompt_embeds)

            cfg_prompt_embeds = prompt_embeds

            # 5. Prepare latent variables
            latents = None
            num_channels_latents = self.transformer.config.out_channels
            latent_model_input, latents = self.prepare_mask_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                latent_length,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
                condition_latents,
                ref_img,
                dummpy_ref_image,
                pose_latents
            )

            # 5. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)
            # if is_progress_bar:
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue
                    
                    self._current_timestep = t
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    t_expand = t.repeat(latent_model_input.shape[0])

                    with torch.autocast(
                        device_type="cuda", dtype=target_dtype, enabled=autocast_enabled
                    ):
                        # predict the noise residual
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=t_expand,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_image=None,
                            attention_kwargs=None,
                            return_dict=False,
                        )[0]
                        # print("====> single step time: ", time.time() - t0)

                    # perform guidance
                    noise_pred = noise_pred[:, :, 2:, :, :]
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop(
                            "negative_prompt_embeds", negative_prompt_embeds
                        )

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                    ):
                        if progress_bar is not None:
                            progress_bar.update()

                    latent_model_input, latents = self.prepare_mask_latents(
                        batch_size * num_videos_per_prompt,
                        num_channels_latents,
                        height,
                        width,
                        latent_length,
                        prompt_embeds.dtype,
                        device,
                        generator,
                        latents,
                        condition_latents,
                        ref_img,
                        dummpy_ref_image,
                        pose_latents
                    )

            self._current_timestep = None

            if not output_type == "latent":
                latents = latents.to(self.vae.dtype)
                latents_mean = self.latents_mean.to(latents.device, latents.dtype)
                latents_std = self.latents_std.to(latents.device, latents.dtype)
                latents = latents / latents_std + latents_mean
                with torch.autocast(
                    device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
                ):
                    image = self.vae.decode(latents, return_dict=False)[0]

                dummpy_ref_image = image[:, :, -overlap_frames:, :, :]
                dummpy_ref_image = retrieve_latents(self.vae.encode(dummpy_ref_image), sample_mode="argmax")
                dummpy_ref_image = (dummpy_ref_image - latents_mean) * latents_std
                debug = False
                if debug:
                    print("----> [debug] saving first img.")
                    from PIL import Image
                    first_img = image[:, :, :1, :, :].squeeze(0).squeeze(1)
                    first_img = rearrange(first_img, "c h w -> h w c")
                    first_img = (first_img / 2 + 0.5).clamp(0, 1)
                    first_img = first_img.cpu().float().numpy()
                    first_img = (first_img * 255).astype(np.uint8)
                    first_img = Image.fromarray(first_img)
                    first_img.save("first_reg_img_debug.png")

                    img = image[:, :, -overlap_frames:, :, :].squeeze(0).squeeze(1)
                    img = rearrange(img, "c h w -> h w c")
                    img = (img / 2 + 0.5).clamp(0, 1)
                    img = img.cpu().float().numpy()
                    img = (img * 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save("last_reg_img_debug.png")
            else:
                image = latents

            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
            image = image.cpu().float()

            image = rearrange(image, "b c f h w -> b f c h w")
            if task_id == 0:
                full_frames.extend(image[0][:-padding_length] if padding_length != 0 else image[0])
            else:
                full_frames.extend(image[0][overlap_frames:-padding_length] if padding_length != 0 else image[0][overlap_frames:])

        full_frames = torch.stack(full_frames, dim=1).unsqueeze(0)
        full_frames = self.video_processor.postprocess_video(full_frames, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (full_frames,)

        return WanPipelineOutput(frames=full_frames)

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        video_length: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        pose_imgs = None,
        ref_img = None,
        ref_pose = None,
        overlap_frames = 1,
        seed = None,
        self_reg_inference = True,
    ):
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [
                random.randint(0, 1_000_000)
                for _ in range(batch_size * num_videos_per_prompt)
            ]
        elif isinstance(seed, int):
            seeds = [
                seed + i
                for _ in range(batch_size)
                for i in range(num_videos_per_prompt)
            ]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [
                    int(seed[i]) + j
                    for i in range(batch_size)
                    for j in range(num_videos_per_prompt)
                ]
            elif len(seed) == batch_size * num_videos_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt(batch_size) or "
                    f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got {seed}."
                )
        else:
            raise ValueError(
                f"Seed must be an integer, a list of integers, or None, got {seed}."
            )
        generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        prompt_embeds = self.prompt_embeds.unsqueeze(0).to(device)
        negative_prompt_embeds = self.n_prompt_embeds.unsqueeze(0).to(device)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds])
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, negative_prompt_embeds])

        # Encode image embedding
        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        stride = video_length - overlap_frames
        num_tasks = math.ceil((len(pose_imgs) - video_length) / stride) + 1
        full_frames = []
        dummpy_ref_image = None
        padding_length = 0
        ref_img = [self.get_latent([img], width, height) for img in ref_img]
        ref_pose = [self.get_latent([img], width, height) for img in ref_pose]
        latent_length = (video_length - 1) // 4 + 1

        target_dtype = torch.bfloat16
        autocast_enabled = (
            target_dtype != torch.float32
        ) and True
        vae_dtype = torch.float16
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and True

        for task_id in range(num_tasks):
            # process self regression
            print("====> task_id: ", task_id)
            pose_image_i = pose_imgs[task_id*stride:(task_id+1)*stride+overlap_frames]
            print(f'start={task_id*stride},end={(task_id+1)*stride+overlap_frames-1},raw_pose_len={len(pose_image_i)}')
            if len(pose_image_i) == overlap_frames:
                continue
            if len(pose_image_i) < video_length:
                padding_length = video_length - len(pose_image_i)
                last_image = pose_image_i[-1]
                for _ in range(padding_length):
                    pose_image_i.append(last_image.copy())

            # Get video latent
            pose_image_i = self.get_latent(pose_image_i, width, height)

            # 5. Prepare latent variables
            latents = None
            num_channels_latents = self.transformer.config.out_channels
            latent_model_input, latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                latent_length,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
                pose_image_i,
                ref_img,
                ref_pose,
                dummpy_ref_image,
            )

            # 5. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)
            # if is_progress_bar:
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue
                    
                    self._current_timestep = t
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    t_expand = t.repeat(latent_model_input.shape[0])

                    with torch.autocast(
                        device_type="cuda", dtype=target_dtype, enabled=autocast_enabled
                    ):
                        # predict the noise residual
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=t_expand,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_image=None,
                            attention_kwargs=None,
                            return_dict=False,
                        )[0]
                        # print("====> single step time: ", time.time() - t0)

                    # perform guidance
                    noise_pred = noise_pred[:, :, 2:, :, :]
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop(
                            "negative_prompt_embeds", negative_prompt_embeds
                        )

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                    ):
                        if progress_bar is not None:
                            progress_bar.update()

                    latent_model_input, latents = self.prepare_latents(
                        batch_size * num_videos_per_prompt,
                        num_channels_latents,
                        height,
                        width,
                        latent_length,
                        prompt_embeds.dtype,
                        device,
                        generator,
                        latents,
                        pose_image_i,
                        ref_img,
                        ref_pose,
                        dummpy_ref_image,
                    )

            self._current_timestep = None

            if not output_type == "latent":
                latents = latents.to(self.vae.dtype)
                latents_mean = self.latents_mean.to(latents.device, latents.dtype)
                latents_std = self.latents_std.to(latents.device, latents.dtype)
                latents = latents / latents_std + latents_mean
                with torch.autocast(
                    device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
                ):
                    image = self.vae.decode(latents, return_dict=False)[0]

                dummpy_ref_image = image[:, :, -overlap_frames:, :, :]
                dummpy_ref_image = retrieve_latents(self.vae.encode(dummpy_ref_image), sample_mode="argmax")
                dummpy_ref_image = (dummpy_ref_image - latents_mean) * latents_std
                debug = False
                if debug:
                    print("----> [debug] saving first img.")
                    from PIL import Image
                    first_img = image[:, :, :1, :, :].squeeze(0).squeeze(1)
                    first_img = rearrange(first_img, "c h w -> h w c")
                    first_img = (first_img / 2 + 0.5).clamp(0, 1)
                    first_img = first_img.cpu().float().numpy()
                    first_img = (first_img * 255).astype(np.uint8)
                    first_img = Image.fromarray(first_img)
                    first_img.save("first_reg_img_debug.png")

                    img = image[:, :, -overlap_frames:, :, :].squeeze(0).squeeze(1)
                    img = rearrange(img, "c h w -> h w c")
                    img = (img / 2 + 0.5).clamp(0, 1)
                    img = img.cpu().float().numpy()
                    img = (img * 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save("last_reg_img_debug.png")
            else:
                image = latents

            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
            image = image.cpu().float()

            image = rearrange(image, "b c f h w -> b f c h w")
            if task_id == 0:
                full_frames.extend(image[0][:-padding_length] if padding_length != 0 else image[0])
            else:
                full_frames.extend(image[0][overlap_frames:-padding_length] if padding_length != 0 else image[0][overlap_frames:])

        full_frames = torch.stack(full_frames, dim=1).unsqueeze(0)
        full_frames = self.video_processor.postprocess_video(full_frames, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (full_frames,)

        return WanPipelineOutput(frames=full_frames)


from diffusers import DDIMScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler
import numpy as np
import torch
from tqdm.auto import tqdm

class SDEdit(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               z_latent=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               **kwargs
               ):
        noise_scheduler = DDIMScheduler(
        #noise_scheduler = DPMSolverMultistepScheduler(
          num_train_timesteps=1000,
          beta_start=0.00085,
          beta_end=0.0120,
          beta_schedule="linear",
          clip_sample=False,
          set_alpha_to_one=False,
        )
        #S = 50
        strength = 0.75
        b = z_latent.shape[0]

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        generator = torch.Generator(device).manual_seed(2023)
        noise_scheduler.set_timesteps(S, device=device)
        init_timestep = min(int(S * strength), S)
        t_start = max(S - init_timestep, 0)
        timesteps = noise_scheduler.timesteps[t_start:].to(device)

        #init_latents = (z_latent * 0.18215).to(device)
        init_latents = z_latent.to(device)

        noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=init_latents.dtype)
        init_latents = noise_scheduler.add_noise(init_latents, noise, timesteps[:1])
        latents = init_latents
        #latents = noise

        for t in tqdm(timesteps):
            latents_model_input = noise_scheduler.scale_model_input(latents, t)
            ts = torch.full((b,), t, device=device, dtype=torch.long)
            model_t = self.model.apply_model(latents_model_input, ts, conditioning)
            model_uncond = self.model.apply_model(latents_model_input, ts, unconditional_conditioning)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

            latents = noise_scheduler.step(model_output, t, latents).prev_sample

        #latents = 1 / 0.18215 * latents

        return latents





"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


class PLMSSampler(object):
    def __init__(self, model, schedule='linear', device=None, **kwargs):
        super().__init__()
        
        # Store the LDM model LatentDiffusion
        self.model = model
        
        # How many steps it took to train the model
        self.ddpm_num_timesteps = model.num_timesteps
        
        # Shedule, which is linear in out case 
        self.schedule = schedule
        if not device:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device(self.device):
                attr = attr.to(torch.float32).to(torch.device(self.device))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        if ddim_eta != 0:
            raise ValueError('ddim_eta must be 0 for PLMS')
        
        # ddim_discretize -> 'uniform'
        # ddim_num_steps -> t=5, diffusion steps for inference. 50 or 100 gets better results
        # self.ddpm_num_timesteps -> Steps during training, so t=1000
        # Maps [0:5] -> [1, 201, 401, 601, 801] -> How we uniformly sample our 1000 into 5
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        
        # We grab the original alphas from our LatentDiffusion model
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        # We now start creating these non-learnable constants needed for sampling
        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    # This function samples from the U-Net decoder and takes encoded latent information as input
    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
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
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        # We are using conditioning based on the prompts
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
        
        # This makes sure we have set the appropiate constants needed to sample
        # For example format inference steps to be in the 1 to 1000 range and;
        # set all these non-learnable constants needed for the sampling process
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for PLMS sampling is {size}')

        # Peform the actual sampling
        samples, intermediates = self.plms_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def plms_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            # Here we generate the initial noise tensor of 64 by 64, because thats latent space size
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        # Grab the timesteps we created with the make_schedule method
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        # probably don't need this at all
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        
        # Reverse the time steps, as with generating (decoding) we start from the end to the original noise
        time_range = list(reversed(range(0,timesteps))) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running PLMS Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
        old_eps = []

        # So for t=5 we start with 801 then 601 etc
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            
            # Grab the timestep like [601]
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            # Code is a bit messsy but this is another part of the sampling
            outs = self.p_sample_plms(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      old_eps=old_eps, t_next=ts_next)
            img, pred_x0, e_t = outs
            
            # Where we keep all the epsilons (predicted noise), for our 4 branches
            old_eps.append(e_t)
            
            # Then FIFO when more than 4, so try to keep at least memory of 4 previously preditected noises
            if len(old_eps) >= 4:
                old_eps.pop(0)
                
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_plms(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, old_eps=None, t_next=None):
        b, *_, device = *x.shape, x.device

        # The forward pass through the U-NET
        def get_model_output(x, t):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, t, c)
            else:
                # Because we do classifier free guidance we have to repeat our
                # input representation, which is currently just noise
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                
                # Now we have to concatunate the unconditinal (so no prompt) and the conditioned output
                c_in = torch.cat([unconditional_conditioning, c])
                
                # We do the forward pass through the U-Net model and get the output representations
                # Right before we concatunate both representations so we can forward pass it once 
                # And then we use the .chunk method to get back 2 decoded representations
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                
                # Combine the noise from when we conditioned the u-net with unconditional condition
                # And also pass the information we got from when we did use the conditioning
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            return e_t

        # We grab these constants
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        # Equation 8 from the PMLS Paper: Pseudo Numerical Methods for Diffusion Models on Manifolds 
        # Very hard to onderstand what's going, so please read paper if you want to understand
        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            # Create some non learnable expressions
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            # First part of Equation 8
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
                
            # direction pointing to x_t
            # Second part of Equation 8
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            
            # Noise = 0, because it's multuiplied by sigma_t = 0
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)

            # First part times sqrt(a_prev) + Second part plus the noise
            # Noise = 0, also stated in paper that they only care about cases when sigma_t = 0 -> noise=0
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        # Basically a forward pass through the U-NET -> the noise prediction
        e_t = get_model_output(x, t)
        
        # Calculations from the PMLS Paper: Pseudo Numerical Methods for Diffusion Models on Manifolds 
        # Bottomline: Diffusion is related to differential equations, and so we can use 
        # familiar solutions from differential equations to attach the diffusion problem
        
        # Equation 22
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            # This will calculate whatever phi is 
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            
            # We again feed that into the decoder to predict noise 
            e_t_next = get_model_output(x_prev, t_next)
            
            # Add this up with the episolon (noise preds) from prev step and divide by 2
            e_t_prime = (e_t + e_t_next) / 2
            
        # Equation 23
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
            
        # Equation no idea?
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
            
        # Equation 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            # Calculate the epsilon prime from the e_t (predicted noise)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24
        
        # Equation 8
        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        # x_prev -> Output image, which is next step in the decoder diffusion process, as we move backwards
        # pred_x0 -> What is predicted 
        # e_t -> The noise prediction that was made
        return x_prev, pred_x0, e_t

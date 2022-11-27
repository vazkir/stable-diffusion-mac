import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?


# Possible loss to use for the generator part of the AE, so for enocder + decoder + logvar
class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        
        # Loss coefficients, so determine on which component of the loss we are looking
        # at will have a different to be used
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        
        # The perceptual loss, idea came from NST
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        # This discriminator will be used for the adverserial loss component of LPIPS
        # Just a normal discriminator, which also downsamples because its patchGAN based from pix2pix
        # Main difference: Instead of having a scalar telling you it's real or fake we now have:
        # - For a Patch will maybe have 32x32 scalars, and all of them will just tell you if 
        # - That specific path is real or fake, gives you more information for our model to train
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        
        # Here we have some hinge loss
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    # From VQGAN paper: Makes sure we are weighting the GAN and the Reconstruction loss appropiatly
    # - Gradients of the reconstruction loss with respect to the last layer of the decoder weights divided;
    # - By the gradients of the GAN loss with respect to the last layer of the decoder weights
    # -> If the gradients of are super big for the reconstruction and smaller gan loss -> lambda big number
    # Which puts a bigger weight on the GAN and vice versa. This way 1 loss if not overwhelming the other
    # when it comes to the contribution of the gradients, so we weight out which one to focus on
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        # Reconstruction loss -> We simply subtract the reconstucted output from the original input image
        # Do note we are working on the contiguous (row oriented) tensors which have shape (1, 3, 256, 256)
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            
            # This time we take the inputs and reconstructions again, however we now:
            # Compare them in the latent space of the pre-trained vgg network so not image space
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            
            # Calculate reconstruction loss as a weighted sum of:
            # - rec_loss: reconstuction loss from the image space
            # - p_loss * weights: The perceptual loss * some weights associated with it
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        
        # Some rescaling and summation
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        
        # This is our final reconstruction loss
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        
        # posterion -> gaussian from the latent space of the encoder of the AE
        # Just compute the KL-Divergence, which is the regularize component
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0] # Rescale again

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                
                # We pass the reconstructed images through the discriminator and we get logits_fake
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            
            # We are working with the patch based discriminator for which we want the negative mean (regular GAN)
            # of the patch that is tried to be reconstructed of the input image. - because this is what is reconstruced
            # Input (1,1,30,30), so 30x30 is the patch size and a mean
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    # The VQGAN loss that weights out the perceptual and GAN loss
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            # This is going to be 0 for the first like 50k iterations, because
            # They first want to train the AE and ignore the GAN loss so that the can form
            # some representations, and than slowly start adding the GAN loss
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            
            # Means that this part "d_weight * disc_factor * g_loss" is toggled of for a big portion 
            # Meaning we only use the kl regularizer loss and the reconstruction loss
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            # Logging all the loss
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        # The discriminator loss calculation
        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                
                # Pass the real and reconstructed input to have discriminator determin of real of or not and
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            # Again for like the first 50k steps, this part of the loss will not be enabled
            # And later it's going to gradually kick in
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            
            # This is basically a hinge loss between the real and fake predictins of discriminator
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


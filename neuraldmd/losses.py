"""
Reusable loss pieces for both Fourier- and Pixel-domain experiments.
"""

import jax, jax.numpy as jnp


# ── generic helpers ────────────────────────────────────────────────────
def sparsity_loss(*arrays):          # ℓ1 sparsity on any number of tensors
    return sum(jnp.mean(jnp.abs(a)) for a in arrays)

def tv_loss(img):                    # Total-variation (pixel experiment)
    return jnp.sum(jnp.abs(img[:-1]-img[1:])) + \
           jnp.sum(jnp.abs(img[:,:-1]-img[:,1:]))

def negative_penalty(x):             # penalise negative intensities
    return jnp.sum(jax.nn.relu(-x)**2)


# ── Pixel experiment composite loss ────────────────────────────────────
def pixel_loss_fn(model, xy, frames, times,
                  *, beta_tv=1e-2, beta_neg=1e-2, beta_sparse=1e-2,
                  frame_max=1.0, frame_min=0.0):
    """
    Returns (total_loss, aux_dict)  for Weather / Pixel tutorial.
    """
    W0, W_half, _ = jax.vmap(model.spatial_forward)(xy)
    al, th = model.temporal_omega()
    Omega       = al + 1j*th
    b0, bh = model.temporal_b()

    Lambda = jnp.exp(Omega[:,None] * times[None,:] * 160.0)       # (r_half,T)
    recon = 2 * jnp.real(jnp.einsum('br,rt,r->bt', W_half, Lambda, bh)) \
            + W0[:,0:1]*b0[0]
    recon = recon * (frame_max - frame_min) + frame_min

    rec_loss = jnp.mean((frames - recon)**2)
    loss = rec_loss
    # loss += beta_tv     * tv_loss(recon.T[0])            # TV on first frame
    loss += beta_neg    * negative_penalty(recon)
    sparsity_value = sparsity_loss(W0, W_half, b0, bh)
    loss += beta_sparse * sparsity_value
    return loss, {"recon": rec_loss, "sparse": sparsity_value}


# ── Fourier experiment composite loss ──────────────────────────────────
def fourier_loss_fn(model, xy, frame_batch, target_vis_batch, num_vis_batch, A_batch, sigma_batch, time_indices, mask_batch, alpha, frame_max, frame_min, *, beta = 1e-3, gamma = 1e-3, b_weight = 1.0):
    W0, W_half, _ = jax.vmap(model.spatial_forward)(xy)
    # Compute global temporal parameters.
    alphas, thetas = model.temporal_omega()  # each shape: (r_half,)
    Omega = alphas + 1j * thetas              # shape: (r_half,)
    b0, b_half = model.temporal_b()           # b0: (1,), b_half: (r_half,)
    
    # Compute temporal evolution.
    lambda_exp = jnp.exp(Omega[:, None] * time_indices[None, :] * 200.0)  # shape: (r_half, T)
    
    # Reconstruction: use spatial mode W_half and combine with temporal evolution and b_half.
    intensities = (2 * jnp.real(jnp.einsum('pr, rt, r -> pt', W_half, lambda_exp, b_half)) + W0[:, 0:1] * b0[0])
    # intensities = intensities 
    # intensities = intensities * (frame_max - frame_min) + frame_min
    intensities = intensities * jax.nn.relu(model.scale)
    reconstruction_loss = jnp.sum(jnp.abs(frame_batch - intensities.T))
    
    vis_pred = jnp.einsum('tvp, pt -> tv', A_batch, intensities.astype(complex))
    vis_diff = jnp.abs(vis_pred - target_vis_batch)
    chi_squared = jnp.sum((vis_diff * mask_batch / sigma_batch)**2) / jnp.sum(num_vis_batch)
    

    total_loss = chi_squared

    negative_penalty = jnp.sum(jax.nn.relu(-intensities)**2)
    
    total_loss += beta * negative_penalty

    sparse_penalty = sparsity_loss(W0, W_half)
    gamma = 1e-3
    total_loss += gamma * sparse_penalty

    b_sparse_penalty = sparsity_loss(b0, b_half)
    b_weight = 1.0
    total_loss += b_weight * b_sparse_penalty
    
    return total_loss, (reconstruction_loss, chi_squared)
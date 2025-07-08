import jax
import jax.numpy as jnp
import optax
import equinox as eqx

@eqx.filter_jit
def train_step_pixels(model, opt_state, xy, target_values, time_indices, optimizer, beta, frame_max, frame_min, loss_fn):
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, xy, target_values, time_indices, beta_tv=beta, frame_max=frame_max, frame_min=frame_min
    )
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    reconstruction_loss, sparsity_loss = aux["recon"], aux["sparse"]
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, reconstruction_loss, sparsity_loss, grads

def train_step_visibilities(model, opt_state, xy, frame_batch, target_vis_batch, num_vis_batch, A_batch, sigma_batch, time_indices, mask_batch, 
               optimizer, alpha, beta, frame_max, frame_min, loss_fn):
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, xy, frame_batch, target_vis_batch, num_vis_batch, A_batch, sigma_batch, time_indices, mask_batch, alpha, frame_max, frame_min
    )
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    reconstruction_loss, orthogonality_loss = aux
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, reconstruction_loss, orthogonality_loss, grads
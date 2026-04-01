#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:05:11 2025

@author: kondo
"""

#import numpy as np
import scipy as sp

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, lax
from functools import partial

import equinox.internal as eqxi


def save_config_to_file(filename='config.txt', **kwargs):
    """
    Save parameters to a text file using keyword arguments.
    Each parameter is written on a separate line in the format: param_name = param_value
    
    Args:
        filename (str): Name of the file to save parameters to (default: 'config.txt')
        **kwargs: Variable-length keyword arguments representing parameters to save
    
    Returns:
        bool: True if parameters were successfully saved, False otherwise
    """
    try:
        with open(filename, 'w') as f:
            for param_name, param_value in kwargs.items():
                f.write(f"{param_name} = {param_value}\n")
        return True
    except Exception as e:
        print(f"Error saving parameters to file: {e}")
        return False


def calc_prion_levels_jax(vol, atlas_label, atlas_keys_jnp):
    # atlas_keys_jnp: atlas_dict.keys() converted to jnp.int32
    def process_key(key):
        roi_mask = atlas_label == key
        mask_sum = jnp.sum(roi_mask)
        # Only compute mean for non-empty regions
        safe_mask_sum = jnp.maximum(mask_sum, 1)  # Avoid division by zero
        # Use sum divided by count of True values for correct mean
        return jnp.sum(jnp.where(roi_mask, vol, 0.0)) / safe_mask_sum
    
    prion_levels = jax.vmap(process_key)(atlas_keys_jnp)
    return prion_levels


def pearson_loss(x, y):# negated Pearson corr coeff
    mean_x = jnp.mean(x)
    mean_y = jnp.mean(y)
    
    var_x = jnp.sum((x - mean_x) ** 2)
    var_y = jnp.sum((y - mean_y) ** 2)
    
    cov_xy = jnp.sum((x - mean_x) * (y - mean_y))
    
    correlation = cov_xy / jnp.sqrt(var_x * var_y)
    
    return -correlation


# eqxi.scan has a checkpointing scheme to reduce memory usage during packpropagation
def eqxi_scan_update_func(prion, phi, args, dt, dx, length, update_func):
    def scan_fn(carry, _):
        carry = update_func(carry, phi, args, dt, dx)
        return carry, None
    prion_new, _ = eqxi.scan(scan_fn, prion, xs=None, length=length, kind='checkpointed')
    return prion_new# length-times-updated prion


def simulate_pet_and_compute_pearson_loss(alpha, prion_init, pet_atlas, phi, args, dt, dx, length, update_func, atlas_label, atlas_keys):
    pet_pred = eqxi_scan_update_func(prion_init, phi, [*args, alpha], dt, dx, length, update_func)
    pet_atlas_pred = calc_prion_levels_jax(pet_pred, atlas_label, atlas_keys)
    return pearson_loss(pet_atlas_pred, pet_atlas)


@partial(jit, static_argnums=(7, 8))
def alpha_loss_and_grad(alpha, prion_init, pet_atlas, phi, args, dt, dx, length, update_func, atlas_label, atlas_keys):
    loss, grad = jax.value_and_grad(simulate_pet_and_compute_pearson_loss, argnums=0, has_aux=False)(
        alpha, prion_init, pet_atlas, phi, args, dt, dx, length, update_func, atlas_label, atlas_keys
    )
    return loss, grad


# Batched loss function and grad function for multiple target data. 'pet_atlases' is a jnp.array of shape (Batch size, Number of ROIs).
batched_pearson_loss = jax.vmap(pearson_loss, in_axes=(None, 0))# batched along the first axis of the second arg of pearson_loss
def batched_simulate_pet_and_compute_pearson_loss(alpha, prion_init, pet_atlases, phi, args, dt, dx, length, update_func, atlas_label, atlas_keys):
    pet_pred = eqxi_scan_update_func(prion_init, phi, [*args, alpha], dt, dx, length, update_func)
    pet_atlas_pred = calc_prion_levels_jax(pet_pred, atlas_label, atlas_keys)
    return jnp.mean(batched_pearson_loss(pet_atlas_pred, pet_atlases))


@partial(jit, static_argnums=(7, 8))
def batched_alpha_loss_and_grad(alpha, prion_init, pet_atlases, phi, args, dt, dx, length, update_func, atlas_label, atlas_keys):
    loss, grad = jax.value_and_grad(batched_simulate_pet_and_compute_pearson_loss, argnums=0, has_aux=False)(
        alpha, prion_init, pet_atlases, phi, args, dt, dx, length, update_func, atlas_label, atlas_keys
    )
    return loss, grad


#
# Code below is for record-keeping and not the part of the main functionality
#
def pet_loss_squared_voxelwise_error(prion0, prion1, phi, alpha, args, dt, dx, length, update_func):
    prion1_pred = eqxi_scan_update_func(prion0, phi, [*args, alpha], dt, dx, length, update_func)
    return jnp.sum((prion1_pred - prion1)**2)#np.mean((prion1_pred - prion1)**2)


def pet_loss_root_squared_voxelwise_error(prion0, prion1, phi, alpha, args, dt, dx, length, update_func):
    prion1_pred = eqxi_scan_update_func(prion0, phi, [*args, alpha], dt, dx, length, update_func)
    err = jnp.sum((prion1_pred - prion1)**2)
    return jnp.sqrt(err)


def pet_loss_squared_voxelwise_error_only_roi(prion0, prion1, phi, alpha, args, dt, dx, length, update_func, mask_roi):
    prion1_pred = eqxi_scan_update_func(prion0, phi, [*args, alpha], dt, dx, length, update_func)
    err = (prion1_pred - prion1)**2
    return jnp.where(mask_roi, err, 0).sum()

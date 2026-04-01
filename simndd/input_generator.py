#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:43:58 2024

@author: kondo
"""

import os
from functools import partial

import numpy as np
import scipy as sp
from scipy import signal, ndimage
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, lax


def calc_MD_scale_factor(lmbdas, mask_brain, mask_gray, MD_target = 100):
    #
    # lmbdas: Field of eigen values of DTI created as
    # lmbdas, eigvecs = np.linalg.eigh(DTI.reshape(-1, 3, 3))
    #
    MD = np.mean(lmbdas, axis=1).reshape(mask_brain.shape)
    med_MD = np.median(MD[(mask_brain - mask_gray) > 0.5])# median MD in white matter
    print('Median MD = ', med_MD)
    return MD_target/med_MD


def convert_diffusion_tensor_field(D, mask_brain, mask_gray, mode='anisotropic'):
    # Preprocessing of DTI as in Weickenmeier, PRL, 2018
    # D: DTI, [x, y, z, 0-8]
    # mask_brain: Binary mask of whole brain
    # mask_gray: Binary mask of all gray matter
    #
    d_gray = 10.0# mm2/year
    d_ortho, d_parall = 1., 100.# mm2/year
    # Isotropic uniform diffusion in gray matter
    D2 = np.zeros_like(D)#D.copy()
    D2[(mask_gray > 0.5), :] = np.eye(3).ravel() * d_gray
    # Anisotropic diffusion along with the first eigen vector
    mask_white = (mask_brain - mask_gray) > 0.5
    # Note that eigvecs[idx, :, -1] is the eigen vector of the largest eigen value at idx (flattened position)
    if mode == 'anisotropic':
        print('mode: ', mode)
        lmbdas, eigvecs = np.linalg.eigh(D.reshape(-1, 3, 3))
        anisotropy_tensor = np.einsum('ij,ik->ijk', eigvecs[:, :, -1], eigvecs[:, :, -1]).reshape(*(mask_brain.shape), -1)
        # print(anisotropy_tensor.shape)
        D2[mask_white, :] = np.eye(3).ravel() * d_ortho + (d_parall - d_ortho)*anisotropy_tensor[mask_white, :]
    elif mode == 'isotropic':
        print('mode: ', mode)
        D2[mask_white, :] = np.eye(3).ravel() * (d_ortho + (d_parall - d_ortho)/3.0) # the same mean diffusivity to the anisotropic case
    else:
        print('Invalid mode flag! the returned DTI is broken')
    # Gray-matter-like isotropic diffusion on brain boundary to suppress numerical instability on phase boundary
    s = ndimage.generate_binary_structure(3, 3)
    boundary_voxels = binary_dilation(mask_brain < 0.5, s) & (mask_brain > 0.5)
    D2[boundary_voxels, :] = np.eye(3).ravel() * d_gray
    # Gray-matter-like virtual diffusion tensor outside of brain for phase field method
    D2[(mask_brain < 0.5), :] = np.eye(3).ravel() * d_gray
    return D2


def laplacian_finite_difference(phi):
    dxx = jnp.roll(phi, -1, axis=0) + jnp.roll(phi, +1, axis=0)
    dyy = jnp.roll(phi, -1, axis=1) + jnp.roll(phi, +1, axis=1)
    dzz = jnp.roll(phi, -1, axis=2) + jnp.roll(phi, +1, axis=2)
    return dxx + dyy + dzz - 6.0*phi


def update_phase_allen_cahn(phi, phi_mask, dt, dx, eta=1.0, eps=0.5):
    diffusion_term = dt/dx/dx * laplacian_finite_difference(phi) * eta
    energy_term = dt * 16. * phi * (1.0 - phi) * (2*phi - 1.0) * eta/eps/eps
    phi = jnp.where(phi_mask, phi + diffusion_term + energy_term, phi)
    return phi


@partial(jit, static_argnums=(3, ))
def scan_update_phase_allen_cahn(phi, dt, dx, length):
    def scan_fn(carry, _):
        carry = update_phase_allen_cahn(carry, phi < 0.9, dt, dx)
        return carry, None
    phase_new, _ = lax.scan(scan_fn, phi, xs=None, length=length)
    return phase_new# length-times-updated phase


def smooth_phase_boundary(mask_brain, use_eroded_mask_for_phase=False):
    kernel = ndimage.generate_binary_structure(3, 1).astype(np.float64) / 7.
    phi = ndimage.convolve(mask_brain.astype(np.float64), kernel, mode='nearest')
    phi[mask_brain > 0.5] = 1.0
    phi = scan_update_phase_allen_cahn(phi, 0.0001, 1.0, 10000)
    phi = np.array(phi)
    phi[phi[:] < 0.0001] = 0
    return phi


def smooth_initial_field(seed_roi_mask, sigma=1.0):
    prion_init = seed_roi_mask.copy().astype(np.float32)
    return ndimage.gaussian_filter(prion_init, sigma=sigma)


def convert_symmetric_diffusion_tensor_to_full(D_sym):
    """
    Convert symmetric tensor field with 6 components to full 3x3 tensor field.
    
    Parameters:
    D_sym: numpy array with shape (N, M, L, 6) where the 6 components are
           D_xx, D_xy, D_xz, D_yy, D_yz, D_zz
    
    Returns:
    D_full: numpy array with shape (N, M, L, 9) where the 9 components are
            D_xx, D_xy, D_xz, D_yx, D_yy, D_yz, D_zx, D_zy, D_zz
    """
    N, M, L, _ = D_sym.shape
    D_full = np.zeros((N, M, L, 9), dtype=D_sym.dtype)
    
    # Fill in components from symmetric tensor
    # D_xx, D_xy, D_xz
    D_full[:, :, :, 0] = D_sym[:, :, :, 0]  # D_xx
    D_full[:, :, :, 1] = D_sym[:, :, :, 1]  # D_xy
    D_full[:, :, :, 2] = D_sym[:, :, :, 2]  # D_xz
    
    # D_yx, D_yy, D_yz
    D_full[:, :, :, 3] = D_sym[:, :, :, 1]  # D_yx = D_xy (symmetric)
    D_full[:, :, :, 4] = D_sym[:, :, :, 3]  # D_yy
    D_full[:, :, :, 5] = D_sym[:, :, :, 4]  # D_yz
    
    # D_zx, D_zy, D_zz
    D_full[:, :, :, 6] = D_sym[:, :, :, 2]  # D_zx = D_xz (symmetric)
    D_full[:, :, :, 7] = D_sym[:, :, :, 4]  # D_zy = D_yz (symmetric)
    D_full[:, :, :, 8] = D_sym[:, :, :, 5]  # D_zz
    
    return D_full
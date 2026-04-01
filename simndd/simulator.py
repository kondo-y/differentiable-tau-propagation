#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:10:37 2024

@author: kondo
"""
from functools import partial

import numpy as np
import scipy as sp

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, lax

def jax_diffusion_on_phase_forward_backward(phi, C, D):
    #
    # forward-backward discretization for xx, yy, and zz derivatives,
    # while central-central discretization for the other ones, such as xy, xz
    #
    def diff_fwd(X, axis):
        return jnp.roll(X, -1, axis=axis) - X
    def diff_cent(X, axis):
        return 0.5 * (jnp.roll(X, -1, axis=axis) - jnp.roll(X, 1, axis=axis))
    def diff_bwd(X, axis):
        return X - jnp.roll(X, 1, axis=axis)
    axes_for_xyz = [0, 1, 2]# row, col, depth as x, y, z, NOT y, x, z.
    Cx_fwd, Cy_fwd, Cz_fwd = [diff_fwd(C, ax) for ax in axes_for_xyz]
    Cx_cent, Cy_cent, Cz_cent = [diff_cent(C, ax) for ax in axes_for_xyz]

    def avg_fwd(X, axis):
        return 0.5 * jnp.roll(X, -1, axis=axis) + 0.5 * X
    def avg_cent(X, axis):
        return 0.5 * jnp.roll(X, -1, axis=axis) + 0.5 * jnp.roll(X, 1, axis=axis)
    indices = [0, 1, 2, 4, 5, 8]# upper-triangular elements
    phi_Dxx, phi_Dxy, phi_Dxz, phi_Dyy, phi_Dyz, phi_Dzz = [jnp.multiply(phi, D[:, :, :, idx]) for idx in indices]
    phi_Dxx_Cx_fwd = jnp.multiply(Cx_fwd, avg_fwd(phi_Dxx, axis=0))
    phi_Dxy_Cy_cent = jnp.multiply(Cy_cent, avg_cent(phi_Dxy, axis=1))
    phi_Dxz_Cz_cent = jnp.multiply(Cz_cent, avg_cent(phi_Dxz, axis=2))
    phi_Dxy_Cx_cent = jnp.multiply(Cx_cent, avg_cent(phi_Dxy, axis=0))
    phi_Dyy_Cy_fwd = jnp.multiply(Cy_fwd, avg_fwd(phi_Dyy, axis=1))
    phi_Dyz_Cz_cent = jnp.multiply(Cz_cent, avg_cent(phi_Dyz, axis=2))
    phi_Dxz_Cx_cent = jnp.multiply(Cx_cent, avg_cent(phi_Dxz, axis=0))
    phi_Dyz_Cy_cent = jnp.multiply(Cy_cent, avg_cent(phi_Dyz, axis=1))
    phi_Dzz_Cz_fwd = jnp.multiply(Cz_fwd, avg_fwd(phi_Dzz, axis=2))

    # divergence of x-components
    diffusion = diff_bwd(phi_Dxx_Cx_fwd, axis=0)
    diffusion += diff_cent(phi_Dxy_Cy_cent, axis=0)
    diffusion += diff_cent(phi_Dxz_Cz_cent, axis=0)
    # divergence of y-components
    diffusion += diff_cent(phi_Dxy_Cx_cent, axis=1)
    diffusion += diff_bwd(phi_Dyy_Cy_fwd, axis=1)
    diffusion += diff_cent(phi_Dyz_Cz_cent, axis=1)
    # divergence of z-components
    diffusion += diff_cent(phi_Dxz_Cx_cent, axis=2)
    diffusion += diff_cent(phi_Dyz_Cy_cent, axis=2)
    diffusion += diff_bwd(phi_Dzz_Cz_fwd, axis=2)

    return diffusion

def jax_diffusion_on_phase_flux_based(phi, C, D):
    #
    # Computing numerical flux for both diagonal diffusion and cross diffusion
    # Interpolation of diffusion coeff at virtual interface gird may use
    # harmonic mean (See Ch 4, Pataskar, Numerical heat transfer and fluid flow, 1980)
    # 2024.11.1
    # Harmonic mean for cross diffusion coeffs is problematic,
    # because cross diffusion coeffs can be negative.
    # A possiblity is to use harmonic mean only for diagonal terms.
    # But for now, arithmetic mean used for all terms.
    #
    def diff_fwd(X, axis):
        return jnp.roll(X, -1, axis=axis) - X
    def diff_cent(X, axis):
        return 0.5 * (jnp.roll(X, -1, axis=axis) - jnp.roll(X, 1, axis=axis))
    def diff_bwd(X, axis):
        return X - jnp.roll(X, 1, axis=axis)
    axes_for_xyz = [0, 1, 2]# row, col, depth as x, y, z, NOT y, x, z.
    Cx_fwd, Cy_fwd, Cz_fwd = [diff_fwd(C, ax) for ax in axes_for_xyz]
    Cx_cent, Cy_cent, Cz_cent = [diff_cent(C, ax) for ax in axes_for_xyz]

    def avg_fwd(X, axis):
        return 0.5 * jnp.roll(X, -1, axis=axis) + 0.5 * X
    def avg_cent(X, axis):
        return 0.5 * jnp.roll(X, -1, axis=axis) + 0.5 * jnp.roll(X, 1, axis=axis)
    def harmonic_avg_fwd(X, axis):
        Xf = jnp.roll(X, -1, axis=axis)
        return 2.0 * jnp.divide(jnp.multiply(Xf, X), Xf + X + 1e-6)
        
    indices = [0, 1, 2, 4, 5, 8]# upper-triangular elements
    phi_Dxx, phi_Dxy, phi_Dxz, phi_Dyy, phi_Dyz, phi_Dzz = [jnp.multiply(phi, D[:, :, :, idx]) for idx in indices]
    # J_x, x-component of numerical flux vector at x = (i + 1/2) * dx
    phi_Dxx_Cx_fwd = jnp.multiply(Cx_fwd, avg_fwd(phi_Dxx, axis=0))
    phi_Dxy_Cy_cent = jnp.multiply(avg_fwd(Cy_cent, axis=0), avg_fwd(phi_Dxy, axis=0))
    phi_Dxz_Cz_cent = jnp.multiply(avg_fwd(Cz_cent, axis=0), avg_fwd(phi_Dxz, axis=0))
    # J_y, y-component of numerical flux vector at y = (j + 1/2) * dx
    phi_Dxy_Cx_cent = jnp.multiply(avg_fwd(Cx_cent, axis=1), avg_fwd(phi_Dxy, axis=1))
    phi_Dyy_Cy_fwd = jnp.multiply(Cy_fwd, avg_fwd(phi_Dyy, axis=1))
    phi_Dyz_Cz_cent = jnp.multiply(avg_fwd(Cz_cent, axis=1), avg_fwd(phi_Dyz, axis=1))
    # J_z, z-component
    phi_Dxz_Cx_cent = jnp.multiply(avg_fwd(Cx_cent, axis=2), avg_fwd(phi_Dxz, axis=2))
    phi_Dyz_Cy_cent = jnp.multiply(avg_fwd(Cy_cent, axis=2), avg_fwd(phi_Dyz, axis=2))
    phi_Dzz_Cz_fwd = jnp.multiply(Cz_fwd, avg_fwd(phi_Dzz, axis=2))

    # divergence of numerical flux vector J = (J_x, J_y, J_z)
    diffusion = diff_bwd(phi_Dxx_Cx_fwd + phi_Dxy_Cy_cent + phi_Dxz_Cz_cent, axis=0)# d/dx J_x
    diffusion += diff_bwd(phi_Dxy_Cx_cent + phi_Dyy_Cy_fwd + phi_Dyz_Cz_cent, axis=1)# d/dy J_y
    diffusion += diff_bwd(phi_Dxz_Cx_cent + phi_Dyz_Cy_cent + phi_Dzz_Cz_fwd, axis=2)# d/dz J_z

    return diffusion


def jax_chemotaxis_upwind(C, phi_D, M):
    def diff_fwd(X, axis):
        return jnp.roll(X, -1, axis=axis) - X
    def diff_cent(X, axis):
        return 0.5 * (jnp.roll(X, -1, axis=axis) - jnp.roll(X, 1, axis=axis))
    def diff_bwd(X, axis):
        return X - jnp.roll(X, 1, axis=axis)
    axes_for_xyz = [0, 1, 2]# row, col, depth as x, y, z, NOT y, x, z.
    Cx_fwd, Cy_fwd, Cz_fwd = [diff_fwd(C, ax) for ax in axes_for_xyz]
    Cx_cent, Cy_cent, Cz_cent = [diff_cent(C, ax) for ax in axes_for_xyz]

    def avg_fwd(X, axis):
        return 0.5 * jnp.roll(X, -1, axis=axis) + 0.5 * X
    def avg_cent(X, axis):
        return 0.5 * jnp.roll(X, -1, axis=axis) + 0.5 * jnp.roll(X, 1, axis=axis)
    def harmonic_avg_fwd(X, axis):
        Xf = jnp.roll(X, -1, axis=axis)
        return 2.0 * jnp.divide(jnp.multiply(Xf, X), Xf + X + 1e-6)
        
    indices = [0, 1, 2, 4, 5, 8]# upper-triangular elements
    phi_Dxx, phi_Dxy, phi_Dxz, phi_Dyy, phi_Dyz, phi_Dzz = [phi_D[:, :, :, idx] for idx in indices]
    # J_x, x-component of numerical flux vector at x = (i + 1/2) * dx
    phi_Dxx_Cx_fwd = jnp.multiply(Cx_fwd, avg_fwd(phi_Dxx, axis=0))
    phi_Dxy_Cy_cent = jnp.multiply(avg_fwd(Cy_cent, axis=0), avg_fwd(phi_Dxy, axis=0))
    phi_Dxz_Cz_cent = jnp.multiply(avg_fwd(Cz_cent, axis=0), avg_fwd(phi_Dxz, axis=0))
    # J_y, y-component of numerical flux vector at y = (j + 1/2) * dx
    phi_Dxy_Cx_cent = jnp.multiply(avg_fwd(Cx_cent, axis=1), avg_fwd(phi_Dxy, axis=1))
    phi_Dyy_Cy_fwd = jnp.multiply(Cy_fwd, avg_fwd(phi_Dyy, axis=1))
    phi_Dyz_Cz_cent = jnp.multiply(avg_fwd(Cz_cent, axis=1), avg_fwd(phi_Dyz, axis=1))
    # J_z, z-component
    phi_Dxz_Cx_cent = jnp.multiply(avg_fwd(Cx_cent, axis=2), avg_fwd(phi_Dxz, axis=2))
    phi_Dyz_Cy_cent = jnp.multiply(avg_fwd(Cy_cent, axis=2), avg_fwd(phi_Dyz, axis=2))
    phi_Dzz_Cz_fwd = jnp.multiply(Cz_fwd, avg_fwd(phi_Dzz, axis=2))

    v_x = phi_Dxx_Cx_fwd + phi_Dxy_Cy_cent + phi_Dxz_Cz_cent
    v_y = phi_Dxy_Cx_cent + phi_Dyy_Cy_fwd + phi_Dyz_Cz_cent
    v_z = phi_Dxz_Cx_cent + phi_Dyz_Cy_cent + phi_Dzz_Cz_fwd

    J_x = v_x * jnp.where(v_x > 0, M, jnp.roll(M, -1, 0))
    J_y = v_y * jnp.where(v_y > 0, M, jnp.roll(M, -1, 1))
    J_z = v_z * jnp.where(v_z > 0, M, jnp.roll(M, -1, 2))

    return diff_bwd(J_x, axis=0) + diff_bwd(J_y, axis=1) + diff_bwd(J_z, axis=2)


# Sample function for "update_func" argument for scan_update_func
def update_fisher(prion, phi, args, dt, dx):
    D, alpha = args
    prion_phi = jnp.multiply(prion, phi)
    diff_term = dt/dx/dx * jax_diffusion_on_phase_flux_based(phi, prion, D)
    prion_phi_diff = diff_term + dt * alpha * phi * prion * (1.0 - prion)
    prion_phi += prion_phi_diff
    prion_phi = jnp.clip(prion_phi, 0, 1.0)
    phi_for_divide = jnp.where(phi > 0.0001, phi, 0.0001)
    prion = jnp.where(phi > 0.0001, jnp.divide(prion_phi, phi_for_divide), prion_phi)
    return prion


# Another sample function for "update_func" argument for scan_update_func where binary volume specified for propagation seed
def update_fisher_from_seed(prion, phi, args, dt, dx):
    roi_seed, D, alpha = args
    # roi_seed should be bool type
    prion = jnp.where(roi_seed, 1.0, prion)
    prion_phi = jnp.multiply(prion, phi)
    diff_term = dt/dx/dx * jax_diffusion_on_phase_flux_based(phi, prion, D)
    prion_phi_diff = diff_term + dt * alpha * phi * prion * (1.0 - prion)
    prion_phi += prion_phi_diff
    prion_phi = jnp.clip(prion_phi, 0, 1.0)
    phi_for_divide = jnp.where(phi > 0.0001, phi, 0.0001)
    prion = jnp.where(phi > 0.0001, jnp.divide(prion_phi, phi_for_divide), prion_phi)
    return prion


@partial(jit, static_argnums=(5, 6))
def scan_update_func(prion, phi, args, dt, dx, length, update_func):
    def scan_fn(carry, _):
        carry = update_func(carry, phi, args, dt, dx)
        return carry, None
    prion_new, _ = lax.scan(scan_fn, prion, xs=None, length=length)
    return prion_new# length-times-updated prion


def simulate_scan(prion, phi, args, dt, dx, n_time, n_step, update_func):
    prion = jnp.asarray(prion)
    phi = jnp.asarray(phi)
    history = [prion.copy()]
    for t in range(n_time):
        print(t*n_step, ' th step...')
        prion = scan_update_func(prion, phi, args, dt, dx, n_step, update_func)
        history.append(prion.copy())
    return history
#
# Below is for simulating multi-component models
#
def update_estavoyer_from_seed(states, phi, args, dt, dx):
    #
    # Eqation 1 from Maxima Estavoyer et al., preprint
    #
    oligo, plaque, monomer, microglia, IL = states
    roi_seed, D, r1, r2, d, gamma0, tau1, tau2, tau3, taup, taus, C, alpha1, alpha2, lambdam, sigma, cmtx_coeff = args
    oligo_phi = jnp.multiply(oligo, phi)
    monomer_phi = jnp.multiply(monomer, phi)
    microglia_phi = jnp.multiply(microglia, phi)
    IL_phi = jnp.multiply(IL, phi)

    oligo_diffusion = dt/dx/dx * jax_diffusion_on_phase_flux_based(phi, oligo, D)
    monomer_diffusion = dt/dx/dx * jax_diffusion_on_phase_flux_based(phi, monomer, D)
    microglia_diffusion = dt/dx/dx * jax_diffusion_on_phase_flux_based(phi, microglia, D)
    IL_diffusion = dt/dx/dx * jax_diffusion_on_phase_flux_based(phi, IL, D)
    gamma = gamma0# + jnp.divide(gamma1 * microglia, 1.0 + gamma2 * microglia)
    stress = jnp.divide(taus * IL, 1.0 + C * oligo * oligo)
    microglia_growth = jnp.divide(alpha1*oligo, 1.0+alpha2*oligo) * (1.0 - microglia) * microglia
    microglia_growth += - sigma * microglia + lambdam
    microglia_chemotaxis = cmtx_coeff * dt/dx/dx * jax_diffusion_on_phase_flux_based(microglia_phi, oligo, D)
    
    oligomerization = r1 * monomer * monomer
    dt_oligo = oligo_diffusion + dt * phi * (oligomerization - gamma * oligo)# - tau0 * oligo
    dt_plaque = dt * (gamma * oligo - taup * plaque)
    dt_monomer = monomer_diffusion + dt * phi* (stress - d*monomer - r2*oligo*monomer - oligomerization) + dt * phi * roi_seed * d
    dt_microglia = microglia_diffusion - microglia_chemotaxis + dt * phi * microglia_growth
    dt_IL = IL_diffusion + dt * phi * (jnp.divide(tau1*oligo, 1.0+tau2*oligo) * microglia - tau3 * IL)

    oligo_phi += dt_oligo
    plaque += dt_plaque
    monomer_phi += dt_monomer
    microglia_phi += dt_microglia
    IL_phi += dt_IL

    oligo_phi = jnp.clip(oligo_phi, 0, 100.0)
    monomer_phi = jnp.clip(monomer_phi, 0, 100.0)

    phi_for_divide = jnp.where(phi > 0.0001, phi, 0.0001)
    oligo = jnp.where(phi > 0.0001, jnp.divide(oligo_phi, phi_for_divide), oligo_phi)
    monomer = jnp.where(phi > 0.0001, jnp.divide(monomer_phi, phi_for_divide), monomer_phi)
    microglia = jnp.where(phi > 0.0001, jnp.divide(microglia_phi, phi_for_divide), microglia_phi)
    IL = jnp.where(phi > 0.0001, jnp.divide(IL_phi, phi_for_divide), IL_phi)
    states = [oligo, plaque, monomer, microglia, IL]
    return states


def update_estavoyer_upwind_from_seed(states, phi, args, dt, dx):
    #
    # Eqation 1 from Maxima Estavoyer et al., preprint
    #
    oligo, plaque, monomer, microglia, IL = states
    roi_seed, D, r1, r2, d, gamma0, tau1, tau2, tau3, taup, taus, C, alpha1, alpha2, lambdam, sigma, cmtx_coeff = args
    oligo_phi = jnp.multiply(oligo, phi)
    monomer_phi = jnp.multiply(monomer, phi)
    microglia_phi = jnp.multiply(microglia, phi)
    IL_phi = jnp.multiply(IL, phi)

    oligo_diffusion = dt/dx/dx * jax_diffusion_on_phase_flux_based(phi, oligo, D)
    monomer_diffusion = dt/dx/dx * jax_diffusion_on_phase_flux_based(phi, monomer, D)
    microglia_diffusion = dt/dx/dx * jax_diffusion_on_phase_flux_based(phi, microglia, D)
    IL_diffusion = dt/dx/dx * jax_diffusion_on_phase_flux_based(phi, IL, D)
    gamma = gamma0# + jnp.divide(gamma1 * microglia, 1.0 + gamma2 * microglia)
    stress = jnp.divide(taus * IL, 1.0 + C * oligo * oligo)
    microglia_growth = jnp.divide(alpha1*oligo, 1.0+alpha2*oligo) * (1.0 - microglia) * microglia
    microglia_growth += - sigma * microglia + lambdam
    microglia_chemotaxis = cmtx_coeff * dt/dx/dx * jax_chemotaxis_upwind(oligo, jnp.multiply(D, phi[:, :, :, None]), microglia)# The only difference of this upwind version is this line
    
    oligomerization = r1 * monomer * monomer
    dt_oligo = oligo_diffusion + dt * phi * (oligomerization - gamma * oligo)# - tau0 * oligo
    dt_plaque = dt * (gamma * oligo - taup * plaque)
    dt_monomer = monomer_diffusion + dt * phi* (stress - d*monomer - r2*oligo*monomer - oligomerization) + dt * phi * roi_seed * d
    dt_microglia = microglia_diffusion - microglia_chemotaxis + dt * phi * microglia_growth
    dt_IL = IL_diffusion + dt * phi * (jnp.divide(tau1*oligo, 1.0+tau2*oligo) * microglia - tau3 * IL)

    oligo_phi += dt_oligo
    plaque += dt_plaque
    monomer_phi += dt_monomer
    microglia_phi += dt_microglia
    IL_phi += dt_IL

    oligo_phi = jnp.clip(oligo_phi, 0, 100.0)
    monomer_phi = jnp.clip(monomer_phi, 0, 100.0)

    phi_for_divide = jnp.where(phi > 0.0001, phi, 0.0001)
    oligo = jnp.where(phi > 0.0001, jnp.divide(oligo_phi, phi_for_divide), oligo_phi)
    monomer = jnp.where(phi > 0.0001, jnp.divide(monomer_phi, phi_for_divide), monomer_phi)
    microglia = jnp.where(phi > 0.0001, jnp.divide(microglia_phi, phi_for_divide), microglia_phi)
    IL = jnp.where(phi > 0.0001, jnp.divide(IL_phi, phi_for_divide), IL_phi)
    states = [oligo, plaque, monomer, microglia, IL]
    return states


def simulate_scan_multistates(states, phi, args, dt, dx, n_time, n_step, update_func):
    states = [jnp.asarray(state) for state in states]
    phi = jnp.asarray(phi)
    history = [jnp.stack(states.copy())]
    for t in range(n_time):
        print(t*n_step, ' th step...')
        states = scan_update_func(states, phi, args, dt, dx, n_step, update_func)
        history.append(jnp.stack(states.copy()))
    return history
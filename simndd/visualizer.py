#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:00:23 2024

@author: kondo
"""
#import os
import numpy as np
#import scipy as sp
#from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def volume_slicer(vol, plane, idx):
    if plane == 'coronal':
        ans = vol[:, idx, :]
    elif plane == 'axial':
        ans = vol[:, :, idx]
    elif plane == 'sagittal':
        ans = vol[idx, :, :]
    return ans


def plot_longitudinal_pet(mask_brain, pet0, pet1, plane='coronal', idx=107):
    mask_brain_slice = volume_slicer(mask_brain, plane, idx)
    pet0_slice = volume_slicer(pet0, plane, idx)
    pet1_slice = volume_slicer(pet1, plane, idx)
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
    ax[0].imshow(mask_brain_slice, interpolation='nearest')
    ax[1].imshow(pet0_slice)
    ax[2].imshow(pet1_slice)
    im = ax[3].imshow(pet1_slice - pet0_slice, cmap='bwr')

    for i in range(1, 4):
        ax[i].axis('off')
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.tight_layout()


def plot_dti(mask_brain, mask_gray, dti, plane='coronal', idx=107, k_rot90=None):
    mask_brain_slice = volume_slicer(mask_brain, plane, idx)
    mask_gray_slice = volume_slicer(mask_gray, plane, idx)
    dti_slice = volume_slicer(dti[:, :, :, 0], plane, idx)
    if k_rot90 is not None:
        mask_brain_slice = np.rot90(mask_brain_slice, k=k_rot90)
        mask_gray_slice = np.rot90(mask_gray_slice, k=k_rot90)
        dti_slice = np.rot90(dti_slice, k=k_rot90)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    ax[0].imshow(mask_brain_slice, interpolation='nearest')
    ax[1].imshow(mask_gray_slice, interpolation='nearest')
    ax[2].imshow(dti_slice)
    ax[2].set_title('D_xx')
    for i in range(0, 3):
        ax[i].axis('off')
    plt.tight_layout()


def plot_phase_field(mask_brain, phase, plane='coronal', idx=107):
    mask_brain_slice = volume_slicer(mask_brain, plane, idx)
    eroded_mask_brain_slice = volume_slicer(binary_erosion(mask_brain), plane, idx)
    phase_slice = volume_slicer(phase, plane, idx)
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
    ax[0].imshow(mask_brain_slice, cmap='gray', interpolation='nearest')
    ax[1].imshow(eroded_mask_brain_slice, cmap='gray', interpolation='nearest')
    ax[2].imshow(phase_slice, cmap='gray', interpolation='none', vmin=0, vmax=1.0)
    ax[3].imshow(phase_slice > 0.0001, cmap='gray', interpolation='none')
    ax_titles = ['Brain mask', 'Eroded brain mask', 'Phase', 'Phase > thresh']
    for i in range(0, 4):
        ax[i].axis('off')
        ax[i].set_title(ax_titles[i])
    plt.tight_layout()


def plot_atlas_roi(mask_brain, mask_gray, mask_roi, plane='coronal', indices=[100, 110], k_rot90=None):
    fig, ax = plt.subplots(nrows=len(indices), ncols=3, figsize=(3*3, 3*len(indices)))
    for i, idx in enumerate(indices):
        mask_brain_slice = volume_slicer(mask_brain, plane, idx)
        mask_gray_slice = volume_slicer(mask_gray, plane, idx)
        mask_roi_slice = volume_slicer(mask_roi, plane, idx)
        masks = [mask_brain_slice, mask_gray_slice, mask_roi_slice]
        if k_rot90 is not None:
            masks = [np.rot90(mask, k=k_rot90) for mask in masks]
        for j, mask in enumerate(masks):
            ax[i, j].imshow(mask, cmap='gray', interpolation='nearest')
        ax_titles = ['Brain mask at ' + str(idx), 'Gray matter', 'ROI']
        for j in range(3):
            ax[i, j].axis('off')
            ax[i, j].set_title(ax_titles[j])


def plot_simulation(history, t_intv, plane='coronal', idx=107, k_rot90=None, mask=None, fig_suptitle='Prion'):
    ts = np.arange(history.shape[0])
    fig, ax = plt.subplots(nrows=1, ncols=len(ts), figsize=(3.5*len(ts), 5))
    for i, t in enumerate(ts):
        im = volume_slicer(history[t, :, :, :], plane, idx)
        if mask is not None:
            im = np.multiply(im, volume_slicer(mask, plane, idx))
        if k_rot90 is not None:
            im = np.rot90(im, k=k_rot90)
        ax[i].imshow(im, vmin=0., vmax=1., interpolation='nearest')
        ax[i].axis('off')
        ax[i].set_title('t = ' + str(t*t_intv))
    plt.suptitle(fig_suptitle)
    plt.subplots_adjust(wspace=0.05)


def plot_total_prion(history, phase):
    ts = np.arange(history.shape[0])
    sums_prion_phi = np.zeros_like(ts)
    for i, t in enumerate(ts):
        M = np.multiply(history[t, :, :, :], phase)
        print('max(prion * phi)', t, np.max(M))
        sums_prion_phi[i] = np.sum(M)
    fig, ax = plt.subplots(figsize=(4, 3))
    plt.plot(ts, sums_prion_phi)
    plt.xlabel('t')
    plt.ylabel('sum(prion * phi)')
import numpy as np
from numba import njit

@njit
def pix_order_list(uniq):
    orders = np.floor(np.log2(uniq/4)/2).astype(np.int64)
    nest_pix_per_order = []
    for order in range(max(orders)+1):
        nside = 2**order
        nest_pix_at_this_order = []
        for pix in uniq[orders == order]:
            nest_pix = pix - 4*nside*nside
            nest_pix_at_this_order.append(nest_pix)
        nest_pix_per_order.append(nest_pix_at_this_order)
    return nest_pix_per_order
@njit
def uniq2nside(uniq):
    order = np.array(np.log2(uniq/4)/2, np.int64)
    return 2**order
@njit
def nest2uniq(nside, ipix):
    return ipix + 4*nside*nside
@njit
def uniq2nest(uniq):
    nside = uniq2nside(uniq)
    npix = uniq - 4 * nside * nside
    return nside, npix
@njit
def nest2range(nside_input, pix, nside_output):
    npix_ratio = nside_output * nside_output // nside_input // nside_input
    return (pix*npix_ratio, (pix+1)*npix_ratio)
@njit
def uniq2range(nside, uniq):
    pix_nside, pix = uniq2nest(uniq)
    return nest2range(pix_nside, pix, nside)

@njit
def split(pixel_to_split, uniq):
    nside = uniq2nside(pixel_to_split)
    next_nside = 2*nside
    substart, substop = uniq2range(next_nside, pixel_to_split)
    subpixels = []
    for subindex in range(substart, substop):
        subpixels.append(nest2uniq(next_nside, subindex))
    subpixels = np.array(subpixels)
    uniq = uniq[uniq != pixel_to_split]
    for i,subpixel in enumerate(subpixels):
        uniq = np.append(uniq, subpixel)
    return uniq

@njit
def combine(pixel_to_combine, uniq):
    nside, ipix = uniq2nest(pixel_to_combine)
    previous_nside = nside//2
    suppixel = uniq2nest(pixel_to_combine)[1]//4
    start, stop = nest2range(previous_nside, suppixel, nside)
    range_of_pixels_to_remove = np.arange(start, stop)
    uniq_of_pixels_to_remove = range_of_pixels_to_remove + 4*nside**2
    mask = (uniq != uniq_of_pixels_to_remove[:,None])
    mask = (np.sum(mask, axis=0) == len(mask))
    uniq = uniq[mask]
    suppixel = nest2uniq(previous_nside, suppixel)
    uniq = np.append(uniq, suppixel)
    return uniq

@njit
def get_available_pixels_for_combining(uniq):
    nest_pix_per_order = pix_order_list(uniq)
    max_nside = uniq2nside(np.max(uniq))
    available_pixels = []
    for order in range(1, int(np.log2(max_nside))+1):
        nside = 2**order
        previous_nside = nside//2
        
        nest_pixels_at_this_order = nest_pix_per_order[order]
        for nest_pixel_at_this_order in nest_pixels_at_this_order:
            previous_nside = nside//2
            suppixel = nest_pixel_at_this_order//4
            start, stop = nest2range(previous_nside, suppixel, nside)
            possible_pixels = np.arange(start, stop)
            mask = (np.array(nest_pix_per_order[order]) == possible_pixels[:,None])
            if np.sum(mask) == 4:
                for possible_pixel in possible_pixels:
                    available_pixels.append(possible_pixel + 4*nside**2)
    return np.unique(np.array(available_pixels))

@njit
def rasterize(uniq_in, P_in):
    RASTERIZE_NSIDE = 8
    sort = np.argsort(uniq_in)
    nsides = 2**np.floor(np.log2(uniq_in[sort]/4)/2)
    num_steps_in_resolution = np.log2(RASTERIZE_NSIDE/nsides)
    num_subpixels = 4**num_steps_in_resolution
    num_subpixels = np.where(num_subpixels > 1, num_subpixels, 1).astype(np.int64)
    U = np.array([0])
    for i,uniq in enumerate(uniq_in[sort]):
        if uniq < 16*RASTERIZE_NSIDE**2:
            U = np.append(U, (uniq*num_subpixels[i] + np.arange(num_subpixels[i])).astype(np.int64))
        else:
            U = np.append(U, np.array([uniq]))
    P = np.repeat(P_in[sort], np.round(num_subpixels).astype(np.int64))
    return U[1:], P, sort, num_subpixels

@njit
def unrasterize(uniq_in, rasterized_map_sorted_by_uniq_and_not_in_log_space):
    rasterized_uniq, _, sort, num_subpixels = rasterize(uniq_in, np.zeros_like(uniq_in))
    mapping = np.repeat(uniq_in[sort], repeats=num_subpixels)
    unrasterized_map = []
    for u in uniq_in:
        unrasterized_map.append(np.mean(rasterized_map_sorted_by_uniq_and_not_in_log_space[np.argsort(np.argsort(rasterized_uniq))][mapping == u]))
    return np.array(unrasterized_map)
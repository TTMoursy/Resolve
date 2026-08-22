import mhealpy as mhp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, jaxopt
import numpy as np, pickle, os
from .pixel_routines import (
    pix_order_list,
    uniq2nside,
    nest2uniq, 
    uniq2nest, 
    nest2range, 
    uniq2range, 
    split, 
    combine, 
    get_available_pixels_for_combining, 
    rasterize, 
    unrasterize
)

def default_prior(uniq_in):
    """
    Uniform priors on meshes.
    """
    return 1

def sample(
    rho,
    C,
    psrs_theta,
    psrs_phi,
    N_iterations = 100_000,
    save_every_N_iterations = 20_000,
    resume = False,
    load_cache = False,
    initial_uniq = None,
    initial_nside = 1,
    max_nside = 64,
    prior = [-7,7],
    outdir = 'resolve_sampler_outdir',
    chain_file_name = 'chain.pkl',
    lnlikes_file_name = 'lnlikes.npy',
    cache_file_name = 'cache.pkl',
    npad = 100,
    prior_on_model = default_prior,
    save_cache = False
):
    """
    Run the sampler.
    
    Args
        rho (jax.Array)
            cross-correlations with shape (Npairs,).
        C (jax.Array)
            pair-covariance matrix with shape (Npairs, Npairs).
        psrs_theta (jax.Array, np.ndarray, list, or tuple)
            theta coordinates for the pulsars; has shape (Npulsars,)
        psrs_phi (jax.Array, np.ndarray, list, or tuple)
            phi coordinates for the pulsars; has shape (Npulsars,)
        
    Optional
        N_iterations (int)
            how many iterations to run the sampler for.
        save_every_N_iterations (int)
            how many iterations to run between saves.
        save_cache (bool)
            whether to save the cache or not.
            it is useful to save the cache if one plans to resume the sampler.
        resume (bool)
            whether to resume from a previous run.
        load_cache (bool)
            whether to load a previously saved cache located in outdir/cache_file_name.
        outdir (str)
            name of directory to save outputs to.
        chain_file_name (str)
            name of chain file (excluding directory name) to save chain to.
        lnlikes_file_name (str)
            name of lnlikes file (excluding directory name) to save lnlikes to.
        cache_file_name (str)
            name of cache file (excluding directory name) to save cache to.
            ignored if save_cache is False.
        initial_uniq (None or np.ndarray)
            an np.ndarray containing UNIQ indices specifying a starting mesh for the sampler.
            if None, the sampler starts with a single-resolution mesh at initial_nside.
        initial_nside (int)
            the nside for the mesh the sampler starts at.
            must be a power of two.
            ignored if initial_uniq is specified.
        max_nside (int)
            maximum allowed nside that the sampler can reach.
            must be a power of two.
        prior (list)
            a two-element list representing a uniform prior [lower, upper] on log_10 angular power densities of pixels.
        prior_on_model (callable)
            callable that accepts an np.ndarray of UNIQ indices and returns a prior weight (float or int) on the model.
        npad (int)
            size of padding for meshes to avoid JIT recompilation due to different size inputs to functions.
    """
    
    prior_range = prior[1] - prior[0]
    max_uniq = 16*(max_nside//2)**2 # this is the max uniq that a split is allowed for; so not really the max uniq allowed

    if resume and load_cache:
        with open(os.path.join(outdir, cache_file_name), 'rb') as fp:
            cache = pickle.load(fp)
        cached_fitted_meshes = cache[0]
        cached_inverse_hessian_cholesky_decompositions = cache[1] 
        cached_constrained_pixels_lists = cache[2]
        cached_unconstrained_pixels_lists = cache[3]
        cached_upper_limits_lists = cache[4]
    else:
        cached_fitted_meshes = dict()
        cached_inverse_hessian_cholesky_decompositions = dict()
        cached_constrained_pixels_lists = dict()
        cached_unconstrained_pixels_lists = dict()
        cached_upper_limits_lists = dict()
    
    psrs_theta = jnp.array(psrs_theta)
    psrs_phi = jnp.array(psrs_phi)
    
    Cinv = jnp.linalg.inv(C)
    Lt = jnp.linalg.cholesky(Cinv).T
    pair_idx = np.array(np.triu_indices(len(psrs_phi),1)).T
    pair_idx_a, pair_idx_b = pair_idx[:,0], pair_idx[:,1]

    @jax.jit
    def signalResponse_fast(ptheta_a, pphi_a, gwtheta_a, gwphi_a, pair_idx_a, pair_idx_b):
        gwphi, pphi = jnp.meshgrid(gwphi_a, pphi_a)
        gwtheta, ptheta = jnp.meshgrid(gwtheta_a, ptheta_a)
        p = jnp.array([jnp.cos(pphi) * jnp.sin(ptheta), jnp.sin(pphi) * jnp.sin(ptheta), jnp.cos(ptheta)])
        Fp, Fc = createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, p)
        R = Fp[pair_idx_a]*Fp[pair_idx_b] + Fc[pair_idx_a]*Fc[pair_idx_b]
        return R
        
    def createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, p):
        Omega = jnp.array([-jnp.sin(gwtheta) * jnp.cos(gwphi), -jnp.sin(gwtheta) * jnp.sin(gwphi), -jnp.cos(gwtheta)])
        mhat = jnp.array([-jnp.sin(gwphi), jnp.cos(gwphi), jnp.zeros(gwphi.shape)])
        nhat = jnp.array([-jnp.cos(gwphi) * jnp.cos(gwtheta), -jnp.cos(gwtheta) * jnp.sin(gwphi), jnp.sin(gwtheta)])
        npixels = Omega.shape[2]
        c = jnp.sqrt(1.5)
        Fp = 0.5 * c * (jnp.sum(nhat * p, axis=0) ** 2 - jnp.sum(mhat * p, axis=0) ** 2) / (1 + jnp.sum(Omega * p, axis=0))
        Fc = c * jnp.sum(mhat * p, axis=0) * jnp.sum(nhat * p, axis=0) / (1 + jnp.sum(Omega * p, axis=0))
        return Fp, Fc
    
    def response(uniq):
        m = mhp.HealpixBase(uniq=uniq)
        gwtheta,gwphi = m.pix2ang(np.arange(m.npix))
        R_abk = signalResponse_fast(psrs_theta, psrs_phi, gwtheta, gwphi, pair_idx_a, pair_idx_b)
        R_abk *= m.pixarea().value / 4 / jnp.pi
        return R_abk
    
    @jax.jit
    def _lnlike(rho, Cinv, R, P, sort, num_subpixels, U):
        p = 10**jnp.repeat(P[sort], repeats=num_subpixels, total_repeat_length=U.shape[0])
        orf = R @ p
        return -0.5*jnp.sum(((rho - orf).T @ Cinv @ (rho - orf)))
    
    def lnlike(uniq_in, P_in):
        U,P,sort,num_subpixels = rasterize(uniq_in, np.array(P_in))
        R = response(U)
        return _lnlike(rho, Cinv, R, P_in, sort, num_subpixels, U)

    def get_pixelmap(uniq_in):
        R = response(uniq_in)
        M = R.T @ Cinv @ R
        X = R.T @ Cinv @ rho
        pixelmap = np.linalg.inv(M + .001*np.diag(np.ones_like(X))) @ X
        orf = R @ pixelmap
        A2 = (orf.T @ Cinv @ rho) / (orf.T @ Cinv @ orf)
        pixelmap *= A2
        pixelmap = np.array(pixelmap)
        pixelmap[pixelmap < 0] = 10**prior[0]
        return jnp.log10(pixelmap)

    @jax.jit
    def residuals(P_input, R, sort, num_subpixels, U, rho, Lt):
        r = rho - R @ 10**jnp.repeat(P_input[sort], repeats=num_subpixels, total_repeat_length=U.shape[0])
        return Lt @ r
    LevenbergMarquardt = jax.jit(jaxopt.LevenbergMarquardt(residuals, materialize_jac=True, jit=True, maxiter=500).run)
    
    @jax.jit
    def objective(P_input, R, sort, num_subpixels, U, rho, Cinv):
        r = rho - R @ 10**jnp.repeat(P_input[sort], repeats=num_subpixels, total_repeat_length=U.shape[0])
        return 0.5*r.T @ Cinv @ r
    LBFGS = jax.jit(jaxopt.LBFGS(objective, jit=True, maxiter=100).run)

    def get_optimalmap(uniq_in):
        _NPAD = np.max((npad, len(uniq_in)))
        key = tuple(sorted(uniq_in))
        sort = np.argsort(uniq_in)
        if key in cached_fitted_meshes.keys():
            opt_P = cached_fitted_meshes[key]
            opt_P = opt_P[np.argsort(sort)]
        else:
            U,P,sort,num_subpixels = rasterize(uniq_in, np.zeros_like(uniq_in))
            R = response(U)
    
            R = jnp.zeros((len(rho), _NPAD+768)).at[:,:len(U)].set(response(U))
            num_subpixels = jnp.zeros(_NPAD).astype(jnp.int64).at[:len(num_subpixels)].set(num_subpixels)
            U = jnp.zeros(_NPAD+768).at[:len(U)].set(U)
            sort = (-1*jnp.ones(_NPAD).astype(jnp.int64)).at[:len(sort)].set(sort)
            
            initial_state = get_pixelmap(uniq_in)
            initial_state = jnp.zeros(_NPAD).at[:len(initial_state)].set(initial_state)
            
            opt_P, state = LevenbergMarquardt(initial_state, R, sort, num_subpixels, U, rho, Lt)
            iters, val, jac = state[0], state[4], state[8]
            # try LBFGS to initialize LM
            if np.any(np.isnan(opt_P)) or np.logical_and(iters < 3, np.any(np.isnan(jac))):
                opt_P_LBFGS, _ = LBFGS(initial_state, R, sort, num_subpixels, U, rho, Cinv)
                opt_P_retry, state_retry = LevenbergMarquardt(opt_P_LBFGS, R, sort, num_subpixels, U, rho, Lt)
                val_retry, jac_retry = state_retry[4], state_retry[8]
                opt_P = [opt_P, opt_P_retry][np.argmin([val, val_retry])][:len(uniq_in)]
            else:
                opt_P = opt_P[:len(uniq_in)]
            cached_fitted_meshes[key] = opt_P[np.argsort(uniq_in)]
        return np.asarray(np.clip(opt_P, prior[0]+1, prior[1]))
    
    def outside_prior(P_in):
        return jnp.logical_or(jnp.any(P_in < prior[0]), jnp.any(P_in > prior[1]))

    @jax.jit
    def profile_lnlike(pix_idx, pix_val, opt_lnlike, opt_map, R, sort, num_subpixels, U, rho, Cinv):
        return jnp.abs(_lnlike(rho, Cinv, R, opt_map.at[pix_idx].set(pix_val), sort, num_subpixels, U) - opt_lnlike) > 0.5
    check_for_constrained_pixels = jax.vmap(profile_lnlike, in_axes=(0,None,None,None,None,None,None,None,None,None))
    scan_lnlike = jax.vmap(
                  jax.vmap(
                        profile_lnlike, 
                  in_axes=(None,0,None,None,None,None,None,None,None,None)), 
                  in_axes=(0,None,None,None,None,None,None,None,None,None))
    
    def analyze_limits(uniq_in):
        _NPAD = np.max((npad, len(uniq_in)))
        sort = np.argsort(uniq_in)
        key = tuple(sorted(uniq_in))
        if key in cached_constrained_pixels_lists.keys():
            return cached_constrained_pixels_lists[key], cached_unconstrained_pixels_lists[key], cached_upper_limits_lists[key]
        else:
            opt_map = get_optimalmap(uniq_in)
            opt_lnlike = lnlike(uniq_in,opt_map)
            opt_map = jnp.array(opt_map)
            U,_,sort,num_subpixels = rasterize(uniq_in, np.zeros_like(uniq_in))
            R = jnp.zeros((len(rho), _NPAD+768)).at[:,:len(U)].set(response(U))
            num_subpixels = jnp.zeros(_NPAD).astype(jnp.int64).at[:len(num_subpixels)].set(num_subpixels)
            U = jnp.zeros(_NPAD+768).at[:len(U)].set(U)
            _sort = (-1*jnp.ones(_NPAD).astype(jnp.int64)).at[:len(sort)].set(sort)
            opt_map = jnp.zeros(_NPAD).at[:len(opt_map)].set(opt_map)
            
            constrained_pixels_list = uniq_in[check_for_constrained_pixels(jnp.arange(len(opt_map)), prior[0], opt_lnlike, opt_map, R, _sort, num_subpixels, U, rho, Cinv)[:len(uniq_in)]]
            unconstrained_pixels_list = np.setdiff1d(uniq_in, constrained_pixels_list)
            unconstrained_pixels_indices = np.nonzero(np.isin(uniq_in, unconstrained_pixels_list, assume_unique=True))[0]
            len_unconstrained_pixels_indices = len(unconstrained_pixels_indices)
            unconstrained_pixels_indices = (-1*jnp.ones(_NPAD).astype(jnp.int64)).at[:len(unconstrained_pixels_indices)].set(unconstrained_pixels_indices)
            limits_per_pixel = scan_lnlike(unconstrained_pixels_indices, np.linspace(prior[0],prior[1],2*int(prior_range)+1), opt_lnlike, opt_map, R, _sort, num_subpixels, U, rho, Cinv)
            limits_per_pixel = limits_per_pixel[:len_unconstrained_pixels_indices]
            upper_limits = np.linspace(prior[0], prior[1], 2*prior_range+1)[2*prior_range+1 - limits_per_pixel.sum(1) - 1]
            cached_upper_limits_lists[key] = upper_limits
            cached_constrained_pixels_lists[key] = constrained_pixels_list
            cached_unconstrained_pixels_lists[key] = unconstrained_pixels_list
            return constrained_pixels_list, unconstrained_pixels_list, upper_limits
        
    @jax.jit
    def _objective(pixels, R, num_subpixels, U, rho, Cinv):
        p = 10**jnp.repeat(pixels, repeats=num_subpixels, total_repeat_length=U.shape[0])
        orf = R @ p
        return -0.5*jnp.sum(((rho - orf).T @ Cinv @ (rho - orf)))
    hessian_of_objective = jax.hessian(_objective)

    def cache_inverse_hessian_cholesky_decomposition(uniq_in):
        _NPAD = np.max((npad, len(uniq_in)))
        sort = np.argsort(uniq_in)
        key = tuple(sorted(uniq_in))
        if key in cached_inverse_hessian_cholesky_decompositions.keys():
            return
        else:
            U,_,_,num_subpixels = rasterize(uniq_in[sort], np.ones_like(uniq_in))
            R = response(U)
            
            R = jnp.zeros((len(rho), _NPAD+768)).at[:,:len(U)].set(response(U))
            num_subpixels = jnp.zeros(_NPAD).astype(jnp.int64).at[:len(num_subpixels)].set(num_subpixels)
            U = jnp.zeros(_NPAD+768).at[:len(U)].set(U)
            _sort = (-1*jnp.ones(_NPAD).astype(jnp.int64)).at[:len(sort)].set(sort)
            
            H = hessian_of_objective(cached_fitted_meshes[key][_sort], R, num_subpixels, U, rho, Cinv)
            H = H[:,:len(uniq_in)][:len(uniq_in),:]
            _, unconstrained_pixels_list, _ = analyze_limits(uniq_in)
            unconstrained_mask = np.isin(uniq_in[sort], unconstrained_pixels_list, assume_unique=True)
            def invert_and_clip(H, unconstrained_mask):
                Hinv = jnp.linalg.inv(-H)
                diagHinv = jnp.diag(Hinv)
                mask = jnp.logical_or(unconstrained_mask,
                                      jnp.logical_or(diagHinv > 0.01, diagHinv < 0))
                Hinv = Hinv.at[mask].set(0).at[:,mask].set(0)
                diagHinv = diagHinv.at[mask].set(0.01)
                Hinv = Hinv - jnp.diag(jnp.diag(Hinv)) + jnp.diag(diagHinv)
                Hinv *= 2.38**2/len(Hinv)
                return Hinv
            Hinv = invert_and_clip(H, unconstrained_mask)
            L = jnp.linalg.cholesky(Hinv)
            if jnp.any(jnp.isnan(L)):
                u,s,v = jnp.linalg.svd(Hinv)
                L = u * jnp.sqrt(s)
            cached_inverse_hessian_cholesky_decompositions[key] = np.array(L)
        return
    
    def change_amplitudes(uniq_in, P_in):
        sort = np.argsort(uniq_in)
        key = tuple(sorted(uniq_in))
        if key not in cached_inverse_hessian_cholesky_decompositions.keys():
            cache_inverse_hessian_cholesky_decomposition(uniq_in)
        L = cached_inverse_hessian_cholesky_decompositions[key]
    
        sorted_P_in = P_in[sort]
        P = L @ np.random.standard_normal(len(sorted_P_in)) + sorted_P_in
        P = np.where(P < prior[0], sorted_P_in, P)
        P = np.where(P > prior[1], sorted_P_in, P)
        return uniq_in, np.asarray(P[np.argsort(sort)]), 1, 1

    def shuffle_brightest_pixel(uniq_in, P_in):
        brightest_pixel = uniq_in[np.argmax(P_in)]
        sup_pixel = brightest_pixel // 4
        sub_pixels = sup_pixel*4 + np.arange(4)
        if not np.all(np.isin(sub_pixels, uniq_in)):
            return uniq_in, P_in, 1, 1
        mask = np.argwhere( (uniq_in == sub_pixels[:,None]).sum(axis=0) )
        P = np.copy(P_in)
        P[mask] = np.random.shuffle(P[mask])
        return uniq_in, P, 1, 1

    def swap_two_brightest_pixels(uniq_in, P_in):
        brightest_two_pixels = uniq_in[np.argsort(P_in)][-2:]
        mask = np.argwhere( (uniq_in == brightest_two_pixels[:,None]).sum(axis=0) )
        P = np.copy(P_in)
        P[mask] = np.flipud(P[mask])
        return uniq_in, P, 1, 1

    def triangular_pdf(x):
        return 2 * (prior[1] - x) / prior_range**2

    def general_split_move(uniq_in, P_in, adjust_other_pixels):
        # choose pixel
        mask = uniq_in < max_uniq # set resolution cap
        if mask.sum() == 0:
            return uniq_in, P_in, 1, 1
        uniq_to_split = np.random.choice(uniq_in[mask])
        uniq = split(uniq_to_split, np.copy(uniq_in))
        opt_amplitudes_forward = get_optimalmap(uniq)
        opt_amplitudes_backward = get_optimalmap(uniq_in)
        cache_inverse_hessian_cholesky_decomposition(uniq)
        cache_inverse_hessian_cholesky_decomposition(uniq_in)
        L_forward = cached_inverse_hessian_cholesky_decompositions[tuple(sorted(uniq))]
        C_forward = L_forward @ L_forward.T
        C_forward *= len(C_forward)/2.38**2
        L_backward = cached_inverse_hessian_cholesky_decompositions[tuple(sorted(uniq_in))]
        C_backward = L_backward @ L_backward.T
        C_backward *= len(C_backward)/2.38**2
        constrained_pixels_forward, unconstrained_pixels_forward, upper_limits_forward = analyze_limits(uniq)
        constrained_pixels_backward, unconstrained_pixels_backward, upper_limits_backward = analyze_limits(uniq_in)
        if not adjust_other_pixels:
            constrained_pixels_forward = np.intersect1d(constrained_pixels_forward, uniq[-4:], assume_unique=True)
            unconstrained_pixels_forward, intersection_mask, _ = np.intersect1d(unconstrained_pixels_forward, uniq[-4:], assume_unique=True, return_indices=True)
            upper_limits_forward = upper_limits_forward[intersection_mask]
            constrained_pixels_backward = np.intersect1d(constrained_pixels_backward, uniq_to_split, assume_unique=True)
            unconstrained_pixels_backward, intersection_mask, _ = np.intersect1d(unconstrained_pixels_backward, uniq_to_split, assume_unique=True, return_indices=True)
            upper_limits_backward = upper_limits_backward[intersection_mask]
        constrained_mask_forward = np.isin(uniq, constrained_pixels_forward, assume_unique=True)
        unconstrained_mask_forward = np.isin(uniq, unconstrained_pixels_forward, assume_unique=True)
        unconstrained_mask_backward = np.isin(uniq_in, unconstrained_pixels_backward, assume_unique=True)
        constrained_mask_backward = np.isin(uniq_in, constrained_pixels_backward, assume_unique=True)
        P = np.concatenate((P_in[np.isin(uniq_in, uniq_to_split, assume_unique=True, invert=True)], np.zeros(4)))
        # forward
        constrained_amplitudes_forward = L_forward @ np.random.standard_normal(len(L_forward)) + opt_amplitudes_forward[np.argsort(uniq)]
        constrained_amplitudes_forward = constrained_amplitudes_forward[np.argsort(np.argsort(uniq))][constrained_mask_forward]
        unconstrained_amplitudes_forward = np.where(np.random.rand(len(unconstrained_pixels_forward)) < 0.9,
                                                    np.random.uniform(prior[0], upper_limits_forward, len(unconstrained_pixels_forward)),
                                                    np.random.triangular(prior[0],prior[0],prior[1], len(unconstrained_pixels_forward)))
        P[constrained_mask_forward] = constrained_amplitudes_forward
        P[unconstrained_mask_forward] = unconstrained_amplitudes_forward
        
        delta_constrained_forward = constrained_amplitudes_forward - opt_amplitudes_forward[constrained_mask_forward]
        unsort = np.argsort(np.argsort(uniq))
        C_slice = C_forward[unsort,:][:,unsort][constrained_mask_forward,:][:,constrained_mask_forward]
        q_forward = (-0.5 * delta_constrained_forward.T @ C_slice @ delta_constrained_forward).sum() - .5*len(constrained_pixels_forward)*np.log(2*np.pi) - np.linalg.slogdet(C_slice)[1]
        q_forward += np.log(.9/(upper_limits_forward-prior[0]) + .1*triangular_pdf(unconstrained_amplitudes_forward)).sum()
    
        # backward
        delta_constrained_backward = P_in[constrained_mask_backward] - opt_amplitudes_backward[constrained_mask_backward]
        unsort = np.argsort(np.argsort(uniq_in))
        C_slice = C_backward[unsort,:][:,unsort][constrained_mask_backward,:][:,constrained_mask_backward]
        q_backward = (-0.5 * delta_constrained_backward.T @ C_slice @ delta_constrained_backward).sum() - .5*len(constrained_pixels_backward)*np.log(2*np.pi) - np.linalg.slogdet(C_slice)[1]
        q_backward += np.log(.9/(upper_limits_backward-prior[0]) + .1*triangular_pdf(P_in[unconstrained_mask_backward])).sum()
        q_ratio = np.exp(q_backward - q_forward)
        pixels_available_for_combining = get_available_pixels_for_combining(uniq)
        j_forward = 1 / mask.sum()
        j_backward = 1 / len(pixels_available_for_combining)
        j_ratio = j_backward / j_forward
        return uniq, P, q_ratio.item(), j_ratio.item()
    
    def general_combine_move(uniq_in, P_in, adjust_other_pixels):
        # choose pixel
        pixels_available_for_combining = get_available_pixels_for_combining(uniq_in)
        if len(pixels_available_for_combining) == 0:
            return uniq_in, P_in, 1, 1
        uniq_to_combine = np.random.choice(pixels_available_for_combining)
        uniq_of_combined_pixels = uniq_to_combine // 4 * 4 + np.arange(4)
        
        uniq = combine(uniq_to_combine, np.copy(uniq_in))
        opt_amplitudes_forward = get_optimalmap(uniq)
        opt_amplitudes_backward = get_optimalmap(uniq_in)
        cache_inverse_hessian_cholesky_decomposition(uniq)
        cache_inverse_hessian_cholesky_decomposition(uniq_in)
        L_forward = cached_inverse_hessian_cholesky_decompositions[tuple(sorted(uniq))]
        C_forward = L_forward @ L_forward.T
        C_forward *= len(C_forward)/2.38**2
        L_backward = cached_inverse_hessian_cholesky_decompositions[tuple(sorted(uniq_in))]
        C_backward = L_backward @ L_backward.T
        C_backward *= len(C_backward)/2.38**2
        constrained_pixels_forward, unconstrained_pixels_forward, upper_limits_forward = analyze_limits(uniq)
        constrained_pixels_backward, unconstrained_pixels_backward, upper_limits_backward = analyze_limits(uniq_in)
        if not adjust_other_pixels:
            constrained_pixels_forward = np.intersect1d(constrained_pixels_forward, uniq[-1], assume_unique=True)
            unconstrained_pixels_forward, intersection_mask, _ = np.intersect1d(unconstrained_pixels_forward, uniq[-1], assume_unique=True, return_indices=True)
            upper_limits_forward = upper_limits_forward[intersection_mask]
            constrained_pixels_backward = np.intersect1d(constrained_pixels_backward, uniq_of_combined_pixels, assume_unique=True)
            unconstrained_pixels_backward, intersection_mask, _ = np.intersect1d(unconstrained_pixels_backward, uniq_of_combined_pixels, assume_unique=True, return_indices=True)
            upper_limits_backward = upper_limits_backward[intersection_mask]
        constrained_mask_forward = np.isin(uniq, constrained_pixels_forward, assume_unique=True)
        unconstrained_mask_forward = np.isin(uniq, unconstrained_pixels_forward, assume_unique=True)
        unconstrained_mask_backward = np.isin(uniq_in, unconstrained_pixels_backward, assume_unique=True)
        constrained_mask_backward = np.isin(uniq_in, constrained_pixels_backward, assume_unique=True)
        P = np.concatenate((P_in[np.isin(uniq_in, uniq_of_combined_pixels, assume_unique=True, invert=True)], np.zeros(1)))
        #forward
        constrained_amplitudes_forward = L_forward @ np.random.standard_normal(len(L_forward)) + opt_amplitudes_forward[np.argsort(uniq)]
        constrained_amplitudes_forward = constrained_amplitudes_forward[np.argsort(np.argsort(uniq))][constrained_mask_forward]
        unconstrained_amplitudes_forward = np.where(np.random.rand(len(unconstrained_pixels_forward)) < 0.9,
                                                    np.random.uniform(prior[0], upper_limits_forward, len(unconstrained_pixels_forward)),
                                                    np.random.triangular(prior[0],prior[0],prior[1], len(unconstrained_pixels_forward)))
        P[constrained_mask_forward] = constrained_amplitudes_forward
        P[unconstrained_mask_forward] = unconstrained_amplitudes_forward
    
        delta_constrained_forward = constrained_amplitudes_forward - opt_amplitudes_forward[constrained_mask_forward]
        unsort = np.argsort(np.argsort(uniq))
        C_slice = C_forward[unsort,:][:,unsort][constrained_mask_forward,:][:,constrained_mask_forward]
        q_forward = (-0.5 * delta_constrained_forward.T @ C_slice @ delta_constrained_forward).sum() - .5*len(constrained_pixels_forward)*np.log(2*np.pi) - np.linalg.slogdet(C_slice)[1]
        q_forward += np.log(.9/(upper_limits_forward-prior[0]) + .1*triangular_pdf(unconstrained_amplitudes_forward)).sum()
    
        # backward
        delta_constrained_backward = P_in[constrained_mask_backward] - opt_amplitudes_backward[constrained_mask_backward]
        unsort = np.argsort(np.argsort(uniq_in))
        C_slice = C_backward[unsort,:][:,unsort][constrained_mask_backward,:][:,constrained_mask_backward]
        q_backward = (-0.5 * delta_constrained_backward.T @ C_slice @ delta_constrained_backward).sum() - .5*len(constrained_pixels_backward)*np.log(2*np.pi) - np.linalg.slogdet(C_slice)[1]
        q_backward += np.log(.9/(upper_limits_backward-prior[0]) + .1*triangular_pdf(P_in[unconstrained_mask_backward])).sum()
        q_ratio = np.exp(q_backward - q_forward)
        
        mask = uniq < max_uniq
        j_forward = 1 / len(pixels_available_for_combining)
        j_backward = 1 / mask.sum()
        j_ratio = j_backward / j_forward
        return uniq, P, q_ratio.item(), j_ratio.item()
    
    def split_a_pixel(uniq_in, P_in):
        return general_split_move(uniq_in, P_in, adjust_other_pixels=True)
    def split_a_pixel_without_adjusting_others(uniq_in, P_in):
        return general_split_move(uniq_in, P_in, adjust_other_pixels=False)
    def combine_a_pixel(uniq_in, P_in):
        return general_combine_move(uniq_in, P_in, adjust_other_pixels=True)
    def combine_a_pixel_without_adjusting_others(uniq_in, P_in):
        return general_combine_move(uniq_in, P_in, adjust_other_pixels=False)

    def delayed_rejection_routine(rejected_uniq, rejected_P, proposal,
                                  rejected_acceptance_ratio, rejected_lnlike, rejected_q_ratio, rejected_j_ratio):
        z_uniq, z_P, y2z_q_ratio, y2z_j_ratio = proposal(rejected_uniq, rejected_P)
        z_point = (z_uniq, z_P)
        z_lnlike = lnlike(*z_point)
        model_prior_ratio = prior_on_model(z_uniq) / prior_on_model(samples[-1][0])
        param_prior_ratio = (1/prior_range) ** (len(z_uniq) - len(samples[-1][0]))
        q_ratios = rejected_q_ratio * y2z_q_ratio
        j_ratios = rejected_j_ratio * y2z_j_ratio
    
        y2z_model_prior_ratio = 2**(prior_on_model(z_uniq) - prior_on_model(rejected_uniq))
        y2z_param_prior_ratio = (1/prior_range) ** (len(z_uniq) - len(rejected_uniq))
        y2z_acceptance_ratio = np.min([1, 
                                       1/( y2z_model_prior_ratio*y2z_param_prior_ratio*np.exp(z_lnlike - rejected_lnlike)*y2z_j_ratio*y2z_q_ratio )])
    
        if outside_prior(z_point[1]):
            stage2_acceptance = 0
        else:
            stage2_acceptance = np.min([1, model_prior_ratio * param_prior_ratio * np.exp(z_lnlike - lnlikes[-1])
                                 * q_ratios * j_ratios * 
                                 (1 - y2z_acceptance_ratio) / (1 - rejected_acceptance_ratio) ])
        return z_point, z_lnlike, stage2_acceptance

    def step():
        outside_prior_checker = False
        proposal = np.random.choice([change_amplitudes, split_a_pixel, combine_a_pixel, split_a_pixel_without_adjusting_others, combine_a_pixel_without_adjusting_others, shuffle_brightest_pixel, swap_two_brightest_pixels], p=[0.46,0.11,0.11,0.11,0.11,0.05,0.05])
        uniq, P, q_ratio, j_ratio = proposal(*samples[-1])
        new_point = (uniq, P)
        prior_ratio = (1/prior_range) ** (len(uniq) - len(samples[-1][0]))
        model_prior_ratio = prior_on_model(uniq) / prior_on_model(samples[-1][0])
        if outside_prior(new_point[1]):
            outside_prior_checker = True
            acceptance = 0
            new_lnlike = -np.inf
        else:
            new_lnlike = lnlike(*new_point)
            previous_lnlike = lnlikes[-1]
            try:
                acceptance = np.min([1, model_prior_ratio*prior_ratio*np.exp(new_lnlike - previous_lnlike)*j_ratio*q_ratio]) # and the Jacobian is 1
            except OverflowError:
                acceptance = 1
        if np.random.rand() < acceptance:
            samples.append(new_point)
            lnlikes.append(new_lnlike)
        elif proposal in (combine_a_pixel, split_a_pixel, combine_a_pixel_without_adjusting_others, split_a_pixel_without_adjusting_others):
            delayed_point, delayed_lnlike, delayed_acceptance = delayed_rejection_routine(new_point[0], new_point[1], proposal,
                                                                                          acceptance, new_lnlike, q_ratio, j_ratio)
            if np.random.rand() < delayed_acceptance:
                samples.append(delayed_point)
                lnlikes.append(delayed_lnlike)
            else:
                samples.append(samples[-1])
                lnlikes.append(lnlikes[-1])
        else:
            samples.append(samples[-1])
            lnlikes.append(lnlikes[-1])

    if resume:
        with open(os.path.join(outdir, chain_file_name), 'rb') as fp:
            samples = pickle.load(fp)
        samples[-1] = (samples[-1][0], np.array(samples[-1][1]))
        lnlikes = np.load(os.path.join(outdir, lnlikes_file_name)).tolist()
        _ = get_optimalmap(samples[-1][0])
        cache_inverse_hessian_cholesky_decomposition(samples[-1][0])
    else:
        if initial_uniq is not None:
            uniq0 = initial_uniq
        else:
            uniq0 = np.arange(4*initial_nside**2, 16*initial_nside**2)
        P0 = get_optimalmap(uniq0)
        x0 = (uniq0, P0)
        samples = [x0]
        lnlikes = [lnlike(*x0)]

    num_saves = (N_iterations - len(lnlikes)) // save_every_N_iterations if resume else N_iterations // save_every_N_iterations 
    for _ in range(num_saves):
        for __ in range(save_every_N_iterations):
            step()

        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, chain_file_name), 'wb') as fp:
            pickle.dump(samples, fp)
        np.save(os.path.join(outdir, lnlikes_file_name), lnlikes)

        if save_cache:
            cache = (cached_fitted_meshes,
                     cached_inverse_hessian_cholesky_decompositions,
                     cached_constrained_pixels_lists,
                     cached_unconstrained_pixels_lists,
                     cached_upper_limits_lists)
            with open(os.path.join(outdir, cache_file_name), 'wb') as fp:
                pickle.dump(cache, fp)

    # some more iterations in case save_every_N_iterations does not evenly divide N_iterations
    leftover_iterations = np.max( (N_iterations - len(lnlikes), 0) )
    if leftover_iterations > 0:
        for __ in range(leftover_iterations):
            step()
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, chain_file_name), 'wb') as fp:
            pickle.dump(samples, fp)
        np.save(os.path.join(outdir, lnlikes_file_name), lnlikes)
    
        if save_cache:
            cache = (cached_fitted_meshes,
                     cached_inverse_hessian_cholesky_decompositions,
                     cached_constrained_pixels_lists,
                     cached_unconstrained_pixels_lists,
                     cached_upper_limits_lists)
            with open(os.path.join(outdir, cache_file_name), 'wb') as fp:
                pickle.dump(cache, fp)
    return
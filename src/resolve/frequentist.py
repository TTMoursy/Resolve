import mhealpy as mhp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, jaxopt
import numpy as np
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

def split_mesh(mesh, log10_density, cdf_threshold):
    pixareas = mhp.HealpixBase(uniq=mesh).pixarea().value
    amplitudes = pixareas * 10**log10_density
    sort = np.argsort(amplitudes)[::-1]
    cdf = amplitudes[sort].cumsum() / amplitudes.sum()
    if cdf[0] > cdf_threshold:
        pix_to_split = [mesh[sort][0]]
    else:
        pix_to_split = mesh[sort][np.argwhere(cdf <= cdf_threshold)].flatten()
    mesh_after_splitting = np.copy(mesh)
    for pixel_to_split in pix_to_split:
        mesh_after_splitting = split(pixel_to_split, mesh_after_splitting)
    return mesh_after_splitting

def information_criterion_analysis(
    rho, 
    C,
    psrs_theta,
    psrs_phi,
    information_criterion = 'BIC',
    log10_density_lower = -7,
    log10_density_upper = 7,
    initial_nside = 1, 
    num_iterations = 10,
    cdf_threshold = 0.25,
    return_uncertainties = True,
    sigma = 1
):
    """
    Perform an information criterion analysis.

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
        information_criterion (str)
            information criterion to use.
            must be 'AIC' or 'BIC'.
        log10_density_lower (int or float)
            lower bound of log_10 angular power densities.
            used to bound the bisection algorithm for obtaining uncertainties.
            ignored if return_uncertainties is False.
        log10_density_upper (int or float)
            upper bound of log_10 angular power densities.
            used to bound the bisection algorithm for obtaining uncertainties.
            ignored if return_uncertainties is False.
        initial_nside (int)
            initial nside for the analysis.
            must be a power of 2.
        num_iterations (int)
            number of split-and-evaluate iterations to perform.
        cdf_threshold (float)
            the threshold integrated angular power density to split at each iteration.
        return_uncertainties (bool)
            whether to return uncertainties on the angular power densities for the preferred mesh.
            generally takes a few minutes to compute the uncertainties.
        sigma (int or float)
            the sigma level for the uncertainties, e.g., 1 for 1-sigma, 2.5 for 2.5-sigma, etc.
            ignored if return_uncertainties is False.
    Returns
        list
            the value of the information criterion at each iteration.
            has length equal to num_iterations.
        list
            a list of np.ndarray objects, each containing the UNIQ indices specifying the mesh at that iteration of the analysis.
        list
            a list of jax.Array objects, each containing the log_10 angular power densities at that iteration of the analysis.
        tuple (only if return_uncertainties)
            a tuple of two jax.Array objects. 
            the first element is the lower values of log_10 angular power densities corresponding to sigma.
            the second element is the upper values of log_10 angular power densities corresponding to sigma.
    """
    
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
    def residuals(P_input, R, sort, num_subpixels, U, rho, Lt):
        r = rho - R @ 10**jnp.repeat(P_input[sort], repeats=num_subpixels, total_repeat_length=U.shape[0])
        return Lt @ r
    
    @jax.jit
    def objective(P_input, R, sort, num_subpixels, U, rho, Cinv):
        r = rho - R @ 10**jnp.repeat(P_input[sort], repeats=num_subpixels, total_repeat_length=U.shape[0])
        return 0.5*r.T @ Cinv @ r

    def get_pixelmap(uniq_in):
        R = response(uniq_in)
        M = R.T @ Cinv @ R
        X = R.T @ Cinv @ rho
        pixelmap = jnp.linalg.solve(M + .001*np.diag(np.ones_like(X)), X)
        orf = R @ pixelmap
        A2 = (orf.T @ Cinv @ rho) / (orf.T @ Cinv @ orf)
        pixelmap *= A2
        pixelmap = np.array(pixelmap)
        pixelmap[pixelmap < 0] = 10**log10_density_lower
        return jnp.log10(pixelmap)
    
    def get_optimalmap(uniq_in):
        U,P,sort,num_subpixels = rasterize(uniq_in, np.zeros_like(uniq_in))
        R = response(U)
        initial_state = get_pixelmap(uniq_in)
        opt_P, state = jax.jit(jaxopt.LevenbergMarquardt(residuals, materialize_jac=True, jit=True, maxiter=1000).run)(initial_state, R, sort, num_subpixels, U, rho, Lt)
        iters, val, jac = state[0], state[4], state[8]
        # try LBFGS to initialize LM
        if np.any(np.isnan(opt_P)) or np.logical_and(iters < 3, np.any(np.isnan(jac))):
            opt_P_LBFGS, _ = jax.jit(jaxopt.LBFGS(objective, jit=True, maxiter=100, unroll=True).run)(initial_state, R, sort, num_subpixels, U, rho, Cinv)
            opt_P_retry, state_retry = jax.jit(jaxopt.LevenbergMarquardt(residuals, materialize_jac=True, jit=True, maxiter=1000).run)(opt_P_LBFGS, R, sort, num_subpixels, U, rho, Lt)
            val_retry, jac_retry = state_retry[4], state_retry[8]
            opt_P = [opt_P, opt_P_retry][np.argmin([val, val_retry])]
        return np.asarray(opt_P), (R, sort, num_subpixels, U)
    
    information_criteria, meshes, log10_densities = [], [], []
    mesh = np.arange(4*initial_nside**2, 16*initial_nside**2)
    for i in range(num_iterations):
        log10_density, aux = get_optimalmap(mesh)
        information_criteria.append(2*objective(log10_density, *aux, rho, Cinv) + np.log(len(rho))*len(mesh) if information_criterion == 'BIC' else 2*objective(log10_density, *aux, rho, Cinv) + 2*len(mesh))
        meshes.append(mesh)
        log10_densities.append(log10_density)
        mesh = split_mesh(mesh, log10_density, cdf_threshold)
    if not return_uncertainties:
        return information_criteria, meshes, log10_densities
    preferred_mesh = meshes[np.argmin(information_criteria)]
    def get_uncertainties(mesh, sigma):
        opt_map, aux = get_optimalmap(mesh)
        R, sort, num_subpixels, U = aux
        opt_lnlike = objective(opt_map, R, sort, num_subpixels, U, rho, Cinv)
        def profile_lnlike(pix_val, pix_idx):
            def _obj(pixels_vals, pix_val, pix_idx):
                r = rho - R @ 10**jnp.repeat((pixels_vals.at[pix_idx].set(pix_val))[sort], repeats=num_subpixels, total_repeat_length=U.shape[0])
                return 0.5*r.T @ Cinv @ r
            _, state = jax.jit(jaxopt.LBFGS(_obj, maxiter=10, jit=True).run)(opt_map, pix_val, pix_idx)
            return jnp.abs(state[1] - opt_lnlike) - 0.5*sigma
        def bisection_wrapper(opt_map, pix_idx, lower, upper):
            return jaxopt.Bisection(profile_lnlike, lower, upper, check_bracket=False).run(opt_map[pix_idx], pix_idx)
        check_for_unconstrained_pixels = jax.vmap(profile_lnlike,(None,0))(log10_density_lower, jnp.arange(len(mesh))) < 0
        unconstrained_pixels_indices = jnp.nonzero(check_for_unconstrained_pixels)[0]
        constrained_pixels_indices = jnp.setdiff1d(jnp.arange(len(mesh)), unconstrained_pixels_indices, assume_unique=True)
        upper_limits, _ = jax.vmap(jax.jit(bisection_wrapper), (None,0,0,None))(opt_map, jnp.arange(len(mesh)), opt_map, log10_density_upper)
        lower_limits, _ = jax.vmap(jax.jit(bisection_wrapper), (None,0,None,0))(opt_map[constrained_pixels_indices], constrained_pixels_indices, log10_density_lower, opt_map[constrained_pixels_indices])
        lower_limits = jnp.zeros_like(opt_map).at[constrained_pixels_indices].set(lower_limits).at[unconstrained_pixels_indices].set(log10_density_lower)
        return lower_limits, upper_limits
    uncertainties = get_uncertainties(preferred_mesh, sigma)
    return information_criteria, meshes, log10_densities, uncertainties
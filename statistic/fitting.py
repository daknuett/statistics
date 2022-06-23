from .levenberg_marquardt import LMFitter, LMFitWorker, ErrorEstimatingFitter, FittingError

def fit(f, t, values, stds, p0, data=None, statistic=None
        , jackknife_method=None, p0_guesses=None, covm=None, use_covm_W=False
        , jackknife_kwargs={}, fit_kwargs={}):
    """
    Perform a Levenberg Marquardt fit minimizing
    
    ..math::
        (f(t, p) - values) \sigma^{-1} (f(t, p) - values)

    where sigma is either ``diag(stds)`` or ``covm`` (if ``covm is not None and use_covm_W is True``).

    If ``p0_guesses is not None`` first an opimal ``p0`` is searched and ``p0`` is ignored.
    If ``data is not None and statistic is not None`` the error is estimated using ``jackknife_method``.
    Else the error is estimated using some error propagation.

    """
    worker = LMFitWorker(f, t, values, stds, p0, covm=covm, use_covm_W=use_covm_W, **fit_kwargs)

    if(p0_guesses is not None):
        fitter = LMFitter(worker, p0_guesses)
        worker = fitter.get_optimized_worker()
        p0 = fitter.get_p0()

    
    p, chi2, J, success = worker.do_fit()

    if not success:
        raise FittingError("fitter did not converge")

    if data is not None and statistic is not None:
        if jackknife_method is not None:
            error_estimator = ErrorEstimatingFitter(worker, statistic, data, jackknife_method=jackknife_method, **jackknife_kwargs)
        else:
            error_estimator = ErrorEstimatingFitter(worker, statistic, data, **jackknife_kwargs)

        p_std, f_std = error_estimator.estimate_error()
    else:
        p_cov = worker.estimate_praram_covm(p)
        p_std = np.sqrt(np.diag(p_cov))
        f_std = worker.estimate_output_std(p)

    return_data = {
            "chi2": chi2
            , "J": J
            , "method": worker.method
            , "p0": p0
            }
    return p, p_std, f_std, return_data


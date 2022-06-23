import numpy as np
from .statistic import jackknife_std
from copy import deepcopy

Default_l0=1e-2
Default_Delta=1e-5
Default_nmax=None
Default_eps1=1e-3
Default_eps2=1e-3
Default_eps3=1e-1
Default_eps4=1e-1
Default_Lup=11
Default_Ldown=9
Default_method=1

def jacobian(y, t, p, Delta):
    def partialdiff(p, Delta, j, t):
        deltapj = np.zeros_like(p, dtype=np.double)
        magn = Delta * (1 + np.abs(p[j]))
        deltapj[j] = magn
        pforward = p + deltapj
        pbackward = p - deltapj
        
        return (y(t, pforward) - y(t, pbackward)) / (2 * magn)
    
    J = np.array([[partialdiff(p, Delta, j, ti) for j in range(p.size)] for ti in t])
    return J

def lm_param_covm(J, stds, covm):
    if(covm is not None):
        W = np.linalg.inv(covm)
    else:
        W = np.diag(1 / stds**2)
    return np.linalg.inv(np.transpose(J) @ W @ J)

def lm_output_std(J, stds, covm):
    return np.sqrt(np.diag(J @ lm_param_covm(J, stds, covm) @ np.transpose(J)))

class LMFitWorker:
    """
    This function implements the Levenberg-Marquardt fitting with error estimation
    described in https://people.duke.edu/~hpgavin/ce281/lm.pdf.

    If ``mnax is None`` it defaults to ``10*len(p0)``.
    ``f`` must take ``t`` as first and ``p`` as second argument. 
    ``t, values, stds, p0`` should be numpy arrays.

    If ``covm is not None`` it will be used for the error estimation and should
    be a numpy array containing the covariance matrix of the ``values``. By
    setting ``use_covm_W is True`` the Levenberg Marquardt will use the
    covariance matrix for the minimization but this might lead to
    instabilities.

    ``method`` can be 1, 2, or 3. It changes the used minimizer.
    """
    def __init__(self, f, t, values, stds, p0
                , covm=None, use_covm_W=False
                , l0=Default_l0, Delta=Default_Delta
                , nmax=Default_nmax, eps1=Default_eps1, eps2=Default_eps2, eps3=Default_eps3, eps4=Default_eps4
                , Lup=Default_Lup, Ldown=Default_Ldown, method=Default_method):
        self.f = f
        self.t = t
        self.values = values 
        self.stds = stds 
        self.p0 = p0
        self.covm = covm
        self.use_covm_W = use_covm_W
        self.l0 = l0
        self.Delta = Delta
        if(nmax is None):
            self.nmax = 10 * len(p0)
        else:
            self.nmax = nmax
        self.eps1 = eps1
        self.eps2 = eps2
        self.eps3 = eps3
        self.eps4 = eps4
        self.Lup = Lup
        self.Ldown = Ldown
        self.method = method

        if(covm is not None and use_covm_W):
            self.W = np.linalg.inv(covm)
        else:
            self.W = np.diag(1 / stds**2)

    def lambda_init1(self, lnot, J):
        return lnot

    def lambda_init2(self, lnot, J):
        return lnot * np.max(np.diag(np.transpose(J) @ self.W @ J))

    def update_method1(self, p, l, nu, J):
        yp = self.f(self.t, p)
        chi2p = np.sum(((self.values - yp) / self.stds)**2)

        u = np.transpose(J) @ self.W @ (self.values - yp)

        if(np.max(np.abs(u)) < self.eps1):
            return True, p, None, None, chi2p
        
        if(chi2p / (np.size(self.values) - np.size(p) + 1) < self.eps3):
            return True, p, None, None, chi2p

        V = np.transpose(J) @ self.W @ J
        A = V + l*np.diag(np.diag(V))
        b = np.transpose(J) @ self.W @ (self.values - yp)   # eq. 13
        h = np.linalg.solve(A, b)
        
        yph = self.f(self.t, p + h)

        chi2ph = np.sum(((self.values - yph) / self.stds)**2)

        rho = (chi2p - chi2ph) / (np.transpose(h) @ (l *  np.diag(np.diag(V)) @ h + u))

        if(rho > self.eps4):
            p = p + h
            l = max(l / self.Ldown, 1e-7)
        else:
            l = min(l * self.Lup, 1e7)
        if(np.max(h / p) < self.eps2):
            return True, p, None, None, chi2p
        return False, p, l, None, chi2p

    def update_method2(self, p, l, nu, J):
        yp = self.f(self.t, p)
        chi2p = np.sum(((self.values - yp) / self.stds)**2)
        u = np.transpose(J) @ self.W @ (self.values - yp)

        if(np.max(np.abs(u)) < self.eps1):
            return True, p, None, None, chi2p
        
        if(chi2p / (np.size(self.values) - np.size(p) + 1) < self.eps3):
            return True, p, None, chi2p
        
        V = np.transpose(J) @ self.W @ J
        A = V + l*np.diag(np.ones_like(np.diag(V)))
        b = np.transpose(J) @ self.W @ (self.values - yp)   # eq. 12
        h = np.linalg.solve(A, b)
        
        yph = self.f(self.t, p + h)

        chi2ph = np.sum(((self.values - yph) / self.stds)**2)
        
        alpha = u @ h / ((chi2ph - chi2p) / 2 + 2 * u@h)
        ypah = self.f(self.t, p + alpha * h)
        chi2pah = np.sum(((self.values - ypah) / self.stds)**2)
        
        rho = (chi2p - chi2pah) / (np.transpose(alpha * h) @ (l * alpha * h + u))
        
        if(rho > self.eps4):
            p = p + alpha * h
            l = max(l / (1 + alpha), 1e-7)
        else:
            l = l + np.abs(chi2pah - chi2p) / (2 * alpha)
            
        if(np.max(h / p) < self.eps2):
            return True, p, None, None, chi2p
        return False, p, l, None, chi2p

    def update_method3(self, p, l, nu, J):
        yp = self.f(self.t, p)
        chi2p = np.sum(((self.values - yp) / self.stds)**2)
        u = np.transpose(J) @ self.W @ (self.values - yp)

        if(np.max(np.abs(u)) < self.eps1):
            return True, p, None, None, chi2p
        
        if(chi2p / (np.size(self.values) - np.size(p) + 1) < self.eps3):
            return True, p, None, chi2p
        
        V = np.transpose(J) @ self.W @ J
        A = V + l*np.diag(np.ones_like(np.diag(V)))
        b = np.transpose(J) @ self.W @ (self.values - yp)   # eq. 12
        h = np.linalg.solve(A, b)

        
        yph = self.f(self.t, p + h)

        chi2ph = np.sum(((self.values - yph) / self.stds)**2)
        
        
        rho = (chi2p - chi2ph) / (np.transpose(h) @ (l * h + u))
        
        if(rho > self.eps4):
            p = p + h
            l = max(1/3, 1 - (2*rho - 1)**3)
            nu = 2
        else:
            l = l * nu
            nu *= 2
                
        if(np.max(h / p) < self.eps2):
            return True, p, None, None, chi2p
        return False, p, l, nu, chi2p

   def do_fit(self, p0=None):
       if p0 is None:
           p0 = self.p0

    update_methods = {1: self.update_method1, 2: self.update_method2, 3: self.update_method3}
    lambda_initializers = {1: self.lambda_init1, 2: self.lambda_init2, 3: self.lambda_init2}

    J = jacobian(self.f, self.t, p, self.Delta)
    l = lambda_initializers[method](self.l0, J)
    nu = None

    for n in range(nmax):
        J = jacobian(self.f, self.t, p, self.Delta)

        do_break, p, l, nu, chi2p = update_methods[method](p, l, nu, J)
        
        if(do_break):
            return p, chi2p, J, True
        
    return p, chi2p, J, False

    def estimate_param_covm(self, p):
        return lm_param_covm(jacobian(self.f, self.t, p, self.Delta), self.stds, self.covm)

    def estimate_output_std(self, p):
        J = jacobian(self.f, self.t, p, self.Delta),
        return np.sqrt(np.diag(J @ lm_param_covm(J, self.stds, self.covm) @ np.transpose(J)))

    def new_from_values(self, values):
        cpy = deepcopy(self)
        cpy.values = values
        return cpy

    # FIXME: I should use param covm computed using jackkinife here!
    def estimate_f_std(self, p, p_std):
        J = jacobian(self.f, self.t, p, self.Delta),
        return np.sqrt(np.diag(J @ np.diag(p_std) @ np.transpose(J)))


class FittingError(Exception):
    pass

class LMFitter:
    def __init__(self, worker, p0_guesses):
        self.worker = worker
        self.p0_guesses = p0_guesses

    def get_p0(self):
        results = [self.worker.do_fit(p0=p0) for p0 in self.p0_guesses]
        chi2s = np.array([r[1] if r[3] else float("inf") for r in results])

        if not np.any(np.isfinite(chi2s)):
            raise FittingError("failed to find p0 for which fit converges")
        return self.p0_guesses(np.argmin(chi2s))
    
    def get_optimized_worker(self):
        p0 = self.get_p0()
        self.worker.p0 = p0
        return self.worker


class ErrorEstimatingFitter:
    def __init__(self, worker, statistic, data, jackknife_method=jackknife_std, **kwargs):
        self.worker = worker
        self.statistic = statistic
        self.data = data
        self.jackknife_method = jackknife_method
        self.kwargs = kwargs

    # FIXME: I should use param covm computed using jackkinife here!
    def estimate_error(self):
        def do_fit(self, sample):
            worker = self.worker.new_from_values(self.statistic(sample))
            p, chi2p, J, success = worker.do_fit()
            if(not success):
                raise FittingError("fit failed in jackknife")
            return p

        p, chi2, J, success = self.worker.do_fit()
        p_std = self.jackknife_method(self.data, do_fit, **kwargs)
        f_std = self.worker.estimate_f_std(p, p_std)

        return p_std, f_std

        


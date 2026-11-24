import math

import torch


FEATURE_MAPS = {
    'linear': lambda x: x,
    'tanh': torch.tanh,
}


class RegressionWindTunnel:
    """In-context Bayesian regression with a closed-form posterior.

    A sequence first draws a task family I ~ Categorical(mix), then weights
    w ~ N(mu_I * 1, tau2 * I_d), then examples

        x_i ~ N(0, I_d),   y_i = w . phi_I(x_i) + eps_i,   eps_i ~ N(0, sigma2).

    Each family is linear in w, so its posterior is Gaussian (a conjugate update), and
    the posterior over families follows from the Gaussian marginal likelihoods. The
    exact posterior predictive at x_q is therefore a mixture of Gaussians.

    Two presets reproduce the regression tasks in the literature:

      agarwal()   Agarwal et al., Paper I Sec. 3.4: one linear family, d = 3,
                  sigma2 = 0.25, k = 6, target binned onto 41 bins over [-5, 5].
      wakayama()  Wakayama & Suzuki (ICML 2026), App. B: a 50/50 mixture of linear and
                  tanh families, d = 5, prior means +3 and -3, sigma = 0.1, k <= 15,
                  scored by squared error against the Bayes predictive mean.

    Input format: one token per example, [x_i, y_i, 0], then the query [x_q, 0, 1].
    The source papers interleave x and y as separate tokens; pairing them in one token
    removes the need for the model to match each y to its x.
    """

    def __init__(self, d=3, sigma2=0.25, tau2=1.0, families=('linear',), mix=None,
                 prior_means=None, output='binned', n_bins=41, y_max=5.0, device='cpu'):
        self.d = d
        self.sigma2 = float(sigma2)
        self.tau2 = float(tau2)
        self.families = tuple(families)
        F_ = len(self.families)
        self.mix = torch.tensor(mix if mix is not None else [1.0 / F_] * F_, device=device)
        self.prior_means = [float(m) for m in (prior_means if prior_means is not None else [0.0] * F_)]
        self.output = output
        self.n_bins = n_bins
        self.device = device
        self.token_dim = d + 2

        self.centres = torch.linspace(-y_max, y_max, n_bins, device=device)
        half = (self.centres[1] - self.centres[0]) / 2
        inf = torch.tensor([math.inf], device=device)
        self.edges = torch.cat([-inf, self.centres[:-1] + half, inf])      # (n_bins + 1,)

    # ------------------------------------------------------------------- presets
    @classmethod
    def agarwal(cls, device='cpu'):
        return cls(d=3, sigma2=0.25, tau2=1.0, families=('linear',), prior_means=[0.0],
                   output='binned', n_bins=41, y_max=5.0, device=device)

    @classmethod
    def wakayama(cls, device='cpu', n_bins=201, y_max=30.0):
        # The model is scored on the scalar predictive mean; n_bins and y_max only set the
        # grid on which a discretised predictive can be drawn.
        return cls(d=5, sigma2=0.01, tau2=1.0, families=('linear', 'tanh'), mix=[0.5, 0.5],
                   prior_means=[3.0, -3.0], output='scalar', n_bins=n_bins, y_max=y_max,
                   device=device)

    # ------------------------------------------------------------------ sampling
    def sample(self, batch_size, k, generator=None):
        """Returns X (B,k,d), y (B,k), x_q (B,d), y_q (B,), w (B,d), fam (B,)."""
        B, d, dev, g = batch_size, self.d, self.device, generator
        u = torch.rand(B, device=dev, generator=g)
        fam = (u.unsqueeze(-1) > torch.cumsum(self.mix, 0)[:-1]).sum(-1)
        mu = torch.tensor(self.prior_means, device=dev)[fam].unsqueeze(-1)
        w = mu + torch.randn(B, d, device=dev, generator=g) * math.sqrt(self.tau2)
        X = torch.randn(B, k, d, device=dev, generator=g)
        x_q = torch.randn(B, d, device=dev, generator=g)
        noise = torch.randn(B, k + 1, device=dev, generator=g) * math.sqrt(self.sigma2)
        Xa = torch.cat([X, x_q.unsqueeze(1)], dim=1)                   # (B, k+1, d)
        ya = torch.zeros(B, k + 1, device=dev)
        for i, name in enumerate(self.families):
            yi = torch.einsum('bkd,bd->bk', FEATURE_MAPS[name](Xa), w)
            ya = torch.where((fam == i).unsqueeze(-1), yi, ya)
        ya = ya + noise
        return X, ya[:, :k], x_q, ya[:, k], w, fam

    # ----------------------------------------------------------------- posterior
    def family_posterior(self, X, y, i):
        """Posterior N(mean, cov) over w under family i, plus log marginal likelihood.

        Everything goes through the d x d posterior precision A, never a k x k matrix:
            A       = I / tau2 + Phi^T Phi / sigma2
            log|C|  = k log sigma2 + d log tau2 + log|A|          (determinant lemma)
            r^T C^-1 r = r^T r / sigma2 - s^T A^-1 s / sigma2^2,  s = Phi^T r   (Woodbury)
        where C = sigma2 I + tau2 Phi Phi^T is the marginal covariance of y.
        """
        B, k, d = X.shape
        Phi = FEATURE_MAPS[self.families[i]](X)
        mu0 = torch.full((B, d), self.prior_means[i], device=X.device, dtype=X.dtype)
        eye = torch.eye(d, device=X.device, dtype=X.dtype).expand(B, d, d)
        A = eye / self.tau2 + Phi.transpose(1, 2) @ Phi / self.sigma2
        L = torch.linalg.cholesky(A)
        cov = torch.cholesky_inverse(L)
        rhs = mu0 / self.tau2 + (Phi.transpose(1, 2) @ y.unsqueeze(-1)).squeeze(-1) / self.sigma2
        mean = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
        if k == 0:
            return mean, cov, torch.zeros(B, device=X.device)
        r = y - (Phi @ mu0.unsqueeze(-1)).squeeze(-1)
        s_ = (Phi.transpose(1, 2) @ r.unsqueeze(-1))
        quad = (r * r).sum(-1) / self.sigma2 - \
            (s_ * torch.cholesky_solve(s_, L)).sum((-1, -2)) / self.sigma2 ** 2
        logdetA = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(-1)
        logdetC = k * math.log(self.sigma2) + d * math.log(self.tau2) + logdetA
        logml = -0.5 * (quad + logdetC + k * math.log(2 * math.pi))
        return mean, cov, logml

    def predictive_components(self, X, y, x_q):
        """Mixture weights pi (B,F), component means (B,F) and variances (B,F)."""
        means, vars_, logps = [], [], []
        for i, name in enumerate(self.families):
            m, cov, logml = self.family_posterior(X, y, i)
            phi_q = FEATURE_MAPS[name](x_q)
            means.append((m * phi_q).sum(-1))
            vars_.append(self.sigma2 + torch.einsum('bi,bij,bj->b', phi_q, cov, phi_q))
            logps.append(torch.log(self.mix[i]) + logml)
        pi = torch.softmax(torch.stack(logps, -1), -1)
        return pi, torch.stack(means, -1), torch.stack(vars_, -1)

    def predictive(self, X, y, x_q):
        """Posterior predictive mean (B,) and variance (B,) at x_q."""
        pi, mu, var = self.predictive_components(X, y, x_q)
        mean = (pi * mu).sum(-1)
        second = (pi * (var + mu ** 2)).sum(-1)
        return mean, (second - mean ** 2).clamp_min(1e-12)

    def binned_components(self, pi, mu, var):
        """Discretise the Gaussian mixture sum_f pi_f N(mu_f, var_f) onto the bins, (B, n_bins)."""
        std = var.sqrt().unsqueeze(-1)                                     # (B,F,1)
        z = (self.edges.view(1, 1, -1) - mu.unsqueeze(-1)) / std
        cdf = torch.special.ndtr(z)
        p = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0.0)                 # (B,F,n_bins)
        p = (pi.unsqueeze(-1) * p).sum(1)
        return p / p.sum(-1, keepdim=True)

    def bayes_target(self, X, y, x_q):
        """(binned predictive, predictive mean, predictive variance)."""
        pi, mu_c, var_c = self.predictive_components(X, y, x_q)
        mean = (pi * mu_c).sum(-1)
        var = ((pi * (var_c + mu_c ** 2)).sum(-1) - mean ** 2).clamp_min(1e-12)
        return self.binned_components(pi, mu_c, var_c), mean, var

    # -------------------------------------------------------------------- tokens
    def tokens(self, X, y, x_q, interleaved=False):
        """Model input. Paired (default): (B, k+1, d+2), one token [x_i, y_i, 0] per
        example then the query [x_q, 0, 1].

        Interleaved (the source papers' format): (B, 2k+1, d+2), x and y as separate
        tokens x_1, y_1, ..., x_k, y_k, x_q with x-tokens [x, 0, 0] and y-tokens
        [0, y, 1]. The model must then match each y to the x before it."""
        if interleaved:
            return self._interleaved_tokens(X, y, x_q)
        B, k, d = X.shape
        ctx = torch.cat([X, y.unsqueeze(-1), torch.zeros(B, k, 1, device=X.device)], dim=-1)
        qry = torch.cat([x_q, torch.zeros(B, 1, device=X.device),
                         torch.ones(B, 1, device=X.device)], dim=-1).unsqueeze(1)
        return torch.cat([ctx, qry], dim=1)

    def _interleaved_tokens(self, X, y, x_q):
        B, k, d = X.shape
        z = lambda *shape: torch.zeros(*shape, device=X.device)
        xt = torch.cat([X, z(B, k, 2)], dim=-1)                                   # (B,k,d+2)
        yt = torch.cat([z(B, k, d), y.unsqueeze(-1), torch.ones(B, k, 1, device=X.device)], dim=-1)
        pairs = torch.stack([xt, yt], dim=2).reshape(B, 2 * k, d + 2)
        q = torch.cat([x_q, z(B, 2)], dim=-1).unsqueeze(1)
        return torch.cat([pairs, q], dim=1)

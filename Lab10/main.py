import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm
import scipy.stats as stats


if __name__ == '__main__':

    n_cluster = [100, 150, 250]
    n_total = sum(n_cluster)
    means = [5, 0, 3]
    std_devs = [2, 2, 2]
    mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))
    az.plot_kde(np.array(mix))
    #plt.show()


    clusters = [2, 3, 4]
    models = []
    idatas = []
    for cluster in clusters:
        with pm.Model() as model:
            p = pm.Dirichlet('p', a=np.ones(cluster))
            means = pm.Normal('means',
                              mu=np.linspace(mix.min(), mix.max(), cluster),
                              sd=10, shape=cluster,
                              transform=pm.distributions.transforms.ordered)

            sd = pm.HalfNormal('sd', sd=10)
            y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=mix)
            idata = pm.sample(1000, tune=2000, target_accept=0.9, random_seed=123, return_inferencedata=True)
            idatas.append(idata)
            models.append(model)

    _, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = np.ravel(ax)
    x = np.linspace(mix.min(), mix.max(), 200)
    for idx, idata_x in enumerate(idatas):
        posterior_x = idata_x.posterior.stack(samples=("chain", "draw"))
    x_ = np.array([x] * clusters[idx]).T
    for i in range(50):
        i_ = np.random.randint(0, posterior_x.samples.size)
    means_y = posterior_x['means'][:, i_]
    p_y = posterior_x['p'][:, i_]
    sd = posterior_x['sd'][i_]
    dist = stats.norm(means_y, sd)
    ax[idx].plot(x, np.sum(dist.pdf(x_) * p_y.values, 1), 'C0', alpha=0.1)
    means_y = posterior_x['means'].mean("samples")
    p_y = posterior_x['p'].mean("samples")
    sd = posterior_x['sd'].mean()
    dist = stats.norm(means_y, sd)
    ax[idx].plot(x, np.sum(dist.pdf(x_) * p_y.values, 1), 'C0', lw=2)
    ax[idx].plot(x, dist.pdf(x_) * p_y.values, 'k--', alpha=0.7)
    az.plot_kde(mix, plot_kwargs={'linewidth': 2, 'color': 'k'}, ax=ax[idx])
    ax[idx].set_title('K = {}'.format(clusters[idx]))
    ax[idx].set_yticks([])
    ax[idx].set_xlabel('x')

    plt.show()

    comp_waic = az.compare(dict(zip([str(c) for c in clusters], idatas)),
                      method='BB-pseudo-BMA', ic="waic", scale="deviance")

    comp_loo = az.compare(dict(zip([str(c) for c in clusters], idatas)),
                      method='BB-pseudo-BMA', ic="loo", scale="deviance")

    az.plot_compare(comp_waic)
    az.plot_compare(comp_loo)



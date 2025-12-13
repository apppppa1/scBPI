import torch
import torch.distributions as D
from torch.autograd import Variable
#import scglue

# # Example Gaussian distributions with size [12, 3]
# dist1 = D.Normal(torch.ones([12, 3]), torch.full([12, 3], 0.1))
# dist2 = D.Normal(torch.full([12, 3], 0.5), torch.full([12, 3], 0.2))
#
# # Define the dictionary of Gaussian distributions with new size
# u = {
#     'expert1': {'mean': dist1.mean, 'logvar': torch.log(dist1.variance)},
#     'expert2': {'mean': dist2.mean, 'logvar': torch.log(dist2.variance)}
# }

def sample_gaussian(mu, logvar):
    std = (0.5*logvar).exp()
    eps = torch.randn_like(std)
    return mu + std*eps
# Example ProductOfExperts class
class ProductOfExperts(torch.nn.Module):
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar
# Instantiate the ProductOfExperts class
# poex = ProductOfExperts()
# # Calculate the combined mean and logvar using ProductOfExperts class
# z_mu, z_logvar = poex(u)  # Shape [12, 3]
# z = sample_gaussian(z_mu, z_logvar)
# print(z_sample.shape)
def prior_expert(size, use_cuda=True):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar

def poe(mus,logvars):
    """
    # Product of Experts
    # :param list mus: The Mean. [mu_1, ..., mu_M], where mu_m is N * K
    # :param list logvars: The log-scaled variance. [logvar_1, ..., logvar_M], where logvar_m is N * K
    # """
    # Extract means and logvars from the dictionary of Gaussian distributions
    # mus = torch.stack([u[k].loc for k in u.keys()], dim=0)  # 直接访问 mean 属性
    # # logvars = torch.stack([2 * torch.log(u[k].std) for k in u.keys()], dim=0)
    # logvars = torch.stack([2 * torch.log(u[k].scale + 1e-8) for k in u.keys()], dim=0)# 直接访问 variance 属性，并取对数
    # mus = [torch.full_like(mus[0], 0)] + list(mus)
    # logvars = [torch.full_like(logvars[0], 0)] + list(logvars)
    #
    # mus_stack = torch.stack(mus, dim=1)  # N * M * K
    # logvars_stack = torch.stack(logvars, dim=1)
    #
    # T = torch.exp(-logvars_stack)  # precision of i-th Gaussian expert at point x
    # T_sum = T.sum(1)  # N * K
    # pd_mu = (mus_stack * T).sum(1) / T_sum
    # pd_var = 1 / T_sum
    # pd_logvar = torch.log(pd_var + 1e-8)
    # return pd_mu, pd_logvar  # N * K
    eps = 1e-8
    var = torch.exp(logvars) + eps
    # precision of i-th Gaussian expert at point x
    T = 1. / (var + eps)
    pd_mu = torch.sum(mus * T, dim=0) / torch.sum(T, dim=0)
    pd_var = 1. / torch.sum(T, dim=0)
    pd_logvar = torch.log(pd_var + eps)
    return pd_mu, pd_logvar
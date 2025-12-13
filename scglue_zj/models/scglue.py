
r"""
Graph-linked unified embedding (GLUE) for single-cell multi-omics
data integration
"""

import copy
import os
import sys
from itertools import chain
from math import ceil
from typing import List, Mapping, Optional, Tuple, Union

import ignite
import networkx as nx
import numpy as np
import pandas as pd
import torch
# from scglue_zj.models import SCGLUEModel, PairedSCGLUEModel
# from .scglue import SCGLUEModel
# import torch.nn as nn
from torch.nn import Parameter
import torch.distributions as D
import torch.nn.functional as F
from anndata import AnnData
import anndata as ad
import torch.nn.functional as nnf
from sklearn.cluster import KMeans
from ..graph import check_graph
from ..num import normalize_edges
from ..utils import AUTO, config, get_chained_attr, logged
from . import sc
from .base import Model
from .data import AnnDataset, ArrayDataset, DataLoader, GraphDataset
from .glue import GLUE, GLUETrainer
from .nn import freeze_running_stats
# from .PoE_test import ProductOfExperts,sample_gaussian,prior_expert
from .PoE_test import sample_gaussian,poe,prior_expert
from torch.distributions import Normal
#---------------------------------- Utilities ----------------------------------

_ENCODER_MAP: Mapping[str, type] = {}
_DECODER_MAP: Mapping[str, type] = {}


def register_prob_model(prob_model: str, encoder: type, decoder: type) -> None:
    r"""
    Register probabilistic model

    Parameters
    ----------
    prob_model
        Data probabilistic model
    encoder
        Encoder type of the probabilistic model
    decoder
        Decoder type of the probabilistic model
    """
    _ENCODER_MAP[prob_model] = encoder
    _DECODER_MAP[prob_model] = decoder


register_prob_model("Normal", sc.VanillaDataEncoder, sc.NormalDataDecoder)
register_prob_model("ZIN", sc.VanillaDataEncoder, sc.ZINDataDecoder)
register_prob_model("ZILN", sc.VanillaDataEncoder, sc.ZILNDataDecoder)
register_prob_model("NB", sc.NBDataEncoder, sc.NBDataDecoder)
register_prob_model("ZINB", sc.NBDataEncoder, sc.ZINBDataDecoder)


#----------------------------- Network definition ------------------------------

class SCGLUE(GLUE):

    r"""
    GLUE network for single-cell multi-omics data integration

    Parameters
    ----------
    g2v
        Graph encoder
    v2g
        Graph decoder
    x2u
        Data encoders (indexed by modality name)
    u2x
        Data decoders (indexed by modality name)
    idx
        Feature indices among graph vertices (indexed by modality name)
    du
        Modality discriminator
    prior
        Latent prior
    u2c
        Data classifier
    """

    def __init__(
            self, g2v: sc.GraphEncoder, v2g: sc.GraphDecoder,
            x2u: Mapping[str, sc.DataEncoder],
            u2x: Mapping[str, sc.DataDecoder],
            idx: Mapping[str, torch.Tensor],
            du: sc.Discriminator,
            prior: sc.Prior,
            recons_z: Mapping[str, sc.Reconstruct_data],
            # mu: torch.Tensor,
            u2c: Optional[sc.Classifier] = None,
            # n_clusters: int = 30,  # 假设聚类数目为10
            # latent_dim: int = 50, # 假设隐空间维度为50
    ) -> None:
        super().__init__(g2v, v2g, x2u, u2x, idx, du, prior)
        # super().__init__(g2v, v2g, x2u, u2x, idx, prior)
        self.recons_z = torch.nn.ModuleDict(recons_z)
        self.u2c = u2c.to(self.device) if u2c else None
        # # 定义可训练的 mu 参数
        # # self.mu = torch.nn.Parameter(mu.to(self.device))
        # self.mu = mu


class IndSCGLUE(SCGLUE):

    r"""
    GLUE network where cell and feature in different modalities are independent

    Parameters
    ----------
    g2v
        Graph encoder
    v2g
        Graph decoder
    x2u
        Data encoders (indexed by modality name)
    u2x
        Data decoders (indexed by modality name)
    idx
        Feature indices among graph vertices (indexed by modality name)
    du
        Modality discriminator
    prior
        Latent prior
    u2c
        Data classifier
    """

    def __init__(
            self, g2v: sc.GraphEncoder, v2g: sc.GraphDecoder,
            x2u: Mapping[str, sc.DataEncoder],
            u2x: Mapping[str, sc.IndDataDecoder],
            idx: Mapping[str, torch.Tensor],
            du: sc.Discriminator, prior: sc.Prior,
            u2c: Optional[sc.Classifier] = None
    ) -> None:
        super().__init__(g2v, v2g, x2u, u2x, idx, du, prior, u2c)


#----------------------------- Trainer definition ------------------------------

DataTensors = Tuple[
    Mapping[str, torch.Tensor],  # x (data)
    Mapping[str, torch.Tensor],  # xrep (alternative input data)
    Mapping[str, torch.Tensor],  # xbch (data batch)
    Mapping[str, torch.Tensor],  # xlbl (data label)
    Mapping[str, torch.Tensor],  # xdwt (modality discriminator sample weight)
    Mapping[str, torch.Tensor],  # xflag (modality indicator)
    torch.Tensor,  # eidx (edge index)
    torch.Tensor,  # ewt (edge weight)
    torch.Tensor  # esgn (edge sign)
]  # Specifies the data format of input to SCGLUETrainer.compute_losses


@logged
class SCGLUETrainer(GLUETrainer):

    r"""
    Trainer for :class:`SCGLUE`

    Parameters
    ----------
    net
        :class:`SCGLUE` network to be trained
    lam_data
        Data weight
    lam_kl
        KL weight
    lam_graph
        Graph weight
    lam_align
        Adversarial alignment weight
    lam_sup
        Cell type supervision weight
    normalize_u
        Whether to L2 normalize cell embeddings before decoder
    modality_weight
        Relative modality weight (indexed by modality name)
    optim
        Optimizer
    lr
        Learning rate
    **kwargs
        Additional keyword arguments are passed to the optimizer constructor
    """

    BURNIN_NOISE_EXAG: float = 1.5  # Burn-in noise exaggeration
    print('SCGLUETrainer')
    def __init__(
            self, net: SCGLUE, lam_data: float = None, lam_kl: float = None,
            lam_graph: float = None, lam_align: float = None,
            lam_sup: float = None, normalize_u: bool = None,
            modality_weight: Mapping[str, float] = None,
            optim: str = None, lr: float = None,
            Fine_tuning: bool = False,
            # n_clusters: int = None,
            **kwargs
    ) -> None:
        super().__init__(
            net, lam_data=lam_data, lam_kl=lam_kl, lam_graph=lam_graph,
            lam_align=lam_align, modality_weight=modality_weight,
            optim=optim, lr=lr , **kwargs
        )

        # n_clusters = len(adatas['rna'].obs['cell_type'].unique())
        # self.mu = Parameter(torch.Tensor(n_clusters, latent_dim), requires_grad=True)
        # required_kwargs = ("lam_sup", "normalize_u","n_clusters")
        required_kwargs = ("lam_sup", "normalize_u")
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")
        self.lam_sup = lam_sup
        self.normalize_u = normalize_u
        # self.n_clusters = n_clusters
        self.freeze_u = False
        self.vae_optim = getattr(torch.optim, optim)(
            chain(
                self.net.g2v.parameters(),
                self.net.v2g.parameters(),
                self.net.x2u.parameters(),
                self.net.u2x.parameters(),
                self.net.recons_z.parameters(),
                # self.net.u2c.parameters()
            ), lr=self.lr, **kwargs
        )
        # if Fine_tuning :
        #     # self.required_losses.append("sup_loss")
        #     self.vae_optim = getattr(torch.optim, optim)(
        #         chain(
        #             self.net.g2v.parameters(),
        #             self.net.v2g.parameters(),
        #             self.net.x2u.parameters(),
        #             self.net.u2x.parameters(),
        #             self.net.recons_z.parameters(),
        #             # self.net.u2c.parameters()
        #             [self.net.mu]
        #         ), lr=self.lr, **kwargs
        #     )
        # else:
        #     self.vae_optim = getattr(torch.optim, optim)(
        #         chain(
        #             self.net.g2v.parameters(),
        #             self.net.v2g.parameters(),
        #             self.net.x2u.parameters(),
        #             self.net.u2x.parameters(),
        #             self.net.recons_z.parameters(),
        #             # self.net.mu.parameters()
        #         ), lr=self.lr, **kwargs
        #     )
        # for param_group in self.vae_optim.param_groups:
        #     for param in param_group['params']:
        #         print(param.shape)
        # print("0")
        # if Fine_tuning :
        #     self.required_losses.append("sup_loss")
        #     self.vae_optim = getattr(torch.optim, optim)(
        #         chain(
        #             self.net.g2v.parameters(),
        #             self.net.v2g.parameters(),
        #             self.net.x2u.parameters(),
        #             self.net.u2x.parameters(),
        #             self.net.recons_z.parameters(),
        #             self.net.u2c.parameters()
        #         ), lr=self.lr, **kwargs
        #     )
        # else:
        #     self.vae_optim = getattr(torch.optim, optim)(
        #         chain(
        #             self.net.g2v.parameters(),
        #             self.net.v2g.parameters(),
        #             self.net.x2u.parameters(),
        #             self.net.u2x.parameters(),
        #             self.net.recons_z.parameters()
        #         ), lr=self.lr, **kwargs
        #     )

    @property
    def freeze_u(self) -> bool:
        r"""
        Whether to freeze cell embeddings
        """
        return self._freeze_u

    @freeze_u.setter
    def freeze_u(self, freeze_u: bool) -> None:
        self._freeze_u = freeze_u
        for item in chain(self.net.x2u.parameters(), self.net.du.parameters()):
            item.requires_grad_(not self._freeze_u)

    def format_data(self, data: List[torch.Tensor]) -> DataTensors:
        r"""
        Format data tensors

        Note
        ----
        The data dataset should contain data arrays for each modality,
        followed by alternative input arrays for each modality,
        in the same order as modality keys of the network.
        """
        device = self.net.device
        keys = self.net.keys
        K = len(keys)
        x, xrep, xbch, xlbl, xdwt, (eidx, ewt, esgn) = \
            data[0:K], data[K:2*K], data[2*K:3*K], data[3*K:4*K], data[4*K:5*K], \
            data[5*K+1:]
        x = {
            k: x[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xrep = {
            k: xrep[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xbch = {
            k: xbch[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xlbl = {
            k: xlbl[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xdwt = {
            k: xdwt[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xflag = {
            k: torch.as_tensor(
                i, dtype=torch.int64, device=device
            ).expand(x[k].shape[0])
            for i, k in enumerate(keys)
        }
        eidx = eidx.to(device, non_blocking=True)
        ewt = ewt.to(device, non_blocking=True)
        esgn = esgn.to(device, non_blocking=True)
        return x, xrep, xbch, xlbl, xdwt, xflag, eidx, ewt, esgn

    def compute_losses(
            self, data: DataTensors, epoch: int, dsc_only: bool = False
    ) -> Mapping[str, torch.Tensor]:
        net = self.net
        x, xrep, xbch, xlbl, xdwt, xflag, eidx, ewt, esgn = data

        u, l = {}, {}
        z_x_only_samp = {}
        z_x_only ,z_x_only_mu,z_x_only_logvar= {}, {},{}
        for k in net.keys:
            u[k], l[k] = net.x2u[k](x[k], xrep[k], lazy_normalizer=dsc_only)
        usamp = {k: u[k].rsample() for k in net.keys}
        if self.normalize_u:
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}
        prior = net.prior()
        for k in net.keys:
            # Extract means and logvars from the dictionary of Gaussian distributions
            mus_x = u[k].loc.unsqueeze(0)  # 直接访问 mean 属性
            # logvars = torch.stack([2 * torch.log(u[k].std) for k in u.keys()], dim=0)
            logvars_x = (2 * torch.log(u[k].scale + 1e-8)).unsqueeze(0)  # 直接访问 variance 属性，并取对数
            z_x_only_mu[k], z_x_only_logvar[k] = poe(mus_x, logvars_x)
            z_x_only_samp[k] = sample_gaussian(z_x_only_mu[k], z_x_only_logvar[k])
            z_x_only[k] = Normal(z_x_only_mu[k], torch.exp(0.5 * z_x_only_logvar[k]))
        mus_joint = torch.stack([u[k].loc for k in u.keys()], dim=0)  # 直接访问 mean 属性
        logvars_joint = torch.stack([2 * torch.log(u[k].scale + 1e-8) for k in u.keys()],
                                    dim=0)  # 直接访问 variance 属性，并取对数
        z_mu, z_logvar = poe(mus_joint, logvars_joint)
        z_scale = torch.exp(0.5 * z_logvar)
        z = Normal(z_mu, z_scale)
        z_samp_joint = sample_gaussian(z_mu, z_logvar)
        z_cat = z.mean
        # u['rna'].mean
        u_cat = torch.cat([u[k].mean for k in net.keys])
        xbch_cat = torch.cat([xbch[k] for k in net.keys])
        xdwt_cat = torch.cat([xdwt[k] for k in net.keys])
        xflag_cat = torch.cat([xflag[k] for k in net.keys])
        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) \
            if self.align_burnin else 0
        if anneal:
            noise = D.Normal(0, u_cat.std(axis=0)).sample((u_cat.shape[0], ))
            u_cat = u_cat + (anneal * self.BURNIN_NOISE_EXAG) * noise
            # noise = D.Normal(0, z.mean.std(axis=0)).sample((z.mean.shape[0],))
            # z_cat = z.mean + (anneal * self.BURNIN_NOISE_EXAG) * noise
        dsc_loss = F.cross_entropy(net.du(u_cat, xbch_cat), xflag_cat, reduction="none")
        dsc_loss = (dsc_loss * xdwt_cat).sum() / xdwt_cat.numel()
        # losses_dsc = {}
        # for k in net.keys:
        #     losses_dsc[k] = F.cross_entropy(net.du(z_cat, xbch[k]), xflag[k], reduction="none")
        # dsc_loss = torch.cat([losses_dsc[k] for k in net.keys])
        # dsc_loss = (dsc_loss * xdwt_cat).sum() / xdwt_cat.numel()

        if dsc_only:
            return {"dsc_loss": self.lam_align * dsc_loss}
        if net.u2c:
            lmsk = xlbl >= 0
            sup_loss = F.cross_entropy(
                net.u2c(z_cat[lmsk]), xlbl[lmsk], reduction="none"
            ).sum() / max(lmsk.sum(), 1)
        else:
            sup_loss = torch.tensor(0.0, device=self.net.device)

        v = net.g2v(self.eidx, self.enorm, self.esgn)
        vsamp = v.rsample()

        g_nll = -net.v2g(vsamp, eidx, esgn).log_prob(ewt)
        pos_mask = (ewt != 0).to(torch.int64)
        n_pos = pos_mask.sum().item()
        n_neg = pos_mask.numel() - n_pos
        g_nll_pn = torch.zeros(2, dtype=g_nll.dtype, device=g_nll.device)
        g_nll_pn.scatter_add_(0, pos_mask, g_nll)
        avgc = (n_pos > 0) + (n_neg > 0)
        g_nll = (g_nll_pn[0] / max(n_neg, 1) + g_nll_pn[1] / max(n_pos, 1)) / avgc
        g_kl = D.kl_divergence(v, prior).sum(dim=1).mean() / vsamp.shape[0]
        g_elbo = g_nll + self.lam_kl * g_kl

        # x_nll = {
        #     k: -net.u2x[k](
        #         z_sample, vsamp[getattr(net, f"{k}_idx")], xbch[k], l[k]
        #     ).log_prob(x[k]).mean()
        #     for k in net.keys
        # }
        x_only_nll = {
            k: -net.u2x[k](
                z_x_only_samp[k], vsamp[getattr(net, f"{k}_idx")], xbch[k], l[k]
            ).log_prob(x[k]).mean()
            for k in net.keys
        }
        # x_kl = {
        #     k: D.kl_divergence(
        #         z, prior
        #     ).sum(dim=1).mean() / x[k].shape[1]
        #     for k in net.keys
        # }
        # x_elbo = {
        #     k: x_nll[k] + self.lam_kl * x_kl[k]
        #     for k in net.keys
        # }
        # x_elbo_sum = sum(self.modality_weight[k] * x_elbo[k] for k in net.keys)
        x_only_KLD = {
            k:(-0.5 * (1 + z_x_only_logvar[k] - z_x_only_mu[k].pow(2) - z_x_only_logvar[k].exp())).sum() / z_x_only_mu[k].size(0)
            for k in net.keys
        }
        x_only_elbo = {
            k: x_only_nll[k] + self.lam_kl * x_only_KLD[k]
            for k in net.keys
        }
        x_joint_nll = {
            k: -net.u2x[k](
                z_samp_joint, vsamp[getattr(net, f"{k}_idx")], xbch[k], l[k]
            ).log_prob(x[k]).mean()
            for k in net.keys
        }
        x_joint_KLD = {
            k: D.kl_divergence(
                z, prior
            ).sum(dim=1).mean() / x[k].shape[1]
            for k in net.keys
        }
        x_joint_elbo = {
            k: x_joint_nll[k] + self.lam_kl * x_joint_KLD[k]
            for k in net.keys
        }
        x_elbo_sum = sum(self.modality_weight[k] * ( x_only_elbo[k] + x_joint_elbo[k] ) for k in net.keys)
        vae_loss = self.lam_data * x_elbo_sum \
            + self.lam_graph * len(net.keys) * g_elbo \
            + self.lam_sup * sup_loss
        gen_loss = vae_loss - self.lam_align * dsc_loss
        # vae_loss = self.lam_data * x_elbo_sum \
        #            + self.lam_graph * len(net.keys) * g_elbo
        # gen_loss = vae_loss

        losses = {
            "dsc_loss": dsc_loss, "vae_loss": vae_loss, "gen_loss": gen_loss,
            "g_nll": g_nll, "g_kl": g_kl, "g_elbo": g_elbo,"x_joint_elbo": x_joint_elbo
        }
        # losses = {
        #     "vae_loss": vae_loss, "gen_loss": gen_loss,
        #     "g_nll": g_nll, "g_kl": g_kl, "g_elbo": g_elbo,"x_joint_elbo": x_joint_elbo
        # }
        # losses = {
        #      "vae_loss": vae_loss, "g_nll": g_nll, "g_kl": g_kl, "g_elbo": g_elbo, "gen_loss": gen_loss
        # }
        # for k in net.keys:
        #     losses.update({
        #         f"x_{k}_nll": x_only_nll[k],
        #         f"x_{k}_kl": x_only_KLD[k],
        #     })
        if net.u2c:
            losses["sup_loss"] = sup_loss
        return losses

    def train_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        self.net.train()
        data = self.format_data(data)
        epoch = engine.state.epoch
        Fine_tuning = self.Fine_tuning
        if self.freeze_u:
            self.net.x2u.apply(freeze_running_stats)
            self.net.du.apply(freeze_running_stats)
        elif Fine_tuning:  # Discriminator step
            losses = self.compute_losses(data, epoch, dsc_only=True)
            self.net.zero_grad(set_to_none=True)
            losses["dsc_loss"].backward()  # Already scaled by lam_align
            self.dsc_optim.step()

        # Generator step
        losses = self.compute_losses(data, epoch)
        self.net.zero_grad(set_to_none=True)
        # print(losses["gen_loss"].shape)
        losses["gen_loss"].backward()
        self.vae_optim.step()

        return losses

    def __repr__(self):
        vae_optim = repr(self.vae_optim).replace("    ", "  ").replace("\n", "\n  ")
        dsc_optim = repr(self.dsc_optim).replace("    ", "  ").replace("\n", "\n  ")
        return (
            f"{type(self).__name__}(\n"
            f"  lam_graph: {self.lam_graph}\n"
            f"  lam_align: {self.lam_align}\n"
            f"  vae_optim: {vae_optim}\n"
            f"  dsc_optim: {dsc_optim}\n"
            f"  freeze_u: {self.freeze_u}\n"
            f")"
        # return (
        #     f"{type(self).__name__}(\n"
        #     f"  lam_graph: {self.lam_graph}\n"
        #     f"  lam_align: {self.lam_align}\n"
        #     f"  vae_optim: {vae_optim}\n"
        #     f"  freeze_u: {self.freeze_u}\n"
        #     f")"
         )


PairedDataTensors = Tuple[
    Mapping[str, torch.Tensor],  # x (data)
    Mapping[str, torch.Tensor],  # xrep (alternative input data)
    Mapping[str, torch.Tensor],  # xbch (data batch)
    Mapping[str, torch.Tensor],  # xlbl (data label)
    Mapping[str, torch.Tensor],  # xdwt (modality discriminator sample weight)
    Mapping[str, torch.Tensor],  # xflag (modality indicator)
    torch.Tensor,  # pmsk (paired mask)
    torch.Tensor,  # eidx (edge index)
    torch.Tensor,  # ewt (edge weight)
    torch.Tensor  # esgn (edge sign)
]  # Specifies the data format of input to PairedSCGLUETrainer.compute_losses


@logged
class PairedSCGLUETrainer(SCGLUETrainer):

    r"""
    Paired trainer for :class:`SCGLUE`

    Parameters
    ----------
    net
        :class:`SCGLUE` network to be trained
    lam_data
        Data weight
    lam_kl
        KL weight
    lam_graph
        Graph weight
    lam_align
        Adversarial alignment weight
    lam_sup
        Cell type supervision weight
    lam_joint_cross
        Joint cross-prediction weight
    lam_real_cross
        Real cross-prediction weight
    lam_cos
        Cosine similarity weight
    normalize_u
        Whether to L2 normalize cell embeddings before decoder
    modality_weight
        Relative modality weight (indexed by modality name)
    optim
        Optimizer
    lr
        Learning rate
    **kwargs
        Additional keyword arguments are passed to the optimizer constructor
    """

    def __init__(
            self,
            net: SCGLUE,
            # adatas: Mapping[str, AnnData],
            lam_data: float = None, lam_kl: float = None,
            lam_graph: float = None, lam_align: float = None, lam_sup: float = None,
            lam_joint_cross: float = None, lam_real_cross: float = None,
            lam_cos: float = None, normalize_u: bool = None,
            modality_weight: Mapping[str, float] = None,
            optim: str = None, lr: float = None,
            # n_clusters: int = None,
            # z_embedding: Optional[np.ndarray] = None,
            Fine_tuning: bool = False,
            **kwargs
    ) -> None:
        # # 检查并移除重复的 'adatas' 参数
        # if 'adatas' in kwargs:
        #     kwargs.pop('adatas')
        super().__init__(
            net, lam_data=lam_data, lam_kl=lam_kl,
            lam_graph=lam_graph, lam_align=lam_align,
            lam_sup=lam_sup, normalize_u=normalize_u,
            modality_weight=modality_weight,
            optim=optim, lr=lr,
            # Fine_tuning=Fine_tuning,
            **kwargs
        )
        # self.mu = None
        # if Fine_tuning:
        #     required_kwargs = ( "lam_joint_cross", "lam_real_cross", "lam_cos", "n_clusters", "z_embedding", "Fine_tuning" ,"adatas")
        # else:
        #     required_kwargs = ("lam_joint_cross", "lam_real_cross", "lam_cos", "n_clusters", "Fine_tuning","adatas")
        required_kwargs = ("lam_joint_cross", "lam_real_cross", "lam_cos", "Fine_tuning")
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")
        self.lam_joint_cross = lam_joint_cross
        self.lam_real_cross = lam_real_cross
        self.lam_cos = lam_cos
        # self.n_clusters = n_clusters
        # # self.z_embedding = z_embedding
        self.Fine_tuning = Fine_tuning
        # if self.Fine_tuning:
        #     self.z_embedding = z_embedding
        #     self.initialize_cluster_centers(z_embedding, n_clusters)

        # self.z_embedding_list = []  # 用于保存每个 epoch 内的 z_samp_joint
        # self.update_interval = 1  # 可以设为传入参数
        # self.batch_size = 128  # 示例 batch size，可通过传入参数设置
        # self.modalities = {}
        # self.adatas = adatas
        # for k, adata in adatas.items():
        #     data_config = copy.deepcopy(adata.uns[config.ANNDATA_KEY])
        #     self.modalities[k] = data_config
        #     # true_labels = data_config["cell_types"]
        # # self.true_labels = true_labels
        # # self.required_losses += ["x_j_loss", "real_cross_loss", "cos_loss"]
        # # print("Initializing cluster centers with kmeans.")
        # # kmeans = KMeans(n_clusters, n_init=50)
        # # latent
        #
        # # self.y_pred = kmeans.fit_predict(Zdata.data.cpu().numpy())
        # # self.y_pred_last = self.y_pred
        # # self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))

    # def initialize_cluster_centers(self, z_embedding, n_clusters):
    #     print("Initializing cluster centers with k-means.")
    #     kmeans = KMeans(n_clusters=n_clusters, n_init=50)
    #     self.y_pred = kmeans.fit_predict(z_embedding)
    #     self.net.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))
    #     self.y_pred_last = self.y_pred
    #     # if hasattr(self.net, 'mu'):
    #     #     self.net.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))
    #     # else:
    #     #     self.net.mu = torch.nn.Parameter(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))
    #
    # @torch.no_grad()
    # def encode_zembedding(self) -> np.ndarray:
    #     # 使用 self.adatas, self.batch_size, self.n_sample 作为参数进行编码
    #     # self.net.eval()
    #     encoder = {}
    #     data, data_loader = {}, {}
    #     u, locs, scales, result, embedding = {}, {}, {}, {}, {}
    #
    #     # 遍历数据集进行编码
    #     for k, adata in self.adatas.items():
    #         encoder[k] = self.net.x2u[k]
    #         data[k] = AnnDataset(
    #             [adata], [self.modalities[k]],
    #             mode="train", getitem_size=self.batch_size
    #         )
    #         data_loader[k] = DataLoader(
    #             data[k], batch_size=1, shuffle=False,
    #             num_workers=config.DATALOADER_NUM_WORKERS,
    #             pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
    #             persistent_workers=False
    #         )
    #         locs[k], scales[k], result[k] = [], [], []
    #         for x, xrep, *_ in data_loader[k]:
    #             u[k] = encoder[k](
    #                 x.to(self.net.device, non_blocking=True),
    #                 xrep.to(self.net.device, non_blocking=True),
    #                 lazy_normalizer=True
    #             )[0]
    #             loc = u[k].loc.detach().cpu()  # 提取均值
    #             scale = u[k].scale.detach().cpu()  # 提取标准差
    #             locs[k].append(loc)
    #             scales[k].append(scale)
    #             result[k].append(u[k].mean.detach().cpu())
    #
    #         locs[k] = torch.cat(locs[k], dim=0)
    #         scales[k] = torch.cat(scales[k], dim=0)
    #         embedding[k] = torch.cat(result[k]).numpy()
    #
    #     # 计算联合分布
    #     mus_joint = torch.stack([locs[k] for k in u.keys()], dim=0)
    #     logvars_joint = torch.stack([2 * torch.log(scales[k] + 1e-8) for k in u.keys()], dim=0)
    #     # mus_joint = torch.stack([u[k].loc for k in u.keys()], dim=0)
    #     # logvars_joint = torch.stack([2 * torch.log(u[k].scale + 1e-8) for k in u.keys()], dim=0)
    #     z_mu, z_logvar = poe(mus_joint, logvars_joint)
    #     z_embedding = z_mu.detach().cpu()
    #     # z_embedding = z_result.numpy()
    #     # z_embedding = sample_gaussian(z_mu, z_logvar)
    #     # z_result = z_mu.detach().cpu()
    #     # z_embedding = z_result.numpy()
    #
    #     return z_embedding

    def format_data(self, data: List[torch.Tensor]) -> DataTensors:
        r"""
        Format data tensors

        Note
        ----
        The data dataset should contain data arrays for each modality,
        followed by alternative input arrays for each modality,
        in the same order as modality keys of the network.
        """
        device = self.net.device
        keys = self.net.keys
        K = len(keys)
        x, xrep, xbch, xlbl, xdwt, pmsk, (eidx, ewt, esgn) = \
            data[0:K], data[K:2*K], data[2*K:3*K], data[3*K:4*K], data[4*K:5*K], \
            data[5*K], data[5*K+1:]
        x = {
            k: x[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xrep = {
            k: xrep[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xbch = {
            k: xbch[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xlbl = {
            k: xlbl[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xdwt = {
            k: xdwt[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xflag = {
            k: torch.as_tensor(
                i, dtype=torch.int64, device=device
            ).expand(x[k].shape[0])
            for i, k in enumerate(keys)
        }
        # pmsk = pmsk.to(device, non_blocking=True)
        eidx = eidx.to(device, non_blocking=True)
        ewt = ewt.to(device, non_blocking=True)
        esgn = esgn.to(device, non_blocking=True)
        return x, xrep, xbch, xlbl, xdwt, xflag, pmsk, eidx, ewt, esgn

    def kmeans_loss(self, z):
        z = z.to('cuda:1')
        net = self.net
        net.mu = Parameter(net.mu.cuda())
        #print(z.is_cuda, self.mu.is_cuda)
        tau = 1.
        dist1 = tau * torch.sum(torch.square(z.unsqueeze(1) - net.mu), dim=2)
        temp_dist1 = dist1 - torch.reshape(torch.mean(dist1, dim=1), [-1, 1])
        q = torch.exp(-temp_dist1)
        q = (q.t() / torch.sum(q, dim=1)).t()
        q = torch.pow(q, 2)
        q = (q.t() / torch.sum(q, dim=1)).t()
        dist2 = dist1 * q
        return dist1, torch.mean(torch.sum(dist2, dim=1))

    # @ignite.engine.Events.EPOCH_COMPLETED  # 在每个 epoch 完成后执行
    # def on_epoch_completed(self, engine):
    #     adatas = self.model.adatas
    #     if not self.Fine_tuning:  # 如果 Fine_tuning 不为 True, 直接返回
    #         return
    #     print("0")
    #     tol = 1e-3
    #     epoch = engine.state.epoch
    #     # 将 z_embedding_list 中的所有 z_samp_joint 拼接成 z_embedding
    #     z_embedding = torch.cat(self.z_embedding_list, dim=0)
    #     self.z_embedding_list = []  # 清空 z_embedding_list，为下个 epoch 做准备
    #
    #     # 执行 kmeans_loss 计算并保存距离
    #     dist, _ = self.kmeans_loss(z_embedding)
    #     # np.savetxt("GSE126074_embedding.csv", z_embedding.numpy(), delimiter=",")
    #
    #     # 更新 y_pred
    #     self.y_pred = torch.argmin(dist, dim=1).cpu().numpy()
    #
    #     print(f"Epoch {epoch} completed, k-means loss computed.")
    #     # check stop criterion
    #     delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / z_embedding.shape[0]
    #     self.y_pred_last = self.y_pred
    #     if epoch > 0 and delta_label < tol:
    #         print('delta_label ', delta_label, '< tol ', tol)
    #         print("Reach tolerance threshold. Stopping training.")
    #         sys.exit()  # 结束整个程序

    def compute_losses(
            self, data: PairedDataTensors, epoch: int, dsc_only: bool = False
    ) -> Mapping[str, torch.Tensor]:
        net = self.net
        Fine_tuning = self.Fine_tuning
        x, xrep, xbch, xlbl, xdwt, xflag, pmsk, eidx, ewt, esgn = data
        batch_size = x['rna'].size(0)
        u, l = {}, {}
        z_x_only_samp = {}
        z_x_only ,z_x_only_mu,z_x_only_logvar= {}, {}, {}
        mu_x, logvar_x = {}, {}
        for k in net.keys:
            u[k], l[k] = net.x2u[k](x[k], xrep[k], lazy_normalizer=dsc_only)
        usamp = {k: u[k].rsample() for k in net.keys}
        # usamp = {}
        # for k in net.keys:
            # usamp[k] = sample_gaussian(u[k].loc, (2 * torch.log(u[k].scale + 1e-8)))

        # h_dims = usamp['rna'].size(1)
        if self.normalize_u:
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}
        prior = net.prior()
        # experts = ProductOfExperts()
        # mus_joint, logvars_joint = prior_expert((1, batch_size, h_dims), )
        # mu, logvar = prior_expert((1, self., self.n_latents),
        #                           use_cuda=use_cuda)
        for k in net.keys:
            # Extract means and logvars from the dictionary of Gaussian distributions
            # mu_x[k] = u[k].loc.unsqueeze(0) # 直接访问 mean 属性
            # logvar_x[k] = (2 * torch.log(u[k].scale + 1e-8)).unsqueeze(0)  # 直接访问 variance 属性，并取对数

            mu_x[k], logvar_x[k] = prior_expert((1, batch_size, 50), )
            mu_x[k] = torch.cat((mu_x[k], u[k].loc.unsqueeze(0)), dim=0)
            logvar_x[k] = torch.cat((logvar_x[k], (2 * torch.log(u[k].scale)+ 1e-8).unsqueeze(0)), dim=0)
            z_x_only_mu[k], z_x_only_logvar[k] = poe(mu_x[k], logvar_x[k])
            # mus_joint = torch.cat((mus_joint, u[k].loc.unsqueeze(0)), dim=0)
            # logvars_joint = torch.cat((logvars_joint, (2 * torch.log(u[k].scale)+ 1e-8).unsqueeze(0)), dim=0)
            # mu_x[k], logvar_x[k] = experts(mu_x[k], logvar_x[k])
            # z_x_only_samp[k] = sample_gaussian(mu_x[k], logvar_x[k])
            # z_x_only_samp[k] = sample_gaussian(z_x_only_mu[k], z_x_only_logvar[k])
            z_x_only[k] = Normal(z_x_only_mu[k], torch.exp(0.5 * z_x_only_logvar[k]))
            z_x_only_samp[k] = z_x_only[k].rsample()

        # mus_joint = torch.stack([u[k].loc for k in u.keys()], dim=0)  # 直接访问 mean 属性
        # logvars_joint = torch.stack([2 * torch.log(u[k].scale + 1e-8) for k in u.keys()], dim=0)# 直接访问 variance 属性，并取对数
        mus_joint, logvars_joint = prior_expert((1, batch_size, 50), )
        for k in net.keys:
            mus_joint =torch.cat((mus_joint, u[k].loc.unsqueeze(0)), dim=0)
            logvars_joint = torch.cat((logvars_joint, (2 * torch.log(u[k].scale)+ 1e-8).unsqueeze(0)), dim=0)  # 直接访问 variance 属性，并取对数
        # z_mu, z_logvar = experts(mus_joint, logvars_joint)

        z_mu, z_logvar = poe(mus_joint,logvars_joint)
        z_scale = torch.exp(0.5 * z_logvar)
        z = Normal(z_mu, z_scale)
        # z_samp_joint = sample_gaussian(z_mu, z_logvar)
        z_samp_joint = z.rsample()
        # print(z_mu.shape)
        # self.y_pred = kmeans.fit_predict(z_samp_joint.data.cpu().numpy())
        # zsample = {k: z_sample for k in net.keys}
        #u['rna'].mean
        # n_clusters = self.n_clusters
        u_cat = torch.cat([u[k].mean for k in net.keys])
        # u_cat = torch.cat([z_x_only_mu[k] for k in net.keys])
        xbch_cat = torch.cat([xbch[k] for k in net.keys])
        xdwt_cat = torch.cat([xdwt[k] for k in net.keys])
        xflag_cat = torch.cat([xflag[k] for k in net.keys])
        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) \
            if self.align_burnin else 0
        # print(anneal)
        if anneal:
            noise = D.Normal(0, u_cat.std(axis=0)).sample((u_cat.shape[0], ))
            u_cat = u_cat + (anneal * self.BURNIN_NOISE_EXAG) * noise
            # noise = D.Normal(0, z.mean.std(axis=0)).sample((z.mean.shape[0],))
            # z_cat = z.mean + (anneal * self.BURNIN_NOISE_EXAG) * noise

        if Fine_tuning:
            dsc_loss = F.cross_entropy(net.du(u_cat, xbch_cat), xflag_cat, reduction="none")
            dsc_loss = (dsc_loss * xdwt_cat).sum() / xdwt_cat.numel()
            if dsc_only:
                return {"dsc_loss": self.lam_align * dsc_loss}
        else:
            dsc_loss = torch.tensor(0.0, device=self.net.device)
        # print(x[k].shape for k in net.keys)
        # losses_dsc = {}
        # for k in net.keys:
        #     losses_dsc[k] = F.cross_entropy(net.du(z_cat, xbch[k]), xflag[k], reduction="none")
        # dsc_loss = torch.cat([losses_dsc[k] for k in net.keys])
        # # dsc_loss = (dsc_loss * xdwt_cat).sum() / xdwt_cat.numel()
        # if dsc_only:
        #     return {"dsc_loss": self.lam_align * dsc_loss}
        # dsc_loss = torch.tensor(0.0, device=self.net.device)
        v = net.g2v(self.eidx, self.enorm, self.esgn)
        vsamp = v.rsample()

        # if Fine_tuning:
        #     xlbl_cat = torch.cat([xlbl[k] for k in net.keys])
        #     lmsk = xlbl_cat >= 0
        #     sup_loss = F.cross_entropy(
        #         net.u2c(u_cat[lmsk]), xlbl_cat[lmsk], reduction="none"
        #     ).sum() / max(lmsk.sum(), 1)
        # else:
        #     sup_loss = torch.tensor(0.0, device=self.net.device)
        # if net.u2c:
        #     lmsk = xlbl >= 0
        #     sup_loss = F.cross_entropy(
        #         net.u2c(z_noise[lmsk]), xlbl[lmsk], reduction="none"
        #     ).sum() / max(lmsk.sum(), 1)
        # else:
        #     sup_loss = torch.tensor(0.0, device=self.net.device)
        sup_loss = torch.tensor(0.0, device=self.net.device)
        g_nll = -net.v2g(vsamp, eidx, esgn).log_prob(ewt)
        pos_mask = (ewt != 0).to(torch.int64)
        n_pos = pos_mask.sum().item()
        n_neg = pos_mask.numel() - n_pos
        g_nll_pn = torch.zeros(2, dtype=g_nll.dtype, device=g_nll.device)
        g_nll_pn.scatter_add_(0, pos_mask, g_nll)
        avgc = (n_pos > 0) + (n_neg > 0)
        g_nll = (g_nll_pn[0] / max(n_neg, 1) + g_nll_pn[1] / max(n_pos, 1)) / avgc
        g_kl = D.kl_divergence(v, prior).sum(dim=1).mean() / vsamp.shape[0]
        g_elbo = g_nll + self.lam_kl * g_kl

        # 消融引导图
        # g_elbo = torch.tensor(0.0, device=self.net.device)

        # x_nll = {
        #     k: -net.u2x[k](
        #         usamp[k], vsamp[getattr(net, f"{k}_idx")], xbch[k], l[k]
        #     ).log_prob(x[k]).mean()
        #     for k in net.keys
        # }
        # x_only_nll = {
        #     k: -net.u2x[k](
        #         (z_x_only_samp[k] @ vsamp[getattr(net, f"{k}_idx")].t()), xbch[k], l[k]
        #     ).log_prob(x[k]).mean()
        #     for k in net.keys
        # }
        # dims_rna = vsamp[getattr(net, "rna_idx")].shape
        # dims_atac = vsamp[getattr(net, "atac_idx")].shape
        # vsamp_rna = torch.randn(dims_rna)  # 平均值 (loc) 使用标准正态分布随机生成
        # vsamp_atac = torch.rand(dims_atac)  # 标准差 (scale) 使用 [0, 1) 的随机值

        x_only_nll = {
            k: -net.u2x[k](
                (usamp[k] @ vsamp[getattr(net, f"{k}_idx")].t()), xbch[k], l[k]
            ).log_prob(x[k]).mean()
            for k in net.keys
        }

        # # 消融引导图
        # dims = {
        #     k: vsamp[getattr(net, f"{k}_idx")].shape
        #     for k in net.keys
        # }
        #
        # vsample_ = {
        #     k: torch.ones(dims[k]).to(self.net.device)  # 将每个张量初始化为全1
        #     for k in self.net.keys
        # }
        # x_only_nll = {
        #     k: -net.u2x[k](
        #         (usamp[k] @ vsample_[k].t()), xbch[k], l[k]
        #     ).log_prob(x[k]).mean()
        #     for k in net.keys
        # }

        # recon_x = {
        #     k: net.u2x[k](
        #         z_x_only_samp[k], vsamp[getattr(net, f"{k}_idx")], xbch[k], l[k]
        #     )
        #     for k in net.keys
        # }
        # x_only_mse = {
        #     k:nnf.mse_loss(recon_x[k], x[k], reduction='sum')
        #     for k in net.keys
        # }
        # x_only_KLD = {
        #     k: D.kl_divergence(
        #         z_x_only[k], prior
        #     ).sum(dim=1).mean() / x[k].shape[1]
        #     for k in net.keys
        # }
        x_only_KLD = {
            k: D.kl_divergence(
                u[k], prior
            ).sum(dim=1).mean() / x[k].shape[1]
            for k in net.keys
        }
        # x_only_KLD = {}
        # u_logvar, u_mu = {},{}
        # for k in net.keys:
        #     u_logvar[k] = (2 * torch.log(u[k].scale) + 1e-8).unsqueeze(0)
        #     u_mu[k] = u[k].loc.unsqueeze(0)
        #     x_only_KLD[k] = torch.mean(
        #         -0.5 * torch.sum(1 + u_logvar[k] - u_mu[k].pow(2) - u_logvar[k].exp(), dim=1)) / x[k].shape[1]
        # for k in net.keys:
        #     x_only_KLD[k] = torch.mean(-0.5 * torch.sum(1 + z_x_only_logvar[k] - z_x_only_mu[k].pow(2) - z_x_only_logvar[k].exp(), dim=1)) / x[k].shape[1]
        # x_only_KLD = sum((-0.5 * (1 + z_x_only_logvar[k] - z_x_only_mu[k].pow(2) - z_x_only_logvar[k].exp())).sum() / z_x_only_mu[k].size(0) for k in net.keys)
        # x_only_KLD = {
        #     k:(-0.5 * torch.sum(1 + logvar_x[k] - mu_x[k].pow(2) - logvar_x[k].exp(), dim=1))
        #     for k in net.keys
        # }
        # x_elbo = {
        #     k: x_nll[k] + self.lam_kl * x_kl[k]
        #     for k in net.keys
        # }
        x_only_elbo = {
            k: x_only_nll[k] +  self.lam_kl * x_only_KLD[k]
            for k in net.keys
        }
        # x_only_elbo =  {
        #     k: torch.mean(x_only_mse[k]+ self.lam_kl * x_only_KLD[k])
        #     for k in net.keys
        # }
        # x_only_elbo_sum = sum(self.modality_weight[k] * x_only_elbo[k] for k in net.keys)
        # vsamp
        # v_get = vsamp[getattr(net, f"{'rna'}_idx")]

        z_hat = {}
        net.cuda()

        for k in net.keys:
            z_hat[k] = net.recons_z[k](z_samp_joint)


        # # 消融-poe
        # z_samp_joint_ = torch.cat([usamp['rna'], usamp['atac']], dim=1)
        # for k in net.keys:
        #     z_hat[k] = net.recons_z[k](z_samp_joint_)

        x_joint_nll = {
            k: -net.u2x[k](
                z_hat[k], xbch[k], l[k]
            ).log_prob(x[k]).mean()
            for k in net.keys
        }

        # x_joint_nll = {
        #     k: -net.u2x[k](
        #         z_samp_joint, vsamp[getattr(net, f"{k}_idx")], xbch[k], l[k]
        #     ).log_prob(x[k]).mean()
        #     for k in net.keys
        # }
        # recon_z = {
        #     k: net.u2x[k](
        #         z_samp_joint, vsamp[getattr(net, f"{k}_idx")], xbch[k], l[k]
        #     )
        #     for k in net.keys
        # }

        # # 消融-poe
        # x_joint_mse = {
        #     k : nnf.mse_loss(z_hat[k], x[k], reduction='mean')
        #     for k in net.keys
        # }

        # x_joint_KLD = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1)
        # x_joint_KLD = {}
        # for k in net.keys:
        #     x_joint_KLD[k] = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1))/ x[k].shape[1]
        x_joint_KLD = {
            k: D.kl_divergence(z, prior).sum(dim=1).mean() / x[k].shape[1]
            for k in net.keys
        }
        # x_joint_elbo = torch.mean(x_joint_mse+ self.lam_kl * x_joint_KLD)

        x_joint_elbo = {
            k: x_joint_nll[k]+ self.lam_kl * x_joint_KLD[k]
               # * 0.5
            for k in net.keys
        }

        # # 消融-poe
        # x_joint_elbo = {
        #     k: x_joint_nll[k]+ self.lam_kl * x_joint_mse[k]
        #        # * 0.5
        #     for k in net.keys
        # }
        x_elbo_sum = sum(self.modality_weight[k] * (x_only_elbo[k]+ x_joint_elbo[k]) for k in net.keys)

        # pmsk = pmsk.T
        # # usamp_stack = torch.stack([usamp[k] for k in net.keys])
        # # pmsk_stack = pmsk.unsqueeze(2).expand_as(usamp_stack)
        # # zsamp = {k: z_sample for k in net.keys}
        # zsamp_stack = z_samp_joint
        # # 打印张量的形状以检查
        # # 调整 pmsk 的形状以匹配 zsamp_stack
        # # 添加一个额外的维度，并扩展它以匹配 zsamp_stack 的形状
        # pmsk_expanded = pmsk.unsqueeze(2).expand(-1, -1, zsamp_stack.size(1))  # 形状变为 [2, 128, 50]
        # # 如果需要，你可以选择最后一个维度的部分
        # pmsk_stack = pmsk_expanded[0]  # 选择第一个部分，形状为 [128, 50]
        # # pmsk_stack = pmsk.unsqueeze(2).expand_as(zsamp_stack)
        # pmsk_stack = pmsk_stack.to(zsamp_stack.device)
        # # print(zsamp_stack.device)
        # # print(pmsk_stack.device)
        # # usamp_mean = (usamp_stack * pmsk_stack).sum(dim=0) / pmsk_stack.sum(dim=0)
        # zsamp_mean = (zsamp_stack * pmsk_stack).sum(dim=0) / pmsk_stack.sum(dim=0)
        # if self.normalize_u:
        #     usamp_mean = F.normalize(usamp_mean, dim=1)

        # if self.lam_joint_cross:
        #     x_joint_cross_nll = {
        #         k: -net.u2x[k](
        #             usamp_mean[m], vsamp[getattr(net, f"{k}_idx")],
        #             xbch[k][m], None if l[k] is None else l[k][m]
        #         ).log_prob(x[k][m]).mean()
        #         for k, m in zip(net.keys, pmsk) if m.sum()
        #     }
        #     joint_cross_loss = sum(
        #         self.modality_weight[k] * nll
        #         for k, nll in x_joint_cross_nll.items()
        #     )
        # else:
        #     joint_cross_loss = torch.as_tensor(0.0, device=net.device)
        # if self.lam_joint_cross:
        #     x_joint_cross_nll = {
        #         k: -net.u2x[k](
        #             zsamp_mean[m], vsamp[getattr(net, f"{k}_idx")],
        #             xbch[k][m], None if l[k] is None else l[k][m]
        #         ).log_prob(x[k][m]).mean()
        #         for k, m in zip(net.keys, pmsk) if m.sum()
        #     }
        #     joint_cross_loss = sum(
        #         self.modality_weight[k] * nll
        #         for k, nll in x_joint_cross_nll.items()
        #     )
        # else:
        #     joint_cross_loss = torch.as_tensor(0.0, device=net.device)
        joint_cross_loss = torch.as_tensor(0.0, device=net.device)
        # if self.lam_real_cross:
        #     x_real_cross_nll = {}
        #     for k_tgt, m_tgt in zip(net.keys, pmsk):
        #         x_tgt_real_cross_nll = torch.as_tensor(0.0, device=net.device)
        #         for k_src, m_src in zip(net.keys, pmsk):
        #             if k_src == k_tgt:
        #                 continue
        #             m = m_src & m_tgt
        #             if m.sum():
        #                 x_tgt_real_cross_nll += -net.u2x[k_tgt](
        #                     usamp[k_src][m], vsamp[getattr(net, f"{k_tgt}_idx")],
        #                     xbch[k_tgt][m], None if l[k_tgt] is None else l[k_tgt][m]
        #                 ).log_prob(x[k_tgt][m]).mean()
        #         x_real_cross_nll[k_tgt] = x_tgt_real_cross_nll
        #     real_cross_loss = sum(
        #         self.modality_weight[k] * nll
        #         for k, nll in x_real_cross_nll.items()
        #     )
        # else:
        #     real_cross_loss = torch.as_tensor(0.0, device=net.device)
        # if self.lam_real_cross:
        #     x_real_cross_nll = {}
        #     for k_tgt, m_tgt in zip(net.keys, pmsk):
        #         x_tgt_real_cross_nll = torch.as_tensor(0.0, device=net.device)
        #         for k_src, m_src in zip(net.keys, pmsk):
        #             if k_src == k_tgt:
        #                 continue
        #             m = m_src & m_tgt
        #             if m.sum():
        #                 x_tgt_real_cross_nll += -net.u2x[k_tgt](
        #                     zsamp[k_src][m], vsamp[getattr(net, f"{k_tgt}_idx")],
        #                     xbch[k_tgt][m], None if l[k_tgt] is None else l[k_tgt][m]
        #                 ).log_prob(x[k_tgt][m]).mean()
        #         x_real_cross_nll[k_tgt] = x_tgt_real_cross_nll
        #     real_cross_loss = sum(
        #         self.modality_weight[k] * nll
        #         for k, nll in x_real_cross_nll.items()
        #     )
        # else:
        #     real_cross_loss = torch.as_tensor(0.0, device=net.device)
        real_cross_loss = torch.as_tensor(0.0, device=net.device)
        # if self.lam_cos:
        #     cos_loss = sum(
        #         1 - F.cosine_similarity(
        #             usamp_stack[i, m], usamp_mean[m]
        #         ).mean()
        #         for i, m in enumerate(pmsk) if m.sum()
        #     )
        # else:
        #     cos_loss = torch.as_tensor(0.0, device=net.device)
        # if self.lam_cos:
        #     cos_loss = sum(
        #         1 - F.cosine_similarity(
        #             zsamp_stack[i, m], zsamp_mean[m]
        #         ).mean()
        #         for i, m in enumerate(pmsk) if m.sum()
        #     )
        # else:
        #     cos_loss = torch.as_tensor(0.0, device=net.device)
        cos_loss = torch.as_tensor(0.0, device=net.device)
        vae_loss = self.lam_data * x_elbo_sum \
            + self.lam_graph * len(net.keys) * g_elbo \
            # + self.lam_sup * sup_loss \
            # + self.lam_joint_cross * joint_cross_loss \
            # + self.lam_real_cross * real_cross_loss \
            # + self.lam_cos * cos_loss
        # print(self.lam_graph)
        # if Fine_tuning:
            # if z_samp_joint is not None:
            #     self.z_embedding_list.append(z_samp_joint.detach().cpu())
            # kmeans = KMeans(n_clusters, n_init=50)
        #     _, cluster_loss = self.kmeans_loss(z_samp_joint)
        # else:
        #     cluster_loss = torch.as_tensor(0.0, device=net.device)
        # gen_loss = vae_loss + 0.5 * cluster_loss
        # vae_loss = self.lam_data * x_elbo_sum \
        #     + self.lam_graph * len(net.keys) * g_elbo \
        #     + self.lam_joint_cross * joint_cross_loss \
        #     + self.lam_real_cross * real_cross_loss \
        #     + self.lam_cos * cos_loss
        gen_loss = vae_loss - self.lam_align * dsc_loss
        # gen_loss = vae_loss

        losses = {
            "dsc_loss": dsc_loss,
            "vae_loss": vae_loss, "gen_loss": gen_loss,
            "g_nll": g_nll, "g_kl": g_kl, "g_elbo": g_elbo,
            # "joint_cross_loss": joint_cross_loss,
            # "real_cross_loss": real_cross_loss,
            # "cos_loss": cos_loss,
            # "joint_kld": x_joint_KLD,
            # "joint_elbo": x_joint_elbo,
            "elbo_sum": x_elbo_sum,
            # "cluster_loss": cluster_loss
        }
        for k in net.keys:
            losses.update({
                f"x_only_{k}_nll": x_only_nll[k],
                f"x_only_{k}_kld": x_only_KLD[k],
                f"x_only_{k}_elbo": x_only_elbo[k],
                f"x_joint_{k}_nll": x_joint_nll[k],
                f"x_joint_{k}_kld": x_joint_KLD[k],
                f"x_joint_{k}_elbo": x_joint_elbo[k]

            })
        # if net.u2c:
        #     losses["sup_loss"] = sup_loss
        return losses


#--------------------------------- Public API ----------------------------------

@logged
class SCGLUEModel(Model):

    r"""
    GLUE model for single-cell multi-omics data integration

    Parameters
    ----------
    adatas
        Datasets (indexed by modality name)
    vertices
        Guidance graph vertices (must cover feature names in all modalities)
    latent_dim
        Latent dimensionality
    h_depth
        Hidden layer depth for encoder and discriminator
    h_dim
        Hidden layer dimensionality for encoder and discriminator
    dropout
        Dropout rate
    shared_batches
        Whether the same batches are shared across modalities
    random_seed
        Random seed
    """

    NET_TYPE = SCGLUE
    TRAINER_TYPE = SCGLUETrainer

    GRAPH_BATCHES: int = 32  # Number of graph batches in each graph epoch
    ALIGN_BURNIN_PRG: float = 8.0  # Effective optimization progress of align_burnin (learning rate * iterations)
    MAX_EPOCHS_PRG: float = 48.0  # Effective optimization progress of max_epochs (learning rate * iterations)
    PATIENCE_PRG: float = 4.0  # Effective optimization progress of patience (learning rate * iterations)
    REDUCE_LR_PATIENCE_PRG: float = 2.0  # Effective optimization progress of reduce_lr_patience (learning rate * iterations)

    def __init__(
            self ,adatas: Mapping[str, AnnData],
            vertices: List[str], latent_dim: int = 50,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2, shared_batches: bool = False,
            random_seed: int = 0
    ) -> None:
        # n_clusters = len(adatas['rna'].obs['cell_type'].unique())
        # # mu = torch.Tensor(n_clusters, latent_dim)
        # # self.net.mu = torch.randn(n_clusters, latent_dim)
        # # mu = torch.randn(n_clusters, latent_dim)  # 初始化 mu 为随机值
        # # self.mu = torch.nn.Parameter(self.mu.cuda())
        # # self.adatas = adatas
        # # mu = Parameter(torch.Tensor(n_clusters, latent_dim), requires_grad=True)
        # # 初始化 mu 为随机值，并设置为可训练的参数
        # mu = Parameter(torch.randn(n_clusters, latent_dim), requires_grad=True)
        # random_seed = int(os.getenv("ITERATION", "1"))
        self.vertices = pd.Index(vertices)
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        # self.n_clusters = len(adatas['rna'].obs['cell_type'].unique())
        g2v = sc.GraphEncoder(self.vertices.size, latent_dim)
        v2g = sc.GraphDecoder()
        self.modalities, idx, x2u, u2x, all_ct = {}, {}, {}, {}, set()
        recons_z = {}
        # features_sum = 0
        for k, adata in adatas.items():
            if config.ANNDATA_KEY not in adata.uns:
                raise ValueError(
                    f"The '{k}' dataset has not been configured. "
                    f"Please call `configure_dataset` first!"
                )
            data_config = copy.deepcopy(adata.uns[config.ANNDATA_KEY])
            if data_config["rep_dim"] and data_config["rep_dim"] < latent_dim:
                self.logger.warning(
                    "It is recommended that `use_rep` dimensionality "
                    "be equal or larger than `latent_dim`."
                )
            idx[k] = self.vertices.get_indexer(data_config["features"]).astype(np.int64)
            if idx[k].min() < 0:
                raise ValueError("Not all modality features exist in the graph!")
            idx[k] = torch.as_tensor(idx[k])
            x2u[k] = _ENCODER_MAP[data_config["prob_model"]](
                data_config["rep_dim"] or len(data_config["features"]), latent_dim,
                h_depth=h_depth, h_dim=h_dim, dropout=dropout
            )
            data_config["batches"] = pd.Index([]) if data_config["batches"] is None \
                else pd.Index(data_config["batches"])
            u2x[k] = _DECODER_MAP[data_config["prob_model"]](
                len(data_config["features"]),
                n_batches=max(data_config["batches"].size, 1)
            )
            # #消融-poe
            # recons_z[k] = sc.Reconstruct_data(len(data_config["features"]), latent_dim*2)
            recons_z[k] = sc.Reconstruct_data(len(data_config["features"]),latent_dim)
            all_ct = all_ct.union(
                set() if data_config["cell_types"] is None
                else data_config["cell_types"]
            )
            self.modalities[k] = data_config
        all_ct = pd.Index(all_ct).sort_values()
        for modality in self.modalities.values():
            modality["cell_types"] = all_ct
        if shared_batches:
            all_batches = [modality["batches"] for modality in self.modalities.values()]
            ref_batch = all_batches[0]
            for batches in all_batches:
                if not np.array_equal(batches, ref_batch):
                    raise RuntimeError("Batches must match when using `shared_batches`!")
            du_n_batches = ref_batch.size
        else:
            du_n_batches = 0
        du = sc.Discriminator(
            latent_dim, len(self.modalities), n_batches=du_n_batches,
            h_depth=h_depth, h_dim=h_dim, dropout=dropout
        )
        prior = sc.Prior()
        super().__init__(
            g2v, v2g, x2u, u2x, idx, du, prior, recons_z,
            # mu,
            u2c=None if all_ct.empty else sc.Classifier(latent_dim, all_ct.size)
        )
        # super().__init__(
        #     g2v, v2g, x2u, u2x, idx, prior,recons_z, mu,
        #     u2c=None if all_ct.empty else sc.Classifier(latent_dim, all_ct.size)
        # )
        # super().__init__(
        #     g2v, v2g, x2u, u2x, idx, prior, recons_z,
        #     u2c=None if all_ct.empty else sc.Classifier(latent_dim, all_ct.size)
        # )


    def freeze_cells(self) -> None:
        r"""
        Freeze cell embeddings
        """
        self.trainer.freeze_u = True

    def unfreeze_cells(self) -> None:
        r"""
        Unfreeze cell embeddings
        """
        self.trainer.freeze_u = False

    def adopt_pretrained_model(
            self, source: "SCGLUEModel", submodule: Optional[str] = None
    ) -> None:
        r"""
        Adopt buffers and parameters from a pretrained model

        Parameters
        ----------
        source
            Source model to be adopted
        submodule
            Only adopt a specific submodule (e.g., ``"x2u"``)
        """
        source, target = source.net, self.net
        if submodule:
            source = get_chained_attr(source, submodule)
            target = get_chained_attr(target, submodule)
        for k, t in chain(target.named_parameters(), target.named_buffers()):
            try:
                s = get_chained_attr(source, k)
            except AttributeError:
                self.logger.warning("Missing: %s", k)
                continue
            if isinstance(t, torch.nn.Parameter):
                t = t.data
            if isinstance(s, torch.nn.Parameter):
                s = s.data
            if s.shape != t.shape:
                self.logger.warning("Shape mismatch: %s", k)
                continue
            s = s.to(device=t.device, dtype=t.dtype)
            t.copy_(s)
            self.logger.debug("Copied: %s", k)

    # def initialize_cluster_centers(self, adatas, n_clusters):
    #     print("Initializing cluster centers with kmeans.")
    #     z_embedding = adatas['rna'].obsm["Z_latent"]
    #     if isinstance(z_embedding, torch.Tensor):
    #         z_embedding = z_embedding.detach().cpu().numpy()
    #
    #     kmeans = KMeans(n_clusters=n_clusters, n_init=50)
    #     self.y_pred = kmeans.fit_predict(z_embedding)
    #     self.y_pred_last = self.y_pred
    #
    #     if hasattr(self.net, 'mu'):
    #         self.net.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))
    #     else:
    #         self.net.mu = torch.nn.Parameter(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))

    def compile(  # pylint: disable=arguments-differ
            self,
            # adatas: Mapping[str, AnnData],
            lam_data: float = 1.0,
            lam_kl: float = 1.0,
            lam_graph: float = 0.02,
            # lam_graph: float = 0.001,
            # lam_graph: float = 0.005,
            lam_align: float = 0.05,
            lam_sup: float = 0.02,
            normalize_u: bool = False,
            modality_weight: Optional[Mapping[str, float]] = None,
            lr: float = 2e-3,
            # n_clusters: int = None,
            **kwargs
    ) -> None:
        r"""
        Prepare model for training

        Parameters
        ----------
        n_clusters
        adatas
        lam_data
            Data weight
        lam_kl
            KL weight
        lam_graph
            Graph weight
        lam_align
            Adversarial alignment weight
        lam_sup
            Cell type supervision weight
        normalize_u
            Whether to L2 normalize cell embeddings before decoder
        modality_weight
            Relative modality weight (indexed by modality name)
        lr
            Learning rate
        **kwargs
            Additional keyword arguments passed to trainer
        """
        # n_clusters = self.n_clusters
        if modality_weight is None:
            modality_weight = {k: 1.0 for k in self.net.keys}
        # if 'adatas' in kwargs:
        #     kwargs.pop('adatas')
        super().compile(
            lam_data=lam_data, lam_kl=lam_kl,
            lam_graph=lam_graph, lam_align=lam_align, lam_sup=lam_sup,
            normalize_u=normalize_u, modality_weight=modality_weight,
            optim="RMSprop", lr=lr,
            # adatas = adatas,
            **kwargs
        )

    def fit(  # pylint: disable=arguments-differ
            self, adatas: Mapping[str, AnnData], graph: nx.Graph,
            neg_samples: int = 10, val_split: float = 0.1,
            data_batch_size: int = 128, graph_batch_size: int = AUTO,
            align_burnin: int = AUTO, safe_burnin: bool = True,
            max_epochs: int = AUTO, patience: Optional[int] = AUTO,
            reduce_lr_patience: Optional[int] = AUTO,
            wait_n_lrs: int = 1, directory: Optional[os.PathLike] = None,
    ) -> None:
        r"""
        Fit model on given datasets

        Parameters
        ----------
        adatas
            Datasets (indexed by modality name)
        graph
            Guidance graph
        neg_samples
            Number of negative samples for each edge
        val_split
            Validation split
        data_batch_size
            Number of cells in each data minibatch
        graph_batch_size
            Number of edges in each graph minibatch
        align_burnin
            Number of epochs to wait before starting alignment
        safe_burnin
            Whether to postpone learning rate scheduling and earlystopping
            until after the burnin stage
        max_epochs
            Maximal number of epochs
        patience
            Patience of early stopping
        reduce_lr_patience
            Patience to reduce learning rate
        wait_n_lrs
            Wait n learning rate scheduling events before starting early stopping
        directory
            Directory to store checkpoints and tensorboard logs
        """
        # self.mu = Parameter(torch.Tensor(self.n_clusters, 50), requires_grad=True)
        # print("Initializing cluster centers with kmeans.")
        # kmeans = KMeans(self.n_clusters, n_init=50)
        # Zdata = adatas['rna'].obsm["Z_latent"]
        # # latent
        # self.y_pred = kmeans.fit_predict(Zdata.data.cpu().numpy())
        # self.y_pred_last = self.y_pred
        # self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        # self.n_clusters = len(adatas['rna'].obs['cell_type'].unique())
        # # 初始化聚类中心
        # self.initialize_cluster_centers(adatas, n_clusters)
        data = AnnDataset(
            [adatas[key] for key in self.net.keys],
            [self.modalities[key] for key in self.net.keys],
            mode="train"
        )
        check_graph(
            graph, adatas.values(),
            cov="ignore", attr="error", loop="warn", sym="warn"
        )
        # self.trainer.
        graph = GraphDataset(
            graph, self.vertices, neg_samples=neg_samples,
            weighted_sampling=True, deemphasize_loops=True
        )

        batch_per_epoch = data.size * (1 - val_split) / data_batch_size
        if graph_batch_size == AUTO:
            graph_batch_size = ceil(graph.size / self.GRAPH_BATCHES)
            self.logger.info("Setting `graph_batch_size` = %d", graph_batch_size)
        if align_burnin == AUTO:
            align_burnin = max(
                ceil(self.ALIGN_BURNIN_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.ALIGN_BURNIN_PRG)
            )
            self.logger.info("Setting `align_burnin` = %d", align_burnin)
        if max_epochs == AUTO:
            max_epochs = max(
                ceil(self.MAX_EPOCHS_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.MAX_EPOCHS_PRG)
            )
            self.logger.info("Setting `max_epochs` = %d", max_epochs)
        if patience == AUTO:
            patience = max(
                ceil(self.PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.PATIENCE_PRG)
            )
            self.logger.info("Setting `patience` = %d", patience)
        if reduce_lr_patience == AUTO:
            reduce_lr_patience = max(
                ceil(self.REDUCE_LR_PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.REDUCE_LR_PATIENCE_PRG)
            )
            self.logger.info("Setting `reduce_lr_patience` = %d", reduce_lr_patience)

        if self.trainer.freeze_u:
            self.logger.info("Cell embeddings are frozen")

        super().fit(
            data, graph, val_split=val_split,
            data_batch_size=data_batch_size, graph_batch_size=graph_batch_size,
            align_burnin=align_burnin, safe_burnin=safe_burnin,
            max_epochs=max_epochs, patience=patience,
            reduce_lr_patience=reduce_lr_patience, wait_n_lrs=wait_n_lrs,
            random_seed=self.random_seed,
            directory=directory
        )

    @torch.no_grad()
    def get_losses(  # pylint: disable=arguments-differ
            self, adatas: Mapping[str, AnnData], graph: nx.Graph,
            neg_samples: int = 10, data_batch_size: int = 128,
            graph_batch_size: int = AUTO
    ) -> Mapping[str, np.ndarray]:
        r"""
        Compute loss function values

        Parameters
        ----------
        adatas
            Datasets (indexed by modality name)
        graph
            Guidance graph
        neg_samples
            Number of negative samples for each edge
        data_batch_size
            Number of cells in each data minibatch
        graph_batch_size
            Number of edges in each graph minibatch

        Returns
        -------
        losses
            Loss function values
        """
        print('get_losses')
        data = AnnDataset(
            [adatas[key] for key in self.net.keys],
            [self.modalities[key] for key in self.net.keys],
            mode="train"
        )
        graph = GraphDataset(
            graph, self.vertices,
            neg_samples=neg_samples,
            weighted_sampling=True,
            deemphasize_loops=True
        )
        if graph_batch_size == AUTO:
            graph_batch_size = ceil(graph.size / self.GRAPH_BATCHES)
            self.logger.info("Setting `graph_batch_size` = %d", graph_batch_size)
        return super().get_losses(
            data, graph, data_batch_size=data_batch_size,
            graph_batch_size=graph_batch_size,
            random_seed=self.random_seed
        )

    @torch.no_grad()
    def encode_graph(
            self, graph: nx.Graph, n_sample: Optional[int] = None
    ) -> np.ndarray:
        r"""
        Compute graph (feature) embedding

        Parameters
        ----------
        graph
            Input graph
        n_sample
            Number of samples from the embedding distribution,
            by default ``None``, returns the mean of the embedding distribution.

        Returns
        -------
        graph_embedding
            Graph (feature) embedding
            with shape :math:`n_{feature} \times n_{dim}`
            if ``n_sample`` is ``None``,
            or shape :math:`n_{feature} \times n_{sample} \times n_{dim}`
            if ``n_sample`` is not ``None``.
        """
        print('encode_graph')
        self.net.eval()
        graph = GraphDataset(graph, self.vertices)
        enorm = torch.as_tensor(
            normalize_edges(graph.eidx, graph.ewt),
            device=self.net.device
        )
        esgn = torch.as_tensor(graph.esgn, device=self.net.device)
        eidx = torch.as_tensor(graph.eidx, device=self.net.device)

        v = self.net.g2v(eidx, enorm, esgn)
        if n_sample:
            return torch.cat([
                v.sample((1, )).cpu() for _ in range(n_sample)
            ]).permute(1, 0, 2).numpy()
        return v.mean.detach().cpu().numpy()

    @torch.no_grad()
    def encode_data(
            self, key: str, adata: AnnData, batch_size: int = 128,
            n_sample: Optional[int] = None
    ) -> np.ndarray:
        r"""
        Compute data (cell) embedding

        Parameters
        ----------
        key
            Modality key
        adata
            Input dataset
        batch_size
            Size of minibatches
        n_sample
            Number of samples from the embedding distribution,
            by default ``None``, returns the mean of the embedding distribution.

        Returns
        -------
        data_embedding
            Data (cell) embedding
            with shape :math:`n_{cell} \times n_{dim}`
            if ``n_sample`` is ``None``,
            or shape :math:`n_{cell} \times n_{sample} \times n_{dim}`
            if ``n_sample`` is not ``None``.
        """
        # print('encode_data')
        self.net.eval()
        encoder = self.net.x2u[key]
        data = AnnDataset(
            [adata], [self.modalities[key]],
            mode="eval", getitem_size=batch_size
        )
        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )
        locs = []  # 用于存储每个批次的 loc (均值)
        scales = []  # 用于存储每个批次的 scale (标准差)
        result = []
        for x, xrep, *_ in data_loader:
            u = encoder(
                x.to(self.net.device, non_blocking=True),
                xrep.to(self.net.device, non_blocking=True),
                lazy_normalizer=True
            )[0]
            # 提取 u 的 loc 和 scale
            loc = u.loc.detach().cpu()  # 提取均值
            scale = u.scale.detach().cpu()  # 提取标准差
            # 将每个批次的 loc 和 scale 添加到列表
            locs.append(loc)
            scales.append(scale)
            if n_sample:
                result.append(u.sample((n_sample, )).cpu().permute(1, 0, 2))
            else:
                result.append(u.mean.detach().cpu())
        # 整合 loc 和 scale
        locs = torch.cat(locs, dim=0)  # 拼接 loc, 形状为 (总样本数, feature_dim)
        scales = torch.cat(scales, dim=0)  # 拼接 scale, 形状为 (总样本数, feature_dim)
        # 创建拼接后的 Normal 分布对象
        u_cat = Normal(locs, scales)
        embedding = torch.cat(result).numpy()
        # save_path = "embedding.npy"
        # np.save(save_path, embedding)
        return embedding,u_cat
        # return torch.cat(result).numpy()

    # @torch.no_grad()
    # def encode_zembedding(
    #         self, adatas: Mapping[str, AnnData], batch_size: int = 128,
    #         n_sample: Optional[int] = None
    # ) -> np.ndarray:
    #     r"""
    #     Compute data (cell) embedding
    #
    #     Parameters
    #     ----------
    #     key
    #         Modality key
    #     adata
    #         Input dataset
    #     batch_size
    #         Size of minibatches
    #     n_sample
    #         Number of samples from the embedding distribution,
    #         by default ``None``, returns the mean of the embedding distribution.
    #
    #     Returns
    #     -------
    #     data_embedding
    #         Data (cell) embedding
    #         with shape :math:`n_{cell} \times n_{dim}`
    #         if ``n_sample`` is ``None``,
    #         or shape :math:`n_{cell} \times n_{sample} \times n_{dim}`
    #         if ``n_sample`` is not ``None``.
    #     """
    #     # print('encode_data')
    #     self.net.eval()
    #     encoder = {}
    #     data, data_loader = {}, {}
    #     u, locs, scales, result, embedding = {}, {}, {}, {}, {}
    #     for k, adata in adatas.items():
    #         encoder[k] = self.net.x2u[k]
    #         data[k] = AnnDataset(
    #             [adata], [self.modalities[k]],
    #             mode="eval", getitem_size=batch_size
    #         )
    #         # data = AnnDataset(
    #         #     [adatas[key] for key in self.net.keys],
    #         #     [self.modalities[key] for key in self.net.keys],
    #         #     mode="eval",
    #         #     getitem_size=batch_size
    #         # )
    #         data_loader[k] = DataLoader(
    #             data[k], batch_size=1, shuffle=False,
    #             num_workers=config.DATALOADER_NUM_WORKERS,
    #             pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
    #             persistent_workers=False
    #         )
    #         locs[k], scales[k], result[k] = [], [], []
    #         for x, xrep, *_ in data_loader[k]:
    #             u[k] = encoder[k](
    #                 x.to(self.net.device, non_blocking=True),
    #                 xrep.to(self.net.device, non_blocking=True),
    #                 lazy_normalizer=True
    #             )[0]
    #             # 提取 u 的 loc 和 scale
    #             loc = u[k].loc.detach().cpu()  # 提取均值
    #             scale = u[k].scale.detach().cpu()  # 提取标准差
    #             # 将每个批次的 loc 和 scale 添加到列表
    #             locs[k].append(loc)
    #             scales[k].append(scale)
    #             if n_sample:
    #                 result[k].append(u[k].sample((n_sample,)).cpu().permute(1, 0, 2))
    #             else:
    #                 result[k].append(u[k].mean.detach().cpu())
    #
    #         locs[k] = torch.cat(locs[k], dim=0)  # 拼接 loc, 形状为 (总样本数, feature_dim)
    #         scales[k] = torch.cat(scales[k], dim=0)  # 拼接 scale, 形状为 (总样本数, feature_dim)
    #         embedding[k] = torch.cat(result[k]).numpy()
    #
    #     mus_joint = torch.stack([u[k].loc for k in u.keys()], dim=0)  # 直接访问 mean 属性
    #     logvars_joint = torch.stack([2 * torch.log(u[k].scale + 1e-8) for k in u.keys()], dim=0)
    #     z_mu, z_logvar = poe(mus_joint, logvars_joint)
    #     z_result = z_mu.detach().cpu()
    #     z_embedding = z_result.numpy()
    #
    #     # locs = []  # 用于存储每个批次的 loc (均值)
    #     # scales = []  # 用于存储每个批次的 scale (标准差)
    #     # result = []
    #     # # 整合 loc 和 scale
    #     # locs = torch.cat(locs, dim=0)  # 拼接 loc, 形状为 (总样本数, feature_dim)
    #     # scales = torch.cat(scales, dim=0)  # 拼接 scale, 形状为 (总样本数, feature_dim)
    #     # 创建拼接后的 Normal 分布对象
    #
    #     # u_cat = Normal(locs, scales)
    #     # embedding = torch.cat(result).numpy()
    #     # save_path = "embedding.npy"
    #     # np.save(save_path, embedding)
    #
    #     return z_embedding
    #     # return torch.cat(result).numpy()

    @torch.no_grad()
    def decode_data(
            self, source_key: str, target_key: str,
            adata: AnnData, graph: nx.Graph,
            target_libsize: Optional[Union[float, np.ndarray]] = None,
            target_batch: Optional[np.ndarray] = None,
            batch_size: int = 128
    ) -> np.ndarray:
        r"""
        Decode data

        Parameters
        ----------
        source_key
            Source modality key
        target_key
            Target modality key
        adata
            Source modality data
        graph
            Guidance graph
        target_libsize
            Target modality library size, by default 1.0
        target_batch
            Target modality batch, by default batch 0
        batch_size
            Size of minibatches

        Returns
        -------
        decoded
            Decoded data

        Note
        ----
        This is EXPERIMENTAL!
        """
        print('0')
        l = target_libsize or 1.0
        if not isinstance(l, np.ndarray):
            l = np.asarray(l)
        l = l.squeeze()
        if l.ndim == 0:  # Scalar
            l = l[np.newaxis]
        elif l.ndim > 1:
            raise ValueError("`target_libsize` cannot be >1 dimensional")
        if l.size == 1:
            l = np.repeat(l, adata.shape[0])
        if l.size != adata.shape[0]:
            raise ValueError("`target_libsize` must have the same size as `adata`!")
        l = l.reshape((-1, 1))

        use_batch = self.modalities[target_key]["use_batch"]
        batches = self.modalities[target_key]["batches"]
        if use_batch and target_batch is not None:
            target_batch = np.asarray(target_batch)
            if target_batch.size != adata.shape[0]:
                raise ValueError("`target_batch` must have the same size as `adata`!")
            b = batches.get_indexer(target_batch)
        else:
            b = np.zeros(adata.shape[0], dtype=int)

        net = self.net
        device = net.device
        net.eval()
        print('decode_data')
        u = self.encode_data(source_key, adata, batch_size=batch_size)
        v = self.encode_graph(graph)
        v = torch.as_tensor(v, device=device)
        v = v[getattr(net, f"{target_key}_idx")]

        data = ArrayDataset(u, b, l, getitem_size=batch_size)
        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )
        decoder = net.u2x[target_key]

        result = []
        for u_, b_, l_ in data_loader:
            u_ = u_.to(device, non_blocking=True)
            b_ = b_.to(device, non_blocking=True)
            l_ = l_.to(device, non_blocking=True)
            result.append(decoder(u_, v, b_, l_).mean.detach().cpu())
        return torch.cat(result).numpy()

    def upgrade(self) -> None:
        if hasattr(self, "domains"):
            self.logger.warning("Upgrading model generated by older versions...")
            self.modalities = getattr(self, "domains")
            delattr(self, "domains")

    def __repr__(self) -> str:
        return (
            f"SCGLUE model with the following network and trainer:\n\n"
            f"{repr(self.net)}\n\n"
            f"{repr(self.trainer)}\n"
        )


@logged
class PairedSCGLUEModel(SCGLUEModel):

    r"""
    GLUE model for partially-paired single-cell multi-omics data integration

    Parameters
    ----------
    adatas
        Datasets (indexed by modality name)
    vertices
        Guidance graph vertices (must cover feature names in all modalities)
    latent_dim
        Latent dimensionality
    h_depth
        Hidden layer depth for encoder and discriminator
    h_dim
        Hidden layer dimensionality for encoder and discriminator
    dropout
        Dropout rate
    shared_batches
        Whether the same batches are shared across modalities
    random_seed
        Random seed
    """

    TRAINER_TYPE = PairedSCGLUETrainer
    def compile(  # pylint: disable=arguments-renamed
            self,
            # adatas: Mapping[str, AnnData],
            lam_data: float = 1.0,
            lam_kl: float = 1.0,
            lam_graph: float = 0.02,
            lam_align: float = 0.05,
            lam_sup: float = 0.02,
            lam_joint_cross: float = 0.02,
            lam_real_cross: float = 0.02,
            lam_cos: float = 0.02,
            normalize_u: bool = False,
            modality_weight: Optional[Mapping[str, float]] = None,
            lr: float = 2e-3,
            # n_clusters: int = 0,
            Fine_tuning: bool = False,
            random_seed: int = 1,
            # z_embedding: Optional[np.ndarray] = None,
            **kwargs
    ) -> None:
        r"""
        Prepare model for training

        Parameters
        ----------
        n_clusters
        n_cluster
        adatas
        lam_data
            Data weight
        lam_kl
            KL weight
        lam_graph
            Graph weight
        lam_align
            Adversarial alignment weight
        lam_sup
            Cell type supervision weight
        lam_joint_cross
            Joint cross-prediction weight
        lam_real_cross
            Real cross-prediction weight
        lam_cos
            Cosine similarity weight
        normalize_u
            Whether to L2 normalize cell embeddings before decoder
        modality_weight
            Relative modality weight (indexed by modality name)
        lr
            Learning rate
        """
        # n_clusters = len(adatas['rna'].obs['cell_type'].unique())

        # 初始化聚类中心
        # self.initialize_cluster_centers(adatas, n_clusters)
        # z_embedding = adatas['rna'].obsm["Z_latent"]
        # adatas = self.adatas
        # for adata in adatas.values():
        #     rna =adata
        # print(lr)
        super().compile(
            # adatas=adatas,
            lam_data=lam_data, lam_kl=lam_kl,
            lam_graph=lam_graph, lam_align=lam_align, lam_sup=lam_sup,
            lam_joint_cross=lam_joint_cross, lam_real_cross=lam_real_cross,
            lam_cos=lam_cos, normalize_u=normalize_u, modality_weight=modality_weight,lr=lr,
            # n_clusters = n_clusters,
            Fine_tuning = Fine_tuning,
            # z_embedding = z_embedding
        )

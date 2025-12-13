r"""
Integration models
"""

import os
from pathlib import Path
from typing import Mapping, Optional
# import scglue_zj as scglue
import dill
import networkx as nx
import numpy as np
import pandas as pd
import torch
import scanpy
from anndata import AnnData
from sklearn.cluster import KMeans
from ..data import estimate_balancing_weight
from ..typehint import Kws
from ..utils import config, logged
from .base import Model
from .dx import integration_consistency
from .nn import autodevice
from .scclue import SCCLUEModel
from .scglue import PairedSCGLUEModel, SCGLUEModel
from scglue_zj import metrics
from .PoE_test import prior_expert,sample_gaussian,poe,ProductOfExperts
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score,accuracy_score
from torch.distributions import Normal
@logged
def configure_dataset(
        adata: AnnData, prob_model: str,
        use_highly_variable: bool = True,
        use_layer: Optional[str] = None,
        use_rep: Optional[str] = None,
        use_batch: Optional[str] = None,
        use_cell_type: Optional[str] = None,
        use_dsc_weight: Optional[str] = None,
        use_obs_names: bool = False
) -> None:
    r"""
    Configure dataset for model training

    Parameters
    ----------
    adata
        Dataset to be configured
    prob_model
        Probabilistic generative model used by the decoder,
        must be one of ``{"Normal", "ZIN", "ZILN", "NB", "ZINB"}``.
    use_highly_variable
        Whether to use highly variable features
    use_layer
        Data layer to use (key in ``adata.layers``)
    use_rep
        Data representation to use as the first encoder transformation
        (key in ``adata.obsm``)
    use_batch
        Data batch to use (key in ``adata.obs``)
    use_cell_type
        Data cell type to use (key in ``adata.obs``)
    use_dsc_weight
        Discriminator sample weight to use (key in ``adata.obs``)
    use_obs_names
        Whether to use ``obs_names`` to mark paired cells across
        different datasets

    Note
    -----
    The ``use_rep`` option applies to encoder inputs, but not the decoders,
    which are always fitted on data in the original space.
    """
    if config.ANNDATA_KEY in adata.uns:
        configure_dataset.logger.warning(
            "`configure_dataset` has already been called. "
            "Previous configuration will be overwritten!"
        )
    data_config = {}
    data_config["prob_model"] = prob_model
    if use_highly_variable:
        if "highly_variable" not in adata.var:
            raise ValueError("Please mark highly variable features first!")
        data_config["use_highly_variable"] = True
        data_config["features"] = adata.var.query("highly_variable").index.to_numpy().tolist()
    else:
        data_config["use_highly_variable"] = False
        data_config["features"] = adata.var_names.to_numpy().tolist()
    if use_layer:
        if use_layer not in adata.layers:
            raise ValueError("Invalid `use_layer`!")
        data_config["use_layer"] = use_layer
    else:
        data_config["use_layer"] = None
    if use_rep:
        if use_rep not in adata.obsm:
            raise ValueError("Invalid `use_rep`!")
        data_config["use_rep"] = use_rep
        data_config["rep_dim"] = adata.obsm[use_rep].shape[1]
    else:
        data_config["use_rep"] = None
        data_config["rep_dim"] = None
    if use_batch:
        if use_batch not in adata.obs:
            raise ValueError("Invalid `use_batch`!")
        data_config["use_batch"] = use_batch
        data_config["batches"] = pd.Index(
            adata.obs[use_batch]
        ).dropna().drop_duplicates().sort_values().to_numpy()  # AnnData does not support saving pd.Index in uns
    else:
        data_config["use_batch"] = None
        data_config["batches"] = None
    if use_cell_type:
        if use_cell_type not in adata.obs:
            raise ValueError("Invalid `use_cell_type`!")
        data_config["use_cell_type"] = use_cell_type
        data_config["cell_types"] = pd.Index(
            adata.obs[use_cell_type]
        ).dropna().drop_duplicates().sort_values().to_numpy()  # AnnData does not support saving pd.Index in uns
    else:
        data_config["use_cell_type"] = None
        data_config["cell_types"] = None
    if use_dsc_weight:
        if use_dsc_weight not in adata.obs:
            raise ValueError("Invalid `use_dsc_weight`!")
        data_config["use_dsc_weight"] = use_dsc_weight
    else:
        data_config["use_dsc_weight"] = None
    data_config["use_obs_names"] = use_obs_names
    adata.uns[config.ANNDATA_KEY] = data_config


def load_model(fname: os.PathLike) -> Model:
    r"""
    Load model from file

    Parameters
    ----------
    fname
        Specifies path to the file

    Returns
    -------
    model
        Loaded model
    """
    fname = Path(fname)
    with fname.open("rb") as f:
        model = dill.load(f)
    model.upgrade()  # pylint: disable=no-member
    model.net.device = autodevice()  # pylint: disable=no-member
    return model

@logged
def fit_SCGLUE(
        adatas: Mapping[str, AnnData], graph: nx.Graph, model: type = SCGLUEModel,
        init_kws: Kws = None, compile_kws: Kws = None, fit_kws: Kws = None,
        balance_kws: Kws = None
) -> SCGLUEModel:
    r"""
    Fit GLUE model to integrate single-cell multi-omics data

    Parameters
    ----------
    adatas
        Single-cell datasets (indexed by modality name)
    graph
        Guidance graph
    model
        Model class, must be one of
        {:class:`scglue.models.scglue.SCGLUEModel`,
        :class:`scglue.models.scglue.PairedSCGLUEModel`}
    init_kws
        Model initialization keyword arguments
        (see the constructor of the ``model`` class,
        either :class:`scglue.models.scglue.SCGLUEModel`
        or :class:`scglue.models.scglue.PairedSCGLUEModel`)
    compile_kws
        Model compile keyword arguments
        (see the ``compile`` method of the ``model`` class,
        either :meth:`scglue.models.scglue.SCGLUEModel.compile`
        or :meth:`scglue.models.scglue.PairedSCGLUEModel.compile`)
    fit_kws
        Model fitting keyword arguments
        (see :meth:`scglue.models.scglue.SCGLUEModel.fit`)
    balance_kws
        Balancing weight estimation keyword arguments
        (see :func:`scglue.data.estimate_balancing_weight`)

    Returns
    -------
    model
        Fitted model object
    """
    init_kws = init_kws or {}
    compile_kws = compile_kws or {}
    fit_kws = fit_kws or {}
    balance_kws = balance_kws or {}
    n_clusters: int
    n_clusters = len(adatas['rna'].obs['cell_type'].unique())
    fit_SCGLUE.logger.info("Pretraining SCGLUE model...")
    pretrain_init_kws = init_kws.copy()
    pretrain_init_kws.update({"shared_batches": False})
    pretrain_fit_kws = fit_kws.copy()
    pretrain_fit_kws.update({"align_burnin": np.inf, "safe_burnin": False})
    if "directory" in pretrain_fit_kws:
        pretrain_fit_kws["directory"] = \
            os.path.join(pretrain_fit_kws["directory"], "pretrain")

    pretrain = model(adatas, sorted(graph.nodes), **pretrain_init_kws)
    pretrain.compile(Fine_tuning=True,**compile_kws)
    # pretrain.compile(adatas, n_clusters = n_clusters, z_embedding = None ,**compile_kws)
    pretrain.fit(adatas, graph, **pretrain_fit_kws)
    if "directory" in pretrain_fit_kws:
        pretrain.save(os.path.join(pretrain_fit_kws["directory"], "pretrain.dill"))
    # pretrain = load_model(os.path.join(pretrain_fit_kws["directory"], "pretrain.dill"))

    # fit_SCGLUE.logger.info("Estimating balancing weight...")
    u = {}
    mus_joint, logvars_joint = prior_expert((1, adatas['rna'].n_obs, 50), )
    for k, adata in adatas.items():
        # 解包 encode_data 的返回值
        embedding, u[k] = pretrain.encode_data(k, adata)
        # 将 embedding 存入 adata.obsm
        adata.obsm[f"X_{config.TMP_PREFIX}"] = embedding
        mus_joint = torch.cat((mus_joint, u[k].loc.cuda().unsqueeze(0)), dim=0)
        logvars_joint = torch.cat((logvars_joint, (2 * torch.log(u[k].scale.cuda()) + 1e-8).unsqueeze(0).cuda()), dim=0)
        # batch_size, h_dims = embedding.shape
        # adata.obsm[f"X_{config.TMP_PREFIX}"] = pretrain.encode_data(k, adata)
    # mus_joint = torch.stack([u[k].loc for k in u.keys()], dim=0)  # 直接访问 mean 属性
    # logvars_joint = torch.stack([2 * torch.log(u[k].scale + 1e-8) for k in u.keys()], dim=0)  #
    # mus_joint, logvars_joint = prior_expert((1, batch_size, h_dims), )
    # for k in u.keys():
    #     mus_joint = torch.cat((mus_joint, u[k].loc.cuda().unsqueeze(0)), dim=0)
    #     logvars_joint = torch.cat((logvars_joint, (2 * torch.log(u[k].scale.cuda()) + 1e-8).unsqueeze(0)), dim=0)
    # experts = ProductOfExperts()
    # z_mu, z_logvar = experts(mus_joint, logvars_joint)
    z_mu, z_logvar = poe(mus_joint, logvars_joint)
    # z_scale = torch.exp(0.5 * z_logvar)
    # z = Normal(z_mu, z_scale)
    # z_samp_joint = sample_gaussian(z_mu, z_logvar)
    # z_result = z.mean.detach().cpu()
    z_result = z_mu.detach().cpu()
    z_embedding = z_result.numpy()
    # adatas['rna'].obsm["Z_latent"] = z_embedding
    # adatas['rna'].obsm["Z_latent"] = z_embedding
    # for k, adata in adatas.items():
    #     adata.obsm["Z_latent"] = z_embedding
    #中间评估
    kmeans = KMeans(n_clusters=n_clusters, n_init=50)
    pred_labels_z = kmeans.fit_predict(z_embedding)
    true_labels = adatas['rna'].obs['cell_type']
    # 计算ARI (Adjusted Rand Index)
    ari_score_z = np.round(adjusted_rand_score(true_labels, pred_labels_z), 5)

    # 计算AMI (Adjusted Mutual Information)
    ami_score_z = np.round(adjusted_mutual_info_score(true_labels, pred_labels_z), 5)

    # 计算NMI (Normalized Mutual Information)
    nmi_score_z = np.round(normalized_mutual_info_score(true_labels, pred_labels_z), 5)
    # 传入对应的函数计算出该结果下的轮廓系数
    Silhouette_score_z = np.round(metrics.avg_silhouette_width(z_embedding, true_labels), 5)
    # foscttm = metrics.foscttm(
    #     adatas['rna'].obsm["X_glue"], adatas['atac'].obsm["X_glue"]
    # )
    mean_average_precision = metrics.mean_average_precision(
       z_embedding, true_labels.to_numpy().ravel()
    )
    acc_score = np.round(metrics.label_transfer(true_labels, pred_labels_z), 5)
    print(f"ARI (Adjusted Rand Index): {ari_score_z}")
    print(f"AMI (Adjusted Mutual Information): {ami_score_z}")
    print(f"NMI (Normalized Mutual Information): {nmi_score_z}")
    print(f"Silhouette (Silhouette Coefficient): {Silhouette_score_z}")
    # print(f"foscttm : {foscttm}")
    # print('F1:', f1_score(true_labels, pred_labels_z, average='micro'))
    print(f"MAP : {mean_average_precision}")
    print(f"ACC : {acc_score}")
    # # adatas['rna'].obsm["Z_latent"] = z_embedding
    # # return pretrain

    if init_kws.get("shared_batches"):
        use_batch = set(
            adata.uns[config.ANNDATA_KEY]["use_batch"]
            for adata in adatas.values()
        )
        use_batch = use_batch.pop() if len(use_batch) == 1 else None
    else:
        use_batch = None
    fit_SCGLUE.logger.info("Estimating balancing weight...")
    estimate_balancing_weight(
        *adatas.values(), use_rep=f"X_{config.TMP_PREFIX}", use_batch=use_batch,
        key_added="balancing_weight", **balance_kws
    )
    # estimate_balancing_weight(
    #     *adatas.values(), use_rep="Z_latent", use_batch=use_batch,
    #     key_added="balancing_weight", **balance_kws
    # )
    for adata in adatas.values():
        adata.uns[config.ANNDATA_KEY]["use_dsc_weight"] = "balancing_weight"
        del adata.obsm[f"X_{config.TMP_PREFIX}"]
    # for adata in adatas.values():
    #     adata.uns[config.ANNDATA_KEY]["use_dsc_weight"] = None
    #     del adata.obsm[f"X_{config.TMP_PREFIX}"]

    fit_SCGLUE.logger.info("Fine-tuning SCGLUE model...")
    finetune_fit_kws = fit_kws.copy()
    if "directory" in finetune_fit_kws:
        finetune_fit_kws["directory"] = \
            os.path.join(finetune_fit_kws["directory"], "fine-tune")

    finetune = model(adatas, sorted(graph.nodes), **init_kws)
    finetune.adopt_pretrained_model(pretrain)
    finetune.compile(Fine_tuning=True, **compile_kws)

    print(f"Current numpy random seed: {finetune.random_seed}")

    fit_SCGLUE.logger.debug("Increasing random seed by 1 to prevent idential data order...")
    finetune.random_seed += 1

    print(f"Current numpy random seed: {finetune.random_seed}")
    finetune.fit(adatas, graph, **finetune_fit_kws)
    if "directory" in finetune_fit_kws:
        finetune.save(os.path.join(finetune_fit_kws["directory"], "fine-tune.dill"))
    # finetune = load_model(os.path.join(finetune_fit_kws["directory"], "fine-tune.dill"))
    u = {}
    # embedding = []
    mus_joint, logvars_joint = prior_expert((1, adatas['rna'].n_obs, 50), )
    for k, adata in adatas.items():
        # 解包 encode_data 的返回值
        embedding, u[k] = finetune.encode_data(k, adata)
        mus_joint = torch.cat((mus_joint, u[k].loc.cuda().unsqueeze(0)), dim=0)
        logvars_joint = torch.cat((logvars_joint, (2 * torch.log(u[k].scale.cuda()) + 1e-8).unsqueeze(0)), dim=0)
    # mus_joint = torch.stack([u[k].loc for k in u.keys()], dim=0)  # 直接访问 mean 属性
    # logvars_joint = torch.stack([2 * torch.log(u[k].scale + 1e-8) for k in u.keys()], dim=0)  #
    # mus_joint, logvars_joint = prior_expert((1, batch_size, h_dims), )
    # for k in u.keys():
    #     mus_joint = torch.cat((mus_joint, u[k].loc.cuda().unsqueeze(0)), dim=0)
    #     logvars_joint = torch.cat((logvars_joint, (2 * torch.log(u[k].scale.cuda()) + 1e-8).unsqueeze(0)), dim=0)
    # experts = ProductOfExperts()
    # z_mu, z_logvar = experts(mus_joint, logvars_joint)
    z_mu, z_logvar = poe(mus_joint, logvars_joint)
    # z_scale = torch.exp(0.5 * z_logvar)
    # z = Normal(z_mu, z_scale)
    # z_samp_joint = sample_gaussian(z_mu, z_logvar)
    # z_result = z.mean.detach().cpu()
    z_result = z_mu.detach().cpu()
    z_embedding = z_result.numpy()
    adatas['rna'].obsm["Z_latent"] = z_embedding
    pred_labels_z = kmeans.fit_predict(z_embedding)
    # 计算ARI (Adjusted Rand Index)
    ari_score = np.round(adjusted_rand_score(true_labels, pred_labels_z), 5)

    # 计算AMI (Adjusted Mutual Information)
    ami_score = np.round(adjusted_mutual_info_score(true_labels, pred_labels_z), 5)

    # 计算NMI (Normalized Mutual Information)
    nmi_score = np.round(normalized_mutual_info_score(true_labels, pred_labels_z), 5)
    # 传入对应的函数计算出该结果下的轮廓系数
    Silhouette_score = np.round(metrics.avg_silhouette_width(z_embedding, true_labels), 5)
    # foscttm = metrics.foscttm(
    #     adatas['rna'].obsm["X_glue"], adatas['atac'].obsm["X_glue"]
    # )
    mean_average_precision = metrics.mean_average_precision(
        z_embedding, true_labels.to_numpy().ravel()
    )
    acc_score = np.round(metrics.label_transfer(true_labels, pred_labels_z), 5)
    print(f"ARI (Adjusted Rand Index): {ari_score}")
    print(f"AMI (Adjusted Mutual Information): {ami_score}")
    print(f"NMI (Normalized Mutual Information): {nmi_score}")
    print(f"Silhouette (Silhouette Coefficient): {Silhouette_score}")
    print(f"MAP : {mean_average_precision}")
    print(f"ACC : {acc_score}")

    return finetune
    # return pretrain

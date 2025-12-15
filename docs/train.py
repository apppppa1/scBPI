import sys
sys.path.append("..")
import os
from itertools import chain
import numpy as np
import anndata as ad
import itertools
import networkx as nx
import pandas as pd
import scanpy as sc
# import scglue
import scglue_zj as scbpi
# print(scglue.__file__)
import torch
from torch.distributions import Normal
# print('11111111111111111')
import seaborn as sns
from matplotlib import rcParams
# import scglue.models.PoE_test
from scglue_zj.models.PoE_test import sample_gaussian,poe,prior_expert,ProductOfExperts
# from scglue.models.PoE_test import ProductOfExperts,sample_gaussian
from scglue_zj.metrics import silhouette

# rna = ad.read_h5ad("../data/GSE126074/rna_GSE126074-prep.h5ad")
# atac = ad.read_h5ad("../data/GSE126074/atac_GSE126074-prep.h5ad")
# guidance = nx.read_graphml("../data/GSE126074/guidance_GSE126074.graphml.gz")
# rna = ad.read_h5ad("../data/GSE126074_AdBrain/Chen-2019-RNA-processed.h5ad")
# atac = ad.read_h5ad("../data/GSE126074_AdBrain/Chen-2019-ATAC-processed.h5ad")
# guidance = nx.read_graphml("../data/GSE126074_AdBrain/guidance_Chen.graphml.gz")
# rna = ad.read_h5ad("../data/PBMC_10K/10x-Pbmc10k-RNA.h5ad")
# atac = ad.read_h5ad("../data/PBMC_10K/10x-Pbmc10k-ATAC.h5ad")
# guidance = nx.read_graphml("../data/PBMC_10K/guidance.graphml.gz")
# rna = ad.read_h5ad("../data/GSE140203_skin/rna-GSE140203.h5ad")
# atac = ad.read_h5ad("../data/GSE140203_skin/atac-GSE140203.h5ad")
# guidance = nx.read_graphml("../data/GSE140203_skin/guidance-GSE140203.graphml.gz")
# rna = ad.read_h5ad("../data/GSE151302_human/rna-GSE151302.h5ad")
# atac = ad.read_h5ad("../data/GSE151302_human/atac-GSE151302.h5ad")
# guidance = nx.read_graphml("../data/GSE151302_human/guidance-GSE151302.graphml.gz")
rna = ad.read_h5ad("../data/PBMC_3K/rna-pbmc3k.h5ad")
atac = ad.read_h5ad("../data/PBMC_3K/atac-pbmc3k.h5ad")
guidance = nx.read_graphml("../data/PBMC_3K/guidance-pbmc3k.graphml.gz")
# rna = ad.read_h5ad("../data/opmulti/rna-final.h5ad")
# atac = ad.read_h5ad("../data/opmulti/atac-final.h5ad")
# guidance = nx.read_graphml("../data/opmulti/guidance.graphml.gz")
# rna = ad.read_h5ad("../data/GSE100866/RNA-processed-2.8.h5ad")
# adt = ad.read_h5ad("../data/GSE100866/Protein-processed-2.8.h5ad")
# guidance = nx.read_graphml("../data/GSE100866/guidance-2.8.graphml.gz")
scbpi.models.configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_layer="counts",
    use_rep="X_pca"
)
scbpi.models.configure_dataset(
    atac, "NB", use_highly_variable=True,
    use_rep="X_lsi"
)
# scbpi.models.configure_dataset(
#     adt, "NB", use_highly_variable=False,
#     use_layer="counts",
#     # use_rep="clr"
# )
guidance_hvf = guidance.subgraph(chain(
    rna.var.query("highly_variable").index,
    atac.var.query("highly_variable").index
)).copy()
# guidance_hvf = guidance.subgraph(chain(
#     rna.var.query("highly_variable").index,
#     adt.var.index
# )).copy()

glue = scbpi.models.fit_SCGLUE(
    {"rna": rna, "atac": atac}, guidance_hvf,
    model=scbpi.models.PairedSCGLUEModel,
    fit_kws={"directory": "glue"}
)

# glue = scglue.models.fit_SCGLUE(
#     {"rna": rna, "adt": adt}, guidance_hvf,
#     model=scglue.models.PairedSCGLUEModel,
#
#     fit_kws={"directory": "glue_true"}
# )
# # 设置模型的保存和加载路径
# model_save_path = os.getenv('MODEL_SAVE_PATH', '../data/GSE126074/glue_GSE126074.dill')
#
# # 保存模型到指定路径
# glue.save("../data/GSE126074_AdBrain/glue_AdBrain.dill")
# feature_embeddings = glue.encode_graph(guidance_hvf)
#
# feature_embeddings = pd.DataFrame(feature_embeddings, index=glue.vertices)
# rna.varm["X_glue"] = feature_embeddings.reindex(rna.var_names).to_numpy()
# atac.varm["X_glue"] = feature_embeddings.reindex(atac.var_names).to_numpy()
# rna.write("../data/GSE126074_AdBrain/rna-emb.h5ad", compression="gzip")
# atac.write("../data/GSE126074_AdBrain/atac-emb.h5ad", compression="gzip")
# nx.write_graphml(guidance_hvf, "../data/GSE126074_AdBrain/guidance-hvf.graphml.gz")
#
# # 从指定路径加载模型
# glue = scglue.models.load_model(model_save_path)


# glue.save("../data/GSE126074/glue_GSE126074.dill")
# glue = scglue.models.load_model("../data/GSE126074/glue_GSE126074.dill")
# glue.save("../data/GSE126074_AdBrain/glue_GSE126074_Ad.dill")
# glue = scglue.models.load_model("../data/GSE126074_AdBrain/glue_GSE126074_Ad.dill")
# glue.save("../data/PBMC_10K/glue.dill")
# glue = scglue.models.load_model("../data/PBMC_10K/glue.dill")
# glue.save("../data/GSE140203_skin/glue.dill")
# glue = scglue.models.load_model("../data/GSE140203_skin/glue.dill")
# glue.save("../data/GSE151302_human/glue.dill")
# glue = scglue.models.load_model("../data/GSE151302_human/glue.dill")
# glue.save("../data/PBMC_3K/glue.dill")
glue = scbpi.models.load_model("../data/PBMC_3K/glue.dill")
# glue.save("../data/opmulti/glue.dill")
# glue = scglue.models.load_model("../data/opmulti/glue.dill")
# glue.save("../data/PBMC_3K/glue-.dill")
# glue = scglue.models.load_model("../data/PBMC_3K/glue-.dill")
# glue.save("../data/GSE100866/glue-2.8.dill")
# glue = scglue.models.load_model("../data/GSE100866/glue-3.dill")
u = {}
# atac = adt.copy()
rna.obsm["X_glue"],u[0] = glue.encode_data("rna", rna)
atac.obsm["X_glue"],u[1]= glue.encode_data("atac", atac)
# rna.obsm["X_glue"] = glue.encode_data("rna", rna)
# atac.obsm["X_glue"] = glue.encode_data("atac", atac)
# mus_joint = torch.stack([u[k].loc for k in u.keys()], dim=0)  # 直接访问 mean 属性
# logvars_joint = torch.stack([2 * torch.log(u[k].scale + 1e-8) for k in u.keys()], dim=0)

# experts = ProductOfExperts()
# batch_size, h_dims = rna.obsm["X_glue"].shape
# mus_joint, logvars_joint = prior_expert((1, batch_size, h_dims), )
mus_joint, logvars_joint = prior_expert((1, rna.n_obs, 50), )
for k in u.keys():
    # mus_joint = torch.cat((mus_joint, u[k].loc.cuda().unsqueeze(0)), dim=0)
    # logvars_joint = torch.cat((logvars_joint, (2 * torch.log(u[k].scale.cuda()) + 1e-8).unsqueeze(0)), dim=0)
    mus_joint = torch.cat((mus_joint, u[k].loc.cuda().unsqueeze(0)), dim=0)
    logvars_joint = torch.cat((logvars_joint, (2 * torch.log(u[k].scale.cuda()) + 1e-8).unsqueeze(0)), dim=0)
# z_mu, z_logvar = experts(mus_joint, logvars_joint)
z_mu, z_logvar = poe(mus_joint, logvars_joint)
# z_scale = torch.exp(0.5 * z_logvar)
# z = Normal(z_mu, z_scale)
# z_result = z.mean.detach().cpu()
n_clusters = len(rna.obs['cell_type'].unique())
true_labels = rna.obs['cell_type']
z_result = z_mu.detach().cpu()
z_embedding = z_result.numpy()
rna.obsm["Z_latent"] = z_embedding
# atac.obsm["Z_latent"] = z_embedding
z_embedding = rna.obsm["Z_latent"]
np.save("../data/PBMC_3K/z_embedding.npy", z_embedding)

# z_embedding = np.load("../data/GSE100866/z_embedding-2.npy")
# rna.obsm["Z_latent"] = z_embedding

# sc.pp.neighbors(rna, use_rep="Z_latent", n_neighbors=20)

# 查看 Z_latent 是否存在且无 NaN 值
# if 'Z_latent' in rna.obsm.keys() and not np.isnan(rna.obsm['Z_latent']).any():
#     sc.pp.neighbors(rna, use_rep="Z_latent", n_neighbors=20)
# else:
#     print("Z_latent 不存在或含有 NaN 值")
# sc.tl.umap(rna)
# sc.pl.umap(rna, color="cell_type",save='GSE128639.pdf', size=50, alpha=0.7)
# sc.pl.umap(rna, color="cell_type",save='GSE100866-2.8.pdf')
true_labels = rna.obs['cell_type']
from scbpi.metrics import embedding_leiden_across_resolutions,embedding_to_knn,knn_purity_score
resolutions, aris, nmis = embedding_leiden_across_resolutions(
        embedding=z_embedding,
        labels=true_labels,
        n_neighbors=10,
        resolutions=[0.1, 0.5, 1.0],
    )
Silhouette_score = np.round(silhouette(z_embedding,true_labels),5)
# Compute a knn from the embedding.
knn = embedding_to_knn(embedding=z_embedding, k=15, metric="euclidean")

# Compute the knn purity score.
knn_score = knn_purity_score(knn=knn, labels=true_labels)
# 打印结果
print("Resolutions:", resolutions)
print("ARI (Adjusted Rand Index) values:", aris)
print("NMI (Normalized Mutual Information) values:", nmis)
print("Silhouette:", Silhouette_score)
print("knn_purity_score:", knn_score)


"""Build pb_full.h5ad from the raw CD19 CAR T-cell single-cell dataset.

Downloads the source dataset (~1.6 GB) from CZ CELLxGENE Discover if it is not
already present locally, then aggregates raw counts into a donor x cell type x
CAR status x exhaustion status pseudobulk object.

Source: https://datasets.cellxgene.cziscience.com/25735b46-c216-43fb-8cae-5b890090b714.h5ad
(CZ CELLxGENE Discover, collection be21c2d1-2392-47d0-96fb-c625d115e0dc;
Deng et al., Nat Med 2020, https://doi.org/10.1038/s41591-020-1061-7)

Usage:
  conda run -n sysbio python build_pseudobulk.py
"""
import os

import anndata as ad
import numpy as np
import pandas as pd
import requests
import scanpy as sc
import scipy.sparse as sp

RAW_H5AD_URL = "https://datasets.cellxgene.cziscience.com/25735b46-c216-43fb-8cae-5b890090b714.h5ad"
RAW_H5AD_PATH = "25735b46-c216-43fb-8cae-5b890090b714.h5ad"


def download_raw_dataset(path, url):
    if os.path.exists(path):
        print(f"Found existing local file: {path}")
        return
    print(f"Downloading raw dataset from {url}")
    print("Size: ~1.6 GB. This is a one-time download; it can take a few minutes.")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        next_report = 200_000_000
        tmp_path = path + ".part"
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded >= next_report:
                    print(f"  ... {downloaded / 1e9:.1f} / {total / 1e9:.1f} GB")
                    next_report += 200_000_000
        os.rename(tmp_path, path)
    print("Download complete.")


def sum_pseudobulk(obs_df, raw_X, group_cols, min_cells):
    pb_X, pb_obs = [], []
    for key, idx in obs_df.groupby(group_cols, observed=True).indices.items():
        if len(idx) < min_cells:
            continue
        summed = np.asarray(raw_X[idx].sum(axis=0)).ravel()
        pb_X.append(summed)
        row = dict(zip(group_cols, key)) if isinstance(key, tuple) else {group_cols[0]: key}
        row["n_cells"] = len(idx)
        pb_obs.append(row)
    return np.vstack(pb_X), pd.DataFrame(pb_obs)


def main():
    download_raw_dataset(RAW_H5AD_PATH, RAW_H5AD_URL)

    print(f"Loading raw single-cell dataset from {RAW_H5AD_PATH} ...")
    adata = sc.read_h5ad(RAW_H5AD_PATH)
    adata.obs["exhaustion_status"] = np.where(adata.obs["Exhausted.T.cells"], "exhausted", "non_exhausted")

    raw = adata.raw.to_adata()
    assert (raw.var_names == adata.var_names).all()
    print(f"{adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # pb_full: donor x cell type x CAR status x exhaustion status (groups with >=10 cells)
    group_cols_full = ["donor_id", "cell_type_in_paper", "CAR", "exhaustion_status"]
    X_full, obs_full = sum_pseudobulk(adata.obs, raw.X, group_cols_full, min_cells=10)
    obs_full.index = (
        obs_full["donor_id"].astype(str) + "_"
        + obs_full["cell_type_in_paper"].astype(str).str.replace(r"\W+", "_", regex=True) + "_"
        + obs_full["CAR"].astype(str) + "_" + obs_full["exhaustion_status"].astype(str)
    )
    case_mask = (obs_full["CAR"] == "CAR+") & (obs_full["exhaustion_status"] == "non_exhausted")
    ctrl_mask = (obs_full["CAR"] == "CAR-") & (obs_full["exhaustion_status"] == "non_exhausted")
    obs_full["group_tag"] = np.select(
        [case_mask, ctrl_mask],
        ["case_CARpos_nonexhausted", "control_CARneg_nonexhausted"],
        default="other_exhausted",
    )

    pb_full = ad.AnnData(X=sp.csr_matrix(X_full), obs=obs_full, var=raw.var.copy())
    pb_full.layers["counts"] = pb_full.X.copy()
    print("pb_full:", pb_full.shape)
    print(pb_full.obs["group_tag"].value_counts())

    pb_full.write_h5ad("pb_full.h5ad", compression="gzip")
    print("\nWrote pb_full.h5ad")


if __name__ == "__main__":
    main()

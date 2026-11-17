"""Shared helpers for the GBM multi-cohort project (written to Drive by NB00).

Design rules
- All paths come from P(); all thresholds come from config.json (00_Project_Config).
- SMOKE mode writes derived outputs to <ROOT>/_smoke/ (raw downloads and config are shared).
- Nothing here fabricates data: failures raise ActionableError with next steps.
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("GBM_ROOT", "/content/drive/MyDrive/GBM_MultiCohort_Research"))

FOLDERS = {
    "config": "00_Project_Config",
    "lit": "01_Literature_Review",
    "raw": "02_Raw_Datasets",
    "meta": "03_Metadata",
    "qc": "04_QC",
    "de": "05_Differential_Expression",
    "meta_analysis": "06_Meta_Analysis",
    "enrich": "07_Functional_Enrichment",
    "network": "08_Network_Analysis",
    "novelty": "09_Novelty_Assessment",
    "validation": "10_Independent_Validation",
    "sc": "11_Single_Cell",
    "clinical": "12_Clinical_Associations",
    "orthogonal": "13_Orthogonal_Evidence",
    "sensitivity": "14_Sensitivity_Analysis",
    "final": "15_Final_Results",
    "figures": "16_Publication_Figures",
    "medcon": "17_MEDCON_Presentation",
    "manuscript": "18_Manuscript",
    "logs": "19_Logs_And_Provenance",
}
SHARED_KEYS = {"config", "raw", "logs"}  # never redirected to _smoke

GEO_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}nnn/{acc}/matrix/{acc}_series_matrix.txt.gz"
TOIL = "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/"
GDC = "https://gdc-hub.s3.us-east-1.amazonaws.com/download/"
CGGA_BASE = "http://www.cgga.org.cn/download?file=download/20200506/"

DEFAULT_CONFIG = {
    "smoke_test": True,
    "random_seed": 20261019,
    "project": "GBM multi-cohort: malignant neural-like vs admixture signal",
    "population": "adult, primary, IDH-wildtype GBM",
    "urls": {
        "toil_counts": [TOIL + "TcgaTargetGtex_gene_expected_count.gz"],
        "toil_phenotype": [TOIL + "TcgaTargetGTEX_phenotype.txt.gz"],
        "toil_probemap": [TOIL + "probeMap%2Fgencode.v23.annotation.gene.probemap"],
        "tcga_survival": [GDC + "TCGA-GBM.survival.tsv.gz"],
        "tcga_mutect2": [GDC + "TCGA-GBM.mutect2_snv.tsv.gz"],
        "tcga_gdc_phenotype": [GDC + "TCGA-GBM.GDC_phenotype.tsv.gz"],
        "cgga693_expr": [CGGA_BASE + "CGGA.mRNAseq_693.RSEM-genes.20200506.txt.zip"],
        "cgga693_clin": [CGGA_BASE + "CGGA.mRNAseq_693_clinical.20200506.txt.zip"],
        "cgga325_expr": [CGGA_BASE + "CGGA.mRNAseq_325.RSEM-genes.20200506.txt.zip"],
        "cgga325_clin": [CGGA_BASE + "CGGA.mRNAseq_325_clinical.20200506.txt.zip"],
        "gse131928_raw": ["https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131928/suppl/GSE131928_RAW.tar"],
        "ivy_expression": [
            "http://glioblastoma.alleninstitute.org/api/v2/well_known_file_download/305873915",
            "http://download.alleninstitute.org/informatics-archive/current-release/ivygap/gene_expression_matrix_2014-11-25.zip",
        ],
        "syngo": ["https://www.syngoportal.org/data/SynGO_bulk_download_release_20231201.zip"],
        "hpa_sc": ["https://www.proteinatlas.org/download/tsv/rna_single_cell_type.tsv.zip"],
    },
    "geo_microarray": {"GSE4290": "GPL570", "GSE50161": "GPL570"},
    "preprocessing": {
        "min_cpm": 1.0,
        "outlier_pc_z": 3.5,
        "array_log2_if_max_above": 100.0,
        "min_group_n": 5,
    },
    "de": {"fdr": 0.05, "min_abs_log2fc": 1.0},
    "meta": {"fdr": 0.05, "i2_heterogeneous": 50.0, "min_studies": 2},
    "enrichment": {"min_size": 10, "max_size": 500, "fdr": 0.05},
    "sc": {
        "min_malignant_cells_per_patient": 50,
        "detect_fraction_in_patient": 0.10,
        "min_patient_fraction": 0.50,
        "min_datasets": 1,
        "cnv_z_threshold": 2.0,
        "adult_patient_regex": "^MGH",
    },
    "survival": {
        "min_events_per_param": 10,
        "training_cohort": "TCGA",
        "validation_cohorts": ["CGGA693", "CGGA325"],
        "h1": {"hr_shift_frac": 0.25, "meta_p": 0.05, "i2_max": 50.0},
    },
    "tiers": {"fdr": 0.05, "i2_max": 50.0, "min_sc_datasets": 1},
    "s_definition": {"preferred": "SynGO", "fallback": "GOCC_SYNAPSE"},
    # Small curated marker lists (cell-type identity); NOT derived from outcome data.
    # Sensitivity analyses should replace these with reference-derived markers.
    "markers": {
        "immune_stromal": ["PTPRC", "CD68", "CD163", "CSF1R", "AIF1", "C1QA", "C1QB", "CD3E", "CD2",
                           "CD14", "LYZ", "PECAM1", "VWF", "CLDN5", "ACTA2", "PDGFRB", "COL1A1", "COL1A2"],
        "neuron_identity": ["RBFOX3", "SLC17A7", "SLC17A6", "GAD1", "GAD2", "NEFL", "NEFM", "NEFH",
                            "MAP2", "TUBB3", "CAMK2A", "SATB2", "CUX2", "STMN2"],
        "oligodendrocyte": ["MBP", "PLP1", "MOG", "MOBP", "MAG"],
        "astrocyte_normal": ["AQP4", "SLC1A2", "SLC1A3", "ALDH1L1", "GJA1"],
        "malignant_cnv_reference": ["PTPRC", "MBP", "PLP1", "MOG", "CD68"],
    },
}


class ActionableError(RuntimeError):
    """An error that tells the user what to do next."""


def _deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def config_path() -> Path:
    return ROOT / FOLDERS["config"] / "config.json"


def load_config() -> dict:
    p = config_path()
    if p.exists():
        return _deep_merge(DEFAULT_CONFIG, json.loads(p.read_text()))
    return json.loads(json.dumps(DEFAULT_CONFIG))


def save_config(cfg: dict) -> None:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2, sort_keys=True))


CFG = load_config()
SMOKE = bool(CFG.get("smoke_test", True))


def P(key: str, *parts: str, mkdir: bool = True) -> Path:
    """Centralised path builder. Directory keys only; extra parts are appended."""
    if key not in FOLDERS:
        raise KeyError(f"Unknown folder key '{key}'. Valid: {sorted(FOLDERS)}")
    base = ROOT / ("_smoke" if (SMOKE and key not in SHARED_KEYS) else "") / FOLDERS[key]
    path = base.joinpath(*parts) if parts else base
    target = path if not parts or not Path(parts[-1]).suffix else path.parent
    if mkdir:
        target.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs() -> None:
    for k in FOLDERS:
        P(k)


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str, level: str = "INFO") -> None:
    line = f"{now_utc()} [{level}] {msg}"
    print(line)
    with open(P("logs", "run_log.txt"), "a") as fh:
        fh.write(line + "\n")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def manifest_update(section: str, key: str, value) -> None:
    p = P("logs", "run_manifest.json")
    data = json.loads(p.read_text()) if p.exists() else {}
    data.setdefault(section, {})[key] = value
    data["_updated_utc"] = now_utc()
    p.write_text(json.dumps(data, indent=2, default=str))


def software_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "seaborn", "requests", "openpyxl"]:
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    try:
        r = subprocess.run(["Rscript", "--version"], capture_output=True, text=True, timeout=30)
        out["R"] = (r.stderr or r.stdout).strip()
    except Exception:
        out["R"] = None
    return out


def resource_report() -> dict:
    import shutil
    rep = {"cpu_count": os.cpu_count()}
    try:
        with open("/proc/meminfo") as fh:
            kb = int(fh.readline().split()[1])
        rep["ram_gb"] = round(kb / 1e6, 1)
    except Exception:
        rep["ram_gb"] = None
    du = shutil.disk_usage("/content" if os.path.exists("/content") else "/")
    rep["disk_free_gb"] = round(du.free / 1e9, 1)
    return rep


def pip_install(pkgs: list[str]) -> None:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], capture_output=True, text=True)
    if r.returncode != 0:
        raise ActionableError(f"pip install failed for {pkgs}:\n{r.stderr[-1500:]}")


def download(urls, dest: Path, min_bytes: int = 1024, retries: int = 3, timeout: int = 180,
             force: bool = False, reject_html: bool = True, key: str = "") -> Path:
    """Download the first working URL to dest. Validates size; refuses HTML error pages."""
    import requests
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(urls, str):
        urls = [urls]
    if dest.exists() and dest.stat().st_size >= min_bytes and not force:
        log(f"cached: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    failures = []
    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                with requests.get(url, stream=True, timeout=timeout,
                                  headers={"User-Agent": "gbm-multicohort/1.0"}) as r:
                    status = r.status_code
                    ctype = r.headers.get("Content-Type", "")
                    if status != 200:
                        raise IOError(f"HTTP {status}")
                    if reject_html and "html" in ctype.lower() and not str(dest).endswith((".html", ".htm")):
                        raise IOError(f"server returned HTML ({ctype}), probably a block or error page")
                    part = dest.with_suffix(dest.suffix + ".part")
                    with open(part, "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                size = part.stat().st_size
                if size < min_bytes:
                    part.unlink()
                    raise IOError(f"file too small ({size} bytes)")
                os.replace(part, dest)
                digest = sha256(dest)
                manifest_update("downloads", key or dest.name,
                                {"url": url, "bytes": size, "sha256": digest, "date_utc": now_utc(), "path": str(dest)})
                log(f"downloaded {dest.name}: {size/1e6:.1f} MB from {url}")
                return dest
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{url} (attempt {attempt}): {exc}")
                time.sleep(2 * attempt)
    raise ActionableError(
        f"Could not download {dest.name}.\nTried:\n  " + "\n  ".join(failures[-6:]) +
        f"\nFix: (1) re-run later; (2) open the first URL in a browser; if it works, save the file as\n  {dest}\n"
        "and re-run this cell (cached files are used automatically)."
    )


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise ActionableError(msg)


def require_files(paths, hint: str = "") -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise ActionableError("Missing input file(s):\n  " + "\n  ".join(missing) + (f"\n{hint}" if hint else ""))


def require_columns(df: pd.DataFrame, cols, name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ActionableError(f"{name}: expected columns {miss} not found. Present: {list(df.columns)[:40]}")


def find_col(df: pd.DataFrame, candidates, required: bool = True):
    """Return the first column whose normalised name matches any candidate."""
    norm = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in norm:
            return norm[key]
    if required:
        raise ActionableError(f"None of {candidates} found. Columns: {list(df.columns)[:50]}")
    return None


def read_geo_series_matrix(path: Path):
    """Parse a GEO series_matrix(.gz). Returns (sample_meta DataFrame, expression DataFrame probes x samples)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    meta, chars, table_lines, in_table = {}, [], [], False
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                table_lines.append(line)
            elif line.startswith("!Sample_"):
                parts = line.split("\t")
                key = parts[0][len("!Sample_"):]
                vals = [v.strip().strip('"') for v in parts[1:]]
                if key == "characteristics_ch1":
                    chars.append(vals)
                else:
                    k, n = key, 1
                    while k in meta:
                        n += 1
                        k = f"{key}_{n}"
                    meta[k] = vals
    require(bool(table_lines), f"{path.name}: no expression table found (file may be truncated or a different format).")
    expr = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="\t", index_col=0, quotechar='"')
    expr.index = expr.index.astype(str).str.strip('"')
    expr.columns = [str(c).strip('"') for c in expr.columns]
    m = pd.DataFrame(meta)
    require("geo_accession" in m.columns, f"{path.name}: no sample accessions in header.")
    m.index = m["geo_accession"]
    for i, vals in enumerate(chars):
        s = pd.Series(vals, index=m.index)
        keyed = s.str.extract(r"^([^:]+):\s*(.*)$")
        if keyed[0].notna().mean() > 0.9 and keyed[0].nunique() == 1:
            m[str(keyed[0].iloc[0]).strip().lower().replace(" ", "_")] = keyed[1]
        else:
            m[f"characteristics_{i}"] = s
    common = [c for c in expr.columns if c in m.index]
    require(len(common) == len(m), f"{path.name}: sample columns do not match header ({len(common)} vs {len(m)}).")
    return m, expr[m.index]


def run_r(script: Path, args=None, timeout: int = 7200) -> str:
    args = [str(a) for a in (args or [])]
    log(f"Rscript {Path(script).name} {' '.join(args)}")
    try:
        r = subprocess.run(["Rscript", "--vanilla", str(script), *args], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise ActionableError("Rscript not found. Run NB00 (it installs R packages) or use a Colab runtime with R.") from exc
    with open(P("logs", "r_log.txt"), "a") as fh:
        fh.write(f"\n===== {now_utc()} {Path(script).name} {args}\n{r.stdout}\n{r.stderr}\n")
    if r.returncode != 0:
        raise ActionableError(f"R script {Path(script).name} failed (exit {r.returncode}). Last stderr lines:\n{r.stderr[-2500:]}")
    return r.stdout


def write_r(name: str, code: str) -> Path:
    p = P("config", "r_scripts", name)
    p.write_text(code)
    return p


def savefig(fig, stem: str, folder: str = "figures", dpi: int = 300) -> None:
    out = P(folder)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    log(f"figure saved: {out / stem}.(png|pdf|svg)")


def stop_if_smoke(reason: str) -> None:
    if SMOKE:
        print(f"SMOKE MODE: stopping here by design. {reason}\nSet smoke_test=false in NB00 for the full analysis.")
        raise SystemExit(0)


def sample_group_labels(series: pd.Series, rules: dict) -> pd.Series:
    """Map free-text labels to groups using ordered {group: [lowercase substrings]} rules."""
    lowered = series.fillna("").astype(str).str.lower()
    out = pd.Series("other", index=series.index)
    for group, needles in rules.items():
        mask = pd.Series(False, index=series.index)
        for n in needles:
            mask |= lowered.str.contains(n, regex=False)
        out[mask & (out == "other")] = group
    return out


def canonical_hash(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(map(str, paths)):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def freeze_definitions(files: list, params: dict) -> dict:
    """Pre-register gene-set definitions before survival analysis. Refuses to overwrite."""
    p = P("config", "frozen_definitions.json")
    require(not p.exists(), f"Definitions already frozen at {p}. Delete it only if you deliberately re-register "
                            "(and note this in the manuscript).")
    payload = {"frozen_at_utc": now_utc(), "files": [str(f) for f in files],
               "sha256": canonical_hash(files), "params": params}
    p.write_text(json.dumps(payload, indent=2))
    manifest_update("preregistration", "frozen_definitions", payload)
    return payload


def verify_frozen() -> dict:
    p = P("config", "frozen_definitions.json")
    require(p.exists(), "Gene-set definitions are not frozen. Run NB10 first (it freezes S, M and R).")
    payload = json.loads(p.read_text())
    require(canonical_hash(payload["files"]) == payload["sha256"],
            "Frozen gene-set files changed after freezing. Restore them or re-register explicitly.")
    return payload


def zscore_df(df: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
    """z-score each row (axis=1: across samples for each gene)."""
    mu = df.mean(axis=axis)
    sd = df.std(axis=axis).replace(0, np.nan)
    if axis == 1:
        return df.sub(mu, axis=0).div(sd, axis=0)
    return df.sub(mu, axis=1).div(sd, axis=1)


def score_geneset(expr_z: pd.DataFrame, genes) -> pd.Series:
    """Mean z-score over the genes present (genes x samples). Returns NaN if <3 genes found."""
    present = [g for g in genes if g in expr_z.index]
    if len(present) < 3:
        return pd.Series(np.nan, index=expr_z.columns)
    return expr_z.loc[present].mean(axis=0)


def dl_meta(Y, V):
    """DerSimonian-Laird random-effects meta-analysis. Y, V: arrays (n_items x n_studies) of estimates and variances."""
    from scipy import stats as _st
    Y = np.asarray(Y, float); V = np.asarray(V, float)
    k = Y.shape[1]
    w = 1 / V; sw = w.sum(1)
    mu_fe = (w * Y).sum(1) / sw
    se_fe = np.sqrt(1 / sw)
    if k == 1:
        z = mu_fe / se_fe
        return dict(mu=mu_fe, se=se_fe, p=2 * _st.norm.sf(np.abs(z)), Q=np.zeros(len(mu_fe)), I2=np.zeros(len(mu_fe)),
                    tau2=np.zeros(len(mu_fe)), mu_fe=mu_fe, se_fe=se_fe)
    Q = (w * (Y - mu_fe[:, None]) ** 2).sum(1)
    C = sw - (w ** 2).sum(1) / sw
    tau2 = np.maximum(0, (Q - (k - 1)) / C)
    ws = 1 / (V + tau2[:, None]); sws = ws.sum(1)
    mu = (ws * Y).sum(1) / sws; se = np.sqrt(1 / sws)
    with np.errstate(divide="ignore", invalid="ignore"):
        I2 = np.where(Q > 0, np.maximum(0, (Q - (k - 1)) / Q) * 100, 0.0)
    return dict(mu=mu, se=se, p=2 * _st.norm.sf(np.abs(mu / se)), Q=Q, I2=I2, tau2=tau2, mu_fe=mu_fe, se_fe=se_fe)


R_COX = r"""
suppressMessages({library(survival); library(data.table)})
args <- commandArgs(TRUE)
data_f <- args[1]; pred_f <- args[2]; adj <- args[3]; out_f <- args[4]
d <- as.data.frame(fread(data_f))
preds <- readLines(pred_f); preds <- preds[nzchar(preds)]
adj_terms <- if (nzchar(adj)) strsplit(adj, ",")[[1]] else character(0)
adj_terms <- adj_terms[adj_terms %in% colnames(d)]
rows <- list()
for (p in preds) {
  if (!(p %in% colnames(d))) next
  cols <- c("time", "event", p, adj_terms)
  dd <- d[complete.cases(d[, cols, drop = FALSE]), cols, drop = FALSE]
  colnames(dd)[3] <- "PRED"
  if (nrow(dd) < 20 || sum(dd$event) < 5 || sd(dd$PRED) == 0) next
  f <- as.formula(paste("Surv(time, event) ~ PRED", if (length(adj_terms)) paste("+", paste(adj_terms, collapse = " + ")) else ""))
  fit <- tryCatch(coxph(f, data = dd), error = function(e) NULL)
  if (is.null(fit)) next
  s <- summary(fit)$coefficients["PRED", ]
  ph <- tryCatch(cox.zph(fit)$table["PRED", "p"], error = function(e) NA_real_)
  rows[[length(rows) + 1]] <- data.frame(predictor = p, n = nrow(dd), events = sum(dd$event), n_params = length(coef(fit)),
        logHR = s["coef"], SE = s["se(coef)"], p = s["Pr(>|z|)"], ph_p = ph, adjusted_for = paste(adj_terms, collapse = ","))
}
out <- if (length(rows)) do.call(rbind, rows) else data.frame()
write.csv(out, out_f, row.names = FALSE)
cat("cox models fitted:", nrow(out), "of", length(preds), "\n")
"""

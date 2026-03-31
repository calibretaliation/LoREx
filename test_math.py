import numpy as np
from lorex.scoring import ms_spectral

# generate random data
np.random.seed(42)
trusted = np.random.randn(5000, 512)
trusted[:, 0] *= 2.0  # make variance unequal
mu = trusted.mean(axis=0)

# compute whitening and pca
from lorex.whitening import compute_whitening_matrix, whiten_and_pca
W = compute_whitening_matrix(trusted - mu, method="cholesky", ridge=1e-3).numpy()

# notebook style
tw_notebook = (trusted - mu) @ W
cov_notebook = np.cov(tw_notebook.T)
ev, evec = np.linalg.eigh(cov_notebook)
idx = np.argsort(ev)[::-1]
top_eigvec = evec[:, idx[0]]

# script style
_, _, _, ev_ms, evec_ms = whiten_and_pca(torch.tensor(trusted).float())

print("Top eigvec diff:", np.max(np.abs(np.abs(top_eigvec) - np.abs(evec_ms[:, 0]))))

cw = np.random.randn(200, 512)
pw = np.random.randn(200, 512)
pw[:, 0] += 5.0 # add signal in top direction

# notebook score
z_cw = (cw - mu) @ W
z_pw = (pw - mu) @ W
score_cw_note = np.abs(z_cw @ top_eigvec)
score_pw_note = np.abs(z_pw @ top_eigvec)

# ms spectral score k=1
from lorex.scoring import spectral_score
st = spectral_score(tw_notebook, evec_ms, ev_ms, 1)
sc = spectral_score(z_cw, evec_ms, ev_ms, 1)
sp = spectral_score(z_pw, evec_ms, ev_ms, 1)

print("Notebook AUC:", __import__('sklearn.metrics').metrics.roc_auc_score(
    np.concatenate([np.zeros(200), np.ones(200)]),
    np.concatenate([score_cw_note, score_pw_note])
))
print("MS Spectral raw AUC K=1:", __import__('sklearn.metrics').metrics.roc_auc_score(
    np.concatenate([np.zeros(200), np.ones(200)]),
    np.concatenate([sc, sp])
))

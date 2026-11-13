"""Zone-level and city-level error metrics."""
import numpy as np

from ..config import MIN_DAY, ND, NH, Z
from ..features.calendar import calendar


def agg_weights(D, target):
    return np.ones(Z) if target == "volume" else D["st"]["cc"].values.astype(float)


def peak_hours(Y3, w, d_train_end, k=6):
    city = (Y3[MIN_DAY : d_train_end + 1] * w).sum(-1).mean(0)
    return np.argsort(-city)[:k]


def truth(D, target):
    cc = D["st"]["cc"].values.astype(float)
    return (D["V"] if target == "volume" else D["O"] / cc).reshape(ND, 24, Z)


class Evaluator:

    def __init__(self, D, target, days, fold_end):
        self.Y3 = truth(D, target)
        self.w = agg_weights(D, target)
        self.days = np.asarray(days)
        self.fe = np.asarray(fold_end)
        self.A = self.Y3[self.days]
        self.cal = calendar().iloc[self.days].reset_index(drop=True)
        self.sc = {}
        for fe in np.unique(self.fe):
            Y = self.Y3.reshape(NH, Z)
            t0, t1 = (MIN_DAY * 24, (fe + 1) * 24)
            dz = np.abs(Y[t0:t1] - Y[t0 - 24 : t1 - 24])
            city = (Y * self.w).sum(1)
            dc = np.abs(city[t0:t1] - city[t0 - 24 : t1 - 24])
            self.sc[fe] = (dz.mean(0), dc.mean(), peak_hours(self.Y3, self.w, fe))

    def daily(self, P):
        ae = np.abs(P - self.A)
        ec = np.abs(((P - self.A) * self.w).sum(-1))
        return (ae.mean((1, 2)), ec.mean(1))

    def metrics(self, P, mask=None):
        m = np.ones(len(self.days), bool) if mask is None else mask
        if m.sum() == 0:
            return {}
        S = dict(
            ae=0.0,
            a=0.0,
            e2=0.0,
            n=0,
            mase_n=0.0,
            nh=0,
            pk_n=0.0,
            pk_d=0.0,
            c_ae=0.0,
            c_a=0.0,
            c_e2=0.0,
            c_n=0,
            c_mase=0.0,
            c_pk_n=0.0,
            c_pk_d=0.0,
            mape=[],
            bias=0.0,
        )
        for fe in np.unique(self.fe[m]):
            k = m & (self.fe == fe)
            sz, sc, pk = self.sc[fe]
            Pk, Ak = (P[k], self.A[k])
            e = Pk - Ak
            ae = np.abs(e)
            Pc, Ac = ((Pk * self.w).sum(-1), (Ak * self.w).sum(-1))
            ec = Pc - Ac
            S["ae"] += ae.sum()
            S["a"] += np.abs(Ak).sum()
            S["e2"] += (e**2).sum()
            S["n"] += e.size
            S["mase_n"] += (ae.sum((0, 1)) / sz).mean()
            S["nh"] += k.sum() * 24
            S["pk_n"] += ae[:, pk].sum()
            S["pk_d"] += np.abs(Ak[:, pk]).sum()
            S["c_ae"] += np.abs(ec).sum()
            S["c_a"] += np.abs(Ac).sum()
            S["c_e2"] += (ec**2).sum()
            S["c_n"] += ec.size
            S["c_mase"] += np.abs(ec).sum() / sc
            S["c_pk_n"] += np.abs(ec[:, pk]).sum()
            S["c_pk_d"] += np.abs(Ac[:, pk]).sum()
            S["mape"] += list(np.abs(ec.sum(1)) / Ac.sum(1))
            S["bias"] += ec.sum()
        return {
            "zh_MAE": S["ae"] / S["n"],
            "zh_RMSE": np.sqrt(S["e2"] / S["n"]),
            "zh_WAPE": S["ae"] / S["a"],
            "zh_MASE": S["mase_n"] / S["nh"],
            "zh_peakWAPE": S["pk_n"] / S["pk_d"],
            "city_MAE": S["c_ae"] / S["c_n"],
            "city_RMSE": np.sqrt(S["c_e2"] / S["c_n"]),
            "city_WAPE": S["c_ae"] / S["c_a"],
            "city_MASE": S["c_mase"] / S["c_n"],
            "city_peakWAPE": S["c_pk_n"] / S["c_pk_d"],
            "city_dailyMAPE": float(np.mean(S["mape"])),
            "city_bias": S["bias"] / S["c_a"] * 1.0,
        }

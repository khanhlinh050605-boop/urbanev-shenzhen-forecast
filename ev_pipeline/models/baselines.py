"""Naive seasonal baselines."""


def baseline_preds(B, d0, d1):
    g = B["groups"]["own"]
    s = slice(d0, d1 + 1)
    return {
        "persist_last": g["cut0"][s],
        "snaive24": g["l24"][s],
        "snaive168": g["l168"][s],
        "hour_dow_mean": g["hd_mean"][s],
        "mean7_samehour": g["hr7"][s],
    }

"""Experiments for the owner-in-the-loop site (aligned with the IRL Lab's recent work).

Exp 1  Ratings vs binary preferences for learning the owner's reward (R4-style ranking MSE,
       Kharyal, Muslimani & Taylor, ICML 2026), scored with Kendall tau-b alignment
       (Trajectory Alignment Coefficient, Muslimani et al., RLJ 2025) and with the share of
       decisions the agent takes differently from the owner.
Exp 2  The owner as a monitor (Monitored MDPs, Kazemipour, Parisi, Taylor & Bowling, ICML 2025):
       feedback is only observed when the owner is available.
The owner model reuses the portfolio's decision profile (case 01); teachers are simulated
from it, so absolute numbers are illustrative, the comparisons are the result.
"""
import json, numpy as np
from scipy.optimize import minimize
from scipy.stats import kendalltau

FEATURES = ["amount", "reversible", "novelty", "brand_risk", "urgency", "evidence"]
W_OWNER = np.array([-2.6, 1.4, -0.6, -0.2, 0.3, 0.9])   # 'spend' row of the case 01 owner profile
B_OWNER = -1.5        # this owner approves only about a third of spending proposals
NOISE = 0.35

def utility(X): return X @ W_OWNER + B_OWNER
def owner_rating(u, rng):            # 0 bad, 1 neutral, 2 good
    z = u + rng.normal(0, NOISE, len(u))
    return np.digitize(z, [-1.0, 1.0])
def owner_prefers(u1, u2, rng):
    return ((u1 - u2) + rng.normal(0, NOISE * np.sqrt(2), len(u1)) > 0).astype(float)

def fit_preferences(Xa, Xb, y):
    def loss(p):
        w = p[:-1]; d = (Xa - Xb) @ w
        return np.mean(np.logaddexp(0, -d * (2 * y - 1))) + 1e-3 * w @ w
    p = minimize(loss, np.zeros(len(FEATURES) + 1), method="L-BFGS-B").x
    return p[:-1], None                                    # preferences carry no absolute level

def soft_rank(g, t=0.3):
    return (1 / (1 + np.exp(-(g[:, None] - g[None, :]) / t))).sum(1) - 0.5
def fit_ratings_rmse(X, r):
    """Ranking MSE in the spirit of R4: soft ranks of predicted returns vs. ordinal rating ranks;
    plus thresholds between rating classes so the agent also learns the owner's absolute bar."""
    order = np.argsort(np.argsort(r, kind="stable"), kind="stable").astype(float)
    target = np.zeros(len(r))
    for c in np.unique(r):
        target[r == c] = order[r == c].mean()               # ties share the average rank
    def loss(p):
        w, b = p[:-1], p[-1]
        g = X @ w + b
        n = len(r)
        return np.mean(((soft_rank(g) - target) / n) ** 2) + 1e-4 * w @ w
    w = minimize(loss, np.zeros(len(FEATURES) + 1) + 0.01, method="L-BFGS-B").x[:-1]
    g = X @ w
    # absolute bar: midpoint between the classes 'bad' and 'good' along the learned score
    lo = g[r == 0].mean() if (r == 0).any() else g.min()
    hi = g[r == 2].mean() if (r == 2).any() else g.max()
    return w, (lo + hi) / 2

def evaluate(w, bar, Xt):
    u = utility(Xt); g = Xt @ w
    tac = kendalltau(u, g, variant="b").statistic
    owner_acts = u > 0
    agent_acts = g > (bar if bar is not None else np.median(g))   # preferences: guess the median
    return float(tac), float(100 * np.mean(owner_acts != agent_acts))

def exp1(seeds=20, budgets=(10, 20, 40, 80, 160)):
    out = {"budgets": list(budgets), "ratings": {"tac": [], "err": []}, "preferences": {"tac": [], "err": []}}
    for B in budgets:
        acc = {"ratings": [], "preferences": []}
        for s in range(seeds):
            rng = np.random.default_rng(s)
            Xt = rng.normal(size=(600, len(FEATURES)))
            X = rng.normal(size=(B, len(FEATURES)))
            r = owner_rating(utility(X), rng)
            acc["ratings"].append(evaluate(*fit_ratings_rmse(X, r), Xt))
            Xa, Xb = rng.normal(size=(B, len(FEATURES))), rng.normal(size=(B, len(FEATURES)))
            y = owner_prefers(utility(Xa), utility(Xb), rng)
            acc["preferences"].append(evaluate(*fit_preferences(Xa, Xb, y), Xt))
        for k in acc:
            a = np.array(acc[k]); out[k]["tac"].append(round(a[:, 0].mean(), 3)); out[k]["err"].append(round(a[:, 1].mean(), 1))
        print(B, {k: (out[k]["tac"][-1], out[k]["err"][-1]) for k in acc})
    return out

def exp2(seeds=8, weeks=12, per_hour_reviews=6):
    """Decisions arrive every hour (vending robots and the online store run 24/7). The owner is
    available h hours per week; only then is feedback observed (a monitored reward). Each available
    hour he reviews the waiting queue (up to per_hour_reviews items) and rates them; the agent refits
    its reward model every week from the ratings it has observed.
    Policies: wait for the owner on every decision; act alone with the learned reward;
    act alone only when confident AND reversible, otherwise wait."""
    res = []
    for h in [5, 10, 20, 30, 40, 60, 80]:
        row = {"hours": h}
        for pol in ["wait", "alone", "alone_if_safe"]:
            waits, errs, n_alone, n = [], 0, 0, 0
            labels_seen = []
            for s in range(seeds):
                rng = np.random.default_rng(100 + s)
                avail = np.zeros(168, bool); avail[rng.choice(168, h, replace=False)] = True
                X0 = rng.normal(size=(10, len(FEATURES)))
                LX, LR = list(X0), list(owner_rating(utility(X0), rng))
                w, bar = fit_ratings_rmse(np.array(LX), np.array(LR)); sd = np.std(np.array(LX) @ w) + 1e-9
                queue = []
                for t in range(weeks * 168):
                    x = rng.normal(size=len(FEATURES)); u = utility(x[None])[0]; g = x @ w
                    safe = abs(g - bar) / sd > 0.75 and x[1] > 0
                    if pol == "alone" or (pol == "alone_if_safe" and safe):
                        errs += int((g > bar) != (u > 0)); n_alone += 1
                        if pol == "alone" and avail[t % 168] and rng.random() < 0.3:
                            LX.append(x); LR.append(owner_rating(np.array([u]), rng)[0])   # occasional audit
                    else:
                        queue.append((t, x, u))
                    if avail[t % 168]:
                        for (t0, xq, uq) in queue[:per_hour_reviews]:
                            waits.append(t - t0); LX.append(xq); LR.append(owner_rating(np.array([uq]), rng)[0])
                        queue = queue[per_hour_reviews:]
                    n += 1
                    if t % 168 == 167:
                        w, bar = fit_ratings_rmse(np.array(LX[-300:]), np.array(LR[-300:])); sd = np.std(np.array(LX[-300:]) @ w) + 1e-9
                waits += [weeks * 168 - t0 for (t0, _, _) in queue]
                labels_seen.append(len(LX))
            row[pol] = {"avg_wait_h": round(float(np.mean(waits)) if waits else 0.0, 1),
                        "errors_per_100": round(100 * errs / n, 2),
                        "share_alone": round(100 * n_alone / n, 1),
                        "ratings_seen": int(np.mean(labels_seen))}
        res.append(row); print(row)
    return res

if __name__ == "__main__":
    json.dump({"exp1": exp1(), "exp2": exp2()}, open("site/data/experiments.json", "w"))

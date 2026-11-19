"""Refresh-aware earners/players forecaster for GTA VI trophy rarity.

Forecasts, for every PS5 trophy of GTA VI, its earn rate on Exophase
at a case's target, as five quantiles (5/25/50/75/95%). Python 3.13,
standard library only, no network, deterministic.

The model, in one paragraph: never forecast the rate from the last
rate. Split every rate into its two counts -- earners and
players-tracked -- and forecast each count separately, because
Exophase's page is a snapshot that refreshes on its own schedule:
when it refreshes, the tracked-player count jumps and every rate is
diluted by the newcomers, even though nobody un-earned a trophy.
So the forecast is earners-at-target divided by players-at-target.
Earners grow two ways: the players already tracked keep playing
(a per-trophy share trend fitted on the history), and the newcomers
joining before the target earn trophies as they play (a newcomer
budget that matures with the horizon, split half in proportion to
each trophy's current share and half onto the earliest trophies --
the ones new players reach first). The spread on the logit scale
combines the disagreement between those two allocation shapes, a
random-walk term that widens with the horizon, and the uncertainty
in the player-count forecast.

Calibration priors come from the kit's own published numbers
(RULES.txt / competition.yaml): launch night sat flat at 638
players tracked and 943 trophies earned, then the first refresh
showed 3,868 players and 4,276 earned. When a case's history shows
no refresh yet, the model expects one pending refresh of that
published size; the newcomer maturity rate is set so the model
reproduces that published refresh in aggregate. Once a history
shows refreshes, the player count is extrapolated from the
observed jumps instead, and the published numbers are not used.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, and the standard normal's
# quantile at each (rounded to 3 decimals, as the kit states them).
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# First refresh, as the kit documents it (launch-night page flat at
# 638 players / 943 earned, refreshed to 3,868 players / 4,276
# earned). Used only as a prior while a case's history shows no
# refresh at all; a history with refreshes extrapolates them instead.
DOC_PLAYERS_BEFORE = 638.0
DOC_PLAYERS_AFTER = 3868.0
DOC_EARNED_BEFORE = 943.0
DOC_EARNED_AFTER = 4276.0
DOC_HORIZON_MIN = 90.0

# How the newcomer budget is split: half in proportion to each
# trophy's share, half onto the earliest (highest-share) trophies.
EARLY_TOP_K = 5
EARLY_MIX = 0.5

# Spread budget, in logit units. The allocation disagreement
# (all-proportional vs all-early newcomer split) is the calibrated
# term: it reproduces the full range of plausible refresh outcomes
# the kit's own numbers allow. The other two terms must stay SMALL
# corrections to it, not dominate it -- at WALK_SIGMA 0.5 and
# PLAYERS_SIGMA_FLAT 0.15 they supplied 83% of the median trophy's
# variance, the 90% band covered 98% of the practice answers, and
# the long-horizon band saturated SIGMA_CAP on 52 of 54 trophies.
# So the walk scale is halved (0.25 per root-day still leaves a
# 0.53 logit walk term over the 4.5-day final horizon), and the
# flat-branch player term is 0.05: that forecast is pinned by the
# kit's published exact refresh (638 -> 3,868 players), not an
# extrapolation, so its residual uncertainty is only the rounding
# of the published figures. The trend branch keeps 0.25 -- jumps
# extrapolated from past refreshes stay genuinely uncertain.
WALK_SIGMA = 0.25
PLAYERS_SIGMA_FLAT = 0.05
PLAYERS_SIGMA_TREND = 0.25
SIGMA_FLOOR = 0.10
SIGMA_CAP = 1.20
SATURATED_SIGMA = 0.02
SATURATED_SHARE = 0.995


def clamp_pct(x):
    """A percent into the range the logit can take."""
    return min(99.99, max(0.01, x))


def logit_pct(pct):
    """logit of a percent, clamped first (0% and 100% have no logit)."""
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    """Percent from a logit value, kept inside (0, 100)."""
    if lg >= 0.0:
        p = 1.0 / (1.0 + math.exp(-lg))
    else:
        e = math.exp(lg)
        p = e / (1.0 + e)
    return 100.0 * p


def round4_half_up(x):
    """Round to 4 decimals, halves up, as the kit's reference does."""
    scaled = math.floor(x * 10000.0 + 0.5) / 10000.0
    return int(scaled) if float(scaled).is_integer() else scaled


def minutes_since_epoch(text):
    """ISO-8601 timestamp (with a trailing Z) to minutes since epoch."""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() / 60.0


def read_minutes(case, fallback):
    """Minutes of a case's checkpoint; falls back to its first read."""
    try:
        return minutes_since_epoch(case["at"])
    except (KeyError, TypeError, ValueError):
        return fallback


def latest_rates(case):
    """Per-trophy latest non-null rate, walking the history backwards.

    Returns (players_last, earned_last, rates) with rates aligned to
    case["trophies"]; a trophy never seen gets None.
    """
    history = case.get("history") or []
    n = len(case["trophies"])
    rates = [None] * n
    players_last = None
    earned_last = None
    for read in reversed(history):
        if players_last is None:
            players_last = read.get("players_tracked")
            earned_last = read.get("trophies_earned")
        row = read.get("rates") or []
        for i in range(min(n, len(row))):
            if rates[i] is None and row[i] is not None:
                rates[i] = row[i]
        if all(r is not None for r in rates):
            break
    return players_last, earned_last, rates


def whole_earners(rates, players, earned_total):
    """Integer earners per trophy from the last rates.

    Rates are printed to 2 decimals, so rate * players / 100 is a
    near-integer count. Round, then fix the total with
    largest-remainder so it matches the page's earned total.
    """
    raw = []
    for r in rates:
        raw.append(0.0 if r is None else r * players / 100.0)
    counts = [int(math.floor(v + 0.5)) for v in raw]
    if earned_total is not None:
        target = int(earned_total)
        drift = target - sum(counts)
        if drift != 0 and abs(drift) <= len(counts):
            order = sorted(range(len(raw)),
                           key=lambda i: (raw[i] - counts[i], -i),
                           reverse=(drift > 0))
            step = 1 if drift > 0 else -1
            for i in order[:abs(drift)]:
                counts[i] = max(0, counts[i] + step)
    return counts


def refresh_jumps(case):
    """(minutes, players) at each point the tracked count changed."""
    points = []
    seen = None
    for read in case.get("history") or []:
        players = read.get("players_tracked")
        if players is None:
            continue
        if seen is None or players != seen:
            points.append((minutes_since_epoch(read["slot"]), float(players)))
            seen = players
    return points


def players_at_target(case, horizon_min, players_now):
    """Forecast players-tracked at the target, and its sigma.

    Flat history (no refresh seen yet): one pending refresh of the
    documented first-refresh size. History with jumps: compound the
    recent jump ratios over the expected number of further jumps.
    """
    points = refresh_jumps(case)
    if len(points) < 2:
        ratio = DOC_PLAYERS_AFTER / DOC_PLAYERS_BEFORE
        return players_now * ratio, PLAYERS_SIGMA_FLAT
    ratios = []
    gaps = []
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        if p0 > 0 and p1 > p0 and t1 > t0:
            ratios.append(math.log(p1 / p0))
            gaps.append(t1 - t0)
    if not ratios or not gaps:
        return players_now, PLAYERS_SIGMA_TREND
    recent = ratios[-3:]
    rho = math.exp(sum(recent) / len(recent))
    gaps_sorted = sorted(gaps)
    gap = gaps_sorted[len(gaps_sorted) // 2]
    jumps = horizon_min / gap if gap > 0 else 0.0
    jumps = min(max(jumps, 0.0), 40.0)
    forecast = players_now * (rho ** jumps)
    forecast = min(max(forecast, players_now), players_now * 200.0)
    return forecast, PLAYERS_SIGMA_TREND


def share_trend(case, idx, rate_now, origin_min, target_min):
    """A trophy's share among already-tracked players at the target.

    Least-squares line of logit(share) against log hours since the
    game's release, over the last day of reads; read at the target.
    Organic earning only raises a share, so the fit is clamped to
    never fall below today's share and never more than triple it.
    Flat history gives a flat line, so this term is exactly zero
    there and all movement comes from dilution plus newcomers.
    """
    if rate_now is None or rate_now <= 0.0:
        return 0.0
    share_now = rate_now / 100.0
    if share_now >= SATURATED_SHARE:
        return share_now
    at_min = read_minutes(case, origin_min)
    xs = []
    ys = []
    for read in case.get("history") or []:
        if read.get("slot") is None:
            continue
        slot = minutes_since_epoch(read["slot"])
        if slot < at_min - 24.0 * 60.0:
            continue
        row = read.get("rates") or []
        if idx >= len(row) or row[idx] is None:
            continue
        share = row[idx] / 100.0
        if share <= 0.001 or share >= 0.999:
            continue
        hours = max(0.0, (slot - origin_min) / 60.0)
        xs.append(math.log1p(hours))
        ys.append(logit_pct(row[idx]))
    if len(xs) < 2 or max(xs) - min(xs) < 1e-9:
        return share_now
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    slope = num / den if den > 0 else 0.0
    intercept = mean_y - slope * mean_x
    hours_target = max(0.0, (target_min - origin_min) / 60.0)
    fitted = inv_logit_pct(intercept + slope * math.log1p(hours_target)) / 100.0
    cap = min(0.99, 3.0 * share_now)
    return min(max(fitted, share_now), cap)


def maturity_tau_min(shares):
    """Minutes for newcomers to reach the tracked cohort's maturity.

    Set from the documented first refresh: the newcomers in it
    earned 103 trophies beyond the one every player holds, over a
    90-minute horizon, so the maturity factor at 90 minutes is
    103 / (3,230 newcomers x the cohort's mean trophies each).
    A case with no measurable shares falls back to a one-day prior.
    """
    mean_share = sum(shares) / len(shares) if shares else 0.0
    if mean_share <= 0.0:
        return 1440.0
    new_players = DOC_PLAYERS_AFTER - DOC_PLAYERS_BEFORE
    extra = DOC_EARNED_AFTER - DOC_EARNED_BEFORE - new_players
    mature_at_90 = extra / (new_players * mean_share * len(shares))
    mature_at_90 = min(max(mature_at_90, 0.005), 0.9)
    return -DOC_HORIZON_MIN / math.log(1.0 - mature_at_90)


def forecast_case(case):
    """One case's forecast: {"trophies": [{"id", "q": [5]}]}."""
    trophies = case["trophies"]
    n = len(trophies)
    horizon = float(case.get("horizon_minutes") or 90)
    horizon = max(horizon, 1.0)
    players_last, earned_last, rates = latest_rates(case)
    if players_last is None or players_last <= 0:
        players_last = DOC_PLAYERS_BEFORE
        rates = [0.5 if r is None else r for r in rates]
    players_now = float(players_last)

    origin_min = minutes_since_epoch(case["game"]["release_date"] + "T00:00:00Z") \
        if case.get("game", {}).get("release_date") else 0.0
    target_min = read_minutes(case, origin_min) + horizon

    earners = whole_earners(rates, players_now, earned_last)
    shares = [e / players_now for e in earners]
    saturated = [s >= SATURATED_SHARE for s in shares]

    players_target, players_sigma = players_at_target(case, horizon, players_now)
    newcomers = max(0.0, players_target - players_now)

    trend = [share_trend(case, i, rates[i], origin_min, target_min)
             for i in range(n)]
    # The trend never lowers a saturated trophy's share.
    for i in range(n):
        if saturated[i]:
            trend[i] = shares[i]

    # Newcomer budget: how many trophies the newcomers hold by the
    # target, from their maturity over the horizon times the cohort's
    # mean trophies each at the target-time profile.
    open_shares = [trend[i] for i in range(n) if not saturated[i]]
    tau = maturity_tau_min(open_shares)
    maturity = 1.0 - math.exp(-horizon / tau)
    budget = newcomers * sum(open_shares) * maturity

    # Allocation weights: proportional to the target-time share
    # (with a small floor so a never-earned trophy is not frozen at
    # zero), and the early-trophy half onto the highest shares.
    mean_trend = (sum(open_shares) / len(open_shares)) if open_shares else 0.0
    floor = 0.02 * mean_trend
    prop_w = []
    for i in range(n):
        prop_w.append(0.0 if saturated[i] else trend[i] + floor)
    prop_total = sum(prop_w) or 1.0
    prop_w = [w / prop_total for w in prop_w]
    ranked = sorted((i for i in range(n) if not saturated[i]),
                    key=lambda i: (-trend[i], trophies[i]["id"]))
    early = set(ranked[:EARLY_TOP_K])
    early_w = [(1.0 / EARLY_TOP_K if i in early else 0.0) for i in range(n)]

    walk_sigma = WALK_SIGMA * math.sqrt(horizon / 1440.0)
    rows = []
    for i in range(n):
        if saturated[i]:
            median = 99.99
            sigma = SATURATED_SIGMA
        else:
            old_earners = players_now * trend[i]
            alloc = (1.0 - EARLY_MIX) * prop_w[i] + EARLY_MIX * early_w[i]
            new_earners = budget * alloc
            rate = (old_earners + new_earners) / players_target * 100.0
            median = clamp_pct(rate)
            # Spread: the two allocation shapes disagree by this much
            # in logit units; the band spans that disagreement.
            all_prop = (old_earners + budget * prop_w[i]) / players_target * 100.0
            all_early = (old_earners + budget * early_w[i]) / players_target * 100.0
            alloc_sigma = abs(logit_pct(all_prop) - logit_pct(all_early)) / 3.29
            sigma = math.sqrt(alloc_sigma ** 2 + walk_sigma ** 2 + players_sigma ** 2)
            sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
        centre = logit_pct(median)
        quantiles = []
        prev = 0.01
        for z in Z_SCORES:
            q = round4_half_up(clamp_pct(inv_logit_pct(centre + z * sigma)))
            q = max(q, prev)
            quantiles.append(q)
            prev = q
        rows.append({"id": trophies[i]["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()

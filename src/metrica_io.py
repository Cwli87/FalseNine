"""Loading and cleaning Metrica sample data (used by notebooks 07-08)."""
import csv
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

FPS = 25
PITCH_L, PITCH_W = 105.0, 68.0


def _find(met_dir, game, kind):
    hits = list(Path(met_dir).rglob(f"Sample_Game_{game}_{kind}.csv"))
    if not hits:
        raise FileNotFoundError(f"Sample_Game_{game}_{kind}.csv not found under {met_dir}")
    return hits[0]


def _read_team(met_dir, game, team):
    path = _find(met_dir, game, f"RawTrackingData_{team}_Team")
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)                                     # row 1: team name
        jerseys = [x for x in next(reader) if x != ""]   # row 2: jersey numbers
        cols = next(reader)                              # row 3: column names
    for i, j in enumerate(jerseys):
        cols[3 + 2 * i] = f"{team}_{j}_x"
        cols[4 + 2 * i] = f"{team}_{j}_y"
    cols[-2], cols[-1] = "ball_x", "ball_y"
    return pd.read_csv(path, names=cols, skiprows=3, index_col="Frame")


def load_game(met_dir, game_id=1):
    """Tracking in metres (origin = centre spot), period 2 rotated so Home always attacks right."""
    home, away = _read_team(met_dir, game_id, "Home"), _read_team(met_dir, game_id, "Away")
    g = home.join(away.drop(columns=["Period", "Time [s]", "ball_x", "ball_y"]))
    xcols = [c for c in g.columns if c.endswith("_x")]
    ycols = [c for c in g.columns if c.endswith("_y")]
    g[xcols] = (g[xcols] - 0.5) * PITCH_L
    g[ycols] = (0.5 - g[ycols]) * PITCH_W
    flip = g.Period == 2
    g.loc[flip, xcols] *= -1
    g.loc[flip, ycols] *= -1

    home_ids = sorted({c.split("_")[1] for c in g.columns if c.startswith("Home_")}, key=int)
    away_ids = sorted({c.split("_")[1] for c in g.columns if c.startswith("Away_")}, key=int)
    first10 = g[g.Period == 1].iloc[: FPS * 600]
    hm = {i: first10[f"Home_{i}_x"].mean() for i in home_ids}
    am = {i: first10[f"Away_{i}_x"].mean() for i in away_ids}
    home_gk = min((k for k in hm if pd.notna(hm[k])), key=hm.get)    # Home defends the left goal
    away_gk = max((k for k in am if pd.notna(am[k])), key=am.get)    # Away defends the right goal

    ev = pd.read_csv(_find(met_dir, game_id, "RawEventsData"))
    sign = np.where(ev.Period == 2, -1, 1)
    for n, (cx, cy) in {"s": ("Start X", "Start Y"), "e": ("End X", "End Y")}.items():
        ev[n + "x"] = (ev[cx] - 0.5) * PITCH_L * sign              # event coordinates in metres, same orientation
        ev[n + "y"] = (0.5 - ev[cy]) * PITCH_W * sign
    ev = ev.sort_values("Start Frame", kind="stable").reset_index(drop=True)

    return SimpleNamespace(tracking=g, events=ev, home_ids=home_ids, away_ids=away_ids,
                           home_gk=home_gk, away_gk=away_gk)


def in_play_flag(index, events):
    """True while the ball is in play: restart = SET PIECE, stop = BALL OUT / FAULT RECEIVED / goal."""
    marks = []
    for t, st, f in zip(events.Type, events.Subtype, events["Start Frame"]):
        if t == "SET PIECE":
            marks.append((int(f), True))
        elif t in ("BALL OUT", "FAULT RECEIVED") or (t == "SHOT" and "GOAL" in str(st).upper()):
            marks.append((int(f), False))
    marks.sort(key=lambda m: m[0])
    s = pd.Series(False, index=index)
    for (f, state), nxt in zip(marks, marks[1:] + [(int(index.max()) + 1, None)]):
        s.loc[f:nxt[0] - 1] = state
    return s


def possession_flag(index, events, in_play):
    """'Home' / 'Away' = team that last played the ball, 'none' = loose ball, 'dead' = ball not in play."""
    s = pd.Series(np.nan, index=index, dtype=object)
    valid = set(index)
    for t, team, f in zip(events.Type, events.Team, events["Start Frame"]):
        f = int(f)
        if f not in valid: continue
        if t in ("PASS", "SHOT", "RECOVERY", "SET PIECE"): s.loc[f] = team
        elif t == "BALL LOST": s.loc[f] = "none"
    s = s.ffill().fillna("none")
    s[~in_play.values] = "dead"
    return s


def draw_pitch(ax):
    import matplotlib.pyplot as plt
    L, W = PITCH_L / 2, PITCH_W / 2
    ax.plot([-L, L, L, -L, -L], [-W, -W, W, W, -W], "w-", lw=1.5); ax.plot([0, 0], [-W, W], "w-", lw=1.5)
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, ec="w", lw=1.5))
    for s in (-1, 1):
        ax.plot([s * L, s * (L - 16.5), s * (L - 16.5), s * L], [-20.16, -20.16, 20.16, 20.16], "w-", lw=1.5)
        ax.plot([s * L, s * (L - 5.5), s * (L - 5.5), s * L], [-9.16, -9.16, 9.16, 9.16], "w-", lw=1.5)
    ax.set_facecolor("#2e7d32"); ax.set_xlim(-57, 57); ax.set_ylim(-38, 38); ax.set_aspect("equal")
"""
RIVAL Rating Engine — engine.py
FastAPI service. Single responsibility: run the RIVAL formula and write results.
Called via Supabase webhook when both captains confirm a match.
"""

import math
import os
import logging
from typing import Literal, Optional
from datetime import date

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, model_validator
from supabase import create_client, Client

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rival.engine")

# ─────────────────────────────────────────────────────────────
# SUPABASE CLIENT (service role — bypasses RLS for writes)
# ─────────────────────────────────────────────────────────────

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="RIVAL Rating Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to Supabase project URL in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# MATCH WEIGHTS
# Universal across all sports. Same as context document Part 2.
# ─────────────────────────────────────────────────────────────

MATCH_WEIGHTS: dict[str, float] = {
    "league":               1.0,
    "tournament_group":     1.1,
    "tournament_knockout":  1.3,
    "scrimmage":            0.6,
    "self_reported":        0.6,
    "unrated":              0.0,
}

# ─────────────────────────────────────────────────────────────
# PYDANTIC MODELS — request/response
# ─────────────────────────────────────────────────────────────

SportFormat = Literal["race_to_x", "timed", "running_score", "sets_and_games"]
MatchType   = Literal["league", "tournament_group", "tournament_knockout", "scrimmage", "self_reported", "unrated"]


class SportParams(BaseModel):
    """Parameters sourced from the sports table — never hardcoded."""
    k_value:        float = 70.0
    max_pd:         float = 20.0
    upset_divisor:  float = 800.0
    elo_divisor:    float = 400.0
    r0:             float = 1000.0


class PlayerRef(BaseModel):
    player_id:      str
    current_rating: float


class CalculateRequest(BaseModel):
    match_id:       str
    sport_id:       str
    game_format:    SportFormat
    match_type:     MatchType
    score_data:     dict           # raw JSONB from matches table
    race_target:    Optional[int] = None
    sport_params:   SportParams
    players_a:      list[PlayerRef]
    players_b:      list[PlayerRef]

    @model_validator(mode="after")
    def validate_race_target(self):
        if self.game_format == "race_to_x" and self.race_target is None:
            raise ValueError("race_target required for race_to_x format")
        if self.game_format != "race_to_x" and self.race_target is not None:
            raise ValueError("race_target only valid for race_to_x format")
        return self

    @field_validator("players_a", "players_b")
    @classmethod
    def non_empty_roster(cls, v):
        if not v:
            raise ValueError("Each team must have at least one player")
        return v


class PlayerResult(BaseModel):
    player_id:      str
    team:           Literal["a", "b"]
    rating_before:  float
    rating_after:   float
    rating_change:  float


class CalculateResponse(BaseModel):
    match_id:       str
    team_a_rating:  float
    team_b_rating:  float
    normalised_pd:  float
    delta_a:        float
    delta_b:        float
    zero_sum_check: float   # must be ~0
    players:        list[PlayerResult]


# ─────────────────────────────────────────────────────────────
# SCORE NORMALISATION
# The only sport-specific part of the system.
# Converts raw match scores → standardised PD in [-max_pd, +max_pd].
# Positive = team A is ahead.
# ─────────────────────────────────────────────────────────────

def normalise_score(
    score_data:   dict,
    game_format:  SportFormat,
    race_target:  Optional[int],
    max_pd:       float,
) -> float:
    """
    Returns a normalised point differential in [-max_pd, +max_pd].
    Positive means team A performed better.
    """

    def cap(value: float) -> float:
        return max(-max_pd, min(max_pd, value))

    if game_format == "race_to_x":
        # score_data: { "team_a": int, "team_b": int }
        # Winner always equals race_target.
        score_a = float(score_data["team_a"])
        score_b = float(score_data["team_b"])

        winner_score = max(score_a, score_b)
        if winner_score != race_target:
            raise ValueError(
                f"Winner score {winner_score} != race_target {race_target}"
            )
        if min(score_a, score_b) >= race_target:
            raise ValueError(
                f"Loser score {min(score_a, score_b)} must be < race_target {race_target}"
            )

        pd_raw  = score_a - score_b
        pd_norm = (pd_raw / race_target) * max_pd
        return cap(pd_norm)

    elif game_format == "timed":
        # score_data: { "team_a": int, "team_b": int }
        # No denominator available — raw differential, capped.
        pd_norm = float(score_data["team_a"]) - float(score_data["team_b"])
        return cap(pd_norm)

    elif game_format == "running_score":
        # Futsal: goals scored
        # score_data: { "team_a": int, "team_b": int }
        pd_norm = float(score_data["team_a"]) - float(score_data["team_b"])
        return cap(pd_norm)

    elif game_format == "sets_and_games":
        # Padel: score_data: { "sets": [[6,3],[4,6],[6,2]] }
        # Each element = [team_a_games, team_b_games] for that set.
        sets: list[list[int]] = score_data["sets"]
        if not sets or len(sets) < 2:
            raise ValueError("Padel match must have at least 2 sets")

        sets_a   = sum(1 for a, b in sets if a > b)
        sets_b   = sum(1 for a, b in sets if b > a)
        games_a  = sum(a for a, _ in sets)
        games_b  = sum(b for _, b in sets)

        set_margin  = sets_a - sets_b
        game_margin = games_a - games_b

        W_SET  = 3.0
        W_GAME = 0.4

        pd_norm = (set_margin * W_SET) + (game_margin * W_GAME)
        return cap(pd_norm)

    else:
        raise ValueError(f"Unknown game_format: {game_format}")


# ─────────────────────────────────────────────────────────────
# CORE RATING ENGINE
# Sport-agnostic. All sport-specific logic is in normalise_score.
# ─────────────────────────────────────────────────────────────

def calculate_deltas(
    ra:             float,      # team A rating (avg of participants)
    rb:             float,      # team B rating (avg of participants)
    normalised_pd:  float,      # from normalise_score()
    match_type:     str,
    params:         SportParams,
) -> tuple[float, float]:
    """
    Returns (delta_a, delta_b). Zero-sum guaranteed: delta_a + delta_b = 0.

    Args:
        ra, rb:         Team ratings (avg of participating players)
        normalised_pd:  From normalise_score(), in [-max_pd, +max_pd]
        match_type:     Determines match weight w
        params:         K, max_pd, upset_divisor, elo_divisor from sports table
    """

    K             = params.k_value
    D             = params.elo_divisor
    max_pd        = params.max_pd
    upset_divisor = params.upset_divisor

    # Weight — unrated matches return zero delta
    w = MATCH_WEIGHTS.get(match_type, 1.0)
    if w == 0.0:
        return 0.0, 0.0

    # Step 1: Win probability (Elo)
    e_a = 1 / (1 + 10 ** ((rb - ra) / D))

    # Step 2: Expected margin
    exp_pd = (2 * e_a - 1) * max_pd

    # Step 3: Performance score
    perf = (normalised_pd - exp_pd) / max_pd

    # Step 4: Base delta
    delta_base = K * w * perf

    # Step 5: Upset multiplier
    # Fires only when the lower-rated team wins outright. Not on draws —
    # a draw is not an upset win, it's just a better-than-expected result
    # which is already captured in perf.
    a_won    = normalised_pd > 0
    is_draw  = normalised_pd == 0
    if a_won and ra < rb:
        m = 1 + abs(rb - ra) / upset_divisor   # underdog A wins
    elif (not a_won) and (not is_draw) and rb < ra:
        m = 1 + abs(rb - ra) / upset_divisor   # underdog B wins
    else:
        m = 1.0

    # Step 6: Zero-sum enforcement
    # For outright wins/losses: sign follows who won the match.
    # For draws: sign follows perf — whoever outperformed their expectation gains.
    # This correctly gives the underdog a positive delta for holding the favourite.
    if is_draw:
        sign = 1 if perf > 0 else (-1 if perf < 0 else 0)
    else:
        sign = 1 if a_won else -1

    delta_a = abs(delta_base * m) * sign
    delta_b = -delta_a

    # Assertion — this is sacred
    assert abs(delta_a + delta_b) < 1e-10, (
        f"Zero-sum violated: delta_a={delta_a}, delta_b={delta_b}, "
        f"sum={delta_a + delta_b}"
    )

    return round(delta_a, 2), round(delta_b, 2)


# ─────────────────────────────────────────────────────────────
# SUPABASE WRITES
# ─────────────────────────────────────────────────────────────

def write_rating_updates(
    match_id:   str,
    sport_id:   str,
    results:    list[PlayerResult],
    team_a_ids: set[str],
):
    """
    1. Update player_sport_ratings (current_rating, peak_rating, totals)
    2. Append to player_ratings audit trail
    3. Update match_players with rating_after and rating_change
    4. Mark matches.rating_processed = TRUE
    """

    today = date.today().isoformat()

    for r in results:
        pid = r.player_id
        new_rating = r.rating_after
        change = r.rating_change
        won = change > 0
        drew = change == 0

        # Determine win/loss/draw from perspective
        if drew:
            win_inc, loss_inc, draw_inc = 0, 0, 1
        elif won:
            win_inc, loss_inc, draw_inc = 1, 0, 0
        else:
            win_inc, loss_inc, draw_inc = 0, 1, 0

        # Upsert player_sport_ratings
        # We use raw SQL via rpc or the postgrest PATCH with returning.
        # Supabase Python SDK v2: supabase.table().update().eq().execute()
        existing = (
            supabase.table("player_sport_ratings")
            .select("current_rating, peak_rating, total_games, wins, losses, draws")
            .eq("player_id", pid)
            .eq("sport_id", sport_id)
            .maybe_single()
            .execute()
        )

        if existing.data:
            peak = max(existing.data["peak_rating"], new_rating)
            (
                supabase.table("player_sport_ratings")
                .update({
                    "current_rating":  new_rating,
                    "peak_rating":     peak,
                    "total_games":     existing.data["total_games"] + 1,
                    "wins":            existing.data["wins"]   + win_inc,
                    "losses":          existing.data["losses"] + loss_inc,
                    "draws":           existing.data["draws"]  + draw_inc,
                    "last_active_date": today,
                })
                .eq("player_id", pid)
                .eq("sport_id", sport_id)
                .execute()
            )
        else:
            # First match for this player in this sport
            (
                supabase.table("player_sport_ratings")
                .insert({
                    "player_id":       pid,
                    "sport_id":        sport_id,
                    "current_rating":  new_rating,
                    "peak_rating":     max(new_rating, 1000.0),
                    "total_games":     1,
                    "wins":            win_inc,
                    "losses":          loss_inc,
                    "draws":           draw_inc,
                    "last_active_date": today,
                })
                .execute()
            )

        # Append audit row
        team_id_query = (
            supabase.table("match_players")
            .select("team_id")
            .eq("match_id", match_id)
            .eq("player_id", pid)
            .single()
            .execute()
        )
        team_id = team_id_query.data["team_id"] if team_id_query.data else None

        (
            supabase.table("player_ratings")
            .insert({
                "player_id":    pid,
                "sport_id":     sport_id,
                "match_id":     match_id,
                "team_id":      team_id,
                "rating_before": r.rating_before,
                "rating_after":  r.rating_after,
                "rating_change": r.rating_change,
                "formula_version": "1.0",
            })
            .execute()
        )

        # Update match_players row
        (
            supabase.table("match_players")
            .update({
                "rating_after":  r.rating_after,
                "rating_change": r.rating_change,
            })
            .eq("match_id", match_id)
            .eq("player_id", pid)
            .execute()
        )

    # Mark match as processed
    (
        supabase.table("matches")
        .update({"rating_processed": True, "processed_at": "NOW()"})
        .eq("match_id", match_id)
        .execute()
    )

    log.info(f"Match {match_id} processed. {len(results)} player ratings updated.")


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Railway health check."""
    return {"status": "ok", "service": "rival-rating-engine", "version": "1.0.0"}


@app.post("/calculate", response_model=CalculateResponse)
async def calculate(req: CalculateRequest):
    """
    Run the RIVAL rating formula for a confirmed match.
    Called by Supabase webhook when confirmed_by_a AND confirmed_by_b = TRUE.

    Idempotency: checks rating_processed flag before running.
    """

    log.info(f"Processing match {req.match_id} | sport={req.sport_id} | format={req.game_format}")

    # ── Idempotency guard ──────────────────────────────────────
    existing = (
        supabase.table("matches")
        .select("rating_processed")
        .eq("match_id", req.match_id)
        .single()
        .execute()
    )
    if existing.data and existing.data.get("rating_processed"):
        log.warning(f"Match {req.match_id} already processed — skipping")
        raise HTTPException(status_code=409, detail="Match already processed")

    # ── Score normalisation ───────────────────────────────────
    try:
        normalised_pd = normalise_score(
            score_data=req.score_data,
            game_format=req.game_format,
            race_target=req.race_target,
            max_pd=req.sport_params.max_pd,
        )
    except (ValueError, KeyError, AssertionError) as e:
        log.error(f"Score normalisation failed for match {req.match_id}: {e}")
        raise HTTPException(status_code=422, detail=f"Score validation error: {e}")

    # ── Team ratings (average of participants) ────────────────
    ra = sum(p.current_rating for p in req.players_a) / len(req.players_a)
    rb = sum(p.current_rating for p in req.players_b) / len(req.players_b)

    # ── Rating delta ──────────────────────────────────────────
    delta_a, delta_b = calculate_deltas(
        ra=ra,
        rb=rb,
        normalised_pd=normalised_pd,
        match_type=req.match_type,
        params=req.sport_params,
    )

    # ── Build per-player results ──────────────────────────────
    player_results: list[PlayerResult] = []

    for p in req.players_a:
        player_results.append(PlayerResult(
            player_id=p.player_id,
            team="a",
            rating_before=p.current_rating,
            rating_after=round(p.current_rating + delta_a, 2),
            rating_change=delta_a,
        ))

    for p in req.players_b:
        player_results.append(PlayerResult(
            player_id=p.player_id,
            team="b",
            rating_before=p.current_rating,
            rating_after=round(p.current_rating + delta_b, 2),
            rating_change=delta_b,
        ))

    # ── Write to Supabase ─────────────────────────────────────
    team_a_ids = {p.player_id for p in req.players_a}
    write_rating_updates(req.match_id, req.sport_id, player_results, team_a_ids)

    return CalculateResponse(
        match_id=req.match_id,
        team_a_rating=round(ra, 2),
        team_b_rating=round(rb, 2),
        normalised_pd=round(normalised_pd, 4),
        delta_a=delta_a,
        delta_b=delta_b,
        zero_sum_check=round(delta_a + delta_b, 12),
        players=player_results,
    )
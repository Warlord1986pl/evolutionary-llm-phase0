"""Phase 1b pilot run — z-score bounded sigmoid population mechanics.

Runs a short 12-generation pilot in the desert biome with a new P_rep / P_death
formulation that replaces the exponential curves from Phase 1.  After the run,
a validation report is printed and saved to experiments/phase1b_pilot/.

New mechanics (override biome_runner's default Bernoulli selection):
    z_i        = (f_i - mean_f) / max(std_f, 0.01)
    P_rep_i    = 0.05 + (0.45 - 0.05)  * sigmoid( 1.0 * z_i)
    P_death_i  = 0.07 + (0.45 - 0.07)  * sigmoid(-1.5 * z_i)

Usage (from repo root):
    python scripts/phase1b_pilot.py
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Pilot configuration (hardcoded)
# ---------------------------------------------------------------------------

BIOME: str = "desert"
GENERATIONS: int = 12
STARTING_AGENTS: int = 12
DOCS_PER_AGENT: int = 30
K_MAX: int = 30
OUTPUT_DIR: str = "experiments/phase1b_pilot"
CONFIG_PATH: str = "config/phase1_single_model.yaml"
CORPUS_MANIFEST: str = "data/v2/corpus_manifest_v3.json"
SEED: int = 42

# Z-score sigmoid constants
_REP_LOW: float = 0.05
_REP_HIGH: float = 0.45
_REP_SLOPE: float = 1.0

_DEATH_LOW: float = 0.07
_DEATH_HIGH: float = 0.45
_DEATH_SLOPE: float = 1.5  # note: applied as sigmoid(-slope * z)

# Validation thresholds
_MIN_GENS_WITH_DEATHS: int = 10   # deaths > 0 in at least this many generations
_POP_LOWER: float = 15.0
_POP_UPPER: float = 29.0
_SIGMA_MIN: float = 0.01          # std_fitness must be >= this in gen 0..7
_LINEAGE_MIN_ROWS: int = GENERATIONS * STARTING_AGENTS  # 144

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("phase1b_pilot")

# ---------------------------------------------------------------------------
# Z-score bounded sigmoid selection
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    # Avoid overflow for large negative x
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _zscore_select_candidates_for_death(
    population: Any,
    rng: np.random.Generator,
) -> list[str]:
    """Z-score bounded sigmoid death selection.

    P_death_i = 0.07 + (0.45 - 0.07) * sigmoid(-1.5 * z_i)

    Parameters
    ----------
    population : Population
        Active population mapping.
    rng : np.random.Generator
        Seeded numpy RNG.

    Returns
    -------
    list[str]
        Agent IDs selected for death.
    """
    agent_ids = list(population.keys())
    fitnesses = np.array(
        [population[aid].metrics.get("fitness", 0.0) for aid in agent_ids],
        dtype=np.float64,
    )

    mean_f = float(np.mean(fitnesses))
    std_f = float(max(np.std(fitnesses), 0.01))

    dying: list[str] = []
    for aid, f in zip(agent_ids, fitnesses):
        z = (float(f) - mean_f) / std_f
        p = _DEATH_LOW + (_DEATH_HIGH - _DEATH_LOW) * _sigmoid(-_DEATH_SLOPE * z)
        if rng.random() < p:
            dying.append(aid)

    return dying


def _zscore_select_candidates_for_reproduction(
    population: Any,
    rng: np.random.Generator,
) -> list[str]:
    """Z-score bounded sigmoid reproduction selection.

    P_rep_i = 0.05 + (0.45 - 0.05) * sigmoid(1.0 * z_i)

    Parameters
    ----------
    population : Population
        Active population mapping.
    rng : np.random.Generator
        Seeded numpy RNG.

    Returns
    -------
    list[str]
        Agent IDs selected to reproduce.
    """
    agent_ids = list(population.keys())
    fitnesses = np.array(
        [population[aid].metrics.get("fitness", 0.0) for aid in agent_ids],
        dtype=np.float64,
    )

    mean_f = float(np.mean(fitnesses))
    std_f = float(max(np.std(fitnesses), 0.01))

    reproducing: list[str] = []
    for aid, f in zip(agent_ids, fitnesses):
        z = (float(f) - mean_f) / std_f
        p = _REP_LOW + (_REP_HIGH - _REP_LOW) * _sigmoid(_REP_SLOPE * z)
        if rng.random() < p:
            reproducing.append(aid)

    return reproducing


# ---------------------------------------------------------------------------
# Monkey-patch biome_runner before importing run_biome
# ---------------------------------------------------------------------------

def _patch_population_mechanics() -> None:
    """Replace the default selection functions in biome_runner's namespace."""
    import src.evolution.biome_runner as runner_mod

    runner_mod.select_candidates_for_death = _zscore_select_candidates_for_death
    runner_mod.select_candidates_for_reproduction = (
        _zscore_select_candidates_for_reproduction
    )

    log.info(
        "Patched biome_runner: select_candidates_for_death and "
        "select_candidates_for_reproduction replaced with z-score sigmoid variants."
    )


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


def _count_phylogeny_rows(output_dir: str, biome: str) -> int:
    """Count lines in phylogeny.jsonl."""
    path = Path(output_dir) / biome / "phylogeny.jsonl"
    if not path.is_file():
        return 0
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _load_generation_logs(output_dir: str, biome: str) -> list[dict]:
    """Read all generation_log.json files in sorted generation order."""
    biome_dir = Path(output_dir) / biome
    logs: list[dict] = []
    for gen_dir in sorted(biome_dir.glob("generation_[0-9][0-9][0-9]")):
        log_file = gen_dir / "generation_log.json"
        if log_file.is_file():
            with open(log_file, encoding="utf-8") as fh:
                logs.append(json.load(fh))
    return logs


def _build_validation_report(
    output_dir: str,
    biome: str,
    n_generations: int,
    starting_agents: int,
) -> dict:
    """Compile validation metrics and GO/NO-GO verdicts.

    Parameters
    ----------
    output_dir : str
        Root experiment output directory.
    biome : str
        Biome name.
    n_generations : int
        Expected number of completed generations.
    starting_agents : int
        Initial agent count (used for lineage_min_rows).

    Returns
    -------
    dict
        JSON-serialisable validation report.
    """
    gen_logs = _load_generation_logs(output_dir, biome)

    if not gen_logs:
        return {
            "error": "No generation logs found.",
            "output_dir": output_dir,
            "biome": biome,
        }

    # Per-generation stats
    per_gen: list[dict] = []
    for gl in gen_logs:
        n_deaths = len(gl.get("deaths", []))
        pop_size = int(gl.get("population_size", 0))
        std_fit = gl.get("std_fitness")
        if std_fit is None or (isinstance(std_fit, float) and math.isnan(std_fit)):
            std_fit = None
        per_gen.append(
            {
                "generation": int(gl["generation"]),
                "deaths": n_deaths,
                "population_size": pop_size,
                "std_fitness": float(std_fit) if std_fit is not None else None,
                "deaths_flag": n_deaths == 0,
                "sigma_flag": (std_fit is not None and std_fit < _SIGMA_MIN),
            }
        )

    deaths_per_gen = [g["deaths"] for g in per_gen]
    pop_sizes = [g["population_size"] for g in per_gen]
    std_fitnesses = [g["std_fitness"] for g in per_gen]

    # Endogenous equilibrium: mean pop over last 5 generations
    last5_pop = pop_sizes[-5:] if len(pop_sizes) >= 5 else pop_sizes
    equilibrium_mean_pop = float(np.mean(last5_pop)) if last5_pop else float("nan")

    # Phylogeny row count
    phylo_rows = _count_phylogeny_rows(output_dir, biome)

    # --- GO/NO-GO criteria ---
    gens_with_deaths = sum(1 for d in deaths_per_gen if d > 0)
    deaths_ok = gens_with_deaths >= _MIN_GENS_WITH_DEATHS

    pop_ok = _POP_LOWER <= equilibrium_mean_pop <= _POP_UPPER

    # sigma_ok: std_fitness >= 0.01 for all of generations 0..7
    early_stds = [
        g["std_fitness"]
        for g in per_gen
        if g["generation"] < 8 and g["std_fitness"] is not None
    ]
    sigma_ok = all(s >= _SIGMA_MIN for s in early_stds) if early_stds else False

    lineage_min = n_generations * starting_agents
    lineage_ok = phylo_rows >= lineage_min

    verdicts = {
        "deaths_ok": deaths_ok,
        "pop_ok": pop_ok,
        "sigma_ok": sigma_ok,
        "lineage_ok": lineage_ok,
    }
    overall_go = all(verdicts.values())

    report = {
        "pilot_config": {
            "biome": biome,
            "generations": n_generations,
            "starting_agents": starting_agents,
            "docs_per_agent": DOCS_PER_AGENT,
            "k_max": K_MAX,
            "output_dir": output_dir,
            "config": CONFIG_PATH,
            "corpus_manifest": CORPUS_MANIFEST,
        },
        "mechanics": {
            "p_rep_formula": "0.05 + 0.40 * sigmoid(1.0 * z)",
            "p_death_formula": "0.07 + 0.38 * sigmoid(-1.5 * z)",
            "z_formula": "(f - mean_f) / max(std_f, 0.01)",
        },
        "per_generation": per_gen,
        "summary": {
            "deaths_per_generation": deaths_per_gen,
            "gens_with_deaths": gens_with_deaths,
            "mean_population_per_generation": [float(p) for p in pop_sizes],
            "std_fitness_per_generation": std_fitnesses,
            "equilibrium_mean_pop_last5": equilibrium_mean_pop,
            "phylogeny_rows": phylo_rows,
            "lineage_min_rows_required": lineage_min,
        },
        "criteria": {
            "deaths_ok": {
                "result": deaths_ok,
                "description": (
                    f"deaths > 0 in >= {_MIN_GENS_WITH_DEATHS} of {n_generations} "
                    f"generations (actual: {gens_with_deaths})"
                ),
            },
            "pop_ok": {
                "result": pop_ok,
                "description": (
                    f"mean pop over last 5 gens between {_POP_LOWER} and {_POP_UPPER} "
                    f"(actual: {equilibrium_mean_pop:.2f})"
                ),
            },
            "sigma_ok": {
                "result": sigma_ok,
                "description": (
                    f"no std_fitness < {_SIGMA_MIN} in first 8 generations "
                    f"(checked {len(early_stds)} gens)"
                ),
            },
            "lineage_ok": {
                "result": lineage_ok,
                "description": (
                    f"phylogeny.jsonl exists and has >= {lineage_min} rows "
                    f"(actual: {phylo_rows})"
                ),
            },
        },
        "verdict": "GO" if overall_go else "NO-GO",
    }

    return report


def _print_report(report: dict) -> None:
    """Print the validation report in a human-readable format."""
    print()
    print("=" * 70)
    print("  PHASE 1b PILOT — VALIDATION REPORT")
    print("=" * 70)

    summary = report.get("summary", {})
    per_gen = report.get("per_generation", [])

    print("\nDeaths per generation:")
    for g in per_gen:
        flag = " *** ZERO DEATHS ***" if g["deaths_flag"] else ""
        print(f"  gen {g['generation']:>2d}: deaths={g['deaths']}{flag}")

    print("\nPopulation size per generation:")
    mean_pop = float(np.mean([g["population_size"] for g in per_gen])) if per_gen else float("nan")
    for g in per_gen:
        print(f"  gen {g['generation']:>2d}: pop={g['population_size']}")
    print(f"  → overall mean: {mean_pop:.2f}")

    print("\nstd_fitness per generation:")
    for g in per_gen:
        s = g["std_fitness"]
        flag = " *** LOW SIGMA ***" if g.get("sigma_flag") else ""
        if s is not None:
            print(f"  gen {g['generation']:>2d}: std_fitness={s:.4f}{flag}")
        else:
            print(f"  gen {g['generation']:>2d}: std_fitness=N/A")

    eq = summary.get("equilibrium_mean_pop_last5", float("nan"))
    print(f"\nEndogenous equilibrium (mean pop, last 5 gens): {eq:.2f}")
    print(f"Phylogeny rows: {summary.get('phylogeny_rows', 'N/A')}")

    print("\nGO/NO-GO criteria:")
    for key, crit in report.get("criteria", {}).items():
        status = "GO " if crit["result"] else "NO-GO"
        print(f"  [{status}] {key}: {crit['description']}")

    print()
    verdict = report.get("verdict", "UNKNOWN")
    print(f"  *** OVERALL VERDICT: {verdict} ***")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Phase 1b pilot and generate a validation report."""
    # Ensure the repo root is on sys.path regardless of invocation directory.
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Patch before run_biome is imported or called.
    _patch_population_mechanics()

    from src.evolution.biome_runner import run_biome  # noqa: E402 (deferred import)

    log.info(
        "Starting Phase 1b pilot — biome=%s, generations=%d, "
        "starting_agents=%d, docs_per_agent=%d, k_max=%d",
        BIOME,
        GENERATIONS,
        STARTING_AGENTS,
        DOCS_PER_AGENT,
        K_MAX,
    )

    # Override k_max in the config at runtime by patching the YAML on-the-fly.
    # We inject k_max into the biome config via a config override wrapper so
    # run_biome reads the correct value without modifying the source YAML file.
    import yaml

    with open(CONFIG_PATH, encoding="utf-8") as fh:
        cfg: dict = yaml.safe_load(fh)

    cfg.setdefault("biomes", {}).setdefault(BIOME, {})["k_max"] = K_MAX

    # Write a temporary patched config that run_biome will read.
    patched_config_path = str(Path(OUTPUT_DIR) / "phase1b_pilot_config.yaml")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(patched_config_path, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)
    log.info("Wrote patched config to %s (k_max=%d)", patched_config_path, K_MAX)

    run_biome(
        biome_name=BIOME,
        config_path=patched_config_path,
        corpus_manifest_path=CORPUS_MANIFEST,
        output_dir=OUTPUT_DIR,
        n_generations=GENERATIONS,
        n_agents=STARTING_AGENTS,
        n_documents_per_agent=DOCS_PER_AGENT,
        resume_from_generation=0,
        seed=SEED,
    )

    log.info("Run complete. Building validation report...")

    report = _build_validation_report(
        output_dir=OUTPUT_DIR,
        biome=BIOME,
        n_generations=GENERATIONS,
        starting_agents=STARTING_AGENTS,
    )

    _print_report(report)

    report_path = Path(OUTPUT_DIR) / "validation_report.json"
    report_tmp = str(report_path) + ".tmp"
    with open(report_tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    os.replace(report_tmp, str(report_path))

    log.info("Validation report saved to %s", report_path)


if __name__ == "__main__":
    main()

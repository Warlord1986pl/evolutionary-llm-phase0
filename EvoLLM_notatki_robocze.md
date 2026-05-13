# EvoLLM — Notatki robocze (running research log)

*Bieżące ustalenia metodologiczne, decyzje i obserwacje. Pełne wnioski per sesja w osobnych plikach.*

---

## Sesja 2026-05-04 — zamknięcie Phase 0

**LD50 zakończony.** Odpowiedź gradualna i liniowa, brak progu krytycznego. Model bazowy odporny na titrację. LD50 klasyczne nieestymowalne — to jest wynik, nie błąd.

**Hipoteza diagnostyczna H_diag:** h_dezorg to wczesny marker toksyczności (reaguje przy T=50-75%), c_x to późny marker (degraduje przy T=75-100%). Do weryfikacji przez analyze_ld50_thresholds.py.

**Walidacja dwupoziomowa korpusu v3:**
- Binarna: c_x p=6.7e-18, h_dezorg p=1.5e-29
- Gradientowa: r=-0.921 dla fitness, 7 punktów titracji bez nieciągłości

**I(X;seed) jako bag-of-words cosine similarity:** świadomy wybór dla przejrzystości i reprodukowalności. Zmiana na embedding-based wymagałaby rekalibracji całego pipeline. Zostawione as-is przez wszystkie papiery. Ograniczenie odnotowane w Discussion Paper 1.

**Odporność pipeline na nieoptymalny seed:** przypadkowo zweryfikowana — c_x i h_dezorg stabilne niezależnie od seed_text. I(X;seed) słabe we wszystkich 3 runach.

**Terminologia:** toxin → toxin (zatruwa, nie poluje). Spójne z LD50, metabolic decay, dose-response.

---

## Sesja 2026-05-09 — Phase 0 v2 domknięty, Phase 1 ready

**Canonical run zamrożony:** `phase0_metrics_20260508T140551Z`, tag `phase0-final-v2`.

**Wyniki Phase 0 v2 — mean metryki:**
- food:  H=4.949 C=0.393 I=0.806 H_dezorg=0.825 fitness=+0.356
- toxin: H=5.061 C=0.331 I=0.772 H_dezorg=0.907 fitness=+0.304
- noise: H=4.690 C=0.295 I=0.752 H_dezorg=0.888 fitness=+0.287

**Pairwise food vs toxin:** H(X) ns (r=0.035, mimikra potwierdzona). C(X) r=-0.373 ***, I r=-0.358 ***, H_dezorg r=0.487 ***, fitness r=-0.465 ***.

**Gradient LD50:** C(X) r=-0.936 p=0.002, H_dezorg r=0.869 p=0.011, I r=-0.887 p=0.008, fitness r=-0.961 p=0.001. Próg krytyczny T=50%.

**Niemonotoniczność:** kompensacja C(X) przy T=25%, supresja H_dezorg przy T=0-25%.

**Per-domain:** H_dezorg i fitness significant 5/5 domen. C(X) 4/5 (cancer ns). I(X;seed) 3/5 (cancer, alt_med ns).

**Koszt metaboliczny:** H_dezorg/C(X) food=2.197, toxin=3.041. Toksyna 38% droższa per jednostkę użytecznej złożoności → mechanizm energy budget dla Phase 2.

**Phase 1 parametry zamrożone:** fitness_threshold=0.3156, k_rep=13.30, beta_death=0.072. Moduły src/evolution/ gotowe (trainer, population, biome_runner, cli). Start: biom savanna, 35+ generacji, 10 agentów, 30 dokumentów/agent/gen.

**Pending:** update notes Claude Project, rekalibacja k/β po generacji 5 jeśli potrzebna.

---

## Sesja 2026-05-07 — domknięcie decyzji i start rerunu po zmianie MI

- Phase 0 closed: commits `5834c53`, `c1d2cd3`, tag `phase0-final`.
- Flash Attention 2.8.3 installed, `CUDA_HOME` set in `.bashrc`.
- `trainer.py` implemented (commit after import fix).
- `population.py` implemented, statistical test for `select_parent` added.
- `biome_runner.py` implemented with lazy GPU imports.
- `mutual_information_proxy` replaced with entropy decomposition.
- Phase 0 rerun in progress (canonical + LD50).
- Pending: rerun results, recalibration of k and beta if needed,
  tag `phase0-final-v2`, then Phase 1 start.

---

## Sesja 2026-05-08 — kalibracja MI, wybór seeda C, analiza gradientu LD50

- MI calibration complete: 4 seedy × 8 implementacji, wynik zamrożony.
- **Seed C + mi_token_ids_nmi** — r_canonical=0.301, kierunek correct.
- systematic_reversal_finding: mi_entropy_decomp / mi_jsd / mi_npmi zawsze
  odwrócone dla wszystkich seedów — problem strukturalny, nie artefakt seeda.
  Materiał do Discussion Paper 1.
- Seed C STABLE: seed_stability_test.py std(h_x)=0.0, std(c_x)=0.0 × 5 RNG seeds.
- Truncation artifact pomijalny: delta H=0.010, delta C=0.001.
- **C(X) i H_dezorg = primary signal**: r=-0.936 i r=0.869 na gradiencie LD50.
- I(X;seed) = mierzy separację kanoniczną, nie gradient dawka-odpowiedź.
- config/phase0_v3.yaml zaktualizowany z Seed C + mi_token_ids_nmi.
- Pending: final rerun Phase 0, rekalibracja k/β, tag phase0-final-v2, Phase 1.

---

## Sesja 2026-05-06 — mini-rerun pod materiał supplementary

**Cel:** domknąć brakujące evidence jakościowe dryfu Q1/Q2/Q3 bez ruszania canonical run N=880.

**Co zrobiono:**
- utworzono mini-korpus `data/v2_mini_v3/` (55 dokumentów: food=25, toxin=25, noise=5),
- dodano konfigurację `config/phase0_mini_rerun_v3_toxin.yaml` z `save_chunk_texts: true`,
- uruchomiono mini-rerun i zapisano wynik: `experiments/phase0_metrics_20260506T083113Z/metrics_phase0.json`,
- potwierdzono obecność pól `gen_text_Q1`, `gen_text_Q2`, `gen_text_Q3` dla dokumentów.

**Wniosek metodologiczny:**
- dryf ilościowy był już policzony wcześniej na pełnym korpusie,
- mini-rerun służy wyłącznie do ilustracji jakościowej (przykładowe trajektorie i excerpty),
- nie nadpisuje canonical wniosków statystycznych Phase 0.

**Repo/publication workflow:**
- publiczne repo Phase 0 oznaczono tagiem zamrażającym `v0.1-phase0`,
- supplementary opublikowano jako osobny artefakt i sekcję na stronie, aby oddzielić canonical od dodatków.

## Sesja 2026-04-22 — kluczowe ustalenia

**Metryki na wyjściach, nie inputach.** H(X), C(X), I(X;seed) muszą być mierzone na outputach modelu po ekspozycji — nie na samych dokumentach. Measuring na inputach tworzy confound długości.

**Minimum 200 tokenów output.** C(X) delta odwraca znak między 150 a 200 tokenami. Poniżej 200 kierunek efektu jest niestabilny.

**Miller-Madow correction wymagana dla H(X).** H_corrected = H_empirical + (k-1)/(2·N). Zawsze raportować wersję skorygowaną.

---

## Sesja 2026-04-23 — kluczowe ustalenia

**Ollama odrzucona, Unsloth jako backend.** qwen3:8b-base niedostępny w Ollama registry. Unsloth już w stacku dla LoRA, logprobs extractable z forward pass — cleaner architecture.

**H_dezorg = perplexity z forward pass, nie Ollama.** Logprobs z `model(**inputs, labels=inputs["input_ids"])` bezpośrednio. `HF_HUB_OFFLINE=1` wymagany przy ładowaniu z cache.

---

## Sesja 2026-04-27 — kluczowe ustalenia

**Anomalia szczepionkowa.** VaccineLies MisT (akademicki styl): I effect = +0.274 (odwrócony). ClimateFever (autentyczny internet): I effect = -0.607. Styl języka ważniejszy od treści dla odpowiedzi modelu.

**Corpus toxin musi być autentyczny.** NaturalNews, Mercola, WUWT — tak. MBIB, LIAR, VaccineLies MisT — nie. Akademicka taksonomia twierdzeń jest informatycznie nieodróżnialna od food.

**Wagi fitness zamrożone po grid search:** w1=0.3, w2=0.5, w3=0.2. Nie zmieniać po Phase 0.

**Jaccard nie jest redundantny.** Korelacja z I(X;seed) < 0.8 we wszystkich typach. Justified for Phase 2.

---

## Sesja 2026-04-30 — kluczowe ustalenia

**Brighteon CTA contamination.** NaturalNews scraper zbierał krótkie strony CTA (subscribe, video link). Efekt: h_x ≈ 0, c_x ≈ 0 dla zainfekowanych dokumentów, toxin fitness > food fitness (artefakt). Filtr: min 300 znaków + blacklist patterns.

**5 aktywnych domen dla Paper 1:** climate, vaccines, alt_med, cancer, gmo. COVID wykluczone (artefakty, nakładanie z vaccines).

**Climate toxin jest najczystszy** (99% retencja po filtrowaniu). Długie artykuły argumentacyjne (Plate Climatology, WUWT) > news aggregators.

---

## Sesja 2026-05-01 — kluczowe ustalenia

**Mercola przez Windows, nie WSL2.** WSL2 blokowany przez Mercola po IP. Scraper uruchamiać z conda `evolllm` (Windows).

**Artykuły Mercola 3-5× dłuższe niż NaturalNews** (avg 16-24k vs 5k znaków). Potential confound przy single-window truncation. Wymaga normalizacji długości inputu — percentile chunking jako rozwiązanie.

**food_gmo = 77/80 akceptowalne.** Ostatnie 3 niedostępne w PMC OA. Zanotować w Methods.

---

## Sesja 2026-05-02 — kluczowe ustalenia

**Entropia nie była bezużyteczna — była źle mierzona.** Single-window truncation nie chwyta heterogeniczności dokumentu. Percentilowe chunki (5 x 20%) rehabilitują H(X): p=0.77→8.2e-21.

**Noise musi być redefiniowany przed Paper 1.** Fragmenty 50/50 to sygnał zdegradowany, nie tło środowiskowe. Wikipedia noise biologicznie i metodologicznie poprawniejszy.

**Docelowe parametry chunkingu dla korpusu v3:** window_size=1024, n_windows=3. Pokrywa 90% toxina przy pełnym profilu, kontekst generacji 2x lepszy.

**toxin_climate wymaga nowego źródła.** CARDS/PlateClimatology to datasety twierdzeń, nie artykułów. WUWT działa przez Selenium. Mercola nie pisze o klimacie (3 artykuły z 120 prób).

---

## Sesja 2026-05-13 — Phase 1 zamknięta, Phase 1b design frozen

**Status eksperymentów**

| Eksperyment | Status | Uwagi |
|---|---|---|
| Phase 1 Savanna | COMPLETE | 35 gen, stara mechanika, deaths=0 |
| Phase 1 Desert | COMPLETE | 35 gen, stara mechanika, deaths>0 |
| Phase 1 Plain | COMPLETE | 35 gen, stara mechanika, deaths=0 |
| Phase 1b Pilot (Desert, 12 gen) | COMPLETE — GO | nowa mechanika, deaths every gen |
| Phase 1b pełny rerun | PENDING | 3 biomy × 3 seeds × 35 gen |

**Dlaczego Phase 1 jest pilotem, nie finalnym eksperymentem**

Phase 1 ujawnił dwie patologie mechaniki populacyjnej:

- deaths=0 w Savanna i Plains przez wszystkie 35 generacji — brak selekcji gradualnej.
- Przyczyna: stara `P_death = max(1 - exp(f/β), 0)` daje praktycznie 0 dla dodatnich fitness.
- Różne `K_max` per biom (Desert=10, Plains=25, Savanna=30) i różny `N0/K_max` ratio konfundują porównania między biomami — struktura populacyjna różna, nie tylko środowisko.

Wyniki Phase 1 są użyteczne jako obserwacja kierunkowa i uzasadnienie rekalibracji.
Nie są finalnym dowodem na H0.

**Wyniki Phase 1 (kierunkowe)**

- Fitness plateau nierozróżnialne między biomami (~0.455-0.459).
- Efekt środowiska widoczny w:
  - `I(X;seed)`: Desert(+0.005) < Plains(+0.012) < Savanna(+0.022)
  - JSD: sawanna homogenizuje (-38%), pustynia utrzymuje różnorodność (-5%)
  - Kształt trajektorii: sawanna monotoniczna, pustynia oscylacyjna z wczesną stagnacją
  - `H_dezorg` redukcja (~-0.10) jest właściwością fine-tuningu, nie środowiska

**Rekalibracja mechaniki populacyjnej**

Grid 4860 kombinacji na realnych rozkładach fitness Phase 1.
Wybrane parametry (rank 2 z variance populacji jako kryterium):

- `alpha_r = 1.0`, `alpha_d = 1.5`
- `p_r_min = 0.05`, `p_r_max = 0.45`
- `p_d_min = 0.07`, `p_d_max = 0.45`
- `sigma_min = 0.01`

Nowa mechanika z-score + bounded sigmoid:

```text
z_i = (f_i - μ_f) / max(σ_f, 0.01)
P_rep_i   = 0.05 + 0.40 * sigmoid( 1.0 * z_i)
P_death_i = 0.07 + 0.38 * sigmoid(-1.5 * z_i)
```

Uzasadnienie: selekcja względna (nie bezwzględna), `p_d_min=0.07` gwarantuje
niezerową śmiertelność zawsze, eliminuje saturację `P_rep` do 1.0.

**Phase 1b FROZEN parameters**

- `N0: 12` — jednolite dla wszystkich biomów
- `K_max: 30` — jednolite dla wszystkich biomów
- `replications: 3` per biom (seeds: 42, 123, 456)
- `generations: 35`
- `docs_per_agent: 30`
- `logging: generation_log + population_json + phylogeny_jsonl + phylogeny_graph`

**Phase 1b Pilot — kluczowe obserwacje**

Desert, 12 gen, `N0=12`, `K_max=30`:

- deaths > 0 w każdym pokoleniu (12/12) — mechanika działa
- Endogeniczne equilibrium: ~13 agentów (poniżej `K_max=30`)
- Gen 6: deaths=7, purifying selection eliminuje dominującą linię, JSD rośnie
- `std_fitness` oscyluje (nie kolapsuje permanentnie) — zdrowa dynamika

Kluczowa obserwacja z drzewa genealogicznego:

- Founder z najwyższym fitness gen0 (f=0.442) wymiera do gen 3
- Founder ze średnim fitness (f=0.419) dominuje przez wszystkie 12 gen
- Frequency-dependent selection emergentny — nie zaprojektowany, wynika z mechaniki
- Długie linie nadmiernie wyspecjalizowane → eliminowane przez toksyczne dokumenty spoza ich rozkładu treningowego (over-specialization penalty)

**Następne eksperymenty (po Phase 1b)**

- Cross-biome transfer: desert survivors gen35 → populacja startowa w sawannie
- H0: pre-adaptacja w pustyni nie przyspiesza adaptacji w sawannie
- Mixed inoculation: survivors ze wszystkich 3 biomów (4 per biom) → nowe środowisko
- Śledź które linie dominują genealogicznie
- Phase 2: archetypes (Id/Ego/Superego LoRA adaptery)
- Wymaga wyników Phase 1b jako baseline

**Pytanie otwarte**

Czy desert survivors są generalistami (szeroki LoRA, odporna na zmienność próbkowania)
czy po prostu słabo wyspecjalizowanymi agentami (niski fitness absolutny)?
Test: porównaj fitness desert survivors vs savanna fresh start w tym samym środowisku.

# Model Selection

**Status:** design, round 1 (2026-11-13). No code. At Todd's request ("use your recommendations to start") every
answer below is Claude's recommendation, marked provisional, until Todd settles it round by round as
`Apps/AssistantOnboarding.md` is settled.

**Area:** how the planner uses Mila's measurements of a model -- to set the context it gives a model, and to choose
between models for a use case. The numbers come from `ModelEval.md` and `ContextProfile.md`; they are published in
the manifest (`ModelDistribution.md`); the planner is `Deployment.md`'s. The first consumer is Mila Assistant's
suggestion (`Apps/AssistantOnboarding.md`, question 3).

---

## 1. The question

The planner answers, for one model on one computer, whether it fits and at which context. Two questions it cannot
answer today:

- **What context is worth giving.** A model's useful context is the lower of what fits and its reliable depth
  (`ContextProfile.md` 1): Llama 3.1 8B fits 131072 on the 16 GB card and drops its system rule at 65536. Memory
  reserved past the reliable depth holds a conversation the model no longer follows.
- **Which of several models that fit to use for a task.** Mila Assistant's suggestion ranks them by download size
  (`+142`), a stand-in nothing measured.

The model cards were to carry these measurements so a person can choose (`ModelEval.md` 8, and Todd's rule of
2026-10-19 that every card carries Mila's quality measurements and the reasoning for its recipe). This spec makes the
same numbers the planner's, so the program chooses as an informed person would.

## 2. Where it stands

- **Reliable depth exists for three configurations, on recall alone** (`ContextProfile.md`, first profiles,
  2026-10-03/04): Gemma 4 26B-A4B Q4_0 with an FP8 global cache, 98304, past what fits; Qwen 3.8 27B 2.82-bit, 74752,
  as far as it fits; Qwen 3.8 27B FP4, 16384, with its loss gate failing from 8K. Instruction retention and tool calls
  at depth are not yet measured arms.
- **Task scores barely exist.** `ModelEval.md`'s tools are built (IFEval, GSM8K, RULER, BFCL, the Hermes set), with
  a smoke run on real weights and Qwen 3.8's first BFCL sample (section 7, Phase 8). The thresholds a use case must
  meet are not set (section 10, decision 2). The noise floor is defined (6.3) and measured for no published model.
- **The manifest carries capabilities, not measurements** (`ModelDistribution.md`, *Capabilities and modalities*).
  `agentic` is Mila's verdict, a yes or no, with no number behind it in the manifest.
- **Reliable depth in the manifest was agreed** (Todd, 2026-11-03, compaction decision 3): an optional field per
  weight format; absent, the context limit alone. The KV cache format was left open. It was not written to a spec
  until this one; `Agents/Compaction.md` uses the depth as its trigger and does not say where it is read from.
- **The planner ranks plans, not models.** `DeploymentPlans` ranks one model's plans across devices.
  `planPublishedModels` (`+142`) is the first ranking across models.
- **Nothing is republished until 0.21 development is finished and the release backlog run down** (Todd,
  2026-11-13), so published manifests gain these fields after that, whatever is decided here.

## 3. Rules it stands on

- **Suitability is measured.** A number the planner reads is one Mila measured; nothing is inferred from a family,
  a name, a size or a parameter count, and a model without a number is unmeasured, not average.
- **A model is measured at its defaults** (`ModelEval.md`, 2026-10-18): the template's thinking, the publisher's
  sampling. A number from another configuration is not that model's.
- **A number is written once** (`ModelEval.md` 8): the card, the website and the manifest are written from the
  same results file.
- **A difference smaller than the benchmark resolves is a tie** (`ModelEval.md` 6.3). The planner does not rank on
  noise.
- **Older manifests keep loading** (`ModelDistribution.md` *Compatibility*). Every field here is optional, and its
  absence has a stated meaning.
- **The planner decides; an application chooses nothing** (`Apps/AssistantOnboarding.md` 3).

## 4. The design, provisional

### 4.1 What the manifest carries

One package is one weight format, so a manifest's measurements are already per weight format. Within it, one entry per
KV cache format measured:

```json
"measurements": {
  "measured_with": "0.21.0-dev+N",
  "measured_on": "cuda",
  "configurations": [
    {
      "kv_cache": "none",
      "reliable_depth": { "tokens": 65536, "measured_to": 131072, "arms": [ "recall", "retention", "tool_call" ] },
      "scores": {
        "ifeval": { "value": 0.812, "resolves": 0.012 },
        "bfcl": { "value": 0.774, "resolves": 0.020 },
        "hermes": { "value": 0.647, "resolves": 0.060 },
        "ruler_65536": { "value": 0.881, "resolves": 0.015 }
      }
    }
  ]
}
```

- **`reliable_depth`** is `ContextProfile.md`'s headline: `tokens`, how far it was measured (`measured_to`), and
  which arms decided it. A depth equal to `measured_to` is a lower bound: the model held as far as it was tested.
- **`scores`** are `ModelEval.md`'s, each the share right at the model's defaults, with `resolves`, the half-width
  the noise floor gives it. A score with no `resolves` is not ranked on.
- **`measured_on`** is the backend. The numbers are the model's, not the card's, but whether a CUDA number holds on
  Metal is not measured (section 5).
- **Written by the exporter** from the evaluation's results file, as the sampling is from `sampling.json`: strict,
  refusing an unknown key or a value out of range. A reader drops what it does not know, never refuses a model.

### 4.2 Reliable depth caps the automatic context

When the request leaves the context to the planner (`withAutomaticContextLength`), its ceiling becomes the lower of
the request's ceiling and the reliable depth for the plan's KV cache format, when that depth is below what was
measured. A depth at `measured_to` caps nothing. A plan so limited says so: `ContextLimit` gains `ReliableDepth`
beside the device and the trained maximum.

- **A fixed context is never changed.** A person or program that fixes one past the depth gets it; the plan reports
  the depth, and an application may say so.
- **What it buys:** the memory the cache would reserve past the depth, which on a shared machine is memory other
  applications keep (`Deployment.md` 12.10), and on a dedicated card room for a larger model.
- **Why it is consistent with compaction:** `Agents/Compaction.md` compacts at the reliable depth, so a conversation
  never uses the context past it.
- **An opt-out on the request** (`withReliableDepthIgnored`), for measuring past the depth: the profile tools need it.

### 4.3 Ranking models for a use case

A **use case** is a minimum context and the score that measures it, from `ModelEval.md` 12.3:

| Use case | Minimum context | Score | Mila Assistant |
|---|---|---|---|
| Chat | the planner's floor | `ifeval` | -- |
| Tool calling | the planner's floor | `bfcl` | -- |
| Agentic work | 65536 (Todd, 2026-10-15) | `hermes`, with `reliable_depth` to the minimum on all three arms | every activity today (onboarding question 9) |
| Long documents | the document band | `ruler_<band>` | -- |

Models are ranked for a use case, best first:

1. **Its useful context reaches the minimum**: the lower of the plan's context and the reliable depth. A model that
   does not reach it is not ranked for that use case.
2. **Its score**, higher first; two scores within the larger `resolves` of the pair are a tie.
3. **Rate** on a tie: the weight bytes read per decoded token, from the plan, so a mixture of experts is priced by
   its active experts (`ModelEval.md` 12.1).
4. **The smaller download** on a tie in all three.

**Unmeasured models rank after measured ones,** among themselves by download, larger first -- today's `+142` rule,
kept only where nothing better is known. An agentic use case takes measured models only: `agentic` is already a
measured verdict.

**Once thresholds exist** (`ModelEval.md` 10, decision 2), "the cheapest configuration that meets them"
(`ModelEval.md` 12) replaces step 2's best score: models that meet the threshold rank by rate, then download, and a
model that misses it ranks after all that meet it.

### 4.4 Where it lives in the library

- **`Distribution::ModelMeasurements`**, parsed from the manifest, carried by the store record, and reported by
  `InstalledModel` and `PublishedModel`.
- **`Deployment::UseCase`**, and `DeploymentRequest::forUseCase( UseCase )`, which sets the automatic context's
  floor to the use case's minimum.
- **The ceiling of 4.2** in `DeploymentPlanner`, the one place the automatic context is chosen.
- **The ranking of 4.3** in `planPublishedModels` and a counterpart over installed models, both given the use case.
  Mila Assistant's suggestion becomes the first model ranked for agentic work.

## 5. Costs and what is not measured

- **Measuring every published model.** RULER alone is about 1.7 hours a band at 32K for the Qwen re-fit, arithmetic
  on two documents (`ModelEval.md` Phase 8); the Hermes set runs overnight; `ContextProfile.md`'s arms per band.
  Each published model, at each KV cache format published, owes all of it before it ranks on numbers.
- **Republishing every manifest** once the numbers exist -- manifests only, not weights -- after 0.21 (section 2).
- **Metal.** No model has been scored on Metal. One family scored on both backends would say whether a CUDA number
  stands for Metal; until then the planner uses the number it has and the manifest says where it came from.
- **The noise floor per task**, which `resolves` needs, is measured for no published model.

## 6. Questions

Numbered for the rounds. Each answer is Claude's, provisional.

1. *Which numbers.* Reliable depth and one score per use case, each with its resolution (4.1).
2. *Whether reliable depth caps the automatic context or only ranks.* It caps, when below what was measured, with
   an opt-out (4.2).
3. *The use cases and the score for each.* The four of 4.3; agentic work is the only one Mila Assistant uses today.
4. *The order of the ranking.* Useful context reaches the minimum, then score with ties, then rate, then download
   (4.3).
5. *Best score or cheapest that meets the threshold.* Best score until thresholds exist, cheapest after (4.3).
6. *What an unmeasured model gets.* Ranked after every measured one, by download; never for agentic work (4.3).
7. *Per KV cache format.* One entry each, as measured; a plan reads the entry for its own format, and a format with no
   entry is unmeasured (4.1).
8. *Numbers from CUDA on Metal.* Used, with the backend recorded, until a measurement says otherwise (5).
9. *Where the numbers are written from.* The evaluation's results file, by the exporter, the source the card and
   website already read (4.1).
10. *What the card says of the planner.* Nothing; the card states the measurements for its reader, and the planner
    reading them is not the reader's concern (CLAUDE.md, *End-User Prose*).

## 7. Not in this design

- The thresholds themselves (`ModelEval.md` 10, decision 2) and the measurements (`ModelEval.md`, `ContextProfile.md`).
- Compaction (`Agents/Compaction.md`), which reads the same depth.
- The card's layout (`ModelEval.md` 8).
- Choosing a model per conversation or per activity in an application (`MilaDesktop.md` 3.2).

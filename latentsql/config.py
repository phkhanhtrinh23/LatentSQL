"""Configuration objects for the LatentSQL pipeline.

Everything is a plain dataclass so configs can round-trip through YAML/JSON
without extra dependencies. ``load_config`` merges a YAML file on top of the
defaults, so config files only need to state what they change.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
@dataclass
class PathConfig:
    """Where data / models / outputs live.

    Defaults follow the layout on the thesis workstation
    (``/mnt/win_d/data_FusionSQL`` plus a HuggingFace hub cache at
    ``/mnt/win_d/hub``). Override in a YAML config or via ``LATENTSQL_*``
    environment variables when running elsewhere.
    """

    bird_root: str = "/mnt/win_d/data_FusionSQL/bird"
    spider_root: str = "/mnt/win_d/data_FusionSQL/spider"
    hf_hub: str = "/mnt/win_d/hub"
    output_dir: str = "outputs"
    cache_dir: str = "outputs/cache"

    def resolve_env(self) -> "PathConfig":
        """Apply ``LATENTSQL_BIRD_ROOT`` / ``LATENTSQL_SPIDER_ROOT`` / ``LATENTSQL_HF_HUB``."""
        for f in dataclasses.fields(self):
            env = os.environ.get("LATENTSQL_" + f.name.upper())
            if env:
                setattr(self, f.name, env)
        return self


# --------------------------------------------------------------------------
# teacher multi-agent system
# --------------------------------------------------------------------------
@dataclass
class LLMConfig:
    """Backend used to drive the teacher MAS (or to run a baseline).

    Local-first by design: the default backend is ``hf``, i.e. an open-weight
    model resolved out of the offline hub at ``paths.hf_hub``
    (``/mnt/win_d/hub``). The OpenAI-compatible backend is fully implemented
    and can drive the same agents, but it is never selected by default - you
    have to ask for it explicitly with ``backend: openai``.
    """

    backend: str = "hf"             # hf (default, local) | openai | mock
    #: teacher model. Resolved against ``paths.hf_hub`` first, then the HF Hub.
    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    #: ordered fallbacks tried when ``model`` is not present in the local hub
    model_fallbacks: List[str] = field(
        default_factory=lambda: [
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        ]
    )
    local_only: bool = True         # never hit the network for backend == "hf"
    base_url: Optional[str] = None  # for vLLM / any OpenAI-compatible server
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.95
    seed: int = 0
    request_timeout: float = 120.0
    max_retries: int = 3
    # only used by backend == "hf"
    device: str = "auto"
    dtype: str = "auto"
    hub_dir: Optional[str] = None   # filled from paths.hf_hub by load_config
    max_batch: int = 8              # batched generation for candidates_per_agent > 1


@dataclass
class MASConfig:
    """The FusionSQL teacher: a hybrid of DIN-SQL / DAIL-SQL / CHESS / CHASE-SQL."""

    # --- retrieval (CHESS "Information Retriever" + CHASE "value retrieval") ---
    use_value_retrieval: bool = True
    value_top_k: int = 5
    value_min_similarity: float = 0.72
    lsh_signature_size: int = 64
    lsh_bands: int = 16
    max_values_indexed_per_column: int = 5000

    # --- schema linking (DIN-SQL module 1 + CHESS "Schema Selector") ---
    use_schema_selector: bool = True
    schema_keep_all_below_columns: int = 40   # skip pruning for tiny schemas
    schema_max_tables: int = 8
    schema_max_columns_per_table: int = 12

    # --- classification & decomposition (DIN-SQL module 2) ---
    use_classifier: bool = True

    # --- candidate generators = the debate panel (CHASE-SQL 3.3 + DAIL-SQL) ---
    #     each entry is an agent id defined in latentsql/mas/agents.py
    agents: List[str] = field(
        default_factory=lambda: ["divide_conquer", "query_plan", "dail_icl"]
    )
    rounds: int = 2                 # IMAD uses m = 2 debate rounds
    candidates_per_agent: int = 1   # >1 => temperature sampling per agent

    # --- fixer (CHASE 3.4 / CHESS "revise" / DIN-SQL module 4) ---
    use_fixer: bool = True
    max_fix_attempts: int = 3

    # --- selection (CHASE 3.5 selection agent + CHESS unit tester) ---
    selector: str = "unit_test"     # exec_consistency | pairwise | unit_test
    n_unit_tests: int = 3

    # --- few-shot pool for the DAIL-SQL in-context-learning agent ---
    dail_shots: int = 5
    dail_query_similarity_threshold: float = 0.85

    execution_timeout: float = 30.0


# --------------------------------------------------------------------------
# distillation (IMAD) + latent collaboration (LatentMAS)
# --------------------------------------------------------------------------
@dataclass
class TraceConfig:
    """How teacher transcripts get turned into an SFT corpus."""

    require_consensus: bool = True   # IMAD 2.1: drop debates with no consensus
    require_correct: bool = True     # keep only traces whose final SQL matches gold
    require_execution: bool = True   # drop traces whose final SQL errors out
    max_trace_chars: int = 12000
    dedup: bool = True
    min_agents_agree: int = 1


@dataclass
class SFTConfig:
    """Stage 1 - debate structure learning (IMAD 2.2)."""

    model_name_or_path: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    output_dir: str = "outputs/sft"
    epochs: float = 3.0
    lr: float = 1e-4
    batch_size: int = 1
    grad_accum: int = 8
    max_seq_len: int = 4096
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    gradient_checkpointing: bool = True
    bf16: bool = True
    # loss is taken over the whole debate trace, not just the final SQL
    # (IMAD 2.2, in contrast to DebateGPT)
    train_on_full_trace: bool = True
    seed: int = 0
    logging_steps: int = 5
    save_steps: int = 200
    device: str = "auto"


@dataclass
class RewardConfig:
    """IMAD eq. (1):  r = w_fmt * R_fmt + w_clip * R(y; l)."""

    w_fmt_start: float = 1.0
    w_fmt_end: float = 0.05
    w_clip: float = 1.0
    # length annealing  l0 -> l*  (IMAD 2.3)
    len_start: int = 2000
    len_end: int = 500
    schedule: str = "linear"        # linear | cosine | step
    # extra text-to-SQL shaping (kept small so IMAD's two terms dominate)
    w_exec_ok: float = 0.1          # query parses and runs without error
    w_tag_order: float = 0.1        # tags appear in the canonical order
    required_tags: List[str] = field(default_factory=list)  # empty => trace schema default


@dataclass
class GRPOConfig:
    """Stage 2 - internalization via GRPO with dynamic reward scheduling."""

    model_name_or_path: str = "outputs/sft"
    output_dir: str = "outputs/grpo"
    epochs: float = 2.0
    lr: float = 1e-6
    group_size: int = 8             # k candidates per query
    batch_prompts: int = 2
    grad_accum: int = 4
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    kl_beta: float = 0.02
    clip_eps: float = 0.2
    max_grad_norm: float = 1.0
    seed: int = 0
    logging_steps: int = 1
    save_steps: int = 100
    device: str = "auto"
    bf16: bool = True
    lora: bool = True
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class LatentConfig:
    """LatentMAS-style latent thoughts + latent working memory (paper 3.1/3.2)."""

    enabled: bool = False
    latent_steps: int = 8           # m latent steps per internal agent
    roles: List[str] = field(
        default_factory=lambda: ["divide_conquer", "query_plan", "dail_icl", "consensus"]
    )
    align: str = "soft_embed"       # soft_embed | linear | identity
    soft_embed_top_k: int = 32
    soft_embed_temperature: float = 1.0
    share_working_memory: bool = True   # prepend previous roles' KV caches
    train_latent_slots: bool = False    # curriculum-replace agent spans with latent slots


@dataclass
class SteeringConfig:
    """Agent-subspace probing via difference-in-means / CAA (IMAD section 3)."""

    layer: int = -8                 # negative = counted from the last layer
    alpha: float = 1.0
    n_pairs: int = 128
    normalize: bool = True
    positions: str = "response_mean"   # response_mean | last_token


@dataclass
class EvalConfig:
    n_samples: int = 100
    split: str = "dev"
    dataset: str = "bird"           # bird | spider | synthetic
    mode: str = "internalized"      # single | mas | sft_trace | internalized | latent
    max_new_tokens: int = 1024
    temperature: float = 0.0
    batch_size: int = 1
    report_ves: bool = True
    seed: int = 0


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    mas: MASConfig = field(default_factory=MASConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def dump(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _merge(dc: Any, patch: Dict[str, Any]) -> Any:
    """Recursively overwrite dataclass fields from a nested dict."""
    for key, value in patch.items():
        if not hasattr(dc, key):
            raise KeyError(f"unknown config key: {key!r} on {type(dc).__name__}")
        current = getattr(dc, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(dc, key, value)
    return dc


def _read_structured(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "pyyaml is required to read YAML configs; "
                "either `pip install pyyaml` or use a .json config"
            ) from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def load_config(path: Optional[str] = None, **overrides: Any) -> Config:
    """Load defaults, apply an optional YAML/JSON file, then keyword overrides."""
    cfg = Config()
    if path:
        _merge(cfg, _read_structured(path))
    if overrides:
        _merge(cfg, overrides)
    cfg.paths.resolve_env()
    # local-first: the HF backend and the student both read from the same hub
    if cfg.llm.hub_dir is None:
        cfg.llm.hub_dir = cfg.paths.hf_hub
    return cfg

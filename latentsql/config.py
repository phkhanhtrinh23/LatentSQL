"""Configuration for the LatentSQL pipeline.

Every setting is a field of a plain dataclass. ``load_config`` reads a YAML or
JSON file on top of the defaults, so a config file only lists what it changes.
The defaults are the hyperparameters of the paper (Appendix A.3, Table 3).
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

#: the eight benchmarks of the paper (Appendix A.1)
BENCHMARKS = (
    "wikisql", "bird", "spider", "kaggledbqa", "sparc", "cosql", "ehrsql", "livesqlbench",
)

#: the three teacher roles (Sec. 4.1): schema selection, candidate generation, revision
ROLES = ("schema_selector", "candidate_generator", "reviser")


@dataclass
class PathConfig:
    #: converted benchmarks (``scripts/prepare_data.py`` writes here)
    data_dir: str = "data"
    output_dir: str = "outputs/latentsql"
    #: LSH value indexes and database instances
    cache_dir: str = "outputs/cache"
    #: optional local HuggingFace cache; ``None`` uses the default HF cache
    hf_hub: Optional[str] = None

    def resolve_env(self) -> "PathConfig":
        """``LATENTSQL_DATA_DIR`` / ``LATENTSQL_OUTPUT_DIR`` / ... override the file."""
        for f in dataclasses.fields(self):
            env = os.environ.get("LATENTSQL_" + f.name.upper())
            if env:
                setattr(self, f.name, env)
        return self


@dataclass
class DataConfig:
    #: benchmarks whose training split is used (Appendix A.1, "Training and evaluation splits")
    train_benchmarks: List[str] = field(default_factory=lambda: list(BENCHMARKS))
    #: benchmarks evaluated on their development/test split
    eval_benchmarks: List[str] = field(default_factory=lambda: list(BENCHMARKS))
    #: cap on training questions per benchmark (``None`` = all)
    max_train_per_benchmark: Optional[int] = None
    #: cap on evaluation questions per benchmark (``None`` = all)
    max_eval_per_benchmark: Optional[int] = None
    #: KaggleDBQA and LiveSQLBench have no training split: random 70:30 split
    heldout_fraction: float = 0.3
    split_seed: int = 0
    #: EHRSQL unanswerable questions (gold query "nan") have no EX; dropped by default
    keep_unanswerable: bool = False
    #: schema text longer than this is truncated in the student prompt
    max_schema_chars: int = 12000


@dataclass
class RetrievalConfig:
    """Database values indexed once with MinHash LSH (Sec. 4.1, after CHESS)."""

    enabled: bool = True
    top_k: int = 5
    min_similarity: float = 0.72
    lsh_signature_size: int = 64
    lsh_bands: int = 16
    max_values_per_column: int = 5000


@dataclass
class TeacherConfig:
    """The three-role teacher (Sec. 4.1). The paper uses GPT-5.4."""

    backend: str = "openai"             # openai | hf | mock
    model: str = "gpt-5.4"
    #: any OpenAI-compatible server (vLLM, SGLang, ...); ``None`` = api.openai.com
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    #: None = the provider default (GPT-5-family models reject other values)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: int = 2048
    seed: int = 0
    request_timeout: float = 120.0
    max_retries: int = 5
    #: the reviser executes the draft, reads the result or error and fixes it;
    #: this many execute-and-fix rounds at most
    max_revisions: int = 1
    execution_timeout: float = 30.0
    # backend == "hf" only
    device: str = "auto"
    dtype: str = "auto"
    quantization: str = "none"
    hub_dir: Optional[str] = None


@dataclass
class StudentConfig:
    """The small language model and its LoRA adapters (Table 3)."""

    model: str = "Qwen/Qwen3-8B"
    bf16: bool = True
    quantization: str = "none"          # none | 4bit | 8bit
    device: str = "auto"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    #: attention and MLP projections
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
    gradient_checkpointing: bool = True


@dataclass
class LatentConfig:
    """Recurrent latent reasoning (Sec. 3.2, 4.2, 4.4; Table 3)."""

    #: J role blocks, in order
    roles: List[str] = field(default_factory=lambda: list(ROLES))
    #: m latent steps per role
    latent_steps: int = 2
    #: n_top and tau of the token-space alignment A (Eq. 12), roles r2 and r3
    top_k: int = 10
    temperature: float = 0.5
    #: schema-space alignment A_S for r1 (Eq. 13-15); false = A for every role
    schema_alignment: bool = True
    #: n_S, tau and beta of A_S (Eq. 13-15)
    schema_top_k: int = 10
    schema_temperature: float = 0.5
    schema_beta: float = 0.3


@dataclass
class CurriculumConfig:
    """Warm-up on teacher traces, then the text-to-latent curriculum (Sec. 4.2)."""

    #: warm-up on the full verified traces (stage 0); false = "w/o role traces"
    use_role_traces: bool = True
    #: internalise one role per stage; false = all roles latent at once ("w/o curriculum")
    curriculum: bool = True
    epochs_per_stage: int = 3
    batch_size: int = 16
    #: learning rate of the warm-up and stage 1
    lr: float = 5e-5
    #: smaller learning rate of stages 2 and 3
    lr_late: float = 1e-5
    max_grad_norm: float = 1.0
    #: lambda and kappa of L_probe (Eq. 18-19)
    probe_weight: float = 0.5
    probe_tokens: int = 24
    #: lambda_S of L_schema (Eq. 16, 19)
    schema_weight: float = 0.5
    #: learning rate of W_S, which lives outside the LoRA adapter
    schema_lr: float = 1e-3
    #: training records longer than this (prompt + written roles) are dropped
    max_record_tokens: int = 4096
    logging_steps: int = 10
    seed: int = 0


@dataclass
class RLConfig:
    """Dr. GRPO with dense and spurious rewards (Sec. 4.3; Table 3)."""

    group_size: int = 8
    clip_eps: float = 0.2
    temperature: float = 1.0
    lr: float = 1e-6
    #: maximum query length in tokens
    max_new_tokens: int = 256
    epochs: float = 1.0
    batch_prompts: int = 2
    grad_accum: int = 4
    max_grad_norm: float = 1.0
    #: reward terms (Eq. 21); each can be switched off for the ablations of Table 2
    use_atomic: bool = True
    use_spurious: bool = True
    use_counterexamples: bool = True
    #: initial instances per gold query and maximum pool size (Appendix B)
    n_initial_instances: int = 5
    max_pool_size: int = 20
    #: counterexample search: "verieql" (paper), or "mutation" (no external
    #: dependency: random small edits of D; for machines without VeriEQL)
    verifier: str = "verieql"
    verieql_python: Optional[str] = None
    verieql_dir: Optional[str] = None
    #: VeriEQL row bound b and timeout t_max
    verieql_bound: int = 5
    verieql_timeout: float = 10.0
    #: delete a question's generated instances after its rollout group
    cleanup_instances: bool = True
    max_train_examples: Optional[int] = None
    logging_steps: int = 1
    save_steps: int = 100
    seed: int = 0


@dataclass
class EvalConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0            # greedy decoding (Table 3)
    execution_timeout: float = 30.0


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def dump(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=4)


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
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    # a config may extend another one: ``base: configs/default.yaml``
    base = data.pop("base", None)
    if base:
        if not os.path.isabs(base) and not os.path.exists(base):
            base = os.path.join(os.path.dirname(path), base)
        merged = _read_structured(base)
        _deep_update(merged, data)
        return merged
    return data


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def parse_override(text: str) -> Dict[str, Any]:
    """``"rl.lr=2e-6"`` -> ``{"rl": {"lr": 2e-06}}`` (value parsed as YAML)."""
    import yaml

    key, _, raw = text.partition("=")
    if not key or not _:
        raise ValueError(f"override must look like section.key=value, got {text!r}")
    value = yaml.safe_load(raw)
    out: Dict[str, Any] = {}
    cur = out
    parts = key.strip().split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value
    return out


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    """Defaults, then an optional YAML/JSON file, then ``section.key=value`` overrides."""
    cfg = Config()
    if path:
        _merge(cfg, _read_structured(path))
    for text in overrides or []:
        _merge(cfg, parse_override(text))
    cfg.paths.resolve_env()
    if cfg.teacher.hub_dir is None:
        cfg.teacher.hub_dir = cfg.paths.hf_hub
    return cfg

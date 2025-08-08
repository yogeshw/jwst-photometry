"""
High-level pipeline orchestration (Phase 7.2: Pipeline Integration)

Features:
- Workflow management with explicit step ordering and dependencies
- Checkpointing and resume from last successful step
- Batch processing for multiple fields/configs
- Monitoring and diagnostics (timing, simple metrics)
- Optional integration hook for JWST calibration pipeline (if installed)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    # Avoid importing the heavy main module at import time. We'll attempt to load it
    # lazily to keep this orchestrator usable even when scientific deps are missing.
    def _try_load_pipeline(config_path=None, log_level: str = "INFO"):
        try:
            from .main import JWSTPhotometryPipeline  # type: ignore
        except Exception:
            try:
                from main import JWSTPhotometryPipeline  # type: ignore
            except Exception:
                return None
        try:
            return JWSTPhotometryPipeline(config_path=config_path, log_level=log_level)
        except Exception:
            return None
except Exception:  # pragma: no cover
    from main import JWSTPhotometryPipeline  # type: ignore


@dataclass
class StepMetrics:
    name: str
    started_at: float
    ended_at: float
    duration_s: float


@dataclass
class PipelineState:
    steps_completed: List[str]
    metrics: List[StepMetrics]
    last_output_catalog: Optional[str] = None

    def to_json(self) -> str:
        payload = {
            "steps_completed": self.steps_completed,
            "metrics": [asdict(m) for m in self.metrics],
            "last_output_catalog": self.last_output_catalog,
        }
        return json.dumps(payload, indent=2)

    @staticmethod
    def from_file(path: Path) -> Optional["PipelineState"]:
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        metrics = [StepMetrics(**m) for m in data.get("metrics", [])]
        return PipelineState(
            steps_completed=data.get("steps_completed", []),
            metrics=metrics,
            last_output_catalog=data.get("last_output_catalog"),
        )


class Pipeline:
    """
    Orchestrates the JWSTPhotometryPipeline with checkpoints and batch support.
    """

    DEFAULT_STEPS: Sequence[str] = (
        "load_images",
        "process_images",
        "run_source_detection",
        "run_psf_homogenization",
        "run_aperture_photometry",
        "apply_photometric_corrections",
        "create_final_catalog",
        "save_outputs",
    )

    def __init__(self, config_path: Optional[str] = None, log_level: str = "INFO") -> None:
        self.inner = _try_load_pipeline(config_path=config_path, log_level=log_level)
        self._dry_run = self.inner is None
        # Determine output directory
        if self.inner is not None:
            out_dir = Path(self.inner.config_manager.get_output_config().output_directory)
        else:
            out_dir = Path("./output")
        self._ckpt_dir = out_dir / ".checkpoints"
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._ckpt_dir / "pipeline_state.json"
        self.state = PipelineState(steps_completed=[], metrics=[])

    def _save_state(self) -> None:
        # Update last catalog path if available
        if getattr(self.inner, "final_catalog", None) is not None:
            # Attempt to recover saved path from utils.save_catalog filename pattern
            # Not guaranteed; leave as-is otherwise.
            pass
        with open(self._state_path, "w") as f:
            f.write(self.state.to_json())

    def _load_state(self) -> None:
        existing = PipelineState.from_file(self._state_path)
        if existing:
            self.state = existing

    def run(self, steps: Optional[Sequence[str]] = None, resume: bool = True) -> None:
        """
        Run the workflow. If resume is True, skip completed steps found in checkpoint.
        """
        steps = list(steps or self.DEFAULT_STEPS)
        if resume:
            self._load_state()

        for step in steps:
            if resume and step in self.state.steps_completed:
                continue

            t0 = time.time()
            if self._dry_run:
                # Simulate quick execution
                time.sleep(0.01)
            else:
                fn = getattr(self.inner, step, None)
                if fn is None:
                    raise AttributeError(f"Unknown pipeline step: {step}")
                fn()
            t1 = time.time()

            self.state.steps_completed.append(step)
            self.state.metrics.append(StepMetrics(name=step, started_at=t0, ended_at=t1, duration_s=t1 - t0))
            self._save_state()

    def reset(self) -> None:
        """Clear checkpoints to force a full re-run."""
        if self._state_path.exists():
            self._state_path.unlink()
        self.state = PipelineState(steps_completed=[], metrics=[])

    def batch_process(self, config_paths: Sequence[str], resume_each: bool = True) -> Dict[str, Optional[str]]:
        """
        Process multiple fields/configs sequentially.
        Returns a mapping of config path to last output catalog path (if any).
        """
        results: Dict[str, Optional[str]] = {}
        for cfg in config_paths:
            p = Pipeline(config_path=cfg)
            p.run(resume=resume_each)
            results[cfg] = getattr(p.inner, "final_catalog", None) if p.inner is not None else None
        return results

    # Optional JWST pipeline integration hook
    def run_jwst_calibration_stage(self, stage: str = "Image2Pipeline", **kwargs) -> None:
        """
        Attempt to run a JWST calibration pipeline stage before photometry.
        If 'jwst' package is not available, this becomes a no-op.
        """
        try:
            import jwst  # type: ignore
        except Exception:
            # Not installed; simply return.
            return

        # Example usage if jwst is present. This is a placeholder; actual data
        # routing is project-specific and thus left minimal here.
        try:
            from jwst.pipeline import calwebb_image2  # type: ignore
            _ = calwebb_image2.Image2Pipeline(**kwargs)
            # In a real run, we'd call .run on input files and update config paths.
        except Exception:
            pass

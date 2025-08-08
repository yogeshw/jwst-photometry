import os
import pytest

# Ensure src on path when running tests directly
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import Pipeline


def test_pipeline_checkpoint_and_resume(tmp_path):
    # Create minimal config.yaml stub path (not strictly used due to defaults)
    cfg = None
    p = Pipeline(config_path=cfg)

    # First run should create checkpoints and complete steps
    p.reset()
    p.run(resume=True)
    assert len(p.state.steps_completed) >= 3  # at least a few steps executed

    # Subsequent run with resume should skip already completed steps
    before = list(p.state.steps_completed)
    p.run(resume=True)
    after = list(p.state.steps_completed)
    assert before == after


def test_pipeline_batch_api():
    p = Pipeline()
    result = p.batch_process([])
    assert isinstance(result, dict)

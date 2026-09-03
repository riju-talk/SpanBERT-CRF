"""Tests for the main.py orchestrator's configuration layer (no training)."""

import argparse

import pytest

main = pytest.importorskip("main")


class TestConfigFromArgs:
    def _parse(self, argv):
        return main.config_from_args(main.build_parser().parse_args(argv))

    def test_defaults_run_both_tasks_with_crf(self):
        cfg = self._parse([])
        assert cfg.tasks == ["qa", "ner"]
        assert cfg.use_crf is True

    def test_no_crf_flag_disables_crf(self):
        assert self._parse(["--no-crf"]).use_crf is False

    def test_negative_sample_caps_become_none(self):
        cfg = self._parse(["--max-train-samples", "-1", "--max-eval-samples", "-1"])
        assert cfg.max_train_samples is None
        assert cfg.max_eval_samples is None

    def test_duplicate_tasks_are_deduped_in_order(self):
        assert self._parse(["--tasks", "ner", "qa", "ner"]).tasks == ["ner", "qa"]

    def test_lora_and_upload_flags(self):
        cfg = self._parse(["--use-lora", "--upload", "--lora-r", "16"])
        assert cfg.use_lora and cfg.upload and cfg.lora_r == 16


class TestCheckpointPath:
    def test_crf_tag_present_for_both_tasks(self):
        cfg = main.PipelineConfig(tasks=["qa", "ner"], use_crf=True, output_dir="out")
        assert cfg.checkpoint_path("qa").endswith("spanbert_qa_crf_base.pt")
        assert cfg.checkpoint_path("ner").endswith("spanbert_ner_crf_base.pt")

    def test_lora_tag(self):
        cfg = main.PipelineConfig(tasks=["qa"], use_crf=True, use_lora=True, output_dir="out")
        assert "crf_lora_" in cfg.checkpoint_path("qa")


class TestAsTrainArgs:
    def test_shape_matches_src_train_expectations(self):
        cfg = main.PipelineConfig(tasks=["qa"])
        ns = main._as_train_args(cfg, "qa")
        assert isinstance(ns, argparse.Namespace)
        for field in ("task", "use_crf", "use_lora", "lora_r", "lora_alpha",
                      "max_train_samples", "max_eval_samples", "max_length",
                      "batch_size", "learning_rate", "num_epochs",
                      "gradient_accumulation", "device", "output_dir",
                      "use_wandb", "upload", "hf_repo_id"):
            assert hasattr(ns, field)
        assert ns.upload is False  # orchestrator owns uploads

    def test_repo_id_switches_by_task(self):
        cfg = main.PipelineConfig(tasks=["qa", "ner"])
        assert main._as_train_args(cfg, "qa").hf_repo_id == cfg.hf_repo_qa
        assert main._as_train_args(cfg, "ner").hf_repo_id == cfg.hf_repo_ner


def test_dry_run_exits_zero_without_training(capsys):
    assert main.main(["--dry-run", "--tasks", "qa"]) == 0
    assert "dry-run" in capsys.readouterr().out

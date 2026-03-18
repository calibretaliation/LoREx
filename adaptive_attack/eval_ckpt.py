import argparse
import os
from pathlib import Path
from typing import Any, Dict

import torch

from train import PROJECT_ROOT, build_models, evaluate_downstream_mlp


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate an adaptive_attack checkpoint (student encoder + UNet generator)"
	)
	parser.add_argument("--ckpt", type=str, required=True, help="Path to adaptive_attack_*.pth")
	parser.add_argument("--gpu", default="0", type=str, help="CUDA_VISIBLE_DEVICES setting")
	parser.add_argument("--downstream_dataset", default=None, type=str, help="Override downstream dataset")
	parser.add_argument(
		"--downstream_batch_size", default=None, type=int, help="Override downstream batch size"
	)
	parser.add_argument(
		"--downstream_eval_mode",
		default=None,
		choices=["generator", "badencoder_patch"],
		help="Override eval mode: generator or badencoder_patch",
	)
	parser.add_argument("--reference_label", default=None, type=int, help="Override target label used for ASR")
	parser.add_argument(
		"--trigger_file",
		default=None,
		type=str,
		help="Override trigger file (only used for badencoder_patch eval)",
	)
	return parser.parse_args()


def _merge_args(saved: Dict[str, Any], override: argparse.Namespace) -> argparse.Namespace:
	merged = dict(saved)
	for key in [
		"gpu",
		"downstream_dataset",
		"downstream_batch_size",
		"downstream_eval_mode",
		"reference_label",
		"trigger_file",
	]:
		val = getattr(override, key)
		if val is not None:
			merged[key] = val

	merged.setdefault("downstream_eval_mode", "generator")
	merged.setdefault("downstream_batch_size", 256)
	return argparse.Namespace(**merged)


def main() -> None:
	cli = parse_args()
	ckpt_path = Path(cli.ckpt)
	ckpt = torch.load(ckpt_path, map_location="cpu")

	saved_args = ckpt.get("args")
	if not isinstance(saved_args, dict):
		raise SystemExit("Checkpoint missing 'args' dict; cannot reconstruct model config")

	args = _merge_args(saved_args, cli)

	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if not getattr(args, "data_dir", ""):
		args.data_dir = str(PROJECT_ROOT / f"data/{args.shadow_dataset.split('_')[0]}/") + "/"

	teacher_encoder, student_encoder, generator, student_model = build_models(args, device)
	student_model.load_state_dict(ckpt["student_state_dict"], strict=True)
	generator.load_state_dict(ckpt["generator_state_dict"], strict=True)

	metrics = evaluate_downstream_mlp(student_encoder, args, device, generator=generator)
	if metrics is None:
		raise SystemExit(1)


if __name__ == "__main__":
	main()
import os
import random
import shutil
from pathlib import Path


def split_dataset(
	source_dir: str,
	output_dir: str,
	train_ratio: float = 0.6,
	val_ratio: float = 0.2,
	test_ratio: float = 0.2,
	seed: int = 42,
) -> None:
	if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
		raise ValueError("Ratios must sum to 1.0")

	src = Path(source_dir)
	out = Path(output_dir)

	if not src.exists():
		raise FileNotFoundError(f"Source directory not found: {src}")

	class_dirs = [d for d in src.iterdir() if d.is_dir()]
	if not class_dirs:
		raise ValueError("No class folders found in source directory")

	random.seed(seed)

	for split in ("train", "val", "test"):
		(out / split).mkdir(parents=True, exist_ok=True)

	for class_dir in class_dirs:
		images = [p for p in class_dir.iterdir() if p.is_file()]
		if not images:
			continue

		random.shuffle(images)
		n_total = len(images)
		n_train = int(n_total * train_ratio)
		n_val = int(n_total * val_ratio)
		n_test = n_total - n_train - n_val

		splits = {
			"train": images[:n_train],
			"val": images[n_train : n_train + n_val],
			"test": images[n_train + n_val :],
		}

		for split_name, split_files in splits.items():
			target_class_dir = out / split_name / class_dir.name
			target_class_dir.mkdir(parents=True, exist_ok=True)
			for file_path in split_files:
				target_path = target_class_dir / file_path.name
				shutil.copy2(file_path, target_path)

	print(f"Done. Output in: {out}")


if __name__ == "__main__":
	# The parent of scripts/ is the project root Embedded_ai/
	project_root = Path(__file__).resolve().parent.parent
	source_dataset = project_root / "data" / "raw" / "Potato Leaf Disease Dataset in Uncontrolled Environment"
	# Output to data/data_split
	output_dataset = project_root / "data" / "data_split"

	split_dataset(
		source_dir=source_dataset,
		output_dir=output_dataset,
		train_ratio=0.6,
		val_ratio=0.2,
		test_ratio=0.2,
		seed=42,
	)

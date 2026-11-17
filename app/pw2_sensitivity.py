"""PW2 - full sensitivity study, printed as Markdown tables for the README.

Run inside the container (takes several minutes):
    python pw2_sensitivity.py            # 10,000 training images (quick)
    python pw2_sensitivity.py --full     # all 60,000 images (slow)
Also prints the teacher's baseline and the "normalized" improvement.
"""

import sys

from services import mlp


def main():
    full = "--full" in sys.argv
    limit = None if full else 10000
    print(f"# PW2 sensitivity study ({'60,000' if full else '10,000'} training images)\n")

    print("## Teacher's model vs. improvements (full data)\n")
    print("| Model | Normalize | Epochs | Params | Test loss | Test accuracy |")
    print("|---|---|---|---|---|---|")
    for name, params in (
        ("Teacher (no hidden layer)", mlp.TEACHER),
        ("Teacher + normalization", dict(mlp.TEACHER, normalize=True)),
        ("128-64-32 relu + normalization", dict(mlp.TEACHER, hidden_layers="128,64,32",
                                                normalize=True)),
    ):
        r = mlp.train_mlp(params)
        print(f"| {name} | {params['normalize']} | {params['epochs']} | {r['num_params']:,} "
              f"| {r['test_loss']:.4f} | {r['test_accuracy']:.4f} |", flush=True)

    base = mlp.DEFAULTS
    print(f"\nBase configuration: {base}\n")
    for parameter in mlp.STUDY_VALUES:
        print(f"## {parameter}\n")
        print(f"| {parameter} | Test accuracy | Test loss | Params | Time (s) |")
        print("|---|---|---|---|---|")
        for row in mlp.run_sensitivity(parameter, base=base, train_limit=limit):
            print(f"| {row['value']} | {row['test_accuracy']:.4f} | {row['test_loss']:.4f} "
                  f"| {row['num_params']:,} | {row['train_time']} |", flush=True)
        print()


if __name__ == "__main__":
    main()

import sys
import numpy as np
from experiments import (
    experiment_1_baseline,
    experiment_2_disinhibition,
    experiment_3_trn_lateral,
    experiment_4_oscillations,
    experiment_5_bg_integration
)


def run_all():
    """Run all 5 experiments and report results."""
    print("=" * 60)
    print("  BioMind-Thalamus: Biologically Faithful Thalamic Circuit")
    print("  Component 2 of the BioMind Conscious Intelligence Project")
    print("=" * 60)

    results = {}

    results['exp1'] = experiment_1_baseline()
    results['exp2'] = experiment_2_disinhibition()
    results['exp3'] = experiment_3_trn_lateral()
    results['exp4'] = experiment_4_oscillations()
    results['exp5'] = experiment_5_bg_integration()

    # Final report
    print("\n" + "x" * 60)
    print("  FINAL RESULTS")
    print("x" * 60)

    exp_names = {
        'exp1': 'Baseline Firing Rates',
        'exp2': 'Disinhibition (Burst-Tonic)',
        'exp3': 'TRN Lateral Inhibition',
        'exp4': 'Oscillation Generation',
        'exp5': 'BG-Thalamus Integration'
    }

    all_pass = True
    for key, name in exp_names.items():
        passed = results[key]
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Experiment {key[-1]}: {name}")
        if not passed:
            all_pass = False

    print("-" * 60)
    if all_pass:
        print("  ALL EXPERIMENTS PASSED")
        print("  BioMind-Thalamus validated. Ready for Component 3.")
    else:
        print("  SOME EXPERIMENTS FAILED")
        print("  Parameter tuning required before proceeding.")
    print("x" * 60)

    return all_pass


def run_single(exp_num):
    """Run a single experiment by number (1-5)."""
    experiments = {
        1: experiment_1_baseline,
        2: experiment_2_disinhibition,
        3: experiment_3_trn_lateral,
        4: experiment_4_oscillations,
        5: experiment_5_bg_integration
    }

    if exp_num not in experiments:
        print(f"Invalid experiment number: {exp_num}")
        print("Valid options: 1-5")
        return False

    return experiments[exp_num]()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            exp_num = int(sys.argv[1])
            run_single(exp_num)
        except ValueError:
            if sys.argv[1] == 'all':
                run_all()
            else:
                print("Usage: python run.py [all | 1-5]")
    else:
        run_all()
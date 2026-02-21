#!/usr/bin/env python3
"""
Detect which models in src/stg/ changed vs. a base ref and run
their unit tests and edge-case tests.

Usage:
    python tests/run_changed_models.py [base_ref]

    base_ref defaults to "origin/main" but can be any git ref,
    e.g. "HEAD~1" for the last commit only.

Exit codes:
    0 – all tests passed (or no models changed)
    1 – one or more tests failed
"""
import sys
import subprocess

# Mapping from model dir-name -> unit test file and pytest -k key
MODEL_MAP = {
    "identity":       ("tests/unit/test_identity.py",        "Identity"),
    "CART":           ("tests/unit/test_cart.py",            "CART"),
    "DPCART":         ("tests/unit/test_dpcart.py",          "DPCART"),
    "CTGAN":          ("tests/unit/test_ctgan.py",           "CTGAN"),
    "PATECTGAN":      ("tests/unit/test_patectgan.py",       "PATECTGAN"),
    "TVAE":           ("tests/unit/test_tvae.py",            "TVAE"),
    "TabDiff":        ("tests/unit/test_tabdiff.py",         "TabDiff"),
    "TabPFGen":       ("tests/unit/test_tabpfgen.py",        "TabPFGen"),
    "TabDDPM":        ("tests/unit/test_tabddpm.py",         "TabDDPM"),
    "AIM":            ("tests/unit/test_aim.py",             "AIM"),
    "SMOTE":          ("tests/unit/test_smote.py",           "SMOTE"),
    "GaussianCopula": ("tests/unit/test_gaussian_copula.py", "GaussianCopula"),
    "AutoDiff":       ("tests/unit/test_autodiff.py",        "AutoDiff"),
    "ARF":            ("tests/unit/test_arf.py",             "ARF"),
    "BayesianNetwork":("tests/unit/test_bayesian_network.py","BayesianNetwork"),
    "GREAT":          ("tests/unit/test_great.py",           "GREAT"),
    "NFlow":          ("tests/unit/test_nflow.py",           "NFlow"),
    "LTM_VAE":        ("tests/unit/test_ltm_vae.py",         "LTM_VAE"),
    "TabSyn":         ("tests/unit/test_tabsyn.py",          "TabSyn"),
}

def get_changed_files(base_ref="origin/main"):
    try:
        result = subprocess.check_output(
            ["git", "diff", "--name-only", base_ref],
            text=True, stderr=subprocess.STDOUT,
        )
        return result.splitlines()
    except subprocess.CalledProcessError as e:
        print(f"git diff failed: {e.output}")
        sys.exit(1)

def detect_changed_models(files):
    models = set()
    for f in files:
        parts = f.split("/")
        if len(parts) >= 3 and parts[0] == "src" and parts[1] == "stg":
            models.add(parts[2])
    return sorted(models)

def run(cmd):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode

def main():
    base_ref = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    changed_files = get_changed_files(base_ref)
    changed_models = detect_changed_models(changed_files)

    if not changed_models:
        print("✅ No model changes detected — skipping targeted tests.")
        sys.exit(0)

    print(f"🔍 Changed models: {', '.join(changed_models)}")
    failures = 0

    for model_dir in changed_models:
        entry = MODEL_MAP.get(model_dir)
        print(f"\n{'='*60}")
        print(f"Testing model: {model_dir}")
        print('='*60)

        # 1. Unit test (if a unit test file exists for this model)
        if entry:
            unit_file, k_key = entry
            rc = run(["pytest", unit_file, "-v", "--tb=short"])
            if rc != 0:
                failures += 1
        else:
            print(f"  ⚠️  No unit test mapping for {model_dir}, skipping unit test.")

        # 2. Edge case tests filtered to this model
        rc = run([
            "pytest", "tests/integration/test_edge_cases.py",
            "-k", model_dir.split("/")[-1],
            "-v", "--tb=short",
        ])
        if rc != 0:
            failures += 1

    if failures:
        print(f"\n❌ {failures} test suite(s) failed.")
        sys.exit(1)
    else:
        print("\n✅ All targeted tests passed!")

if __name__ == "__main__":
    main()

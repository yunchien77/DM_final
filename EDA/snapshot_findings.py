"""
Run every section in eda.py once and dump the structured `findings` dicts to
JSON for use as a planning baseline. No figures are rendered.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import eda

OUT_PATH = Path(__file__).parent / "findings_snapshot.json"


def _json_safe(obj):
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (pd.Series,)):
        return _json_safe(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return _json_safe(obj.to_dict(orient="list"))
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    return str(obj)


def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("  EDA SNAPSHOT — running all sections, dumping findings to JSON", flush=True)
    print("=" * 70, flush=True)

    all_findings = []

    def _record(section_name, fig_findings_pair):
        figs, findings = fig_findings_pair
        # Close any opened figures (saves memory; we don't persist plots).
        import matplotlib.pyplot as plt
        for f in figs:
            plt.close(f)
        all_findings.append(findings)
        print(f"[+] {section_name} done  (cumulative {time.time()-t0:.1f}s)", flush=True)

    # --- Section 1 -----------------------------------------------------------
    _record("section_data_overview", eda.section_data_overview())

    # --- Shared loads --------------------------------------------------------
    print("\n[load] Full scored train rows ...", flush=True)
    scored_df = eda._load_full_train_scored()
    print(f"  loaded {len(scored_df):,} rows  ({time.time()-t0:.1f}s)", flush=True)

    print("[load] Train sample (100 regions) ...", flush=True)
    train_sample = eda._load_train_sample(n_regions=100)
    print(f"  loaded {train_sample.shape}  ({time.time()-t0:.1f}s)", flush=True)

    print("[load] Test ...", flush=True)
    test_df = eda._load_test()
    print(f"  loaded {test_df.shape}  ({time.time()-t0:.1f}s)", flush=True)
    print()

    # --- Sections 2-12 -------------------------------------------------------
    _record("section_score_distribution",        eda.section_score_distribution(scored_df))
    _record("section_feature_distributions",     eda.section_feature_distributions(train_sample))
    _record("section_timeseries_viz",            eda.section_timeseries_viz())
    _record("section_weekly_patterns",           eda.section_weekly_patterns(scored_df))
    _record("section_test_analysis",             eda.section_test_analysis(scored_df))
    _record("section_outlier_analysis",          eda.section_outlier_analysis(train_sample))
    _record("section_distribution_shape",        eda.section_distribution_shape(train_sample))
    _record("section_train_vs_test_shift",       eda.section_train_vs_test_shift(train_sample, test_df))
    _record("section_score_transitions",         eda.section_score_transitions(scored_df))
    _record("section_region_clustering",         eda.section_region_clustering(scored_df, n_clusters=8))
    _record("section_lag_cross_correlation",     eda.section_lag_cross_correlation(scored_df, max_lag=12))

    print()
    eda.print_aggregated_findings(all_findings)

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": round(time.time() - t0, 1),
        "sections": [_json_safe(f) for f in all_findings],
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\n[snapshot] Wrote {OUT_PATH}  ({len(json.dumps(payload)):,} bytes)", flush=True)


if __name__ == "__main__":
    main()

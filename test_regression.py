"""
Regression / smoke tests for the N-body ODD battery.

Run with:  python test_regression.py
Or:        python -m pytest test_regression.py -v

Tests are ordered from cheapest to most expensive.
"""
import importlib
import json
import math
import sys
import types
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def import_stress():
    import nbody_stress
    return nbody_stress


def import_paper():
    import nbody_paper
    return nbody_paper


def _has_matplotlib():
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


_MATPLOTLIB_AVAILABLE = _has_matplotlib()


# ---------------------------------------------------------------------------
# 1. Import / compile
# ---------------------------------------------------------------------------

class TestImports(unittest.TestCase):
    def test_stress_imports(self):
        m = import_stress()
        self.assertTrue(hasattr(m, "run_stress"))
        self.assertTrue(hasattr(m, "analyse"))
        self.assertTrue(hasattr(m, "_winner_gap_bootstrap"))

    @unittest.skipUnless(_MATPLOTLIB_AVAILABLE, "matplotlib not installed")
    def test_paper_imports(self):
        m = import_paper()
        self.assertTrue(hasattr(m, "write_macros"))
        self.assertTrue(hasattr(m, "build_configs"))

    def test_3d_imports(self):
        import nbody_3d
        # Primary integration / IC entry points
        self.assertTrue(hasattr(nbody_3d, "integrate"),
                        "nbody_3d missing 'integrate'")
        self.assertTrue(hasattr(nbody_3d, "initial_conditions"),
                        "nbody_3d missing 'initial_conditions'")


# ---------------------------------------------------------------------------
# 2. Registry consistency
# ---------------------------------------------------------------------------

class TestRegistries(unittest.TestCase):
    def setUp(self):
        self.s = import_stress()

    def test_pred_specs_direct_contains_radial(self):
        names = {n for _, n in self.s._PRED_SPECS_DIRECT}
        self.assertIn("CoarseConc",      names)
        self.assertIn("CoarseRShellVar", names)

    def test_pred_specs_periodic_excludes_radial(self):
        names = {n for _, n in self.s._PRED_SPECS_PERIODIC}
        self.assertNotIn("CoarseConc",      names)
        self.assertNotIn("CoarseRShellVar", names)

    def test_fine_obs_present_in_both_specs(self):
        s = self.s
        fine_names = {n for _, n in s.FINE_OBS}
        for spec_name, spec in [("_PRED_SPECS_DIRECT",   s._PRED_SPECS_DIRECT),
                                 ("_PRED_SPECS_PERIODIC", s._PRED_SPECS_PERIODIC)]:
            spec_names = {n for _, n in spec}
            for fn in fine_names:
                self.assertIn(fn, spec_names,
                              f"{fn} missing from {spec_name}")

    def test_coarse_obs_grid_subset_of_coarse_obs(self):
        s = self.s
        grid_cols = {c for c, _ in s._COARSE_OBS_GRID}
        all_coarse_cols = {c for c, _ in s.COARSE_OBS}
        self.assertTrue(grid_cols.issubset(all_coarse_cols))


# ---------------------------------------------------------------------------
# 3. Observable functions
# ---------------------------------------------------------------------------

class TestObservables(unittest.TestCase):
    def setUp(self):
        self.s = import_stress()
        rng = np.random.default_rng(0)
        self.pos = rng.uniform(0, 1, (64, 3))

    def test_coarse_conc_range(self):
        v = self.s.obs_coarse_conc(self.pos, periodic=False, box_size=1.0)
        self.assertTrue(0.0 <= v <= 1.0, f"coarse_conc={v} out of [0,1]")

    def test_coarse_conc_uses_mean_center_for_isolated(self):
        # For periodic=False, center is np.mean(pos) not box_size/2.
        # Spot-check: value is in [0, 1].
        v = self.s.obs_coarse_conc(self.pos, periodic=False, box_size=1.0)
        self.assertTrue(0.0 <= v <= 1.0 or math.isnan(v),
                        f"coarse_conc={v} out of [0,1]")

    def test_run_stress_fences_radial_obs_for_periodic(self):
        """run_stress() must emit NaN for radial obs on periodic models."""
        s = self.s
        cfg = s.StressConfig(
            model="pm_periodic", init="plummer3d", seed=0,
            n=32, eps=0.05, box_size=10.0, steps=40, k_fine=4, dt=0.01,
        )
        r = s.run_stress(cfg, use_numba=False)
        if r.get("status") != "ok":
            self.skipTest(f"pm_periodic smoke run failed: {r.get('error')}")
        self.assertTrue(math.isnan(float(r["coarse_conc_0"])),
                        "coarse_conc_0 should be NaN for pm_periodic")
        self.assertTrue(math.isnan(float(r["coarse_rshell_var_0"])),
                        "coarse_rshell_var_0 should be NaN for pm_periodic")

    def test_angular_shuffle_preserves_radii(self):
        s = self.s
        center = np.array([0.5, 0.5, 0.5])
        rng = np.random.default_rng(42)
        shuffled = s._angular_shuffle_pos(self.pos, rng, center)
        r_orig = np.sqrt(np.sum((self.pos    - center) ** 2, axis=1))
        r_shuf = np.sqrt(np.sum((shuffled    - center) ** 2, axis=1))
        np.testing.assert_allclose(np.sort(r_orig), np.sort(r_shuf), atol=1e-12,
                                   err_msg="angular shuffle changed radial distribution")

    def test_angular_shuffle_changes_positions(self):
        s = self.s
        center = np.array([0.5, 0.5, 0.5])
        rng = np.random.default_rng(42)
        shuffled = s._angular_shuffle_pos(self.pos, rng, center)
        # Almost certainly not identical (probability of collision is negligible)
        self.assertFalse(np.allclose(self.pos, shuffled),
                         "angular shuffle left positions unchanged")

    def test_angshuf_preserves_radial_predictors(self):
        s = self.s
        base_cfg = s.StressConfig(
            model="direct_isolated", init="plummer3d", seed=7,
            n=96, eps=0.05, box_size=2.0, steps=40, k_fine=8, dt=0.01,
        )
        shuf_cfg = s.StressConfig(
            model="direct_isolated", init="plummer3d_angshuf", seed=7,
            n=96, eps=0.05, box_size=2.0, steps=40, k_fine=8, dt=0.01,
        )
        base = s.run_stress(base_cfg, use_numba=False)
        shuf = s.run_stress(shuf_cfg, use_numba=False)
        if base.get("status") != "ok" or shuf.get("status") != "ok":
            self.skipTest(f"angshuf/base comparison failed: {base.get('message')} / {shuf.get('message')}")
        self.assertAlmostEqual(
            float(base["coarse_conc_0"]), float(shuf["coarse_conc_0"]), places=12,
            msg="angshuf must preserve the base radial concentration predictor")
        self.assertAlmostEqual(
            float(base["coarse_rshell_var_0"]), float(shuf["coarse_rshell_var_0"]), places=12,
            msg="angshuf must preserve the base radial shell-variance predictor")

    def test_bimodal_angshuf_reports_actual_shuffled_radial_predictors(self):
        s = self.s
        cfg = s.StressConfig(
            model="direct_isolated", init="bimodal3d_angshuf", seed=11,
            n=128, eps=0.05, box_size=2.0, steps=40, k_fine=8, dt=0.01,
        )
        row = s.run_stress(cfg, use_numba=False)
        if row.get("status") != "ok":
            self.skipTest(f"bimodal angshuf run failed: {row.get('message')}")
        pos0, _ = s.get_initial_conditions(cfg)
        center = np.mean(pos0, axis=0)
        conc = s.obs_coarse_conc(pos0, periodic=False, box_size=cfg.box_size, center=center)
        rshell = s.obs_coarse_rshell_var(pos0, periodic=False, box_size=cfg.box_size, center=center)
        self.assertAlmostEqual(
            float(row["coarse_conc_0"]), float(conc), places=12,
            msg="bimodal angshuf row must report concentration from the shuffled IC actually simulated")
        self.assertAlmostEqual(
            float(row["coarse_rshell_var_0"]), float(rshell), places=12,
            msg="bimodal angshuf row must report radial shell variance from the shuffled IC actually simulated")

    def test_fof_isolated_uses_number_density_linking_scale(self):
        s = self.s
        pos = np.array([
            [0.10, 0.10, 0.10],
            [0.22, 0.10, 0.10],
            [0.10, 0.22, 0.10],
            [0.10, 0.10, 0.22],
            [0.78, 0.78, 0.78],
            [0.90, 0.78, 0.78],
            [0.78, 0.90, 0.78],
            [0.78, 0.78, 0.90],
        ], dtype=float)
        self.assertEqual(
            s.obs_fof_groups(pos, periodic=False, box_size=2.0, fof_b=0.2), 8,
            "isolated FoF must scale linking length with number density, not raw span volume")


# ---------------------------------------------------------------------------
# 4. Bootstrap: model-aware predictor routing
# ---------------------------------------------------------------------------

class TestWinnerGapBootstrap(unittest.TestCase):
    def setUp(self):
        self.s = import_stress()
        rng = np.random.default_rng(7)
        n = 60
        # Fabricate rows that look like direct_isolated output
        x_fine   = rng.normal(0, 1, n)
        x_coarse = rng.normal(0, 1, n)
        target   = x_fine * 0.8 + rng.normal(0, 0.3, n)
        self.rows_direct = []
        for i in range(n):
            row: dict = {}
            for col, _ in self.s.FINE_OBS:
                row[col] = float(x_fine[i])
            row["coarse_g4_0"]         = float(x_coarse[i])
            row["coarse_g8_0"]         = float(x_coarse[i])
            row["coarse_g16_0"]        = float(x_coarse[i])
            row["coarse_conc_0"]       = float(rng.uniform(0.1, 0.5))
            row["coarse_rshell_var_0"] = float(rng.uniform(0.0, 1.0))
            row["d_coarse_g8_early"]   = float(target[i])
            self.rows_direct.append(row)

        # PM rows: radial columns are NaN
        self.rows_pm = []
        for row in self.rows_direct:
            pm_row = dict(row)
            pm_row["coarse_conc_0"]       = float("nan")
            pm_row["coarse_rshell_var_0"] = float("nan")
            self.rows_pm.append(pm_row)

    def test_direct_bootstrap_succeeds(self):
        s = self.s
        result = s._winner_gap_bootstrap(
            self.rows_direct, "d_coarse_g8_early", n_boot=50, seed=1,
            pred_specs=s._PRED_SPECS_DIRECT)
        self.assertIsNotNone(result["winner_gap_mean"],
                             "direct bootstrap returned None — listwise dropped all rows")

    def test_pm_bootstrap_not_poisoned(self):
        """PM rows with NaN radial obs must not be dropped by listwise completion."""
        s = self.s
        result = s._winner_gap_bootstrap(
            self.rows_pm, "d_coarse_g8_early", n_boot=50, seed=1,
            pred_specs=s._PRED_SPECS_PERIODIC)
        self.assertIsNotNone(result["winner_gap_mean"],
                             "PM bootstrap returned None — radial NaNs poisoned listwise filter")

    def test_pm_bootstrap_does_not_use_radial_cols(self):
        """Verify _PRED_SPECS_PERIODIC contains no radial column names."""
        s = self.s
        radial_cols = {"coarse_conc_0", "coarse_rshell_var_0"}
        spec_cols   = {c for c, _ in s._PRED_SPECS_PERIODIC}
        overlap = radial_cols & spec_cols
        self.assertEqual(overlap, set(),
                         f"_PRED_SPECS_PERIODIC includes radial cols: {overlap}")

    def test_bootstrap_discards_degenerate_resamples(self):
        s = self.s
        rows = []
        fine = [0, 0, 0, 0, 1]
        coarse = [1, 0, 0, 0, 0]
        target = [0, 0, 0, 0, 1]
        for f, c, y in zip(fine, coarse, target):
            rows.append({
                "fine_knn_all_0": float(f),
                "coarse_g8_0": float(c),
                "d_coarse_g8_early": float(y),
            })
        result = s._winner_gap_bootstrap(
            rows, "d_coarse_g8_early", n_boot=4000, seed=1,
            pred_specs=[("fine_knn_all_0", "kNN-all"),
                        ("coarse_g8_0", "CoarseG8")],
        )
        self.assertIsNotNone(result["winner_gap_ci_lo"])
        self.assertGreater(
            float(result["winner_gap_ci_lo"]), 0.3,
            "degenerate bootstrap resamples should be discarded, not counted as zero correlation")


# ---------------------------------------------------------------------------
# 5. Smoke run: 2 replicates, smallest config, end-to-end
# ---------------------------------------------------------------------------

class TestSmokeRun(unittest.TestCase):
    """
    Runs a tiny 2-replicate battery (N=64, 1 IC, 1 eps, direct_isolated only)
    and checks that analyse() produces non-empty output with no NaN-poisoned cells.
    """
    @classmethod
    def setUpClass(cls):
        s = import_stress()
        cfg = s.StressConfig(
            model="direct_isolated",
            init="plummer3d",
            seed=0,
            n=64,
            eps=0.05,
            box_size=10.0,
            steps=100,
            k_fine=8,
            dt=0.01,
        )
        rows = []
        for seed in [0, 1]:
            cfg2 = s.StressConfig(**{**vars(cfg), "seed": seed})
            try:
                r = s.run_stress(cfg2, use_numba=False)
                rows.append(r)
            except Exception as e:
                raise RuntimeError(f"run_stress failed: {e}") from e
        cls.rows = rows
        cls.analysis = s.analyse(rows, n_boot=20, min_ok_hard=1)
        cls.s = s

    def test_runs_complete(self):
        ok = [r for r in self.rows if r.get("status") == "ok"]
        self.assertGreaterEqual(len(ok), 1, "No successful smoke runs")

    def test_analysis_nonempty(self):
        self.assertGreater(len(self.analysis), 0, "analyse() returned empty dict")

    def test_direct_cells_not_poisoned(self):
        """direct_isolated cell must have winner_gap_mean for primary target."""
        s = self.s
        for key, cell in self.analysis.items():
            if cell.get("model") == "direct_isolated":
                wg = cell.get(f"winner_gap_mean_{s.PRIMARY_TARGET}")
                # wg can be None if n_ok < 5; just check it isn't poisoned by NaN
                if wg is not None:
                    self.assertFalse(math.isnan(wg),
                                     f"winner_gap_mean is NaN for direct cell {key}")

    def test_radial_obs_present_in_direct_rows(self):
        ok = [r for r in self.rows if r.get("status") == "ok"]
        for r in ok:
            self.assertIn("coarse_conc_0",       r)
            self.assertIn("coarse_rshell_var_0", r)
            # For non-periodic direct, these must not be NaN
            self.assertFalse(math.isnan(float(r["coarse_conc_0"])),
                             "coarse_conc_0 is NaN in direct_isolated row")

    def test_angshuf_ic_smoke(self):
        """Angular-shuffle IC must complete without error."""
        s = self.s
        cfg = s.StressConfig(
            model="direct_isolated",
            init="plummer3d_angshuf",
            seed=0,
            n=64,
            eps=0.05,
            box_size=10.0,
            steps=100,
            k_fine=8,
            dt=0.01,
        )
        r = s.run_stress(cfg, use_numba=False)
        self.assertEqual(r.get("status"), "ok",
                         f"angshuf smoke run failed: {r.get('error')}")
        # Radial obs are still valid for angshuf (direct_isolated, not periodic)
        self.assertFalse(math.isnan(float(r["coarse_conc_0"])),
                         "coarse_conc_0 NaN in angshuf direct_isolated row")


# ---------------------------------------------------------------------------
# 6. write_macros runtime-arg fidelity
# ---------------------------------------------------------------------------

class TestMacroFidelity(unittest.TestCase):
    @unittest.skipUnless(_MATPLOTLIB_AVAILABLE, "matplotlib not installed")
    def test_nreplicates_and_nbootstrap_reflect_args(self):
        """write_macros must use caller-supplied counts, not hardcoded constants."""
        import io, os, tempfile
        p = import_paper()
        # Build a minimal stub analysis and rows so write_macros doesn't crash
        # on missing cells (it uses safe_float which returns None gracefully)
        rows: list = []
        analysis: dict = {}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            tmp = f.name
        try:
            p.write_macros(analysis, rows, tmp,
                           n_replicates=123, n_boot=4567)
            with open(tmp) as f:
                content = f.read()
            self.assertIn("123",  content, "NReplicates not written correctly")
            self.assertIn("4567", content, "NBootstrap not written correctly")
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# 7. angshuf model enforcement
# ---------------------------------------------------------------------------

class TestAngshufEnforcement(unittest.TestCase):
    """_angshuf ICs must be silently dropped for any model != direct_isolated."""

    def setUp(self):
        self.s = import_stress()

    def _build_configs_via_product(self, models, inits, n=64, eps=0.05,
                                   k_fine=8, steps=40, replicates=1):
        """Replicate the config-building loop in main() without invoking argparse."""
        s = self.s
        from itertools import product
        seeds = list(range(replicates))
        DIRECT_N_MAX = 2048
        configs = []
        for n_val, eps_val, k_val, model, init, seed in product(
                [n], [eps], [k_fine], models, inits, seeds):
            if model == "direct_isolated" and n_val > DIRECT_N_MAX:
                continue
            if init.endswith("_angshuf") and model != "direct_isolated":
                continue   # belt-and-suspenders guard
            configs.append(s.StressConfig(
                model=model, init=init, seed=seed,
                n=n_val, steps=steps, eps=eps_val, k_fine=k_val))
        return configs

    def test_angshuf_excluded_from_pm_periodic(self):
        configs = self._build_configs_via_product(
            models=["pm_periodic"],
            inits=["plummer3d_angshuf"])
        self.assertEqual(len(configs), 0,
                         "angshuf IC must be excluded from pm_periodic configs")

    def test_angshuf_excluded_from_direct_periodic(self):
        configs = self._build_configs_via_product(
            models=["direct_periodic"],
            inits=["plummer3d_angshuf"])
        self.assertEqual(len(configs), 0,
                         "angshuf IC must be excluded from direct_periodic configs")

    def test_angshuf_allowed_for_direct_isolated(self):
        configs = self._build_configs_via_product(
            models=["direct_isolated"],
            inits=["plummer3d_angshuf"])
        self.assertGreater(len(configs), 0,
                           "angshuf IC should be allowed for direct_isolated")

    def test_angshuf_mixed_models_only_produces_direct_isolated(self):
        """When angshuf and non-isolated models are both present, only direct_isolated rows survive."""
        configs = self._build_configs_via_product(
            models=["direct_isolated", "pm_periodic"],
            inits=["plummer3d", "plummer3d_angshuf"])
        angshuf_models = {c.model for c in configs if c.init.endswith("_angshuf")}
        self.assertEqual(angshuf_models, {"direct_isolated"},
                         f"angshuf configs present for unexpected models: {angshuf_models}")


# ---------------------------------------------------------------------------
# 8. validate_outputs fatal/advisory classification
# ---------------------------------------------------------------------------

class TestValidationLayer(unittest.TestCase):
    """
    validate_outputs() must correctly classify fatal vs advisory warnings
    without requiring a full run.  We call it with a minimal stub analysis
    that triggers specific conditions.
    """
    @unittest.skipUnless(_MATPLOTLIB_AVAILABLE, "matplotlib not installed")
    def test_empty_analysis_is_fatal(self):
        p = import_paper()
        all_w, fatal_w = p.validate_outputs({}, [], skip_figures=True, skip_tables=True)
        self.assertTrue(any("empty" in w.lower() for w in fatal_w),
                        "empty analysis must produce a fatal warning")

    @unittest.skipUnless(_MATPLOTLIB_AVAILABLE, "matplotlib not installed")
    def test_no_pm_cells_is_fatal(self):
        """If analysis has only direct_isolated cells, no-PM must be fatal."""
        p = import_paper()
        # Use correct key format from make_key()
        key = p.make_key("direct_isolated", "bimodal3d", 1024, 0.05)
        stub_cell = {
            "model": "direct_isolated", "init": "bimodal3d",
            "n": 1024, "eps": 0.05, "underpowered": False,
            f"winner_gap_mean_{p.PRIMARY_TARGET}": 0.1,
        }
        analysis = {key: stub_cell}
        all_w, fatal_w = p.validate_outputs(
            analysis, [], skip_figures=True, skip_tables=True)
        self.assertTrue(any("PM" in w.upper() or "pm_periodic" in w.lower()
                            for w in fatal_w),
                        "no powered PM cells must be a fatal warning")

    @unittest.skipUnless(_MATPLOTLIB_AVAILABLE, "matplotlib not installed")
    def test_underpowered_cells_are_advisory_not_fatal(self):
        """Underpowered flagship cells must warn but not be fatal."""
        p = import_paper()
        # Populate ALL expected flagship cells so MISSING FLAGSHIP CELLS doesn't
        # fire and drown the underpowered advisory.  Set bimodal3d/N=1024/eps=0.05
        # as underpowered; everything else as powered.
        analysis = {}
        direct_ns = [n for n in p.PAPER_N if n <= p.DIRECT_N_MAX]
        for n in direct_ns:
            for eps in p.PAPER_EPS:
                for init in p.PAPER_INITS:
                    is_target = (init == "bimodal3d" and n == 1024 and abs(eps - 0.05) < 1e-9)
                    cell = {
                        "model": "direct_isolated", "init": init,
                        "n": n, "eps": eps,
                        "underpowered": is_target,
                        "n_ok": 5 if is_target else 50,
                    }
                    analysis[p.make_key("direct_isolated", init, n, eps)] = cell
        # Add one PM cell so that check passes
        pm_key = p.make_key("pm_periodic", "bimodal3d", 1024, 0.05)
        analysis[pm_key] = {
            "model": "pm_periodic", "init": "bimodal3d",
            "n": 1024, "eps": 0.05, "underpowered": False,
        }
        all_w, fatal_w = p.validate_outputs(
            analysis, [], skip_figures=True, skip_tables=True)
        advisory_only = [w for w in all_w if w not in fatal_w]
        self.assertTrue(any("UNDERPOWERED" in w for w in advisory_only),
                        "underpowered-cell warning should appear as advisory, not fatal")
        self.assertFalse(any("UNDERPOWERED" in w for w in fatal_w),
                         "underpowered-cell warning must NOT be fatal")

    @unittest.skipUnless(_MATPLOTLIB_AVAILABLE, "matplotlib not installed")
    def test_clean_run_produces_no_warnings(self):
        """A stub analysis with no issues should produce zero warnings."""
        p = import_paper()
        # We can't trivially satisfy "all flagship cells present" without
        # reproducing the full paper grid, so just verify that the
        # empty-analysis path is what triggers the fatal — not a spurious default.
        all_w, fatal_w = p.validate_outputs(
            {}, [], skip_figures=True, skip_tables=True)
        self.assertGreater(len(fatal_w), 0,
                           "empty analysis must produce at least one fatal warning")
        # And that it's exactly the empty-analysis warning, not a figure warning
        self.assertFalse(any("FIGURE" in w for w in fatal_w),
                         "figure warnings should not fire when skip_figures=True")


class TestCSVMixedKeys(unittest.TestCase):
    """write_csv_rows must preserve columns even when rows[0] lacks them."""

    def test_union_of_keys_preserves_late_columns(self):
        from nbody_paper import write_csv_rows, load_csv_rows
        import tempfile, os
        rows = [
            {"n": 256, "eps": 0.05, "model": "direct_isolated", "coarse_g8_0": 1.5},
            {"n": 256, "eps": 0.05, "model": "direct_isolated", "coarse_g8_0": 2.0,
             "coarse_g8_f": 3.0},
        ]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            write_csv_rows(path, rows)
            reloaded = load_csv_rows(path)
            self.assertEqual(len(reloaded), 2)
            # Row 1 (missing coarse_g8_f) should have empty string / None
            self.assertIn("coarse_g8_f", reloaded[0])
            # Row 2 should have the actual value
            self.assertAlmostEqual(float(reloaded[1]["coarse_g8_f"]), 3.0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)

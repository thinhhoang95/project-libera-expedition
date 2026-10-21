"""Job queue: state transitions, recovery after a crash, and a worker running, pausing and resuming a step."""

import json
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

from landscape import config, jobs


class QueueTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.patches = [mock.patch.object(jobs, "APP_DIR", self.tmp / "app"),
                        mock.patch.object(jobs, "DB_FILE", self.tmp / "app" / "app.sqlite"),
                        mock.patch.object(config, "PROJECTS", self.tmp / "projects")]
        for p in self.patches:
            p.start()
        self.con = jobs.connect()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def step(self, body: str, name="fake"):
        f = self.tmp / f"{name}.py"
        f.write_text(textwrap.dedent(body))
        return (name, name.capitalize(), ["script", str(f)])

    def wait(self, jid, until, timeout=30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            j = jobs.job_dict(self.con, jid)
            if until(j):
                return j
            time.sleep(0.2)
        self.fail(f"timed out; job is {jobs.job_dict(self.con, jid)}")

    def test_enqueue_uses_the_catalogue(self):
        jid = jobs.enqueue(self.con, "demo", jobs.TEMPLATES["map"], title="map")
        j = jobs.job_dict(self.con, jid)
        self.assertEqual([s["name"] for s in j["steps"]], jobs.TEMPLATES["map"])
        self.assertEqual(j["steps"][0]["argv"], ["cli", "expand", "--run-id", "demo"])
        self.assertEqual(j["queue_position"], 1)

    def test_reorder_and_hold(self):
        a = jobs.enqueue(self.con, "demo", ["portfolio"], title="a")
        b = jobs.enqueue(self.con, "demo", ["portfolio"], title="b")
        jobs.control(self.con, b, "up")
        self.assertEqual(jobs.job_dict(self.con, b)["queue_position"], 1)
        self.assertEqual(jobs.control(self.con, a, "pause")["status"], "paused")
        self.assertEqual(jobs.control(self.con, a, "resume")["status"], "queued")

    def test_recover_requeues_running_jobs_and_marks_the_step(self):
        jid = jobs.enqueue(self.con, "demo", ["expand", "extract"], title="t")
        self.con.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (jid,))
        self.con.execute("UPDATE steps SET status = 'done' WHERE job_id = ? AND idx = 0", (jid,))
        self.con.execute("UPDATE steps SET status = 'running' WHERE job_id = ? AND idx = 1", (jid,))
        self.assertEqual(jobs.recover(self.con, "pipeline"), 1)
        j = jobs.job_dict(self.con, jid)
        self.assertEqual(j["status"], "queued")
        self.assertEqual([s["status"] for s in j["steps"]], ["done", "interrupted"])

    def test_worker_runs_pauses_and_resumes(self):
        done_hook = []
        quick = self.step("""
            import sys, time
            for i in range(1, 4):
                print(f"work {i}/3", flush=True)
                time.sleep(0.1)
        """, "quick")
        slow = self.step("""
            import sys, time
            print("starting", flush=True)
            for i in range(600):
                sys.stderr.write(f"\\r {i * 100 // 600}%|##| {i}/600")
                sys.stderr.flush()
                time.sleep(0.05)
        """, "slow")
        jid = jobs.enqueue(self.con, "demo", [quick, slow], title="t")
        w = jobs.Worker(lanes=("pipeline",), on_step_done=lambda job, s: done_hook.append(s["name"]), poll_s=0.2)
        self.assertEqual(w.start(), ["pipeline"])
        try:
            j = self.wait(jid, lambda j: j["steps"][1]["status"] == "running" and (j["steps"][1]["progress"] or 0) > 0)
            self.assertEqual(j["steps"][0]["status"], "done")
            self.assertEqual(done_hook, ["quick"])
            jobs.control(self.con, jid, "pause")
            j = self.wait(jid, lambda j: j["status"] == "paused")
            self.assertEqual(j["steps"][1]["status"], "interrupted")
            log = Path(j["steps"][1]["log"]).read_text()
            self.assertIn("starting", log)
            self.assertNotIn("%|##|", log)  # carriage-return progress updates stay out of the log
            jobs.control(self.con, jid, "cancel")
            self.assertEqual(jobs.job_dict(self.con, jid)["status"], "cancelled")
            # run again (with a short body): finished steps are kept, the interrupted one restarts
            self.con.execute("UPDATE steps SET argv = ? WHERE job_id = ? AND idx = 1", (json.dumps(quick[2]), jid))
            jobs.control(self.con, jid, "resume")
            j = self.wait(jid, lambda j: j["status"] in ("done", "failed"))
            self.assertEqual(j["status"], "done", j.get("error"))
            self.assertEqual([s["attempts"] for s in j["steps"]], [1, 2])
        finally:
            w.shutdown()

    def test_failed_step_fails_the_job_with_the_log_tail(self):
        bad = self.step("""
            print("about to fail", flush=True)
            raise SystemExit(3)
        """, "bad")
        jid = jobs.enqueue(self.con, "demo", [bad], title="t")
        w = jobs.Worker(lanes=("pipeline",), poll_s=0.2)
        w.start()
        try:
            j = self.wait(jid, lambda j: j["status"] == "failed")
            self.assertEqual(j["steps"][0]["exit_code"], 3)
            self.assertIn("about to fail", j["error"])
        finally:
            w.shutdown()


class CatalogTest(unittest.TestCase):
    """The catalogue reaches every experiment script, and every saved experiment has a step and a decision."""

    def test_every_script_step_and_experiment_script_exists(self):
        from landscape import experiments

        for s in jobs.CATALOG:
            if s.command[0] == "script":
                self.assertTrue((config.ROOT / s.command[1]).exists(), s.command[1])
        for step, path, _, exps in experiments.RUNS:
            self.assertIn(step, jobs.STEPS)
            self.assertTrue((config.ROOT / path).exists(), path)
            for e in exps:
                self.assertIn(e, experiments.PILOT_DECISIONS)
        on_disk = {str(p.relative_to(config.ROOT)) for d in ("phase8", "phase8b", "phase10")
                   for p in (config.ROOT / "scripts" / d).glob("*.py")
                   if p.name != "common.py" and not p.name.startswith("._")}  # ._*: macOS metadata on non-APFS disks
        covered = {path for _, path, _, _ in experiments.RUNS} | {
            s.command[1] for s in jobs.CATALOG if s.command[0] == "script"}
        # superseded by `landscape external` (a CLI step)
        self.assertEqual(on_disk - covered, {"scripts/phase8b/external_mine.py"})

    def test_templates_name_known_steps(self):
        for name, steps in jobs.TEMPLATES.items():
            self.assertTrue(set(steps) <= set(jobs.STEPS), name)
            self.assertIn(name, jobs.TEMPLATE_TITLES)
        self.assertEqual(jobs.TEMPLATES["full"][-1], "portfolio")  # rebuilt after the experiments

    def test_project_models_win_over_the_defaults(self):
        tmp = Path(tempfile.mkdtemp())
        with mock.patch.object(config, "PROJECTS", tmp / "projects"):
            self.assertEqual(config.model_file("demo@T2021", "early_work"), config.DEFAULT_MODELS / "early_work.json")
            own = config.project_dir("demo", "models") / "early_work.json"
            own.write_text("{}")
            self.assertEqual(config.model_file("demo@T2021", "early_work"), own)
            self.assertEqual(config.model_file("other", "early_work"), config.DEFAULT_MODELS / "early_work.json")


class RunExperimentsTest(unittest.TestCase):
    """scripts/run_experiments.py: runs a step's scripts in order, resumes after a failure, reports spend."""

    def test_resume_after_failure_and_spend(self):
        import importlib.util

        from landscape import experiments

        spec = importlib.util.spec_from_file_location("run_experiments", config.ROOT / "scripts" / "run_experiments.py")
        rx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rx)
        tmp = Path(tempfile.mkdtemp())
        log, flag = tmp / "ran.txt", tmp / "fail-once"
        flag.touch()
        a, b = tmp / "a.py", tmp / "b.py"
        a.write_text(f"open({str(log)!r}, 'a').write('a\\n')\n")
        b.write_text(textwrap.dedent(f"""
            import os
            open({str(log)!r}, 'a').write('b\\n')
            print("@@spend llm_experiments=0.25", flush=True)
            if os.path.exists({str(flag)!r}):
                os.remove({str(flag)!r})
                raise SystemExit(4)
        """))
        spent = []
        with mock.patch.object(config, "PROJECTS", tmp / "projects"), \
             mock.patch.dict(experiments.STEP_SCRIPTS, {"experiments-test": [(str(a), ()), (str(b), ())]}), \
             mock.patch.object(rx, "record_spend", lambda run, s: spent.append(s)), \
             mock.patch("sys.argv", ["run_experiments.py", "--step", "experiments-test", "--run-id", "demo"]):
            self.assertEqual(rx.main(), 4)
            state = experiments.results_dir("demo") / ".progress-experiments-test.json"
            self.assertEqual(json.loads(state.read_text())["done"], [str(a)])
            self.assertEqual(rx.main(), 0)  # resumes at b
            self.assertFalse(state.exists())
        self.assertEqual(log.read_text().split(), ["a", "b", "b"])
        self.assertEqual(spent, [{"llm_experiments": 0.25}] * 2)


if __name__ == "__main__":
    unittest.main()

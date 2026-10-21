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


if __name__ == "__main__":
    unittest.main()

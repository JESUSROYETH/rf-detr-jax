"""Internal experiment records must not return through Git or built packages."""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RepositoryHygieneTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "candidate"
        self.root.mkdir()
        for name in ("LICENSE", "NOTICE", "README.md", "AGENTS.md", "CONTRIBUTING.md", "SECURITY.md", ".gitignore"):
            shutil.copy(ROOT / name, self.root / name)
        for name in ("docs", "scripts", "tests/data"):
            shutil.copytree(
                ROOT / name, self.root / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*-measurements", "tpu.local.sh",
                                              "tpu.md", "gpu.md", "comparison.md"),
            )
        self.dist = Path(self.temporary.name) / "dist"
        self.dist.mkdir()
        self.wheel = self.dist / "example-0.1-py3-none-any.whl"
        with zipfile.ZipFile(self.wheel, "w") as archive:
            for path in (ROOT / "src").rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, str(path.relative_to(ROOT / "src")))
            for name in ("LICENSE", "NOTICE"):
                archive.write(ROOT / name, f"example-0.1.dist-info/licenses/{name}")
        self.sdist = self.dist / "example-0.1.tar.gz"
        shutil.copy(ROOT / "Makefile", self.root / "Makefile")
        self._write_sdist()

    def _write_sdist(self, injected=None):
        with tarfile.open(self.sdist, "w:gz") as archive:
            archive.add(self.root, arcname="example-0.1")
            if injected:
                content = b"internal experiment sentinel\n"
                member = tarfile.TarInfo(f"example-0.1/{injected}")
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))

    def _check(self, distributions=False):
        command = [sys.executable, str(self.root / "scripts" / (
            "check_distributions.py" if distributions else "check_repository.py"
        ))]
        if distributions:
            command.append(str(self.dist))
        return subprocess.run(command, cwd=self.root, capture_output=True, text=True)

    def test_clean_export_and_distributions_keep_reference_fixtures(self):
        for distributions in (False, True):
            result = self._check(distributions)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_force_tracked_internal_report_is_rejected(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True)
        report = self.root / "docs/tpu.md"
        report.write_text("internal experiment sentinel\n")
        subprocess.run(["git", "-C", str(self.root), "add", "-f", "docs/tpu.md"], check=True)
        result = self._check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("private/generated file in source: docs/tpu.md", result.stderr)

    def test_extracted_internal_records_are_rejected(self):
        for name in (
            "docs/gpu-measurements/raw.json", "artifacts/run.json", ".venv-review/pyvenv.cfg",
            "client.pem", "src/rfdetr_jax/client.key",
        ):
            with self.subTest(name=name):
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("internal experiment sentinel\n")
                result = self._check()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name, result.stderr)
                path.unlink()

    def test_sdist_internal_records_are_rejected(self):
        for name in (
            "docs/tpu.md", "docs/gpu-measurements/raw.json", "artifacts/run.json", ".venv-review/pyvenv.cfg",
            "scripts/client.pem", "src/rfdetr_jax/client.key",
        ):
            with self.subTest(name=name):
                self._write_sdist(name)
                result = self._check(distributions=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("source distribution contains local-only content", result.stderr)

    def test_private_identity_markers_are_rejected(self):
        cases = {
            "docs/notes.md": "Measured on NO" "DE-77 by owner@" "private-lab.io for about $0." "70 per hour.\n",
            "docs/zone.md": "The node ran in Europe-" "North9-z.\n",
            "scripts/helper.sh": "cd /ho" "me/someone/rf-detr-jax && ./run.sh\n",
            "docs/cloud.md": "Upload to gs:" "//private-bucket/results for project my-project-" "123456.\n",
            "docs/tools.md": "Drafted with cla" "ude on instance-2026" "0101-010101.\n",
            "docs/budget.md": "Budget: about €" "4.50 per run, 12 EU" "R total, project-0" "123abcd-ef01.\n",
        }
        for name, content in cases.items():
            with self.subTest(name=name):
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
                result = self._check()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"private content marker: {name}", result.stderr)
                path.unlink()

    def test_allowed_identity_spellings_pass(self):
        path = self.root / "docs/examples.md"
        path.write_text(
            "Set TPU_ACCOUNT=you@example.com and TPU_PROJECT=your-project. "
            "Write to ~/.roboflow/models or $HOME/runs. Contact noreply@github.com or "
            "12345+someone@users.noreply.github.com. Versions 2.10.0 and 0.25.0; sha 6674d858.\n"
            "Placeholders stay generic: TPU_NAME=<tpu-name> TPU_ZONE=<zone>. Script args use $1, $2 and $10.\n"
        )
        result = self._check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        path.unlink()

    def test_wheel_nested_internal_records_are_rejected(self):
        clean_wheel = self.wheel.read_bytes()
        for name in ("rfdetr_jax/artifacts/run.json", "rfdetr_jax/client.pem", "rfdetr_jax/client.key"):
            with self.subTest(name=name):
                self.wheel.write_bytes(clean_wheel)
                with zipfile.ZipFile(self.wheel, "a") as archive:
                    archive.writestr(name, "internal experiment sentinel\n")
                result = self._check(distributions=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("wheel contains local-only content", result.stderr)


if __name__ == "__main__":
    unittest.main()

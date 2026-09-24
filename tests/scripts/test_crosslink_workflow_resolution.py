import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SELECTED_AUTOFIX_TOKEN = (
    "${{ steps.app-token.outputs.has_app == 'true' "
    "&& steps.app-token.outputs.token "
    "|| secrets.AUTOFIX_BOT_PAT || github.token }}"
)


def _load_yaml(path: str):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def _step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_ci_timing_report_never_receives_the_privileged_autofix_pat():
    # Upstream split the old monolithic ci.yml and removed its timing job.
    # Keep the security invariant architecture-independent: if any timing job
    # exists in any workflow, it must not receive the privileged autofix PAT.
    timing_jobs = []
    for path in sorted((ROOT / ".github/workflows").glob("*.*")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for job_id, job in (workflow.get("jobs") or {}).items():
            name = str(job.get("name") or "") if isinstance(job, dict) else ""
            if "timing" in f"{job_id} {name}".lower():
                timing_jobs.append((path.name, job_id, job))

    for path_name, job_id, job in timing_jobs:
        assert "AUTOFIX_BOT_PAT" not in str(job), (path_name, job_id)


def test_workflows_do_not_reference_unprovisioned_larger_runner_labels():
    forbidden = {
        "ubuntu-latest-32-core",
        "ubuntu-latest-32-arm-core",
        "ubuntu-latest-96-core",
        "windows-latest-32-core",
    }
    offenders = []
    for path in sorted((ROOT / ".github/workflows").glob("*.*")):
        text = path.read_text(encoding="utf-8")
        for label in forbidden:
            if label in text:
                offenders.append((path.name, label))

    assert offenders == []


def test_python_suite_budget_preserves_full_standard_runner_coverage():
    job = _load_yaml(".github/workflows/tests.yml")["jobs"]["test"]
    run = _step(job, "Run tests")

    # Capacity floor for the expanded suite, not a per-file timeout increase.
    assert job["timeout-minutes"] >= 90
    assert job["runs-on"] == "ubuntu-latest"
    assert run["run"].strip().splitlines()[-1] == "scripts/run_tests.sh"
    assert run["env"]["HERMES_TEST_WORKERS"] == 4
    assert not job.get("continue-on-error", False)
    assert not run.get("continue-on-error", False)
    env = {**job.get("env", {}), **run.get("env", {})}
    assert not ({"HERMES_TEST_FILE_TIMEOUT", "HERMES_TEST_PATHS", "HERMES_TEST_SLICE"} & env.keys())


def test_ci_detection_budget_covers_full_history_checkout_on_standard_runner():
    # A one-minute budget cancelled main CI during actions/checkout, before
    # detect-changes could schedule Python and upgrade E2E jobs.
    workflow = _load_yaml(".github/workflows/ci.yaml")
    assert workflow["jobs"]["detect"]["timeout-minutes"] >= 10


def test_release_tag_picker_accepts_crosslink_release_tags(tmp_path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "--allow-empty", "-q", "-m", "init"],
        check=True,
    )
    for tag in (
        "crosslink-v2026.5.14",
        "crosslink-v2026.7.1",
        "crosslink-v2026.8.11",
        "backup/not-a-release",
        "v2026.9.11",  # inherited upstream tags must not displace fork releases
    ):
        subprocess.run(["git", "-C", str(repo), "tag", tag], check=True)

    result = subprocess.run(
        [
            str(ROOT / "scripts/sandbox/pick-release-tags.sh"),
            "--repo",
            str(repo),
            "--count",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [
        "crosslink-v2026.5.14",
        "crosslink-v2026.8.11",
    ]

    workflow = (ROOT / ".github/workflows/install-e2e.yml").read_text(
        encoding="utf-8"
    )
    assert "crosslink-v[0-9]+.[0-9]+.[0-9]+" in workflow

    harness = (ROOT / "tests/install/installer-script-e2e.sh").read_text(
        encoding="utf-8"
    )
    assert '[[ "$text" == *"$2"* ]]' in harness
    assert 'installer_supports "$1" "--migrate-legacy-origin"' in harness
    assert "flags+=(--migrate-legacy-origin)" in harness


def test_js_autofix_restores_app_auth_with_crosslink_pat_fallback():
    workflow = _load_yaml(".github/workflows/js-autofix.yml")
    job = workflow["jobs"]["apply-patch"]

    assert job["environment"] == "trusted-automation"
    assert "github.ref == 'refs/heads/main'" in job["if"]
    assert "github.ref_protected == true" in job["if"]

    checkout = next(step for step in job["steps"] if "actions/checkout@" in step.get("uses", ""))
    assert checkout["with"]["persist-credentials"] is False

    app_token = _step(job, "Get GitHub App token")
    assert app_token["with"] == {
        "client-id": "${{ vars.APP_CLIENT_ID }}",
        "private-key": "${{ secrets.APP_PRIVATE_KEY }}",
    }

    push = _step(job, "Apply patch and push to bot branch")
    assert push["env"]["GH_TOKEN"] == SELECTED_AUTOFIX_TOKEN
    assert push["run"].index("gh auth setup-git") < push["run"].index("git push --force")

    create_pr = _step(job, "Create/update PR and enable auto-merge")
    wait = _step(job, "Wait for merge, auto-close on failure or stale")
    assert create_pr["env"]["GH_TOKEN"] == SELECTED_AUTOFIX_TOKEN
    assert wait["env"]["GH_TOKEN"] == SELECTED_AUTOFIX_TOKEN


def test_app_token_action_requires_both_credentials_and_exposes_selection_state():
    action = _load_yaml(".github/actions/get-app-token/action.yml")
    check = _step(action["runs"], "Check if App credentials exist")

    assert check["env"] == {
        "CLIENT_ID": "${{ inputs.client-id }}",
        "PRIVATE_KEY": "${{ inputs.private-key }}",
    }
    assert '[ -n "$CLIENT_ID" ] && [ -n "$PRIVATE_KEY" ]' in check["run"]
    assert action["outputs"]["has_app"]["value"] == "${{ steps.check.outputs.has_app }}"


def test_crosslink_main_and_release_runs_publish_multiarch_images():
    workflow = _load_yaml(".github/workflows/docker.yml")
    expected = (
        "contains(fromJSON('[\"NousResearch/hermes-agent\","
        "\"crosslink-ch/hermes-agent\"]'), github.repository) && "
        "(github.event_name == 'push' && github.ref == 'refs/heads/main' "
        "|| github.event_name == 'release')"
    )

    assert workflow["jobs"]["publish"]["if"] == expected
    assert workflow["jobs"]["merge"]["if"] == "${{ !cancelled() && " + expected + " }}"
    assert expected.split(" && ", 1)[0] in workflow["jobs"]["build"]["if"]
    assert workflow["jobs"]["build"]["needs"] == ["detect"]
    assert workflow["jobs"]["publish"]["needs"] == ["build"]
    assert workflow["jobs"]["merge"]["needs"] == ["publish"]
    assert workflow["jobs"]["publish"]["environment"] == "container-publish"
    assert workflow["jobs"]["merge"]["environment"] == "container-publish"
    assert workflow["env"]["IMAGE_NAME"] == (
        "${{ github.repository == 'crosslink-ch/hermes-agent' && "
        "'crosslinkch/hermes-agent' || 'nousresearch/hermes-agent' }}"
    )
    for job_name in ("build", "publish"):
        matrix = workflow["jobs"][job_name]["strategy"]["matrix"]
        assert matrix["arch"] == ["amd64", "arm64"]
        assert matrix["variant"] == ["slim", "desktop"]
    assert workflow["jobs"]["merge"]["strategy"]["matrix"]["include"] == [
        {"variant": "slim", "suffix": ""},
        {"variant": "desktop", "suffix": "-desktop"},
    ]
    manifest = _step(workflow["jobs"]["merge"], "Create manifest list and push")["run"]
    assert 'if [ "${#args[@]}" -ne 2 ]; then' in manifest
    assert '"${IMAGE_NAME}:main${SUFFIX}"' in manifest


def test_upstream_site_deployment_is_not_activated_on_crosslink():
    workflow = _load_yaml(".github/workflows/deploy-site.yml")
    for job in workflow["jobs"].values():
        assert "github.repository == 'NousResearch/hermes-agent'" in job["if"]

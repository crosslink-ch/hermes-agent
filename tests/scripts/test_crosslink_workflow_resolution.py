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
    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
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
    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        for label in forbidden:
            if label in text:
                offenders.append((path.name, label))

    assert offenders == []


def test_js_autofix_restores_app_auth_with_crosslink_pat_fallback():
    workflow = _load_yaml(".github/workflows/js-autofix.yml")
    job = workflow["jobs"]["apply-patch"]

    assert job["environment"] == "trusted-automation"

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
    assert workflow["jobs"]["merge"]["if"] == expected

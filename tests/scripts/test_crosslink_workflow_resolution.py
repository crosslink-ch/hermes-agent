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
    workflow = _load_yaml(".github/workflows/ci.yml")
    collect = _step(workflow["jobs"]["ci-timings"], "Collect timings and generate report")

    assert collect["env"]["GITHUB_TOKEN"] == "${{ github.token }}"


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

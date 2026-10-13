"""Profile-owned safe task snapshots, independent of invocation/typing lifetime."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

MAX_TERMINAL_TASKS = 256
TERMINAL_RETENTION_SECONDS = 7 * 24 * 60 * 60
TERMINAL = frozenset({"finished", "failed", "stopped", "unknown"})


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def token(value):
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 255
    ):
        raise ValueError("Invalid activity identity")
    return value


class ActivityStore:
    def __init__(self, home, transport_key):
        self.home = Path(home).resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        self.transport_key = transport_key
        self.lock = threading.RLock()
        self.db = sqlite3.connect(
            self.home / "background_activity.db", check_same_thread=False
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, transport TEXT NOT NULL, session_key TEXT NOT NULL, execution TEXT NOT NULL, snapshot TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS delivery (id TEXT PRIMARY KEY, sequence INTEGER NOT NULL, event TEXT NOT NULL, snapshot TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS runtime (id TEXT PRIMARY KEY, parent_active INTEGER NOT NULL, stop_requested INTEGER NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS bindings (task TEXT NOT NULL, link TEXT NOT NULL, PRIMARY KEY(task, link))"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS routes (task TEXT PRIMARY KEY, chat_id TEXT NOT NULL, thread_id TEXT)"
        )
        self.db.commit()
        self.reconcile()

    def reconcile(self):
        with self.lock:
            rows = self.db.execute(
                "SELECT snapshot FROM tasks WHERE transport=?", (self.transport_key,)
            ).fetchall()
            for (raw,) in rows:
                snapshot = json.loads(raw)
                if snapshot["parent"]["state"] in TERMINAL:
                    continue
                now = timestamp()
                snapshot["parent"] = {
                    "state": "unknown",
                    "action": None,
                    "updatedAt": now,
                }
                for worker in snapshot["workers"]:
                    if worker["state"] not in TERMINAL:
                        worker.update(state="unknown", updatedAt=now)
                for activity in snapshot["activities"]:
                    if activity["state"] == "running":
                        activity.update(state="unknown", finishedAt=now)
                self.save(snapshot)

    def pending(self):
        with self.lock:
            return [
                json.loads(row[0])
                for row in self.db.execute(
                    "SELECT d.snapshot FROM delivery d JOIN tasks t ON t.id=d.id WHERE t.transport=? ORDER BY t.rowid LIMIT 64",
                    (self.transport_key,),
                )
            ]

    def acknowledge(self, snapshot):
        with self.lock:
            self.db.execute(
                "DELETE FROM delivery WHERE id=? AND sequence=? AND event=?",
                (snapshot["taskId"], snapshot["sequence"], snapshot["eventId"]),
            )
            self.db.commit()

    def heartbeat(self):
        with self.lock:
            for (raw,) in self.db.execute(
                "SELECT snapshot FROM tasks WHERE transport=?", (self.transport_key,)
            ).fetchall():
                snapshot = json.loads(raw)
                if snapshot["parent"]["state"] not in TERMINAL:
                    self.save(snapshot)
            self.prune()

    def prune(self):
        with self.lock:
            retained = 0
            now = datetime.now(timezone.utc)
            rows = self.db.execute(
                "SELECT id, snapshot FROM tasks WHERE transport=? ORDER BY rowid DESC",
                (self.transport_key,),
            ).fetchall()
            for task_id, raw in rows:
                snapshot = json.loads(raw)
                if snapshot["parent"]["state"] not in TERMINAL:
                    continue
                retained += 1
                age = (
                    now - datetime.fromisoformat(snapshot["occurredAt"])
                ).total_seconds()
                if retained <= MAX_TERMINAL_TASKS and age <= TERMINAL_RETENTION_SECONDS:
                    continue
                for table, key in (
                    ("delivery", "id"),
                    ("runtime", "id"),
                    ("bindings", "task"),
                    ("routes", "task"),
                    ("tasks", "id"),
                ):
                    self.db.execute(f"DELETE FROM {table} WHERE {key}=?", (task_id,))
            self.db.commit()

    def completion_tasks(self, session_key, session_id, links):
        with self.lock:
            matches = []
            for link in links:
                rows = self.db.execute(
                    "SELECT t.id, t.snapshot FROM tasks t JOIN bindings b ON t.id=b.task WHERE t.transport=? AND t.session_key=? AND b.link=?",
                    (self.transport_key, session_key, link),
                )
                matches.extend(
                    task_id
                    for task_id, raw in rows
                    if json.loads(raw)["sessionId"] == session_id
                    and json.loads(raw)["parent"]["state"] not in TERMINAL
                )
            return tuple(dict.fromkeys(matches))

    def begin(
        self,
        session_key,
        session_id,
        origin=None,
        verification=False,
        target_task_id=None,
    ):
        token(session_id)
        with self.lock:
            row = None
            if verification:
                rows = self.db.execute(
                    "SELECT id, snapshot FROM tasks WHERE transport=? AND session_key=? ORDER BY rowid DESC",
                    (self.transport_key, session_key),
                )
                candidates = [
                    r
                    for r in rows
                    if json.loads(r[1])["parent"]["state"] not in TERMINAL
                    and (target_task_id is None or r[0] == target_task_id)
                ]
                row = candidates[0] if len(candidates) == 1 else None
                if row is None:
                    return None
            execution = str(uuid.uuid4())
            if row:
                task_id, raw = row
                runtime = self.db.execute(
                    "SELECT stop_requested FROM runtime WHERE id=?", (task_id,)
                ).fetchone()
                if runtime and runtime[0]:
                    return None
                snapshot = json.loads(raw)
                if snapshot["sessionId"] != session_id:
                    return None
                snapshot["parent"] = {
                    "state": "verifying",
                    "action": None,
                    "updatedAt": timestamp(),
                }
                self.db.execute(
                    "UPDATE tasks SET execution=? WHERE id=?", (execution, task_id)
                )
            else:
                if not origin:
                    return None
                token(origin)
                task_id = str(uuid.uuid4())
                now = timestamp()
                snapshot = {
                    "schemaVersion": 1,
                    "eventId": str(uuid.uuid4()),
                    "taskId": task_id,
                    "sequence": 0,
                    "originInvocationId": origin,
                    "sessionId": session_id,
                    "occurredAt": now,
                    "parent": {"state": "working", "action": None, "updatedAt": now},
                    "workers": [],
                    "activities": [],
                }
                self.db.execute(
                    "INSERT INTO tasks VALUES (?, ?, ?, ?, ?)",
                    (
                        task_id,
                        self.transport_key,
                        session_key,
                        execution,
                        json.dumps(snapshot),
                    ),
                )
            self.db.execute(
                "INSERT OR REPLACE INTO runtime VALUES (?, 1, 0)", (task_id,)
            )
            self.save(snapshot)
            return ActivityTurn(self, task_id, execution, verification=bool(row))

    def request_stop(self, session_key, target_task_id=None):
        targets = []
        requested = 0
        with self.lock:
            rows = self.db.execute(
                "SELECT id, execution, snapshot FROM tasks WHERE transport=? AND session_key=?",
                (self.transport_key, session_key),
            ).fetchall()
            for task_id, execution, raw in rows:
                if target_task_id is not None and task_id != target_task_id:
                    continue
                snapshot = json.loads(raw)
                if snapshot["parent"]["state"] in TERMINAL:
                    continue
                requested += 1
                self.db.execute(
                    "UPDATE runtime SET stop_requested=1 WHERE id=?", (task_id,)
                )
                snapshot["parent"].update(
                    action="Stop requested", updatedAt=timestamp()
                )
                turn = ActivityTurn(self, task_id, execution)
                turn._settle_stop(snapshot)
                self.save(snapshot)
                targets.extend(
                    w["id"]
                    for w in snapshot["workers"]
                    if w["kind"] == "process" and w["state"] not in TERMINAL
                )
            return requested, targets

    def bind_route(self, task_id, chat_id, thread_id=None):
        with self.lock:
            self.db.execute(
                "INSERT OR IGNORE INTO routes VALUES (?, ?, ?)",
                (
                    task_id,
                    token(chat_id),
                    token(thread_id) if thread_id is not None else None,
                ),
            )
            self.db.commit()

    def control_target(self, task_id, session_id, chat_id, thread_id=None):
        with self.lock:
            row = self.db.execute(
                "SELECT t.session_key, t.snapshot FROM tasks t JOIN routes r ON r.task=t.id WHERE t.transport=? AND t.id=? AND r.chat_id=? AND r.thread_id IS ?",
                (self.transport_key, task_id, chat_id, thread_id),
            ).fetchone()
            if not row or json.loads(row[1])["sessionId"] != session_id:
                return None
            return row[0]

    def control_runtime(self, task_id):
        with self.lock:
            row = self.db.execute(
                "SELECT parent_active, stop_requested FROM runtime WHERE id=?",
                (task_id,),
            ).fetchone()
            return tuple(row) if row else (0, 0)

    def control_links(self, task_id):
        with self.lock:
            return tuple(
                row[0]
                for row in self.db.execute(
                    "SELECT link FROM bindings WHERE task=?", (task_id,)
                )
            )

    def snapshot(self, task_id):
        with self.lock:
            row = self.db.execute(
                "SELECT snapshot FROM tasks WHERE id=? AND transport=?",
                (task_id, self.transport_key),
            ).fetchone()
            return json.loads(row[0]) if row else None

    def save(self, snapshot):
        snapshot["sequence"] += 1
        snapshot["eventId"] = str(uuid.uuid4())
        snapshot["occurredAt"] = timestamp()
        raw = json.dumps(snapshot, ensure_ascii=False)
        if len(raw.encode()) > 65536:
            raise ValueError("Activity snapshot exceeds wire limit")
        self.db.execute(
            "UPDATE tasks SET snapshot=? WHERE id=? AND transport=?",
            (raw, snapshot["taskId"], self.transport_key),
        )
        self.db.execute(
            "INSERT OR REPLACE INTO delivery VALUES (?, ?, ?, ?)",
            (snapshot["taskId"], snapshot["sequence"], snapshot["eventId"], raw),
        )
        self.db.commit()


class ActivityTurn:
    def __init__(self, store, task_id, execution_id, *, verification=False):
        self.store, self.task_id, self.execution_id = store, task_id, execution_id
        self.verification = verification

    def snapshot(self):
        return self.store.snapshot(self.task_id)

    def _current_execution(self):
        row = self.store.db.execute(
            "SELECT execution FROM tasks WHERE id=? AND transport=?",
            (self.task_id, self.store.transport_key),
        ).fetchone()
        return row is not None and row[0] == self.execution_id

    def _settle_stop(self, snapshot):
        runtime = self.store.db.execute(
            "SELECT parent_active, stop_requested FROM runtime WHERE id=?",
            (self.task_id,),
        ).fetchone()
        if (
            runtime
            and runtime[1]
            and not runtime[0]
            and not any(w["state"] not in TERMINAL for w in snapshot["workers"])
        ):
            now = timestamp()
            snapshot["parent"] = {"state": "stopped", "action": None, "updatedAt": now}
            for activity in snapshot["activities"]:
                if activity["state"] == "running":
                    activity.update(state="stopped", finishedAt=now)

    def event(self, event_type, tool_name=None, **metadata):
        with self.store.lock:
            snapshot = self.snapshot()
            if not snapshot or snapshot["parent"]["state"] in TERMINAL:
                return
            now = timestamp()
            worker_id = metadata.get("process_id") or metadata.get("subagent_id")
            if worker_id is None and not self._current_execution():
                return
            worker = next(
                (w for w in snapshot["workers"] if w["id"] == worker_id), None
            )
            if event_type in {
                "subagent.spawn_requested",
                "subagent.start",
                "process.started",
            }:
                if worker is not None:
                    if event_type != "subagent.start" or worker["state"] != "waiting":
                        return
                    worker.update(state="running", updatedAt=now)
                    if metadata.get("delegation_id"):
                        self.store.db.execute(
                            "INSERT OR IGNORE INTO bindings VALUES (?, ?)",
                            (self.task_id, token(metadata["delegation_id"])),
                        )
                    self.store.save(snapshot)
                    return
                if not worker_id:
                    return
                if len(snapshot["workers"]) >= 32:
                    expired = next(
                        (
                            old
                            for old in snapshot["workers"]
                            if old["state"] in TERMINAL
                        ),
                        None,
                    )
                    if expired is None:
                        return
                    snapshot["workers"].remove(expired)
                    snapshot["activities"] = [
                        a
                        for a in snapshot["activities"]
                        if a["workerId"] != expired["id"]
                    ]
                worker = {
                    "id": token(worker_id),
                    "kind": "process"
                    if event_type.startswith("process.")
                    else "subagent",
                    "parentId": metadata.get("subagent_id")
                    if event_type.startswith("process.")
                    else metadata.get("parent_id"),
                    "goal": "Background process"
                    if event_type.startswith("process.")
                    else "Delegated work",
                    "state": "waiting"
                    if event_type == "subagent.spawn_requested"
                    else "running",
                    "action": None,
                    "startedAt": now,
                    "updatedAt": now,
                    "lastActionAt": None,
                }
                snapshot["workers"].append(worker)
                for link in (worker_id, metadata.get("delegation_id")):
                    if link:
                        self.store.db.execute(
                            "INSERT OR IGNORE INTO bindings VALUES (?, ?)",
                            (self.task_id, token(link)),
                        )
            elif event_type in {"subagent.complete", "process.completed"}:
                if not worker or worker["state"] in TERMINAL:
                    return
                status = metadata.get("status", "completed")
                worker["state"] = {
                    "completed": "finished",
                    "interrupted": "stopped",
                    "killed": "stopped",
                    "unknown": "unknown",
                }.get(status, "failed")
                worker["updatedAt"] = now
                for activity in snapshot["activities"]:
                    if (
                        activity["workerId"] == worker_id
                        and activity["state"] == "running"
                    ):
                        activity["state"], activity["finishedAt"] = "unknown", now
                if snapshot["parent"]["state"] == "waiting_worker" and not any(
                    w["state"] not in TERMINAL for w in snapshot["workers"]
                ):
                    snapshot["parent"] = {
                        "state": "verifying",
                        "action": None,
                        "updatedAt": now,
                    }
            elif event_type in {
                "tool.started",
                "subagent.tool",
                "tool.completed",
                "subagent.tool.completed",
            }:
                if not tool_name or tool_name.startswith("_") or len(tool_name) > 100:
                    return
                call_id = metadata.get("tool_call_id")
                if not call_id:
                    return
                activity_id = f"{worker_id or self.execution_id}:{call_id}"
                activity = next(
                    (a for a in snapshot["activities"] if a["id"] == activity_id), None
                )
                if event_type in {"tool.started", "subagent.tool"}:
                    if activity or (worker and worker["state"] in TERMINAL):
                        return
                    activity = {
                        "id": token(activity_id),
                        "workerId": worker_id,
                        "toolName": tool_name,
                        "state": "running",
                        "startedAt": now,
                        "finishedAt": None,
                    }
                    snapshot["activities"].append(activity)
                    snapshot["activities"] = snapshot["activities"][-40:]
                elif activity and activity["state"] == "running":
                    activity["state"] = (
                        "failed" if metadata.get("is_error") else "finished"
                    )
                    activity["finishedAt"] = now
                else:
                    return
                if worker:
                    worker.update(action=tool_name, lastActionAt=now, updatedAt=now)
                else:
                    snapshot["parent"].update(action=tool_name, updatedAt=now)
            else:
                return
            self._settle_stop(snapshot)
            self.store.save(snapshot)

    def parent_returned(self, *, failed=False, stopped=False):
        with self.store.lock:
            snapshot = self.snapshot()
            if snapshot["parent"]["state"] in TERMINAL or not self._current_execution():
                return
            live = any(w["state"] not in TERMINAL for w in snapshot["workers"])
            state = (
                "waiting_worker"
                if live
                else "stopped"
                if stopped
                else "failed"
                if failed
                else "verifying"
            )
            snapshot["parent"] = {
                "state": state,
                "action": None,
                "updatedAt": timestamp(),
            }
            self.store.db.execute(
                "UPDATE runtime SET parent_active=0 WHERE id=?", (self.task_id,)
            )
            self._settle_stop(snapshot)
            self.store.save(snapshot)

    def finish(self, failed=False, stopped=False):
        with self.store.lock:
            snapshot = self.snapshot()
            row = self.store.db.execute(
                "SELECT execution FROM tasks WHERE id=?", (self.task_id,)
            ).fetchone()
            if (
                not snapshot
                or snapshot["parent"]["state"] in TERMINAL
                or not row
                or row[0] != self.execution_id
            ):
                return
            live = any(w["state"] not in TERMINAL for w in snapshot["workers"])
            state = (
                "waiting_worker"
                if live
                else "stopped"
                if stopped
                else "failed"
                if failed
                else "verifying"
                if snapshot["workers"] and not self.verification
                else "finished"
            )
            snapshot["parent"] = {
                "state": state,
                "action": None,
                "updatedAt": timestamp(),
            }
            self.store.db.execute(
                "UPDATE runtime SET parent_active=0 WHERE id=?", (self.task_id,)
            )
            self._settle_stop(snapshot)
            self.store.save(snapshot)

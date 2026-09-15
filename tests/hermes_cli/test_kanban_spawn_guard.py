"""Configured Kanban spawn guards are a fail-closed admission boundary."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_configured_spawn_guard_wraps_custom_dispatch_spawn(monkeypatch):
    calls = []
    task = object()

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return 4242

    def guard(task, workspace, board, native_spawn):
        assert task is not None
        return native_spawn(task, workspace, board=board)

    monkeypatch.setattr(kb, "_load_configured_spawn_guard", lambda: guard)
    assert kb._spawn_with_guard(task, "/tmp/workspace", "board", native) == 4242
    assert len(calls) == 1


def test_configured_spawn_guard_failure_never_calls_native_spawn(monkeypatch):
    called = []

    def native(*args, **kwargs):
        called.append(True)
        return 4242

    def guard(*_args, **_kwargs):
        raise kb.SpawnAdmissionError("governor unavailable")

    monkeypatch.setattr(kb, "_load_configured_spawn_guard", lambda: guard)
    with pytest.raises(kb.SpawnAdmissionError, match="governor unavailable"):
        kb._spawn_with_guard(object(), "/tmp/workspace", "board", native)
    assert called == []


def test_guard_deferral_releases_claim_without_creating_a_phantom_worker(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Healthy policy denial returns a task to ready without failure accounting.

    Direct regression coverage for the live incident (t_7588ce5b): a
    ``reserve threshold reached`` / ``admission-unavailable`` denial from the
    token-governor spawn guard must never consume the task's bounded
    ``consecutive_failures`` retry budget, and must never auto-block the
    card. Before this fix, EVERY ``SpawnAdmissionError`` (including these
    healthy, provider-side denials) fell into the generic
    ``_record_spawn_failure`` path and ticked the same counter as a genuine
    task bug — two spawn attempts during a provider dip were enough to trip
    the breaker and permanently stall a perfectly healthy task.
    """
    native_spawn = Mock(return_value=4242)

    def guard(*_args, **_kwargs):
        raise kb.SpawnAdmissionDeferred("admission-unavailable: reserve threshold reached")

    monkeypatch.setattr(kb, "_load_configured_spawn_guard", lambda: guard)
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="defer safely", assignee="alice")
        result = kb.dispatch_once(conn, spawn_fn=native_spawn)
        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id) if hasattr(kb, "latest_run") else None
    finally:
        conn.close()

    native_spawn.assert_not_called()
    assert result.spawned == []
    assert task is not None
    assert task.status == "ready"
    assert task.claim_lock is None
    assert task.worker_pid is None
    assert task.current_run_id is None
    assert task.consecutive_failures == 0


def test_repeated_deferrals_never_auto_block_the_task(kanban_home, all_assignees_spawnable, monkeypatch):
    """A provider that stays unavailable across MANY ticks must still never
    trip the auto-block circuit breaker — only genuine task failures may.
    """
    native_spawn = Mock(return_value=4242)

    def guard(*_args, **_kwargs):
        raise kb.SpawnAdmissionDeferred("admission-unavailable: provider snapshot is stale")

    monkeypatch.setattr(kb, "_load_configured_spawn_guard", lambda: guard)
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="stays healthy", assignee="alice")
        for _ in range(10):
            kb.dispatch_once(conn, spawn_fn=native_spawn)
        task = kb.get_task(conn, task_id)
    finally:
        conn.close()

    native_spawn.assert_not_called()
    assert task.status == "ready"
    assert task.consecutive_failures == 0


def test_genuine_spawn_error_still_counts_toward_the_breaker(kanban_home, all_assignees_spawnable, monkeypatch):
    """Control: a real (non-deferred) SpawnAdmissionError must still count —
    this fix must not accidentally silence real admission failures.
    """
    native_spawn = Mock(return_value=4242)

    def guard(*_args, **_kwargs):
        raise kb.SpawnAdmissionError("task requires an explicit model and provider pin")

    monkeypatch.setattr(kb, "_load_configured_spawn_guard", lambda: guard)
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="genuinely broken pin", assignee="alice")
        kb.dispatch_once(conn, spawn_fn=native_spawn)
        task = kb.get_task(conn, task_id)
    finally:
        conn.close()

    native_spawn.assert_not_called()
    assert task.consecutive_failures == 1

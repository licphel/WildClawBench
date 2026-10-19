"""A checker returning zero after a judge failure is an invalid evaluation."""
import json
from types import SimpleNamespace

from src.utils import grading


def prepare(monkeypatch, audits):
    scopes = iter(['scope-1', 'scope-2', 'scope-3'])
    monkeypatch.setattr(grading, '_judge_relay_base_url', lambda: next(scopes))
    monkeypatch.setattr(grading, '_relay_audit', lambda scope: audits[scope])
    monkeypatch.setattr(grading, '_grader_interpreter', lambda task: 'python')
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout='{"overall_score": 0.0}', stderr='')
    monkeypatch.setattr(grading.subprocess, 'run', run)
    return calls


def test_caught_capability_failure_is_not_a_valid_zero(tmp_path, monkeypatch):
    calls = prepare(monkeypatch, {'scope-1': [{'route': 'web_search', 'status': 502,
                                              'error': 'web search upstream is not configured'}]})
    scores = grading.run_grading('task', 'def grade(**kwargs): return {}', tmp_path,
                                write_error_score=True)
    assert scores['grading_status'] == 'infrastructure_error'
    assert not any(isinstance(value, (int, float)) for value in scores.values())
    assert json.loads((tmp_path / 'score.json').read_text()) == scores
    assert (tmp_path / 'grading_stdout.txt').read_text() == '{"overall_score": 0.0}'
    assert len([call for call in calls if call[1] == 'exec']) == 1


def test_successful_judge_can_still_score_zero(tmp_path, monkeypatch):
    prepare(monkeypatch, {'scope-1': [{'route': 'web_search', 'status': 200}]})
    assert grading.run_grading('task', '', tmp_path) == {'overall_score': 0.0}


def test_transient_judge_failure_retries_with_clean_scope(tmp_path, monkeypatch):
    calls = prepare(monkeypatch, {
        'scope-1': [{'route': 'web_search', 'status': 503, 'error': 'upstream_http_error'}],
        'scope-2': [{'route': 'web_search', 'status': 200}],
    })
    assert grading.run_grading('task', '', tmp_path) == {'overall_score': 0.0}
    exec_calls = [call for call in calls if call[1] == 'exec']
    assert len(exec_calls) == 2
    assert 'OPENROUTER_BASE_URL=scope-1' in exec_calls[0]
    assert 'OPENROUTER_BASE_URL=scope-2' in exec_calls[1]
    assert (tmp_path / 'judge_relay_audit_attempt_1.json').exists()
    assert (tmp_path / 'judge_relay_audit_attempt_2.json').exists()


def test_permanent_upstream_rejection_is_invalid(tmp_path, monkeypatch):
    prepare(monkeypatch, {'scope-1': [{'route': 'web_search', 'status': 400,
                                     'error': 'upstream_http_error'}]})
    assert grading.run_grading('task', '', tmp_path)['grading_status'] == 'infrastructure_error'


def test_unfinished_request_cannot_become_valid_zero(tmp_path, monkeypatch):
    prepare(monkeypatch, {scope: [{'route': 'web_search', 'status': None}]
                          for scope in ['scope-1', 'scope-2', 'scope-3']})
    assert grading.run_grading('task', '', tmp_path)['grading_status'] == 'infrastructure_error'

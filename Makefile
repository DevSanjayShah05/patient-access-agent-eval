PY := .venv/bin/python
export PYTHONPATH := src:.

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
eval:        ## baseline + guardrails v1 + v2, 100 scenarios each (~1.5-2.5 h each on an M2; resumable). Needs `ollama pull qwen2.5:7b`
	./scripts/run_evals.sh
evaluate:
	$(PY) -m front_desk.cli evaluate
test:
	$(PY) -m pytest -q
api:
	.venv/bin/uvicorn api.main:app --reload --port 8002
dashboard:
	.venv/bin/streamlit run dashboard/app.py
.PHONY: setup eval evaluate test api dashboard

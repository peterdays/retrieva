local-api:
	uvicorn api.app:app --host 0.0.0.0 --port 3333 --reload

local-dash:
	python3.10 -m webapp.main

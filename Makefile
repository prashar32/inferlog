# InferLog — common tasks. `make up` is all you need to run the demo.

.PHONY: up down clean logs ps test seed

up:                       ## build + start the whole stack
	docker compose up -d --build
	@echo ""
	@echo "  InferLog is up → http://localhost:8088"
	@echo "  (give the services a few seconds to finish booting)"

down:                     ## stop the stack
	docker compose down

clean:                    ## stop and wipe the database volume
	docker compose down -v

logs:                     ## tail logs from every service
	docker compose logs -f

ps:                       ## show container status
	docker compose ps

test:                     ## run SDK + gateway + ingestion test suites
	docker compose up -d --wait postgres redis
	@echo "--- inferlog SDK ---"
	docker compose run --rm --no-deps -T gateway pytest -q /tmp/sdk/tests
	@echo "--- gateway ---"
	docker compose run --rm --no-deps -T gateway pytest -q
	@echo "--- ingestion ---"
	docker compose run --rm --no-deps -T ingestion-api pytest -q

seed:                     ## push synthetic inference logs so the dashboard has data
	python3 scripts/seed.py

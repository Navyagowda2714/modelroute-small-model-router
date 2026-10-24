"""Use the router from Python: train (or load) the small model and route a few tickets.

    python examples/route_tickets.py
"""

from __future__ import annotations

from modelroute.config import get_settings
from modelroute.evaluation.benchmark import ensure_dataset, train_small
from modelroute.evaluation.pricing import load_prices
from modelroute.llm import get_llm_client
from modelroute.models import LLMClassifier, SmallClassifier, keyword_handler
from modelroute.routing import Router, RouterConfig

settings = get_settings()
if settings.small_model_path.exists():
    small = SmallClassifier.load(settings.small_model_path)
else:
    small = train_small(settings, ensure_dataset(settings)["train"])

# LLM_PROVIDER=anthropic|openai|ollama selects a real large model; default is offline.
large = LLMClassifier(get_llm_client(offline_handler=keyword_handler))
router = Router(small, large, load_prices(settings.prices_path), RouterConfig(threshold=0.8))

for text in [
    "my parcel is stuck in transit for a week",
    "I don't want to cancel anything, I just need to put my new card on the account",
    "Our accountant needs the billing documents for last quarter",
]:
    d = router.route(text)
    print(f"{d.route:5} {d.prediction.intent:24} conf={d.small_prediction.confidence:.2f} "
          f"reasons={d.reasons}")

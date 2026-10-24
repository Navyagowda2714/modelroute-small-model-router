"""``modelroute`` command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import get_settings
from .logging_utils import configure_logging

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="ModelRoute: smaller models, bigger impact.")

DEMO_TICKETS = [
    "Hi, I was charged twice for my Pro subscription this month, $49.00 two times.",
    "cant login, the page just spins forever",
    "I'm not trying to cancel my subscription, the Team plan is just too expensive for us now, "
    "is there something smaller we can keep?",
    "Where is my order? It should have been here by Monday. This is urgent.",
    "The headphones arrived with a cracked case. Also, can you send an invoice for the order?",
    "It would be great if you added dark mode. No rush on this.",
]


@app.callback()
def _main(log_level: str = typer.Option("WARNING", help="Log level for JSON logs.")) -> None:
    configure_logging(log_level)


@app.command()
def data(n: int = typer.Option(None, help="Number of tickets to generate."),
         seed: int = typer.Option(None)) -> None:
    """Generate, clean and split the synthetic ticket dataset."""
    from .data.preprocess import build_dataset

    s = get_settings()
    stats = build_dataset(s.data_dir, n=n or s.n_tickets, seed=seed or s.seed)
    typer.echo(json.dumps({"cleaning": stats["cleaning"],
                           "rows": {k: v["rows"] for k, v in stats["splits"].items()}},  # type: ignore[union-attr]
                          indent=2))


@app.command()
def train() -> None:
    """Train the small TF-IDF model (generates data first if needed)."""
    from .evaluation.benchmark import ensure_dataset, train_small

    s = get_settings()
    splits = ensure_dataset(s)
    train_small(s, splits["train"])
    typer.echo(f"saved {s.small_model_path}")


@app.command("eval")
def evaluate(docs_dir: Path = typer.Option(Path("docs"))) -> None:
    """Benchmark small-only vs large-only vs routed; write charts + JSON to docs/."""
    from .evaluation.benchmark import markdown_table, run_benchmark

    report = run_benchmark(get_settings(), docs_dir=docs_dir)
    typer.echo(markdown_table(report["results"]))
    typer.echo(f"\ntuned threshold: {report['tuned_threshold']:.2f} "
               f"({report['tuning_rule']})")
    typer.echo(f"small-model ECE (test): {report['small_model_ece']:.3f}")
    sv = report["savings_vs_large"]
    typer.echo(f"tuned router vs large-only: cost -{sv['cost_reduction']:.1%}, "
               f"p50 latency -{sv['p50_latency_reduction']:.1%}, "
               f"macro-F1 {sv['macro_f1_delta']:+.3f}")
    typer.echo(f"wrote {docs_dir / 'results' / 'eval_results.json'} and charts in "
               f"{docs_dir / 'images'}")


@app.command()
def classify(text: str, force: str = typer.Option(None, help="small|large")) -> None:
    """Route a single ticket and print the decision as JSON."""
    from .serving.app import build_router

    decision = build_router(get_settings()).route(text, force=force)  # type: ignore[arg-type]
    typer.echo(decision.model_dump_json(indent=2, exclude={"small_prediction"}))


@app.command()
def demo() -> None:
    """Train if needed, then route a handful of example tickets."""
    from .evaluation.benchmark import ensure_dataset, train_small
    from .serving.app import build_router

    s = get_settings()
    if not s.small_model_path.exists():
        typer.echo("training small model (first run)...")
        train_small(s, ensure_dataset(s)["train"])
    router = build_router(s)
    typer.echo(f"router: small={router.small.name} large={router.large.name} "
               f"threshold={router.config.threshold:.2f}\n")
    for text in DEMO_TICKETS:
        d = router.route(text)
        p = d.prediction
        typer.echo(f"> {text}")
        typer.echo(f"  route={d.route:<5} intent={p.intent} priority={p.priority} "
                   f"small_conf={d.small_prediction.confidence:.2f} "
                   f"reasons={','.join(d.reasons) or '-'} cost=${d.cost_usd:.6f}\n")


@app.command("export-sft")
def export_sft_cmd() -> None:
    """Write chat-formatted SFT files for LoRA fine-tuning."""
    from .evaluation.benchmark import ensure_dataset
    from .finetune.formatting import export_sft

    s = get_settings()
    splits = ensure_dataset(s)
    for name in ("train", "val"):
        path = s.data_dir / f"sft_{name}.jsonl"
        typer.echo(f"{path}: {export_sft(splits[name], path)} examples")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI service."""
    import uvicorn

    uvicorn.run("modelroute.serving.app:create_app", factory=True, host=host, port=port)


if __name__ == "__main__":
    app()

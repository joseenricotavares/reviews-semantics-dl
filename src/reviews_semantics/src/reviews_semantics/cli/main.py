from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="reviews-semantics: serve, predict-batch, and train the Olist review-score baseline.")


@app.command("predict-batch")
def predict_batch_command(
    input: Path = typer.Option(..., "--input", help="CSV file with a review_comment_message column."),
    output: Path = typer.Option(..., "--output", help="Where to write the scored CSV."),
    text_column: str = typer.Option("review_comment_message", "--text-column"),
) -> None:
    """Scores every row of a CSV file - the offline batch-execution case."""
    from reviews_semantics.cli.batch import predict_csv

    count = predict_csv(input, output, text_column=text_column)
    typer.echo(f"Scored {count} rows -> {output}")


@app.command("train-baseline")
def train_baseline_command(
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    output_dir: Path = typer.Option(Path("models/bilstm-baseline"), "--output-dir"),
    epochs: int = typer.Option(20, "--epochs"),
    learning_rate: float = typer.Option(1e-3, "--learning-rate"),
    mlflow: bool = typer.Option(True, "--mlflow/--no-mlflow"),
) -> None:
    """Trains the BiLSTM baseline from scratch - the training-container entrypoint."""
    from reviews_semantics.cli.train_baseline import run_train_baseline

    run_train_baseline(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        enable_mlflow=mlflow,
    )
    typer.echo(f"Trained baseline written to {output_dir}")


@app.command("serve")
def serve_command(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Runs the FastAPI prediction service - the online API case."""
    import uvicorn

    uvicorn.run("reviews_semantics.api.app:app", host=host, port=port)


if __name__ == "__main__":
    app()

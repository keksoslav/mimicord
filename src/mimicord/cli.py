import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Build Discord bots that talk like a real person."""


@app.command()
def version() -> None:
    """Print the mimicord version."""
    from mimicord import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()

"""
Interactive command-line chat over your ingested documents.

Usage:
    python ingest.py      # once, or whenever docs change
    python cli.py          # start asking questions
"""
from rich.console import Console
from rich.markdown import Markdown

from rag_pipeline import RAGPipeline

console = Console()


def main():
    console.print("[bold cyan]RAG CLI[/bold cyan] — loading index...")
    try:
        pipeline = RAGPipeline()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return
    except EnvironmentError as e:
        console.print(f"[red]{e}[/red]")
        return

    console.print(f"[green]Ready.[/green] Index has {len(pipeline.store)} chunks. "
                   f"Type your question, or 'exit' to quit.\n")

    while True:
        try:
            query = console.input("[bold yellow]You:[/bold yellow] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        #result = pipeline.ask(query, verbose=True)
        #console.print("[bold cyan]Assistant:[/bold cyan]")
        #console.print(Markdown(result["answer"]))

        #sources = sorted({c["source"] for c in result["chunks"]})
        #if sources:
        #   console.print(f"[dim]Sources: {', '.join(sources)}[/dim]\n")
        #else:
        #   console.print("[dim]No sources retrieved.[/dim]\n")


        result = pipeline.ask(query, verbose=False)
        console.print("[bold cyan]Assistant:[/bold cyan]")
        console.print(Markdown(result["answer"]))
        console.print()

if __name__ == "__main__":
    main()

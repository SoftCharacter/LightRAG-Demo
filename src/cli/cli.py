"""
Command Line Interface (CLI) Module
Provides command-line access to the RAG system
"""

import argparse
import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config_loader import load_config
from src.core.rag_engine import RAGEngine


console = Console()


async def build_knowledge_graph(args):
    """Build knowledge graph from documents"""
    console.print(Panel.fit("📚 Building Knowledge Graph", style="bold blue"))
    
    # Load configuration
    config = load_config(args.config)
    console.print(f"✓ Configuration loaded from: {args.config}", style="green")
    
    # Initialize RAG engine
    engine = RAGEngine(config)
    await engine.initialize()
    console.print("✓ RAG Engine initialized", style="green")
    
    # Process documents with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing documents...", total=None)
        
        def update_progress(ratio, desc):
            progress.update(task, description=desc)
        
        stats = await engine.process_documents(args.docs, update_progress)
    
    # Display statistics
    table = Table(title="Knowledge Graph Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Processed Files", str(stats.get("processed_files", 0)))
    table.add_row("Failed Files", str(stats.get("failed_files", 0)))
    table.add_row("Total Entities", str(stats.get("total_entities", 0)))
    table.add_row("Total Relationships", str(stats.get("total_relationships", 0)))
    
    console.print(table)
    console.print("✓ Knowledge graph built successfully!", style="bold green")
    
    await engine.cleanup()


async def query_knowledge_graph(args):
    """Query the knowledge graph"""
    console.print(Panel.fit("🔍 Querying Knowledge Graph", style="bold blue"))
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize RAG engine
    engine = RAGEngine(config)
    await engine.initialize()
    console.print("✓ RAG Engine initialized", style="green")
    
    # Execute query
    console.print(f"\n[bold]Question:[/bold] {args.query}")
    console.print(f"[dim]Query Mode:[/dim] {args.device}\n")
    
    with console.status("[bold green]Searching knowledge graph..."):
        result = await engine.query(args.query, mode=args.device)
    
    # Display answer
    console.print(Panel(
        result["answer"],
        title="Answer",
        border_style="green"
    ))
    
    await engine.cleanup()


async def interactive_mode(args):
    """Interactive question-answering mode"""
    console.print(Panel.fit("💬 Interactive QA Mode", style="bold blue"))
    console.print("[dim]Type 'exit' or 'quit' to stop, 'stats' for statistics[/dim]\n")
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize RAG engine
    engine = RAGEngine(config)
    await engine.initialize()
    console.print("✓ RAG Engine initialized\n", style="green")
    
    while True:
        try:
            # Get user input
            question = console.input("[bold cyan]❓ Your question:[/bold cyan] ")
            
            if question.lower() in ['exit', 'quit', 'q']:
                console.print("Goodbye! 👋", style="yellow")
                break
            
            if question.lower() == 'stats':
                stats = await engine.get_statistics()
                table = Table(title="Knowledge Graph Statistics")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")
                for key, value in stats.items():
                    table.add_row(key, str(value))
                console.print(table)
                continue
            
            if not question.strip():
                continue
            
            # Query
            with console.status("[bold green]Searching..."):
                result = await engine.query(question, mode=args.device)
            
            # Display answer
            console.print(Panel(
                result["answer"],
                title="Answer",
                border_style="green"
            ))
            console.print()
            
        except KeyboardInterrupt:
            console.print("\nGoodbye! 👋", style="yellow")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
    
    await engine.cleanup()


async def export_graph(args):
    """Export knowledge graph"""
    console.print(Panel.fit("💾 Exporting Knowledge Graph", style="bold blue"))
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize RAG engine
    engine = RAGEngine(config)
    await engine.initialize()
    
    # Export
    console.print(f"Exporting to {args.format} format...")
    data = await engine.export_graph(format=args.format)
    
    # Save to file
    output_file = args.output or f"knowledge_graph.{args.format}"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(data)
    
    console.print(f"✓ Exported to: {output_file}", style="green")
    
    await engine.cleanup()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="LightRAG Knowledge Graph QA System - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build knowledge graph from documents")
    build_parser.add_argument("--docs", required=True, help="Path to documents folder")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge graph")
    query_parser.add_argument("--query", required=True, help="Question to ask")
    query_parser.add_argument("--device", default=None, choices=["naive", "local", "global", "hybrid"],
                             help="Query mode (default: from config)")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive QA mode")
    interactive_parser.add_argument("--mode", default=None, choices=["naive", "local", "global", "hybrid"],
                                   help="Query mode (default: from config)")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export knowledge graph")
    export_parser.add_argument("--format", default="json", choices=["json", "graphml"],
                              help="Export format (default: json)")
    export_parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Run async command
    try:
        if args.command == "build":
            asyncio.run(build_knowledge_graph(args))
        elif args.command == "query":
            asyncio.run(query_knowledge_graph(args))
        elif args.command == "interactive":
            asyncio.run(interactive_mode(args))
        elif args.command == "export":
            asyncio.run(export_graph(args))
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()


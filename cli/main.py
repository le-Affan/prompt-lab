import typer
import json
import dataclasses
from typing import Any, Optional
from contextlib import contextmanager
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree as RichTree
from rich.text import Text

from src.prompt_versioning import (
    VersionStore,
    create_version as lib_create_version,
    branch_version as lib_branch_version,
    rollback_version as lib_rollback_version,
    get_version as lib_get_version,
    get_children as lib_get_children,
    get_version_lineage as lib_get_version_lineage,
    get_branch_versions as lib_get_branch_versions,
    get_version_tree as lib_get_version_tree,
    compute_char_diff,
    compute_line_diff,
)

app = typer.Typer(name="brain-cli", help="Project Brain CLI")
console = Console()

class CLISession:
    """Holds shared context for the CLI session."""
    def __init__(self, format_type: str = "pretty"):
        self.store = VersionStore()
        self.format_type = format_type
        # You could also add a mechanism here to hydrate self.store from disk.

@app.callback()
def main(
    ctx: typer.Context,
    format: str = typer.Option("pretty", help="Output format: pretty or json")
):
    """
    Callback that runs before any command.
    It initializes the CLISession and injects it into the Typer context.
    """
    ctx.obj = CLISession(format_type=format)

version_app = typer.Typer(help="Manage prompt versions")
app.add_typer(version_app, name="version")

def parse_metadata(meta_list: Optional[list[str]]) -> dict[str, str]:
    if not meta_list:
        return {}
    metadata = {}
    for item in meta_list:
        if "=" in item:
            k, v = item.split("=", 1)
            metadata[k] = v
        else:
            metadata[item] = "true"
    return metadata

@contextmanager
def handle_cli_errors():
    try:
        yield
    except KeyError as e:
        # e.args[0] usually contains the string message "version_id '...' not found..."
        err_msg = e.args[0] if e.args else str(e)
        console.print(f"[bold red]Error: Resource not found -[/bold red] {err_msg}")
        raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"[bold red]Error: Invalid argument -[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        raise typer.Exit(code=1)

# --- Formatting Helpers ---

def format_version(version: Any, format_type: str):
    if format_type == "json":
        print(json.dumps(dataclasses.asdict(version), default=str, indent=2))
    else:
        content_text = Text(version.content)
        panel = Panel(
            content_text, 
            title=f"Version: [bold cyan]{version.id}[/bold cyan]", 
            subtitle=f"(Branch: [bold magenta]{version.branch_name}[/bold magenta]) Message: {version.commit_message}"
        )
        console.print(panel)

def format_versions_list(versions: list[Any], format_type: str):
    if format_type == "json":
        print(json.dumps([dataclasses.asdict(v) for v in versions], default=str, indent=2))
    else:
        if not versions:
            console.print("[dim]No versions found.[/dim]")
            return
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", width=36)
        table.add_column("Branch", style="green")
        table.add_column("Created At")
        table.add_column("Message")
        for v in versions:
            dt_str = v.created_at.strftime("%Y-%m-%d %H:%M:%S") if v.created_at else ""
            table.add_row(v.id, v.branch_name, dt_str, v.commit_message)
        console.print(table)

def dict_to_rich_tree(node_dict: dict, rich_tree: Optional[RichTree] = None) -> RichTree:
    label = f"[cyan]{node_dict['id'][:8]}[/cyan] ({node_dict['branch_name']}) - {node_dict['commit_message']}"
    if rich_tree is None:
        rich_tree = RichTree(label)
        curr_tree = rich_tree
    else:
        curr_tree = rich_tree.add(label)
    for child in node_dict.get('children', []):
        dict_to_rich_tree(child, curr_tree)
    return rich_tree

def format_tree(tree_data: list[dict], format_type: str):
    if format_type == "json":
        print(json.dumps(tree_data, default=str, indent=2))
    else:
        if not tree_data:
            console.print("[dim]No tree available.[/dim]")
            return
        for root in tree_data:
            t = dict_to_rich_tree(root)
            console.print(t)

def format_diff(diff_result: list[list[str]], format_type: str):
    if format_type == "json":
        print(json.dumps(diff_result, default=str, indent=2))
    else:
        text = Text()
        for op, val in diff_result:
            if op == "=":
                text.append(val, style="dim")
            elif op == "+":
                text.append(val, style="green")
            elif op == "-":
                text.append(val, style="red strike")
        console.print(text)

# --- Commands ---

@version_app.command("create")
def create_version(
    ctx: typer.Context,
    prompt_id: str = typer.Option(..., "--prompt-id", help="ID of prompt"),
    content: str = typer.Option(..., "--content", help="Raw text"),
    message: str = typer.Option(..., "--message", "-m", help="Commit message"),
    branch: str = typer.Option("main", "--branch", "-b", help="Branch name"),
    parent: Optional[str] = typer.Option(None, "--parent", help="Parent version ID"),
    meta: Optional[list[str]] = typer.Option(None, "--meta", help="Metadata in key=value format")
):
    """Create a new prompt version."""
    store = ctx.obj.store
    with handle_cli_errors():
        metadata = parse_metadata(meta)
        version = lib_create_version(
            prompt_id=prompt_id,
            content=content,
            commit_message=message,
            branch_name=branch,
            parent_version_id=parent,
            metadata=metadata,
            store=store,
        )
        format_version(version, ctx.obj.format_type)

@version_app.command("branch")
def branch_version(
    ctx: typer.Context,
    base_id: str = typer.Option(..., "--base-id", help="Version to fork from"),
    new_branch: str = typer.Option(..., "--new-branch", help="New branch name"),
    message: str = typer.Option(..., "--message", "-m", help="Commit message"),
    meta: Optional[list[str]] = typer.Option(None, "--meta", help="Metadata in key=value format")
):
    """Create a new branch from an existing version."""
    store = ctx.obj.store
    with handle_cli_errors():
        metadata = parse_metadata(meta)
        version = lib_branch_version(
            base_version_id=base_id,
            new_branch_name=new_branch,
            commit_message=message,
            metadata=metadata,
            store=store,
        )
        format_version(version, ctx.obj.format_type)

@version_app.command("rollback")
def rollback_version(
    ctx: typer.Context,
    target_id: str = typer.Option(..., "--target-id", help="Version ID to clone content from"),
    message: str = typer.Option(..., "--message", "-m", help="Commit message"),
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Target branch override")
):
    """Rollback to a previous version."""
    store = ctx.obj.store
    with handle_cli_errors():
        version = lib_rollback_version(
            target_version_id=target_id,
            commit_message=message,
            branch_name=branch,
            store=store,
        )
        format_version(version, ctx.obj.format_type)


@version_app.command("get")
def get_version(
    ctx: typer.Context,
    version_id: str = typer.Argument(..., help="Version ID to retrieve")
):
    """Get details of a specific version."""
    store = ctx.obj.store
    with handle_cli_errors():
        version = lib_get_version(version_id=version_id, store=store)
        format_version(version, ctx.obj.format_type)

@version_app.command("list")
def list_versions(
    ctx: typer.Context,
    prompt_id: str = typer.Option(..., "--prompt-id", help="ID of prompt"),
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Filter by branch name")
):
    """List all versions for a prompt."""
    store = ctx.obj.store
    with handle_cli_errors():
        if branch is not None:
            versions = lib_get_branch_versions(branch_name=branch, prompt_id=prompt_id, store=store)
        else:
            versions = [v for v in store._versions.values() if v.prompt_id == prompt_id]
            versions.sort(key=lambda v: v.created_at)
        
        format_versions_list(versions, ctx.obj.format_type)

@version_app.command("children")
def children_versions(
    ctx: typer.Context,
    version_id: str = typer.Argument(..., help="Parent version ID")
):
    """List direct children of a version."""
    store = ctx.obj.store
    with handle_cli_errors():
        children = lib_get_children(version_id=version_id, store=store)
        format_versions_list(children, ctx.obj.format_type)

@version_app.command("lineage")
def lineage_versions(
    ctx: typer.Context,
    version_id: str = typer.Argument(..., help="Target version ID")
):
    """Show the lineage path of a version from root."""
    store = ctx.obj.store
    with handle_cli_errors():
        lineage = lib_get_version_lineage(version_id=version_id, store=store)
        format_versions_list(lineage, ctx.obj.format_type)

@version_app.command("tree")
def get_version_tree(
    ctx: typer.Context,
    prompt_id: str = typer.Option(..., "--prompt-id", help="ID of prompt")
):
    """Show the version tree for a prompt."""
    store = ctx.obj.store
    with handle_cli_errors():
        tree = lib_get_version_tree(prompt_id=prompt_id, store=store)
        format_tree(tree, ctx.obj.format_type)

@version_app.command("diff")
def diff_versions(
    ctx: typer.Context,
    old: str = typer.Option(..., "--old", help="Previous prompt content"),
    new: str = typer.Option(..., "--new", help="New prompt content"),
    mode: str = typer.Option("line", "--mode", help="Diff mode: char or line")
):
    """Compute the diff between two texts."""
    store = ctx.obj.store
    with handle_cli_errors():
        if mode == "char":
            diff_result = compute_char_diff(old=old, new=new)
        elif mode == "line":
            diff_result = compute_line_diff(old=old, new=new)
        else:
            raise ValueError(f"Invalid mode: '{mode}'. Must be 'char' or 'line'.")
        
        format_diff(diff_result, ctx.obj.format_type)

if __name__ == "__main__":
    app()


import rich
import click
from mttl.models.modifiers.expert_containers.expert_library import ExpertLibrary
from rich.console import Console
from rich.table import Table


@click.group()
def cli():
    pass


@cli.command("le")
@click.argument("name")
def list_experts(name):
    """
    Analyze the library with the given name.
    """
    library = ExpertLibrary.get_expert_library(name)
    table = Table(title="Expert Name")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Expert", justify="left", style="magenta", no_wrap=True)
    for i, expert_name in enumerate(library.keys()):
        table.add_row(str(i + 1), expert_name)

    console = Console()
    console.print(table)


@cli.command("laux")
@click.argument("name")
def list_aux(name):
    console = Console()
    console.print(table)

    library = ExpertLibrary.get_expert_library(name)
    table = Table(title="Auxiliary Data")
    table.add_column("Hash", justify="left", style="cyan", no_wrap=True)
    table.add_column("# Elements", justify="left", style="magenta", no_wrap=True)
    table.add_column("Config", justify="left", style="magenta", no_wrap=False)
    aux_data = library.list_auxiliary_data()
    for hash, (count, config) in aux_data.items():
        table.add_row(hash, str(count), config)
    console.print(table)


if __name__ == "__main__":
    cli()

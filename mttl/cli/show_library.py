"""
A basic CLI for displaying contents of a library of experts.

"""

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
    library = ExpertLibrary.get_expert_library(name)
    table = Table(title="Auxiliary Data")
    table.add_column("Data type", justify="left", style="cyan", no_wrap=True)
    table.add_column("# Elements", justify="left", style="magenta", no_wrap=True)
    table.add_column("Config", justify="left", style="magenta", no_wrap=False)
    aux_data = library.list_auxiliary_data()
    for hash, (count, config) in aux_data.items():
        table.add_row(hash, str(count), config)

    console = Console()
    console.print(table)


@cli.command("raux")
@click.argument("name")
@click.argument("data_type")
def remove_aux(name, data_type):
    library = ExpertLibrary.get_expert_library(name)
    library.remove_auxiliary_data(data_type)


@cli.command("diff")
@click.argument("names", nargs=2)
def list_experts(names):
    """
    Compare the experts of two libraries. Useful to check if experts are missing from one.
    """
    name_a, name_b = names
    library_a = ExpertLibrary.get_expert_library(name_a)
    library_b = ExpertLibrary.get_expert_library(name_b)

    def init_table(name):
        table = Table(title=name)
        table.add_column("Expert", justify="left", style="magenta", no_wrap=True)
        return table

    table_a = init_table(name_a)
    table_b = init_table(name_b)
    common = init_table("Common Experts")

    all_experts = set(library_a.keys()) | set(library_b.keys())

    for i, expert_name in enumerate(list(all_experts)):
        if expert_name in library_a.keys() and expert_name in library_b.keys():
            common.add_row(expert_name)
        elif expert_name in library_a.keys():
            table_a.add_row(expert_name)
        else:
            table_b.add_row(expert_name)

    console = Console()
    console.print(common)
    if len(table_a.rows) > 0:
        console.print(table_a)
    if len(table_b.rows) > 0:
        console.print(table_b)


if __name__ == "__main__":
    cli()

"""Rich/tqdm progress helpers."""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.table import Table


@dataclass
class ProgressSettings:
    mode: str = "both"
    quiet: bool = False
    no_color: bool = False

    @property
    def use_tqdm(self) -> bool:
        return not self.quiet and self.mode in {"tqdm", "both"}

    @property
    def use_rich(self) -> bool:
        return not self.quiet and self.mode in {"rich", "both"}


class RichRunReporter:
    def __init__(self, settings: ProgressSettings):
        self.settings = settings
        self.console = Console(no_color=settings.no_color, quiet=settings.quiet)

    def status(self, message: str) -> None:
        if self.settings.use_rich:
            self.console.print(f"[bold cyan]{message}[/bold cyan]")

    def summary_table(self, title: str, rows: list[dict[str, object]]) -> None:
        if not self.settings.use_rich:
            return
        table = Table(title=title)
        for column in ["method", "target_score", "fixed_pool_regret", "source_calls", "target_calls", "total_dollars"]:
            table.add_column(column)
        for row in rows:
            table.add_row(
                str(row.get("method", "")),
                f"{float(row.get('target_score', 0.0)):.3f}",
                f"{float(row.get('fixed_pool_regret', 0.0)):.3f}",
                str(row.get("source_calls", 0)),
                str(row.get("target_calls", 0)),
                f"{float(row.get('total_dollars', 0.0)):.6f}",
            )
        self.console.print(table)

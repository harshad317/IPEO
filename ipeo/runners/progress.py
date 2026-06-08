"""Rich/tqdm progress helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel
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

    def method_summary_panels(self, rows: list[dict[str, Any]]) -> None:
        if self.settings.quiet or self.settings.mode == "off" or not rows:
            return
        split_order = {"train": 0, "val": 1, "test": 2, "optimization": 3}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["method"]), []).append(row)
        for method in sorted(grouped):
            table = Table(show_header=True)
            table.add_column("Split")
            table.add_column("Score", justify="right")
            table.add_column("StdDev", justify="right")
            table.add_column("AvgTok", justify="right")
            table.add_column("API calls", justify="right")
            for row in sorted(grouped[method], key=lambda item: split_order.get(str(item["split"]), 99)):
                table.add_row(
                    str(row["split"]),
                    self._format_float(row.get("score")),
                    self._format_float(row.get("stddev")),
                    self._format_float(row.get("avg_tokens"), digits=1),
                    str(int(row.get("api_calls", 0) or 0)),
                )
            self.console.print(Panel(table, title=method, expand=False))

    @staticmethod
    def _format_float(value: object, digits: int = 3) -> str:
        if value is None:
            return "—"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "—"

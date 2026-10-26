from __future__ import annotations

import time
from dataclasses import dataclass

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

console = Console()


@dataclass
class ConsoleColors:
    blue = "#0096FF"
    green = "#00FF00"
    red = "#FF0000"
    yellow = "#FFFF00"
    cyan = "#00FFFF"


class TimeStampColumn(ProgressColumn):
    def render(self, task) -> Text:
        timestamp = console.get_datetime().strftime("%X")
        return Text(f"[{timestamp}]", style="log.time")


class SpinnerDescriptionColumn(ProgressColumn):
    _frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def render(self, task) -> Text:
        if task.finished:
            return Text.from_markup(f"[bold {ConsoleColors.green}]{task.description}")
        frame_index = int(time.time() * 10) % len(self._frames)
        frame = self._frames[frame_index]
        return Text.from_markup(
            f"{frame} [bold {ConsoleColors.cyan}]{task.description}"
        )


def get_progress() -> Progress:
    return Progress(
        TimeStampColumn(),
        SpinnerDescriptionColumn(),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

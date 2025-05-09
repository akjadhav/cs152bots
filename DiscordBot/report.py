# report.py
import enum
import time
from dataclasses import dataclass, field
from typing import Optional, List


class Violation(enum.Enum):
    SEXUAL_EXPLOITATION = "Sexual exploitation"
    GROOMING            = "Grooming"
    SEXUAL_HARASSMENT   = "Sexual harassment"


class Priority(enum.IntEnum):
    EXTREME_URGENT = 0
    URGENT         = 1
    NORMAL         = 2


class ModOutcome(enum.Enum):
    NO_VIOLATION   = "No violation"
    REMOVE_MESSAGE = "Remove message"
    WARN_USER      = "Warn user"
    SUSPEND_USER   = "Suspend / Ban user"
    ESCALATE_LE    = "Escalate to law-enforcement"


@dataclass
class Report:
    # core
    reporter_id: int
    guild_id: int
    channel_id: int
    message_id: int
    target_user_id: int
    reason: Violation

    # additions from the new user-flow
    subcategory: Optional[str]        = None
    evidence_text: str                = ""
    attachment_urls: List[str]        = field(default_factory=list)
    reporter_wants_block: bool        = False

    created_at: float                 = field(default_factory=time.time)
    priority: Priority                = field(init=False)
    outcome: Optional[ModOutcome]     = None
    resolved_by: Optional[int]        = None  # moderator’s ID

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.priority = {
            Violation.GROOMING:            Priority.EXTREME_URGENT,
            Violation.SEXUAL_EXPLOITATION: Priority.EXTREME_URGENT,
            Violation.SEXUAL_HARASSMENT:   Priority.URGENT,
        }[self.reason]

    @property
    def is_open(self) -> bool:
        return self.outcome is None

    def close(self, outcome: ModOutcome, moderator_id: int) -> None:
        self.outcome     = outcome
        self.resolved_by = moderator_id

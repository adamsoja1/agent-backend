
from dataclasses import dataclass, field


@dataclass
class Conversation:
    id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    system_prompt: str = ''
    summarized_history: str = "" #to be added later on
    
    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict[str, str]]:
        return self.messages
    
    def clear(self) -> None:
        self.messages.clear()

    def _prepare_system_prompt(self) -> dict[str, str]:
        return {"role": "system", "content": self.system_prompt}

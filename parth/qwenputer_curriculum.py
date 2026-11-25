"""Qwenputer-inspired executable CPU traces used as curriculum data.

The leaderboard requires a stock Hugging Face architecture, so Parth does not use
Qwenputer's nonstandard state-as-RAM layer directly. Instead, this module generates
exact virtual-machine execution traces that teach the transformer the same load,
operate, store, and branch discipline in textual form.
"""

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Instruction:
    op: str
    args: tuple[int, ...]


class TinyCPU:
    def __init__(self, registers: int = 6):
        self.registers = [0] * registers
        self.memory = [0] * 8
        self.trace: list[str] = []

    def run(self, instructions: list[Instruction]) -> int:
        pc = 0
        steps = 0
        while 0 <= pc < len(instructions) and steps < 80:
            steps += 1
            ins = instructions[pc]
            op, args = ins.op, ins.args
            before = self.registers.copy()
            jump = pc + 1
            if op == "LOAD":
                r, value = args
                self.registers[r] = value
                detail = f"R{r}<-{value}"
            elif op == "LOADM":
                r, addr = args
                self.registers[r] = self.memory[addr]
                detail = f"R{r}<-M{addr}={self.registers[r]}"
            elif op == "STORE":
                addr, r = args
                self.memory[addr] = self.registers[r]
                detail = f"M{addr}<-R{r}={self.registers[r]}"
            elif op in {"ADD", "SUB", "MUL", "DIV", "MOD"}:
                dest, a, b = args
                x, y = self.registers[a], self.registers[b]
                if op == "ADD":
                    value = x + y
                elif op == "SUB":
                    value = x - y
                elif op == "MUL":
                    value = x * y
                elif op == "DIV":
                    value = x // y if y else 0
                else:
                    value = x % y if y else 0
                self.registers[dest] = value
                symbol = {"ADD":"+","SUB":"-","MUL":"*","DIV":"//","MOD":"%"}[op]
                detail = f"R{dest}<-{x}{symbol}{y}={value}"
            elif op == "JNZ":
                r, target = args
                taken = self.registers[r] != 0
                jump = target if taken else pc + 1
                detail = f"JNZ R{r} {'taken' if taken else 'not-taken'}"
            elif op == "HALT":
                detail = "HALT"
                self.trace.append(f"pc={pc} {op}{args} | {detail} | R={before}-> {self.registers.copy()} M={self.memory.copy()}")
                break
            else:
                raise ValueError(f"unknown op {op}")
            self.trace.append(f"pc={pc} {op}{args} | {detail} | R={before}-> {self.registers.copy()} M={self.memory.copy()}")
            pc = jump
        return self.registers[0]


def arithmetic_program(index: int) -> tuple[list[Instruction], int]:
    rng = random.Random(0xC0FFEE + index)
    a = rng.randint(12, 480)
    b = rng.randint(7, 240)
    c = rng.randint(2, 60)
    op = rng.choice(["ADD", "SUB", "MUL", "DIV", "MOD"])
    if op in {"DIV", "MOD"}:
        b = max(2, b % 20 + 2)
        if op == "DIV":
            a = b * rng.randint(3, 32)
    instructions = [
        Instruction("LOAD", (0, a)),
        Instruction("LOAD", (1, b)),
        Instruction(op, (0, 0, 1)),
        Instruction("STORE", (0, 0)),
        Instruction("LOAD", (2, c)),
        Instruction("ADD", (0, 0, 2)),
        Instruction("HALT", ()),
    ]
    first = {"ADD": a + b, "SUB": a - b, "MUL": a * b, "DIV": a // b, "MOD": a % b}[op]
    return instructions, first + c


def cpu_trace_document(index: int) -> str:
    instructions, result = arithmetic_program(index)
    cpu = TinyCPU()
    observed = cpu.run(instructions)
    assert observed == result
    listing = "\n".join(f"{pc}: {ins.op} {' '.join(map(str, ins.args))}".rstrip() for pc, ins in enumerate(instructions))
    trace = "\n".join(cpu.trace)
    return (
        "Execute this tiny CPU program exactly. Track registers and memory, then give R0.\n"
        f"Program:\n{listing}\n\nExecution trace:\n{trace}\n\nFinal answer: R0 = {result}"
    )


if __name__ == "__main__":
    print(cpu_trace_document(0))

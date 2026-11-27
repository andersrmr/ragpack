"""Run-scoped checklist state for the native Plan-and-Solve strategy."""

from collections import Counter
from dataclasses import asdict, dataclass, replace


REACT = "ReAct"
PLAN_AND_SOLVE = "Plan-and-Solve"
PLANNING_STRATEGIES = (REACT, PLAN_AND_SOLVE)


def render_checklist(tasks):
    return "\n".join(
        f"- [{'x' if task['status'] == 'done' else ' '}] {index}. {task['description']}"
        for index, task in enumerate(tasks, 1)
    )


@dataclass(frozen=True)
class PlanTask:
    task_id: int
    description: str
    status: str = "pending"


@dataclass(frozen=True)
class PlanState:
    revision: int = 0
    tasks: tuple[PlanTask, ...] = ()
    next_task_id: int = 1

    def snapshot(self):
        return {"revision": self.revision, "tasks": [asdict(task) for task in self.tasks],
                "pending_count": sum(task.status == "pending" for task in self.tasks)}


class NativePlanAndSolve:
    """Supply planning instructions and validated tools to the ReAct harness."""

    tool_names = frozenset({"update_plan", "mark_done"})
    instructions = (
        "## Plan-and-Solve\n\n"
        "For non-trivial multistep tasks, call update_plan to outline the work. "
        "Use permitted tools as needed and call mark_done with the current one-based "
        "position after completing a task. Revise the plan when observations change "
        "what work is needed, keeping retained descriptions stable. A completion mark "
        "records your declaration of progress. For trivial requests, answer directly. "
        "Each user task starts a fresh checklist. Plan observations from earlier "
        "conversation turns are historical."
    )

    def __init__(self, current_run):
        self._current_run = current_run

    def register(self, tools):
        tools.add_tool("update_plan", self.update_plan)
        tools.add_tool("mark_done", self.mark_done)

    def _run(self):
        run = self._current_run()
        if run is None or run.plan is None or run.outcome != "active":
            raise RuntimeError("Planning requires an active Plan-and-Solve step.")
        return run

    @staticmethod
    def _checklist(plan):
        return f"Plan revision {plan.revision}\n" + render_checklist(plan.snapshot()["tasks"])

    def update_plan(self, tasks: list[str]) -> str:
        """Create or revise the plan using an ordered list of non-empty task descriptions."""
        run = self._run()
        if not isinstance(tasks, list) or not tasks or any(
            not isinstance(task, str) or not task.strip() for task in tasks
        ):
            raise ValueError("tasks must be a non-empty list of non-blank strings.")
        descriptions = [" ".join(task.split()) for task in tasks]
        plan = run.plan
        if descriptions == [task.description for task in plan.tasks]:
            return self._checklist(plan)

        new_keys = [description.casefold() for description in descriptions]
        old_counts = Counter(task.description.casefold() for task in plan.tasks)
        new_counts = Counter(new_keys)
        retained = {task.description.casefold(): task for task in plan.tasks
                    if old_counts[task.description.casefold()] == 1}
        revised = []
        next_id = plan.next_task_id
        for description, key in zip(descriptions, new_keys):
            if key in retained and new_counts[key] == 1:
                revised.append(replace(retained[key], description=description))
            else:
                revised.append(PlanTask(next_id, description))
                next_id += 1
        run.plan = PlanState(plan.revision + 1, tuple(revised), next_id)
        return self._checklist(run.plan)

    def mark_done(self, index: int) -> str:
        """Declare the task at the given one-based checklist position done."""
        run = self._run()
        plan = run.plan
        if not plan.tasks:
            raise ValueError("No plan exists. Call update_plan first.")
        if type(index) is not int or not 1 <= index <= len(plan.tasks):
            raise ValueError(f"index must be a one-based task number from 1 to {len(plan.tasks)}.")
        task = plan.tasks[index - 1]
        if task.status != "done":
            tasks = plan.tasks[:index - 1] + (replace(task, status="done"),) + plan.tasks[index:]
            run.plan = replace(plan, revision=plan.revision + 1, tasks=tasks)
        return self._checklist(run.plan)

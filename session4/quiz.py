"""Explicit quiz lifecycle and deterministic Markdown artifact management."""

import json
import re
from datetime import datetime
from pathlib import Path


SKILL_PATH = Path(__file__).parent / "skills" / "quiz_me.md"


def _json_object(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("The model did not return a JSON quiz.")
    return json.loads(text[start:end + 1])


def _safe_topic(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return (slug[:48].strip("-") or "general")


class QuizManager:
    def __init__(self, artifact_dir: str | Path | None = None):
        self.artifact_dir = Path(artifact_dir or Path(__file__).parent / "quizzes")
        self.skill_text = SKILL_PATH.read_text(encoding="utf-8")
        self.quiz = None

    @property
    def active(self) -> bool:
        return self.quiz is not None

    def start(self, agent, topic: str = "") -> str:
        if self.active:
            return f"Quiz already active on **{self.quiz['topic']}**. Answer the current question or use `/quiz end`."
        topic = topic.strip() or "the Session 4 TinyAgent course material available in this conversation"
        if self.skill_text not in agent.system_instructions:
            agent.system_instructions.append(self.skill_text)
        prompt = (f"Create a 5-question interactive quiz about: {topic}. Return only a JSON object "
                  'with keys "topic" and "questions". Each question must have "question", '
                  '"answer", and "explanation" string fields. Use concise, unambiguous questions. '
                  "Do not include markdown fences.")
        try:
            response = agent.run(prompt, run_metadata={"kind": "quiz_generation", "topic": topic}, remember=False)
        except Exception:
            agent.system_instructions = [text for text in agent.system_instructions if text != self.skill_text]
            raise
        try:
            document = _json_object(response.content)
            if not isinstance(document, dict):
                raise ValueError("Quiz response must be a JSON object.")
            questions = document.get("questions")
            if not isinstance(questions, list) or not questions:
                raise ValueError("Quiz questions must be a non-empty list.")
            normalized = []
            for item in questions:
                if not isinstance(item, dict) or not all(
                    isinstance(item.get(key), str) and item[key].strip()
                    for key in ("question", "answer", "explanation")
                ):
                    raise ValueError("Every question needs question, answer, and explanation text.")
                normalized.append({key: item[key].strip() for key in ("question", "answer", "explanation")})
        except Exception as error:
            agent.system_instructions = [text for text in agent.system_instructions if text != self.skill_text]
            raise ValueError("Could not validate quiz questions. Try `/quiz <topic>` again.") from error
        now = datetime.now().astimezone()
        self.quiz = {"topic": str(document.get("topic") or topic), "questions": normalized,
                     "current": 0, "results": [], "created_at": now.isoformat(), "path": None}
        try:
            self._create_artifact()
        except Exception:
            self.quiz = None
            agent.system_instructions = [text for text in agent.system_instructions if text != self.skill_text]
            raise
        return self.current_question

    @property
    def current_question(self) -> str:
        if not self.active:
            return "No quiz is active. Use `/quiz <topic>` to start one."
        index = self.quiz["current"]
        if index >= len(self.quiz["questions"]):
            return "Quiz complete. Use `/quiz end` to return to ordinary tutoring."
        return f"**Question {index + 1} of {len(self.quiz['questions'])}:** {self.quiz['questions'][index]['question']}"

    def answer(self, agent, answer: str) -> str:
        if not self.active:
            raise ValueError("No quiz is active.")
        index = self.quiz["current"]
        if index >= len(self.quiz["questions"]):
            return self.current_question
        item = self.quiz["questions"][index]
        if self.skill_text not in agent.system_instructions:
            agent.system_instructions.append(self.skill_text)
        prompt = (f"The learner answered question {index + 1} of {len(self.quiz['questions'])}: "
                  f"{item['question']}\nExpected answer: {item['answer']}\nExplanation: {item['explanation']}\n"
                  f"Learner answer: {answer}\n"
                  "Return feedback on this answer only: say what is correct or missing and briefly explain why. "
                  "Do not ask a question, repeat or rephrase this question, introduce or preview another question, "
                  "or give an example phrased as a quiz question. Do not write a 'Question N' heading. "
                  "The quiz harness will present the next question after your feedback. Stop after the feedback.")
        response = agent.run(prompt, run_metadata={"kind": "quiz_answer", "topic": self.quiz["topic"], "question": index + 1})
        # Keep learner-visible conversation in Memory; the fuller grading
        # instruction remains available in the trajectory request snapshot.
        agent.memory.messages[-2]["content"] = answer
        self.quiz["results"].append({"answer": answer, "feedback": response.content})
        self.quiz["current"] += 1
        self._write_artifact()
        if self.quiz["current"] >= len(self.quiz["questions"]):
            score_prompt = ("Based only on this quiz interaction and the supplied quiz results, give a short "
                            "final summary of performance and review recommendations: "
                            + json.dumps(self.quiz["results"], ensure_ascii=False))
            final = agent.run(score_prompt, run_metadata={"kind": "quiz_summary", "topic": self.quiz["topic"]}, remember=False)
            self.quiz["summary"] = final.content
            self._write_artifact()
            result = response.content + "\n\n" + final.content + "\n\nQuiz complete. Use `/quiz end` to return to ordinary tutoring."
        else:
            result = response.content + "\n\n" + self.current_question
        agent.memory.messages[-1]["content"] = result
        return result

    def end(self, agent) -> str:
        if not self.active:
            return "Quiz mode is already inactive."
        self._write_artifact()
        agent.system_instructions = [text for text in agent.system_instructions if text != self.skill_text]
        path = self.quiz["path"]
        self.quiz = None
        return f"Quiz ended. Artifact saved at `{path}`. Ordinary tutoring is active."

    def _create_artifact(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        stem = f"quiz-{stamp}-{_safe_topic(self.quiz['topic'])}"
        for n in range(1, 10000):
            suffix = "" if n == 1 else f"-{n}"
            path = self.artifact_dir / f"{stem}{suffix}.md"
            try:
                with path.open("x", encoding="utf-8") as file:
                    file.write("")
                self.quiz["path"] = path
                self._write_artifact()
                return
            except FileExistsError:
                continue
        raise OSError("Could not allocate a unique quiz artifact filename.")

    def _write_artifact(self) -> None:
        quiz = self.quiz
        sections = [f"# Quiz: {quiz['topic']}", f"\nCreated: {quiz['created_at']}",
                    f"\nProgress: {min(quiz['current'], len(quiz['questions']))}/{len(quiz['questions'])} answered"]
        for index, item in enumerate(quiz["questions"], 1):
            sections.append(f"\n## Question {index}\n\n{item['question']}\n\n**Answer:** {item['answer']}\n\n**Explanation:** {item['explanation']}")
            if index <= len(quiz["results"]):
                result = quiz["results"][index - 1]
                sections.append(f"\n**Learner response:** {result['answer']}\n\n**Feedback:** {result['feedback']}")
        if quiz.get("summary"):
            sections.append(f"\n## Summary\n\n{quiz['summary']}")
        quiz["path"].write_text("\n".join(sections) + "\n", encoding="utf-8")

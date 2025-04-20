import operator
from typing import List, Literal, Union, NamedTuple, Optional, Dict, Any, Type
from pydantic import BaseModel, Field
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
import requests
import csv


from typing_extensions import Annotated, TypedDict
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver


csv_data = requests.get(
    "https://storage.googleapis.com/benchmarks-artifacts/game-of-24/24.csv"
).content.decode("utf-8")
# Get just the Puzzles column (column index 1)
puzzles = [row[1].strip() for row in csv.reader(csv_data.splitlines()[1:])]

print(f"Example puzzles: {puzzles[:3]}")

OperatorType = Literal["+", "-", "*", "/"]
TokenType = Union[float, OperatorType]

## We use these schemas to prompt the LLM to generate equations that evaluate to 24.


class Equation(BaseModel):
    """The formula combining the provided numbers to reach the target of 24."""

    tokens: List[TokenType] = Field(
        description="The stack of tokens and operators in reverse-polish notation. Example: [3, 4, '+', -1, '*'] would evaluate to (3 + 4) * -1 = -7.",
    )

    def compute(self) -> float:
        op_funcs = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
        }
        stack = []
        for token in self.tokens:
            if isinstance(token, float):
                stack.append(token)
            else:
                b, a = stack.pop(), stack.pop()
                stack.append(op_funcs[token](a, b))

        return stack[0]


class GuessEquations(BaseModel):
    """Submit multiple equations as guesses."""

    reasoning: str = Field(
        description="The reasoning behind the submitted guesses. Explain how you arrived at these equations."
    )

    equations: List[Equation] = Field(
        description="The list of equations to submit as guesses."
    )
model_registry = {"GuessEquations": GuessEquations, "Equation": Equation}


## These objects will represent a single "candidate" (or scored candidate) within our agent's state.
# You can update the candidate object to match your own task.
class Candidate(NamedTuple):
    candidate: Equation
    score: Optional[float] = None
    feedback: Optional[str] = None

    def __str__(self):
        try:
            computed = self.candidate.compute()
        except Exception as e:
            computed = f"Invalid equation: {self.candidate.tokens}; Error: {repr(e)}"

        return f"Equation({self.candidate.tokens}) = {computed} (Reward: {self.score})"


def generate_prompt_from_model(
    model: Type[BaseModel],
    model_registry: Dict[str, Type[BaseModel]],
    indent: int = 0,
    max_indent: int = 3
) -> str:
    if indent > max_indent:
        return "  " * indent + "... (truncated due to depth limit)"
    
    schema = model.model_json_schema()
    fields = schema.get("properties", {})
    required_fields = schema.get("required", [])
    lines = []
    indent_str = "  " * indent

    for name, details in fields.items():
        field_type = details.get("type", "unknown")
        description = details.get("description", "No description")
        required = "*" if name in required_fields else ""
        
        # 处理复杂类型
        if field_type == "array":
            items = details.get("items", {})
            if "$ref" in items:
                ref_model_name = items["$ref"].split("/")[-1]
                description += f" (array of {ref_model_name} objects)"
        
        if "enum" in details:
            enum_values = ", ".join(map(str, details["enum"]))
            description += f" [Allowed values: {enum_values}]"
        
        lines.append(f"{indent_str}- {required}{name} ({field_type}): {description}")

        # 处理嵌套模型
        if "$ref" in details:
            ref_model_name = details["$ref"].split("/")[-1]
            nested_model = model_registry.get(ref_model_name)
        elif "items" in details and "$ref" in details["items"]:
            ref_model_name = details["items"]["$ref"].split("/")[-1]
            nested_model = model_registry.get(ref_model_name)
        else:
            nested_model = None
        
        if nested_model and issubclass(nested_model, BaseModel):
            nested_lines = generate_prompt_from_model(
                nested_model, model_registry, indent + 1, max_indent
            )
            lines.append(nested_lines)
    
    # 添加示例
    if indent == 0:
        example = model.model_json_schema()
        lines.append(f"\nExample structure:\n{example}")
    
    return "\n".join(lines)


class ScoredCandidate(Candidate):
    candidate: Equation
    score: float
    feedback: str

model_description = generate_prompt_from_model(GuessEquations, model_registry)

parser = PydanticOutputParser(pydantic_object=GuessEquations)
def parse_output(output: str) -> GuessEquations:
    """Parse the output of the LLM into a GuessEquations object."""
    try:
        print(output)
        json_str = output.split("```json")[1].split("```")[0].strip()
        return parser.parse(json_str)
    except ValueError as e:
        raise ValueError(f"Failed to parse output: {output}. Error: {e}") from e

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are playing the Game of 24. Using the provide numbers, create an equation that evaluates to 24.\n
            
            You are tasked with submitting guesses for equations. The input should follow this Json structure STRICTLY!:
            ```json{model_description}```
            请严格按照以上Json格式响应！！！Please provide your guesses in the correct format.

            Create EXACTLY {k} guesses for this round.
            """,
        ),
        ("user", "Solve the 24 game for these numbers: {problem}.{candidate}"),
    ],
).partial(candidate="", model_description=parser.get_format_instructions())

llm = OllamaLLM(model="deepseek-r1:32b", base_url="http://localhost:11434")
solver = prompt | llm | RunnableLambda(parse_output)

def compute_score(problem: str, candidate: Candidate) -> ScoredCandidate:
    numbers = list(map(int, problem.split()))
    # Check that the candidate equation uses all 4 numbers exactly once
    used_numbers = [
        token for token in candidate.candidate.tokens if isinstance(token, float)
    ]
    if sorted(used_numbers) != sorted(numbers):
        score = 0
        feedback = "The equation must use all 4 numbers exactly once."
        return ScoredCandidate(
            candidate=candidate.candidate, score=score, feedback=feedback
        )
    try:
        result = candidate.candidate.compute()
        score = 1 / (1 + abs(24 - result))
        feedback = f"Result: {result}"
    except Exception as e:
        score = 0
        feedback = f"Invalid equation. Error: {repr(e)}"
    return ScoredCandidate(
        candidate=candidate.candidate, score=score, feedback=feedback
    )


def update_candidates(
    existing: Optional[list] = None,
    updates: Optional[Union[list, Literal["clear"]]] = None,
) -> List[str]:
    if existing is None:
        existing = []
    if updates is None:
        return existing
    if updates == "clear":
        return []
    # Concatenate the lists
    return existing + updates


class ToTState(TypedDict):
    problem: str
    candidates: Annotated[List[Candidate], update_candidates]
    scored_candidates: Annotated[List[ScoredCandidate], update_candidates]
    depth: Annotated[int, operator.add]


class Configuration(TypedDict, total=False):
    max_depth: int
    threshold: float
    k: int
    beam_size: int


def _ensure_configurable(config: RunnableConfig) -> Configuration:
    """Get params that configure the search algorithm."""
    configurable = config.get("configurable", {})
    return {
        **configurable,
        "max_depth": configurable.get("max_depth", 10),
        "threshold": config.get("threshold", 0.9),
        "k": configurable.get("k", 5),
        "beam_size": configurable.get("beam_size", 3),
    }


class ExpansionState(ToTState):
    seed: Optional[Candidate]


def expand(state: ExpansionState, *, config: RunnableConfig) -> Dict[str, List[str]]:
    """Generate the next state."""
    configurable = _ensure_configurable(config)
    if not state.get("seed"):
        candidate_str = ""
    else:
        candidate_str = "\n\n" + str(state["seed"])

    equation_submission = solver.invoke(
        {
            "problem": state["problem"],
            "candidate": candidate_str,
            "k": configurable["k"],
        },
        config=config,
    )
    print(equation_submission)
    # except Exception as e:
    #     raise RuntimeError(
    #         f"Failed to generate equations. Error: {repr(e)}"
    #     ) from e
    #     return {"candidates": []}
    new_candidates = [
        Candidate(candidate=equation) for equation in equation_submission.equations
    ]
    return {"candidates": new_candidates}


def score(state: ToTState) -> Dict[str, List[float]]:
    """Evaluate the candidate generations."""
    candidates = state["candidates"]
    scored = []
    for candidate in candidates:
        scored.append(compute_score(state["problem"], candidate))
    return {"scored_candidates": scored, "candidates": "clear"}


def prune(
    state: ToTState, *, config: RunnableConfig
) -> Dict[str, List[Dict[str, Any]]]:
    scored_candidates = state["scored_candidates"]
    beam_size = _ensure_configurable(config)["beam_size"]
    organized = sorted(
        scored_candidates, key=lambda candidate: candidate[1], reverse=True
    )
    pruned = organized[:beam_size]
    return {
        # Update the starting point for the next iteration
        "candidates": pruned,
        # Clear the old memory
        "scored_candidates": "clear",
        # Increment the depth by 1
        "depth": 1,
    }


def should_terminate(
    state: ToTState, config: RunnableConfig
) -> Union[Literal["__end__"], Send]:
    configurable = _ensure_configurable(config)
    if not state["candidates"]:
        return "__end__"
    solved = state["candidates"][0].score >= configurable["threshold"]
    if solved or state["depth"] >= configurable["max_depth"]:
        return "__end__"
    return [
        Send("expand", {**state, "somevalseed": candidate})
        for candidate in state["candidates"]
    ]


# Create the graph
builder = StateGraph(state_schema=ToTState, config_schema=Configuration)

# Add nodes
builder.add_node(expand)
builder.add_node(score)
builder.add_node(prune)

# Add edges
builder.add_edge("expand", "score")
builder.add_edge("score", "prune")
builder.add_conditional_edges("prune", should_terminate, path_map=["expand", "__end__"])

# Set entry point
builder.add_edge("__start__", "expand")

# Compile the graph
graph = builder.compile(checkpointer=MemorySaver())
config = {
    "configurable": {
        "thread_id": "test_1",
        "depth": 10,
    }
}
for step in graph.stream({"problem": puzzles[42]}, config):
    print(step)
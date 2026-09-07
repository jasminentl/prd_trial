import os
import json
from typing import TypedDict, Optional, Literal, Dict
from dotenv import load_dotenv

from neo4j import GraphDatabase
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()

# =========================================================
# 1. Global Singletons Created at Import Time
# =========================================================
embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT"),
    openai_api_version=os.getenv("EMBEDDING_API_VERSION"),
    azure_endpoint=os.getenv("EMBEDDING_AZURE_ENDPOINT"),
    api_key=os.getenv("EMBEDDING_API_KEY"),
)

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
)

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

TARGET_DATABASE = os.getenv("NEO4J_DATABASE")

# JSON Cleanser Helper Function
def clean_json_response(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)

# Manager Praise Validator (Supportive HR Validator)
def validate_manager_praises(praises: list) -> dict:
    """Validates if manager's edited praises are genuine and not just empty phrases."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Supportive HR Quality Validator evaluating 5 teacher performance affirmations.

EVALUATION CRITERIA:
Pass any affirmation that is genuine, contextual, or specific. Do NOT force a rigid formula.
Valid affirmations include:
1. Action & Impact: "You used bilingual charts, which helped students."
2. Effort & Growth Mindset: "I appreciate your effort in trying the new IT system even when it crashed."
3. Value & Support: "Thank you for staying calm and helping new colleagues during the busy exam week."

FAIL ONLY IF: 
The praise is purely empty, generic, or extremely short (e.g., under 4 words) without any context.
- Fail Examples: "Good job.", "Great teacher.", "Nice work.", "You are passionate.", "Excellent."

Evaluate each of the 5 praises. Respond strictly in valid JSON format:
{{
  "all_passed": true/false,
  "evaluations": [
    {{
      "index": 1,
      "passed": true/false,
      "reason": "If failed, state briefly why (e.g., 'Too generic/empty praise'). If passed, write 'OK'."
    }}
  ]
}}"""),
        ("user", "Praises to evaluate:\n1. {p1}\n2. {p2}\n3. {p3}\n4. {p4}\n5. {p5}")
    ])
    
    while len(praises) < 5:
        praises.append("")
        
    try:
        response = (prompt | llm).invoke({
            "p1": praises[0], "p2": praises[1], "p3": praises[2], "p4": praises[3], "p5": praises[4]
        })
        return clean_json_response(response.content)
    except Exception:
        return {"all_passed": True, "evaluations": []}

def classify_custom_objective(objective_text: str, role: str) -> str:
    """Classifies custom self-written objective into a readiness quadrant using LLM."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an Executive Performance Coach analyzing a self-written 90-day Dev-OKR objective by an employee.
Employee Role: {role}

Analyze the psychological intent, risk orientation, and workload focus of the text, then determine which of the 4 readiness quadrants it aligns with best:
- Q1: Bold stretch goals, innovation.
- Q2: 5-minute low-risk micro-habits, lowering anxiety.
- Q3: Calibrating overused strengths, setting boundaries.
- Q4: Administrative time-saving, hassle reduction.

CRITICAL RULE:
If the input text is gibberish (e.g., "asdfasdf"), completely irrelevant to professional development (e.g., "I want to eat pizza"), or too short to convey a goal, you MUST classify it as "INVALID".

Respond strictly with valid JSON:
{{"classified_quadrant": "Q1" | "Q2" | "Q3" | "Q4" | "INVALID", "reasoning": "..."}}"""),
        ("user", "Custom Objective Text: {objective_text}")
    ])
    try:
        response = (prompt | llm).invoke({
            "objective_text": objective_text,
            "role": role
        })
        data = clean_json_response(response.content)
        q = data.get("classified_quadrant", "INVALID")
        return q if q in ["Q1", "Q2", "Q3", "Q4", "INVALID"] else "INVALID"
    except Exception:
        return "INVALID"

# =========================================================
# 2. Pipeline State Schema & Quadrant Registry
# =========================================================
class PipelineState(TypedDict):
    feedback: str
    role: str
    threat_sensitivity: Literal["High", "Low"]
    growth_motivation: Literal["Mastery/Approval", "Workload/Ego Protection"]
    sub_persona: Optional[str]
    quadrant_code: Optional[str]
    quadrant_config: Optional[dict]
    matched_onet: Optional[dict]
    coaching_card_51: Optional[dict]
    candidate_objectives: Optional[Dict[str, dict]]
    selected_option_key: Optional[str]
    selected_objective_text: Optional[str]
    selected_option_quadrant: Optional[str]
    approval_status: Optional[str]
    manager_feedback: Optional[str]
    final_okr: Optional[dict]
    protected_growth_plan: Optional[dict]

QUADRANT_REGISTRY = {
    "Q1": {
        "name": "Q1: The Learning Explorer",
        "mindset": "High Intrinsic Growth / Low Threat Sensitivity",
        "strategy": "Autonomy & Focus. Offer stretch goals and peer-mentoring roles. Narrow down choices to prevent overcommitment.",
        "diagnostic_protocol": "Feedforward Protocol (Marshall Goldsmith) - Crowdsources 1-2 future-focused suggestions without debating past mistakes.",
        "five_to_one_purpose": "Fuel for Curiosity - Validates risk-taking, innovative problem-solving, and peer mentorship.",
        "gap_framing_rule": "Frame growth gap as a stretch capability or leadership extension."
    },
    "Q2": {
        "name": "Q2: The Defensive Prover",
        "mindset": "High Output / High Threat Sensitivity (Perfectionist)",
        "strategy": "Scope Capping & Safety. Shrink goals into 1-action micro-habits using BJ Fogg's Tiny Habits framework. Frame setbacks strictly as 'valuable diagnostic data'.",
        "diagnostic_protocol": "Feedforward Interview / FFI (Kluger & Nir) - Mines past peak performance stories to identify missing enabling conditions.",
        "five_to_one_purpose": "Anxiety Reducer - Lowers perfectionist pressure; affirms that self-worth is not tied to flawless execution.",
        "gap_framing_rule": "Frame growth gap as a 5-minute low-stakes micro-experiment using Tiny Habits."
    },
    "Q3": {
        "name": "Q3: The Avoidant Shield",
        "mindset": "Ego Defense / High Threat Sensitivity (Cynical Veteran)",
        "strategy": "5:1 Affirmation & Objective Mirroring. Generate 5 specific process praises first. Frame growth gaps strictly as 'Overused Strengths' using unedited objective data.",
        "diagnostic_protocol": "Immunity to Change Map (Kegan & Lahey) - Uncovers hidden 'competing commitments' causing stagnation.",
        "five_to_one_purpose": "Threat Shield (Non-Negotiable) - Deactivates brain threat SCARF responses before delivering 1 growth gap.",
        "gap_framing_rule": "Frame growth gap explicitly as an 'Overused Strength' (e.g., high standards overused into inflexibility)."
    },
    "Q4": {
        "name": "Q4: The Pragmatic Functionalist",
        "mindset": "Energy Conservation / Low Engagement (Comfort-Zone)",
        "strategy": "Friction Reduction & Utility. Strip away corporate jargon. Direct Dev-OKRs strictly toward administrative time-saving and daily workflow ease.",
        "diagnostic_protocol": "Friction & Energy Audit (Fogg / Wrzesniewski) - Identifies daily operational annoyances, bottlenecks, and time drains.",
        "five_to_one_purpose": "Efficiency Validator - Proves that baseline reliability, practical output, and time-saving habits are valued.",
        "gap_framing_rule": "Frame growth gap strictly as an administrative time-saver or daily hassle eliminator."
    }
}

PRESET_TEST_DATA = {
    "Q1: High School Science Teacher (Learning Explorer)": {
        "role": "High School Biology / STEM Teacher",
        "threat_sensitivity": "Low",
        "growth_motivation": "Mastery/Approval",
        "feedback": "Constantly brings bold interdisciplinary AI-lab ideas to department meetings. Extremely eager to innovate, but needs help narrowing focus so core IB syllabus coverage doesn't get overwhelmed."
    },
    "Q2: IB Diploma Coordinator (Defensive Prover)": {
        "role": "IB DP Coordinator & Higher Math Teacher",
        "threat_sensitivity": "High",
        "growth_motivation": "Mastery/Approval",
        "feedback": "Extremely dedicated, stays late creating flawless exam reviews. However, becomes visibly anxious and defensive during mid-term curriculum audits or when IB assessment rubrics change."
    },
    "Q3: Veteran Humanities Teacher (Avoidant Shield)": {
        "role": "Middle School Humanities Teacher (15+ Yrs Tenure)",
        "threat_sensitivity": "High",
        "growth_motivation": "Workload/Ego Protection",
        "feedback": "Dismissive of new school-wide digital portfolio tools during staff retros, publicly blaming student attention spans and IT infrastructure to deflect from adopting new reporting formats."
    },
    "Q4: Primary EAL Specialist (Pragmatic Functionalist)": {
        "role": "Primary School EAL Specialist Teacher",
        "threat_sensitivity": "Low",
        "growth_motivation": "Workload/Ego Protection",
        "feedback": "Flawlessly punctual with yard duty, attendance logging, and student progress reports. Consistently declines joining optional curriculum committees or testing new HR reflection frameworks."
    }
}

MANAGER_REJECTION_PRESETS = [
    "[Q1 Focus Issue] This objective is too tactical. As a Q1 profile, let's elevate this into a cross-departmental pilot opportunity.",
    "[Q2 Focus Issue] The selected objective is too broad for report card season. Please re-frame it into a 5-minute daily micro-habit to lower stress.",
    "[Q3 Focus Issue] The objective wording sounds slightly defensive. Let's re-frame it to focus on calibrating an overused strength into a shared team practice.",
    "[Q4 Focus Issue] This objective introduces extra meeting overhead. Refocus strictly on saving 30 minutes of administrative time per week.",
    "[General Rephrase] The wording sounds too evaluative. Rephrase to focus on behavioral practice rather than a performance target."
]

def resolve_quadrant(threat: str, motivation: str) -> str:
    if threat == "Low" and motivation == "Mastery/Approval": return "Q1"
    elif threat == "High" and motivation == "Mastery/Approval": return "Q2"
    elif threat == "High" and motivation == "Workload/Ego Protection": return "Q3"
    else: return "Q4"

# =========================================================
# 3. LangGraph Nodes
# =========================================================
def diagnostic_mapping_node(state: PipelineState) -> dict:
    feedback = state["feedback"]
    try:
        vector = embeddings_model.embed_query(feedback)
        query = """
        CALL db.index.vector.queryNodes('onet_vector_index', 1, $embedding)
        YIELD node, score
        RETURN 
            node.element_id AS element_id, 
            node.name AS name, 
            node.description AS description, 
            score
        """
        with neo4j_driver.session(database=TARGET_DATABASE) as session:
            res = session.run(query, embedding=vector).single()
            if res:
                matched_onet = {"element_id": res["element_id"], "name": res["name"], "description": res["description"], "score": round(res["score"], 4)}
            else:
                matched_onet = {"element_id": "1.D.1.f", "name": "Adaptability/Flexibility", "description": "Being open to change and variety.", "score": 0.85}
    except Exception:
        matched_onet = {"element_id": "1.D.1.f", "name": "Adaptability/Flexibility", "description": "Being open to change and variety.", "score": 0.85}
    return {"matched_onet": matched_onet}

def candidate_generation_node(state: PipelineState) -> dict:
    q_code = resolve_quadrant(state["threat_sensitivity"], state["growth_motivation"])
    q_config = QUADRANT_REGISTRY[q_code]
    onet = state["matched_onet"]
    sub_persona = state.get("sub_persona", "General Profile")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an Executive Performance Coach for an elite International School.
Generate a 5:1 Coaching Card and EXACTLY 4 Candidate Dev-OKR Objectives—one tailored for each of the 4 readiness quadrants—in JSON format.

Quadrant Context: {q_name} ({q_mindset})
Specific Sub-Persona: {sub_persona}
5:1 Praise Purpose: {five_to_one_purpose}
Gap Framing Rule: {gap_framing_rule}

Target Competency: [{element_id}] {name} - {description}
Staff Role: {role}

MANDATORY RULES FOR 5 PROCESS AFFIRMATIONS:
Provide 5 high-quality draft affirmations. These should recognize specific actions, effort, or teamwork.
- Good Example: "You created 3 bilingual diagrams for the unit review, boosting comprehension."
- Good Example: "I appreciate your effort in adopting the new reporting system despite the IT issues."
- BAD Example: "Great job."

INSTRUCTIONS FOR THE 4 CANDIDATE OBJECTIVE PROBES:
- Option A: Stretch goals, interdisciplinary innovation, peer mentoring.
- Option B: 5-minute low-risk micro-habit or perfectionism-reducing experiment.
- Option C: Calibrating an "Overused Strength" to reduce ego defensiveness.
- Option D: Administrative time-saving, hassle reduction, workflow ease.

Respond strictly with valid JSON using this structure:
{{
  "praises": [
    "Praise 1...",
    "Praise 2...",
    "Praise 3...",
    "Praise 4...",
    "Praise 5..."
  ],
  "observation": "Write a clear, non-evaluative, single-paragraph Growth Observation here.",
  "options": {{
    "Option A (Q1 Stretch Goal)": {{ "text": "...", "target_q": "Q1" }},
    "Option B (Q2 Low-Risk Micro-Habit)": {{ "text": "...", "target_q": "Q2" }},
    "Option C (Q3 Strength Calibration)": {{ "text": "...", "target_q": "Q3" }},
    "Option D (Q4 Administrative Time-Saver)": {{ "text": "...", "target_q": "Q4" }}
  }}
}}"""),
        ("user", "Feedback: {feedback}")
    ])

    response = (prompt | llm).invoke({
        "q_name": q_config["name"], "q_mindset": q_config["mindset"], "sub_persona": sub_persona,
        "five_to_one_purpose": q_config["five_to_one_purpose"], "gap_framing_rule": q_config["gap_framing_rule"],
        "element_id": onet["element_id"], "name": onet["name"], "description": onet["description"],
        "role": state["role"], "feedback": state["feedback"]
    })

    try:
        data = clean_json_response(response.content)
    except Exception:
        data = {
            "praises": [
                "You introduced structured IB rubrics during co-planning, directly reducing grading confusion among department peers.",
                "You incorporated formative exit-tickets in biology classes, enabling real-time tracking of student understanding.",
                "I appreciate your ongoing effort in logging student intervention notes within 24 hours.",
                "You designed interactive digital lab exercises that increased student engagement during review sessions.",
                "Thank you for sharing assessment templates with new department members and supporting their onboarding."
            ],
            "observation": f"Growth Opportunity: Consider framing project scope shifts as 5-minute diagnostic micro-trials.",
            "options": {
                "Option A (Q1 Stretch Goal)": {"text": "Pilot an interdisciplinary AI project.", "target_q": "Q1"},
                "Option B (Q2 Low-Risk Micro-Habit)": {"text": "Test a 5-minute daily pause before responding.", "target_q": "Q2"},
                "Option C (Q3 Strength Calibration)": {"text": "Calibrate high standards into a weekly checklist.", "target_q": "Q3"},
                "Option D (Q4 Administrative Time-Saver)": {"text": "Automate student progress logging to save time.", "target_q": "Q4"}
            }
        }

    return {
        "quadrant_code": q_code,
        "quadrant_config": q_config,
        "coaching_card_51": {
            "five_to_one_purpose": q_config["five_to_one_purpose"],
            "praises": data.get("praises", []),
            "observation": data.get("observation", "Growth Observation placeholder.")
        },
        "candidate_objectives": data.get("options", {})
    }

def final_plan_generation_node(state: PipelineState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an Executive Performance Coach. The Line Manager has APPROVED the selected Dev-OKR Objective.
Generate 3 Key Results (low-friction micro-habits) and a 2-Hour/Month Protected Growth Time Plan in JSON format.
Role: {role}
Selected Objective: {selected_obj}

Respond strictly with valid JSON using this structure:
{{
  "key_results": ["Key Result 1: ...", "Key Result 2: ...", "Key Result 3: ..."],
  "protected_growth": {{
    "hour_1": "Hour 1 (Self-Study & FYI Deep Dive): ...",
    "hour_2": "Hour 2 (Reflection & Peer Exchange): ..."
  }}
}}"""),
        ("user", "Generate final execution plan.")
    ])
    response = (prompt | llm).invoke({"role": state["role"], "selected_obj": state["selected_objective_text"]})
    try:
        data = clean_json_response(response.content)
    except Exception:
        data = {"key_results": ["KR1", "KR2", "KR3"], "protected_growth": {"hour_1": "H1", "hour_2": "H2"}}
    return {"final_okr": {"objective": state["selected_objective_text"], "key_results": data.get("key_results", [])}, "protected_growth_plan": data.get("protected_growth", {})}

# =========================================================
# 4. Stage Workflows & Runners
# =========================================================
def run_stage_1(input_state: PipelineState) -> PipelineState:
    graph = StateGraph(PipelineState)
    graph.add_node("diagnostic_mapping", diagnostic_mapping_node)
    graph.add_node("candidate_generation", candidate_generation_node)
    graph.add_edge(START, "diagnostic_mapping")
    graph.add_edge("diagnostic_mapping", "candidate_generation")
    graph.add_edge("candidate_generation", END)
    return graph.compile().invoke(input_state)

def run_stage_2(current_state: PipelineState) -> PipelineState:
    graph = StateGraph(PipelineState)
    graph.add_node("final_plan_generation", final_plan_generation_node)
    graph.add_edge(START, "final_plan_generation")
    graph.add_edge("final_plan_generation", END)
    return graph.compile().invoke(current_state)
import streamlit as st
import time
from step2_langgraph_pipeline import (
    run_stage_1,
    run_stage_2,
    classify_custom_objective,
    validate_manager_praises,
    PRESET_TEST_DATA,
    MANAGER_REJECTION_PRESETS,
    QUADRANT_REGISTRY
)

# Page Configuration
st.set_page_config(
    page_title="Performance Review & Development System",
    page_icon=None,
    layout="wide"
)

# =========================================================
# Deep-Targeted Google Stitch CSS 
# =========================================================
STITCH_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --primary-color: #1a73e8;       /* Google Blue */
        --primary-hover: #1557b0;
        --bg-surface: #f8f9fa;
        --card-bg: #ffffff;
        --text-main: #202124;
        --text-subtle: #5f6368;
        --border-color: #dadce0;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-main);
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 98% !important;
    }

    [data-testid="stSidebar"] {
        background-color: var(--bg-surface);
        border-right: 1px solid var(--border-color);
    }

    h1 {
        font-size: clamp(1.4rem, 2.5vw, 1.8rem) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-main) !important;
        padding-bottom: 0 !important;
        margin-bottom: 0.2rem !important;
        line-height: 1.3 !important;
    }
    
    h5 {
        font-size: clamp(0.9rem, 1.2vw, 1rem) !important;
        font-weight: 400 !important;
        color: var(--text-subtle) !important;
        margin-top: -0.2rem !important;
        line-height: 1.5 !important;
    }
    
    hr {
        margin-top: 1rem !important;
        margin-bottom: 2rem !important;
        border-color: var(--border-color) !important;
    }

    .stitch-card-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-subtle);
        margin-bottom: 0.75rem;
    }

    .badge-tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .badge-primary { background-color: #e8f0fe; color: #1a73e8; }
    .badge-success { background-color: #e6f4ea; color: #137333; }
    .badge-neutral { background-color: #f1f3f4; color: #3c4043; }

    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.875rem;
        border: 1px solid var(--border-color);
        transition: all 0.2s ease;
    }
    
    .stButton>button[kind="primary"] {
        background-color: var(--primary-color);
        color: #ffffff;
        border: none;
    }
    
    .stButton>button[kind="primary"]:hover {
        background-color: var(--primary-hover);
    }

    div[role="radiogroup"] label {
        white-space: normal !important;
        word-wrap: break-word !important;
    }

    .waiting-box {
        text-align: center;
        padding: 4rem 2rem;
        background-color: var(--bg-surface);
        border-radius: 8px;
        border: 2px dashed var(--border-color);
        margin-top: 2rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(STITCH_CSS, unsafe_allow_html=True)

# Session State Initialization
if "workflow_stage" not in st.session_state:
    st.session_state.workflow_stage = "1_MANAGER_INIT"
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "stage1_generated" not in st.session_state:
    st.session_state.stage1_generated = False

# Sidebar: Role Switcher & Workflow Status
with st.sidebar:
    st.markdown("<div class='stitch-card-title'>PR&D CONTROL PANEL</div>", unsafe_allow_html=True)
    current_role = st.radio("ACTIVE ROLE / PERSPECTIVE", ["Line Manager View", "Employee View"])
    st.divider()
    st.markdown("<div class='stitch-card-title'>WORKFLOW TRACKER</div>", unsafe_allow_html=True)
    
    stage_labels = {
        "1_MANAGER_INIT": "Stage 1: Manager Diagnostics",
        "2_EMPLOYEE_CHOICE": "Stage 2: Employee Selection",
        "3_MANAGER_APPROVAL": "Stage 3: Manager Approval Gateway",
        "4_FINAL_APPROVED": "Stage 4: Approved & Active",
        "3B_REVISION_REQUESTED": "Revision Requested"
    }
    st.markdown(f"<span class='badge-tag badge-primary'>{stage_labels[st.session_state.workflow_stage]}</span>", unsafe_allow_html=True)
    st.write("")
    
    if st.button("Reset Workflow Session", use_container_width=True):
        st.session_state.workflow_stage = "1_MANAGER_INIT"
        st.session_state.pipeline_state = None
        st.session_state.stage1_generated = False
        st.rerun()

# Header
st.markdown("# INTERNATIONAL SCHOOL PR&D SYSTEM")
st.markdown("##### Performance Review & Development Platform — Non-Evaluative Growth Framework")
st.divider()

# =========================================================
# STAGE 1: LINE MANAGER DIAGNOSTICS & EDITABLE 5:1 CARD
# =========================================================
if st.session_state.workflow_stage == "1_MANAGER_INIT":
    if "Line Manager View" not in current_role:
        # 💡 Clear Waiting State instead of tiny warning
        st.markdown("""
        <div class='waiting-box'>
            <h3 style='color: #5f6368;'>⏳ Awaiting Manager Diagnostics</h3>
            <p style='color: #5f6368; margin-top: 1rem;'>The workflow is currently in Stage 1. The Line Manager is preparing the initial feedback.</p>
            <p style='font-weight: 600; color: #1a73e8; margin-top: 2rem;'>👉 Switch to 'Line Manager View' in the sidebar to process this stage.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### Stage 1: Diagnostic Setup & 5:1 Coaching Mirror")
        preset_choice = st.selectbox("LOAD TEST DATA PRESET", ["Custom (Use Quiz Below)"] + list(PRESET_TEST_DATA.keys()))
        
        q1_idx, q2_idx, q3_idx, default_role, default_feedback = 0, 0, 0, "High School Teacher", ""
        if preset_choice != "Custom (Use Quiz Below)":
            q_num = preset_choice[:2]
            q1_idx = q2_idx = q3_idx = int(q_num[1]) - 1
            default_role = PRESET_TEST_DATA[preset_choice]["role"]
            default_feedback = PRESET_TEST_DATA[preset_choice]["feedback"]

        col_inp1, col_inp2 = st.columns([1, 2.5])
        with col_inp1:
            st.markdown("<div class='stitch-card-title'>OBSERVATION & ROLE</div>", unsafe_allow_html=True)
            role_input = st.text_input("School Role", value=default_role)
            feedback_text = st.text_area("Sanitized Feedback Mirror", value=default_feedback, height=180)

            with st.expander("OPERATIONAL SIGNALS REFERENCE (PRD STANDARD)", expanded=False):
                st.markdown("""
                | Quadrant Profile | Behavioural & Operational Signals |
                | :--- | :--- |
                | **Q1: Learning Explorer** | High opt-in rates for stretch assignments; frequent usage of learning stipends; balanced self/peer scores. |
                | **Q2: Defensive Prover** | Sets >5 complex milestones; 100% completion rates; high self-reported stress; asks managers to assign goals. |
                | **Q3: Avoidant Shield** | High self-ratings paired with low peer/student scores; history of disputing review notes; rapid click-through on forms. |
                | **Q4: Pragmatic Functionalist** | Consistent "Meets Baseline" on KPIs; zero opt-in to optional initiatives; static year-over-year development goals. |
                """)

        with col_inp2:
            st.markdown("<div class='stitch-card-title'>DIAGNOSTIC QUIZ (RADIO BUTTONS)</div>", unsafe_allow_html=True)
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                q1_ans = st.radio("1. Reaction to constructive feedback", [
                    "Accepts calmly & asks clarifying questions [Low Threat]",
                    "Gets anxious, over-explains, or worries about standard [High Threat]",
                    "Gets defensive, blames external factors, or acts cynical [High Threat]",
                    "Nods passively, complies minimally, checks workload [Low Threat]"
                ], index=q1_idx)
            with col_q2:
                q2_ans = st.radio("2. Workplace focus driver", [
                    "Internal curiosity & craft mastery [Mastery/Approval]",
                    "Maintaining perfect record & pleasing leadership [Mastery/Approval]",
                    "Protecting reputation & avoiding blame [Workload/Ego Protection]",
                    "Protecting personal time, energy, work-life balance [Workload/Ego Protection]"
                ], index=q2_idx)
            
            st.write("")
            q3_ans = st.radio("3. Reaction to unexpected scope changes", [
                "Eager to test new methods & innovate [Q1 Explorer]",
                "Anxious about missing perfectionist standards [Q2 Defensive]",
                "Pushes back, blames IT/infrastructure [Q3 Avoidant]",
                "Complies if it adds zero extra admin hassle [Q4 Pragmatic]"
            ], index=q3_idx)
            
            threat_score = 1 if ("High Threat" in q1_ans or "Q2" in q3_ans or "Q3" in q3_ans) else 0
            mot_score = 1 if ("Workload" in q2_ans or "Q3" in q3_ans or "Q4" in q3_ans) else 0
            
            if threat_score == 0 and mot_score == 0:
                derived_threat, derived_motivation, sub_persona, calc_code, framing_guide = "Low", "Mastery/Approval", "Continuous Learner", "Q1", "Framing Strategy: 'What area do you want to explore next, and how can I support you?'"
            elif threat_score == 1 and mot_score == 0:
                derived_threat, derived_motivation, sub_persona, calc_code, framing_guide = "High", "Mastery/Approval", "Perfectionist / Approval-Seeker", "Q2", "Framing Strategy: 'It is 100% safe to test a small habit and fail—learning is the metric here.'"
            elif threat_score == 1 and mot_score == 1:
                derived_threat, derived_motivation, sub_persona, calc_code, framing_guide = "High", "Workload/Ego Protection", "Blind-Spot Profile / Cynical Veteran", "Q3", "Framing Strategy: 'Here are 5 things you do brilliantly; let's look at 1 pattern as an overused strength.'"
            else:
                derived_threat, derived_motivation, sub_persona, calc_code, framing_guide = "Low", "Workload/Ego Protection", "Comfort-Zone Staff / Tech Specialist", "Q4", "Framing Strategy: 'What is 1 daily hassle we can use this Dev-OKR to eliminate for you?'"

            st.info(f"[CALCULATED PROFILE] {calc_code} ({QUADRANT_REGISTRY[calc_code]['name']})\n\nThreat: {derived_threat} | Motivation: {derived_motivation}")
            st.caption(f"{framing_guide}")

        st.divider()
        if st.button("Generate Draft 5:1 Card & Candidate Package", type="primary"):
            if not feedback_text.strip():
                st.error("Observation feedback required.")
            else:
                # 💡 Enhanced Loading Status Box
                with st.status("Executing LangGraph Pipeline Stage 1...", expanded=True) as status:
                    st.write("🔍 Analyzing operational signals...")
                    time.sleep(0.5)
                    st.write("🧠 Mapping psychological quadrant via Azure OpenAI...")
                    time.sleep(0.5)
                    st.write("✍️ Drafting 5:1 affirmations and Dev-OKR options...")
                    
                    st.session_state.pipeline_state = run_stage_1({
                        "feedback": feedback_text, "role": role_input, "threat_sensitivity": derived_threat,
                        "growth_motivation": derived_motivation, "sub_persona": sub_persona
                    })
                    status.update(label="✅ Stage 1 Generation Complete!", state="complete", expanded=False)
                    
                st.session_state.stage1_generated = True
                st.rerun()

        # Step 1B: Manager Review & Customization
        if st.session_state.stage1_generated and st.session_state.pipeline_state:
            st.divider()
            st.markdown("### Manager Review & Customization")
            
            st.info(
                "[GUIDANCE] Manager Customization (Natural Voice):\n\n"
                "AI has drafted specific praises. Feel free to edit them using your authentic voice. "
                "The system encourages recognizing effort, teamwork, or specific actions. "
                "It will only flag purely empty or generic compliments (e.g., 'Good job' or 'Nice work')."
            )

            p = st.session_state.pipeline_state
            card = p['coaching_card_51']
            
            col_ed1, col_ed2 = st.columns([1.2, 1])
            edited_praises = []
            
            with col_ed1:
                st.markdown("<div class='stitch-card-title'>5 PROCESS AFFIRMATIONS (MAX 250 CHARS)</div>", unsafe_allow_html=True)
                for i, praise in enumerate(card['praises'], 1):
                    ep = st.text_area(
                        f"Affirmation {i}", 
                        value=praise, 
                        height=75, 
                        max_chars=250, 
                        key=f"edit_praise_{i}",
                        help="Focus on specific actions, efforts, or team support. Avoid empty compliments."
                    )
                    edited_praises.append(ep)
                    
            with col_ed2:
                st.markdown("<div class='stitch-card-title'>1 GROWTH OBSERVATION</div>", unsafe_allow_html=True)
                edited_obs = st.text_area("Growth Observation", value=card.get('observation', ''), height=200, max_chars=500, key="edit_obs")
                st.markdown(f"<span class='badge-tag badge-neutral'>Mapped O*NET: [{p['matched_onet']['element_id']}] {p['matched_onet']['name']}</span>", unsafe_allow_html=True)

            st.write("")
            if st.button("Confirm & Transmit Package to Employee", type="primary"):
                # 💡 Enhanced Loading Status Box for Validation
                with st.status("Validating praises against psychological criteria...", expanded=True) as status:
                    st.write("🤖 LLM evaluating manager's feedback tone...")
                    validation_result = validate_manager_praises(edited_praises)
                    
                    if validation_result.get("all_passed", True):
                        status.update(label="✅ Validation Passed! Transmitting...", state="complete", expanded=False)
                        st.session_state.pipeline_state['coaching_card_51']['praises'] = edited_praises
                        st.session_state.pipeline_state['coaching_card_51']['observation'] = edited_obs
                        st.session_state.workflow_stage = "2_EMPLOYEE_CHOICE"
                        time.sleep(1) # Small pause to let user read the success message
                        st.rerun()
                    else:
                        status.update(label="❌ Validation Failed", state="error", expanded=True)
                        st.error("[ALERT] Praise Validation Failed: Some affirmations are too generic. Please revise:")
                        for eval_data in validation_result.get("evaluations", []):
                            if not eval_data.get("passed", True):
                                st.warning(f"Affirmation {eval_data['index']} failed: {eval_data.get('reason', 'Too generic or empty.')}")
                        st.stop()

# =========================================================
# STAGE 2: EMPLOYEE SELF-DIRECTED CHOICE
# =========================================================
elif st.session_state.workflow_stage in ["2_EMPLOYEE_CHOICE", "3B_REVISION_REQUESTED"]:
    if "Employee View" not in current_role:
        # 💡 Clear Waiting State instead of tiny warning
        st.markdown("""
        <div class='waiting-box'>
            <h3 style='color: #5f6368;'>✅ Package Transmitted</h3>
            <p style='color: #5f6368; margin-top: 1rem;'>The coaching package has been successfully sent. <br>The workflow is currently paused waiting for the employee to select their objective.</p>
            <p style='font-weight: 600; color: #1a73e8; margin-top: 2rem;'>👉 Switch to 'Employee View' in the sidebar to continue the demo.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        p = st.session_state.pipeline_state
        st.markdown("### Stage 2: Developmental Mirror & Candidate Selection")
        
        if st.session_state.workflow_stage == "3B_REVISION_REQUESTED":
            st.error(f"[REVISION REQUESTED] Feedback from Manager: {p.get('manager_feedback', '')}")

        card = p['coaching_card_51']
        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            st.markdown("<div class='stitch-card-title'>PROCESS AFFIRMATIONS (5)</div>", unsafe_allow_html=True)
            for i, praise in enumerate(card['praises'], 1):
                st.write(f"**{i}.** {praise}")
        with col_c2:
            st.markdown("<div class='stitch-card-title'>GROWTH OBSERVATION (1)</div>", unsafe_allow_html=True)
            st.warning(f"{card.get('observation', '')}")
            st.markdown(f"<span class='badge-tag badge-primary'>Competency Focus: [{p['matched_onet']['element_id']}] {p['matched_onet']['name']}</span>", unsafe_allow_html=True)

        st.divider()
        st.markdown("### Select or Formulate 90-Day Objective")
        opts = p['candidate_objectives']
        option_keys = list(opts.keys()) + ["Option E (Custom Employee Formulated Objective)"]
        
        selected_key = st.radio("Select 1 option:", option_keys, format_func=lambda k: f"**{k}**: {opts[k]['text']}" if k in opts else f"**{k}**: (Formulate custom objective below)")
        
        if selected_key == "Option E (Custom Employee Formulated Objective)":
            custom_okr_text = st.text_area("Formulate Custom Objective:", placeholder="Type your 90-day objective here...", height=100)
            final_selected_text = custom_okr_text.strip()
        else:
            final_selected_text = opts[selected_key]["text"]
            target_q = opts[selected_key]["target_q"]

        st.write("")
        if st.button("Submit Choice for Line Manager Approval", type="primary"):
            if selected_key == "Option E (Custom Employee Formulated Objective)":
                if not final_selected_text:
                    st.error("Custom objective text required before submission.")
                    st.stop()
                
                # 💡 Enhanced Loading Status Box
                with st.status("Analyzing custom objective mindset via LLM...", expanded=True) as status:
                    st.write("🧠 Reading intent and psychological alignment...")
                    target_q = classify_custom_objective(final_selected_text, p.get("role", "School Staff"))
                    
                    if target_q == "INVALID":
                        status.update(label="❌ Invalid Input Detected", state="error", expanded=True)
                        st.error("[ALERT] Invalid input detected. Please formulate a genuine and meaningful professional development goal.")
                        st.stop()
                    
                    status.update(label="✅ Mindset Analyzed!", state="complete", expanded=False)

            st.session_state.pipeline_state["selected_option_key"] = selected_key
            st.session_state.pipeline_state["selected_objective_text"] = final_selected_text
            st.session_state.pipeline_state["selected_option_quadrant"] = target_q
            st.session_state.workflow_stage = "3_MANAGER_APPROVAL"
            st.rerun()

# =========================================================
# STAGE 3: LINE MANAGER APPROVAL GATEWAY
# =========================================================
elif st.session_state.workflow_stage == "3_MANAGER_APPROVAL":
    if "Line Manager View" not in current_role:
        # 💡 Clear Waiting State instead of tiny warning
        st.markdown("""
        <div class='waiting-box'>
            <h3 style='color: #5f6368;'>⏳ Awaiting Manager Approval</h3>
            <p style='color: #5f6368; margin-top: 1rem;'>Your objective selection has been submitted successfully. <br>The Line Manager is now reviewing it.</p>
            <p style='font-weight: 600; color: #1a73e8; margin-top: 2rem;'>👉 Switch to 'Line Manager View' in the sidebar to process this stage.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        p = st.session_state.pipeline_state
        st.markdown("### Stage 3: Manager Approval & Alignment Gateway")
        st.markdown(f"**Submitted Selection ({p['selected_option_key']}):**")
        st.info(f"\"{p['selected_objective_text']}\"")
        
        initial_q, selected_q = p["quadrant_code"], p["selected_option_quadrant"]
        
        if initial_q != selected_q:
            st.warning(f"[STATUS NOTICE] Quadrant Misalignment Detected.\n\nInitial manager diagnosis was {initial_q}, but employee selection aligns with {selected_q} ({QUADRANT_REGISTRY[selected_q]['name']}).\nGuidance Note: The employee exhibits a {selected_q} focus. Consider this context for ongoing check-ins.")
        else:
            st.success(f"[STATUS CONFIRMED] Objective alignment matches initial diagnosis ({initial_q}).")

        st.divider()
        col_app, col_rej = st.columns(2)
        with col_app:
            if st.button("Approve Objective & Generate Key Results", type="primary", use_container_width=True):
                st.session_state.pipeline_state["approval_status"] = "APPROVED"
                
                # 💡 Enhanced Loading Status Box
                with st.status("Generating Key Results & Protected Growth Plan...", expanded=True) as status:
                    st.write("🛠️ Breaking down objective into actionable micro-habits...")
                    time.sleep(0.5)
                    st.write("⏱️ Structuring protected growth time schedule...")
                    
                    st.session_state.pipeline_state = run_stage_2(st.session_state.pipeline_state)
                    status.update(label="✅ Final Plan Generated!", state="complete", expanded=False)
                    
                st.session_state.workflow_stage = "4_FINAL_APPROVED"
                st.rerun()
        with col_rej:
            with st.popover("Request Revision", use_container_width=True):
                preset_rej = st.selectbox("Preset Feedback:", MANAGER_REJECTION_PRESETS)
                custom_rej = st.text_area("Custom Rejection Feedback:", value=preset_rej)
                if st.button("Transmit Revision Request"):
                    st.session_state.pipeline_state["manager_feedback"] = custom_rej
                    st.session_state.workflow_stage = "3B_REVISION_REQUESTED"
                    st.rerun()

# =========================================================
# STAGE 4: FINAL APPROVED DEV-OKR (EXECUTIVE SUMMARY)
# =========================================================
elif st.session_state.workflow_stage == "4_FINAL_APPROVED":
    p = st.session_state.pipeline_state
    st.markdown("<span class='badge-tag badge-success'>[STATUS: APPROVED & ACTIVE]</span>", unsafe_allow_html=True)
    st.markdown("### Final Approved 90-Day Developmental Plan")
    
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        st.markdown(f"**Objective:** {p['final_okr']['objective']}")
        st.write("")
        st.markdown("<div class='stitch-card-title'>ACTIONABLE KEY RESULTS (MICRO-HABITS)</div>", unsafe_allow_html=True)
        for kr in p['final_okr']['key_results']: 
            st.write(f"- {kr}")
    with col_f2:
        st.markdown("<div class='stitch-card-title'>METADATA</div>", unsafe_allow_html=True)
        st.write(f"**Profile:** {p['quadrant_config']['name']}")
        st.write(f"**Mapped Skill:** [{p['matched_onet']['element_id']}] {p['matched_onet']['name']}")

    st.divider()
    pg = p['protected_growth_plan']
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("<div class='stitch-card-title'>HOUR 1: DEEP DIVE & SCAFFOLDING</div>", unsafe_allow_html=True)
        st.write(pg.get('hour_1', ''))
    with col_g2:
        st.markdown("<div class='stitch-card-title'>HOUR 2: REFLECTION & PEER EXCHANGE</div>", unsafe_allow_html=True)
        st.write(pg.get('hour_2', ''))
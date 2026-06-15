!pip install crewai pysqlite3-binary -qqq

# ==========================================
# 0. SYSTEM FIXES FOR VOCAREUM/JUPYTER
# ==========================================
# Swap out the outdated sqlite3 with the new one to prevent ChromaDB errors.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 

import os
import asyncio
from crewai import Agent, Task, Crew, Process, LLM
from textwrap import dedent

# ==========================================
# 1. SETUP NATIVE CREWAI GEMINI LLM
# ==========================================
# Set your Gemini API Key here
os.environ["GEMINI_API_KEY"] = "AQ.Ab8RN6LCyde2umCo25Bn-uFSlRVoPHBsp-cOouiq-cssEfm6iw"

# FIX: Added "-latest" to the model name to resolve the 404 Not Found error
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.2
)

# ==========================================
# 2. DEFINE AGENTS (Powered by Gemini)
# ==========================================

issue_agent = Agent(
    role="Order Issue Identification Agent",
    goal="Extract factual details from the customer query and assess sentiment.",
    backstory="You are the data-extraction frontline of Pureplate AgroFood support. You do not solve problems; you only structure raw text into clean, factual data points regarding orders, issues, and customer emotional state.",
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm
)

policy_agent = Agent(
    role="Policy Interpretation Agent",
    goal="Evaluate the extracted issue against Pureplate's exact return and refund policies.",
    backstory=dedent("""\
        You are the strict policy enforcer for Pureplate AgroFood. You apply rules without emotion.
        PUREPLATE POLICY RULES:
        - Eligible Returns: Damaged, broken, leaking, missing, incorrect, expired, or spoiled items. Seal broken at delivery.
        - Non-Returnable: Opened/used items for hygiene reasons. "Don't like the taste" is not valid.
        - Timeframes: Must report within 24 hours for fresh/highly perishable items. 48 hours for all other grocery/wellness products.
        - Evidence: Clear photos or videos of the issue are mandatory for damage/quality claims.
        """),
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm
)

resolution_agent = Agent(
    role="Resolution Recommendation Agent",
    goal="Draft a customer-friendly response based strictly on the policy verdict.",
    backstory="You are an empathetic customer support representative. You translate rigid policy verdicts (Eligible/Ineligible/Need Evidence) into polite, professional emails to the customer.",
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm
)

escalation_agent = Agent(
    role="Escalation Agent",
    goal="Determine if the ticket requires manual human intervention based on risk thresholds.",
    backstory="You are the Customer Support Manager. You review automated drafts. You mandate escalation if the customer threatens legal action, mentions health hazards (e.g., food poisoning), uses extreme profanity, or if the situation is highly urgent. You approve standard automated responses for routine in-policy or out-of-policy requests.",
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm
)

# ==========================================
# 3. DEFINE TASKS
# ==========================================

def create_tasks(user_query):
    identify_task = Task(
        description=f"Analyze this customer query: '{user_query}'. Extract Order ID, Item Name, Issue Type, Time Since Delivery, and assess Sentiment (Normal/Angry/Urgent) and Risk Level (Low/High).",
        expected_output="Structured text summarizing: Order ID, Item, Issue, Timeframe, Sentiment, Risk Level.",
        agent=issue_agent
    )

    policy_task = Task(
        description="Review the extracted facts. Determine if the request is IN POLICY, OUT OF POLICY, or NEEDS EVIDENCE based on the Pureplate rules. Cite the specific rule used.",
        expected_output="A verdict (Eligible, Ineligible, Needs Evidence) and the policy justification.",
        agent=policy_agent
    )

    resolution_task = Task(
        description="Read the policy verdict. Draft a polite response to the customer. If eligible, offer replacement/refund. If ineligible, politely explain why based on the rule. If needs evidence, ask for a photo.",
        expected_output="A drafted email response to the customer.",
        agent=resolution_agent
    )

    escalation_task = Task(
        description=f"Review the original query ('{user_query}') and the drafted response. Apply escalation logic: Escalate ONLY if Risk Level is High (legal threats, health hazards, extreme urgency). Do NOT escalate standard angry rejections. Output final JSON.",
        expected_output='JSON format: {"escalation_required": true/false, "escalation_reason": "string (or null)", "final_response": "The approved draft or a message saying a manager will contact them."}',
        agent=escalation_agent
    )
    
    return [identify_task, policy_task, resolution_task, escalation_task]

# ==========================================
# 4. EXECUTE THE WORKFLOW (ASYNC FOR VOCAREUM)
# ==========================================

async def process_query(query):
    try:
        print(f"\n[{'-'*10} PROCESSING NEW QUERY {'-'*10}]")
        print(f"USER INPUT: \"{query}\"\n")
        
        support_crew = Crew(
            agents=[issue_agent, policy_agent, resolution_agent, escalation_agent],
            tasks=create_tasks(query),
            process=Process.sequential
        )
        
        result = await support_crew.kickoff_async()
        
        print("\n--- FINAL SYSTEM OUTPUT ---")
        print(result)
        return result

    except Exception as e:
        print(f"Error processing workflow: {str(e)}")
        return '{"escalation_required": true, "escalation_reason": "System Error", "final_response": "We are experiencing technical difficulties."}'

# ==========================================
# 5. RUN SAMPLE TEST INPUTS
# ==========================================

async def run_tests():
    test_queries = [
        "My order #111 arrived 2 hours ago. The milk packet is leaking everywhere.",
        "I got my order #333 last week. Just opened the pulses and they are spoiled. Refund me.",
        "Order #555 arrived damaged. I threw it away already but I want a refund.",
        "You guys sent me expired baby food in Order #666! My child is sick, I am calling my lawyer immediately!"
    ]
    
    for q in test_queries:
        await process_query(q)

await run_tests()

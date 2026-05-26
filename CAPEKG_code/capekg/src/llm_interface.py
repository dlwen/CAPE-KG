"""
LLM Interface.

Thin wrapper around the OpenAI completion / chat APIs used by CAPE-KG for:
  - Multi-hop question decomposition (paper Appendix D, Table 4)
  - Factual sub-question answering when the KG cannot answer (Table 5)
  - Edit-relevant fallback that injects the edit triple into the LLM
    prompt context (Table 6, paper Sec. 3.3's Failure Stage)

API credentials are taken from the environment variable
``OPENAI_API_KEY``; the framework never hard-codes secrets.
"""

import os
import time

import openai


# ---------------------------------------------------------------------------
# OpenAI client setup
# ---------------------------------------------------------------------------

def configure_openai(api_key_env="OPENAI_API_KEY",
                    api_base="https://api.openai.com/v1"):
    """Set the OpenAI client credentials from environment."""
    openai.api_key = os.environ.get(api_key_env, "")
    openai.api_base = api_base


# ---------------------------------------------------------------------------
# Output sanity checks
# ---------------------------------------------------------------------------

def is_bad_llm_output(text):
    """Reject obviously-broken or echo-style LLM outputs that should trigger
    a retry or a fallback path."""
    if text is None:
        return True
    t = text.strip()
    if not t:
        return True
    if "[INST]" in t and not any(c.isalnum() for c in t):
        return True
    low = t.lower()
    if any(p in low for p in ("i am unable", "i cannot", "unknown",
                              "not specified", "i don't know")):
        return True
    return False


# ---------------------------------------------------------------------------
# Decomposition (paper Appendix D, Table 4)
# ---------------------------------------------------------------------------

def run_llm_divide(prompt, model="gpt-3.5-turbo-instruct", max_retries=3):
    """Decompose a multi-hop question using GPT with a KNN few-shot prompt."""
    for attempt in range(max_retries):
        try:
            if model == "gpt-3.5-turbo-instruct":
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.0,
                    top_p=1.0,
                    stop=["\n\nQuestion:", "\n\n"],
                )
                return response.choices[0].text.strip()
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system",
                         "content": "You decompose multi-hop questions into "
                                    "sub-questions. Use [ENT] as a placeholder "
                                    "for entities passed between sub-questions."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=200,
                    temperature=0.0,
                )
                return response.choices[0].message.content.strip()
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(3)
    return "What is the answer?\nAnswer: [ENT]"


# ---------------------------------------------------------------------------
# Sub-question answering (paper Appendix D, Tables 5 and 6)
# ---------------------------------------------------------------------------

def run_llm_answer(prompt, model="gpt-3.5-turbo-instruct", max_retries=3):
    """Answer a single sub-question. The prompt is expected to follow
    Table 5 (factual fallback) or Table 6 (edit-relevant fallback with the
    edit triple injected into context)."""
    for attempt in range(max_retries):
        try:
            if model == "gpt-3.5-turbo-instruct":
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=64,
                    temperature=0.0,
                    top_p=1.0,
                )
                return response.choices[0].text.strip()
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system",
                         "content": "You answer factual questions with short, "
                                    "specific entity names. Provide only the "
                                    "essential answer without explanation."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=64,
                    temperature=0.0,
                )
                return response.choices[0].message.content.strip()
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
    return "Unknown"


def gpt_select_entity_from_candidates(question, candidates,
                                      model="gpt-3.5-turbo-instruct"):
    """Low-confidence stage of progressive retrieval (paper Sec. 3.3).

    When every entity candidate's score gphi(q, k) falls below the
    threshold tau, ask the LLM to pick the most plausible candidate from
    the original pool K.
    """
    prompt = (
        "Select the most relevant entity for this question:\n\n"
        f"Question: {question}\n\nCandidates:\n"
        + "\n".join(f"{i}. {c}" for i, c in enumerate(candidates, 1))
        + "\n\nReturn only the entity name:"
    )
    try:
        if model == "gpt-3.5-turbo-instruct":
            response = openai.Completion.create(
                model=model, prompt=prompt, max_tokens=30,
                temperature=0.1, stop=["\n"],
            )
            result = response.choices[0].text.strip()
        else:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "Select the most relevant entity from candidates."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=30, temperature=0.1,
            )
            result = response.choices[0].message.content.strip()
    except Exception:
        return candidates[0] if candidates else None

    for c in candidates:
        if c.lower() in result.lower() or result.lower() in c.lower():
            return c
    return candidates[0] if candidates else None

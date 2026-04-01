# synonyms.py
# Purpose: Query expansion for police policies + Nebraska statutes (Chapter 28)
# Usage:
#   from synonyms import expand_query
#   expanded = expand_query("non-emergency call response policy", max_expansions_per_key=6)

import re
from typing import Dict, List, Tuple

# -----------------------------
# SYNONYM DICTIONARY
# -----------------------------
SYNONYMS: Dict[str, List[str]] = {
    #  Call Types & Response
    "non_emergency": [
        "non-emergency",
        "non emergency",
        "routine call",
        "service call",
        "priority 3",
        "low priority call",
        "delayed response",
        "administrative call",
    ],
    "emergency": [
        "emergency",
        "emergency call",
        "priority 1",
        "priority 2",
        "exigent circumstance",
        "immediate response",
        "urgent call",
        "in-progress call",
    ],

    #  Officer Actions & Duties
    "officer_response": [
        "officer response",
        "deputy response",
        "law enforcement response",
        "police response",
        "response protocol",
        "response procedure",
        "response requirement",
    ],
    "duty_to_act": [
        "duty to act",
        "obligation to respond",
        "official duty",
        "law enforcement duty",
        "scope of duty",
    ],

    #  Policies & Procedures
    "policy": [
        "policy",
        "policies",
        "department policy",
        "sheriff policy",
        "agency policy",
        "administrative policy",
        "general order",
        "sop",
        "standard operating procedure",
        "directive",
        "guideline",
    ],
    "procedure": [
        "procedure",
        "procedures",
        "protocol",
        "process",
        "operational steps",
        "required steps",
    ],

    # Legal & Statutory Language (Chapter 28)
    "statute": [
        "statute",
        "statutes",
        "nebraska statute",
        "state law",
        "criminal code",
        "chapter 28",
        "revised statute",
    ],
    "crime": [
        "crime",
        "criminal offense",
        "offense",
        "unlawful act",
        "violation",
    ],
    "felony": [
        "felony",
        "felony offense",
        "criminal felony",
        "class i felony",
        "class ii felony",
        "class iii felony",
        "class iv felony",
        "class ia felony",
        "class ib felony",
        "class ic felony",
        "class id felony",
        "class iia felony",
        "class iiia felony",
    ],
    "misdemeanor": [
        "misdemeanor",
        "minor offense",
        "lower-level offense",
    ],

    # Arrest, Detention & Force
    "arrest": [
        "arrest",
        "custodial arrest",
        "take into custody",
        "apprehension",
        "lawful arrest",
    ],
    "detention": [
        "detention",
        "detain",
        "investigative detention",
        "temporary detention",
        "terry stop",
        "stop and frisk",
    ],
    "use_of_force": [
        "use of force",
        "force",
        "reasonable force",
        "necessary force",
        "physical force",
        "force continuum",
    ],

    # Evidence & Documentation
    "evidence": [
        "evidence",
        "physical evidence",
        "digital evidence",
        "electronic evidence",
        "seized evidence",
        "collected evidence",
    ],
    "body_camera": [
        "body camera",
        "body-worn camera",
        "bwc",
        "body cam",
        "recording device",
    ],

    # Reports & Documentation
    "report": [
        "report",
        "police report",
        "incident report",
        "offense report",
        "case report",
        "narrative",
    ],
    "documentation": [
        "documentation",
        "record",
        "records",
        "official record",
        "case file",
    ],

    # Jurisdiction & Authority
    "jurisdiction": [
        "jurisdiction",
        "authority",
        "legal authority",
        "law enforcement authority",
        "county jurisdiction",
        "territorial jurisdiction",
    ],
    "probable_cause": [
        "probable cause",
        "reasonable grounds",
        "legal cause",
        "sufficient cause",
    ],

    #  Legal Standards
    "reasonable_suspicion": [
        "reasonable suspicion",
        "articulable suspicion",
        "lawful suspicion",
    ],
    "exigent_circumstances": [
        "exigent circumstances",
        "emergency conditions",
        "urgent necessity",
    ],

    #  Court & Legal Outcomes
    "charge": [
        "charge",
        "charges",
        "criminal charge",
        "filing charges",
        "offense charged",
    ],
    "conviction": [
        "conviction",
        "guilty finding",
        "adjudication",
    ],
    "sentence": [
        "sentence",
        "sentencing",
        "punishment",
        "penalty",
    ],

    #  Common policy/statute words that cause misses (extra helpful)
    "non_emergency_response": [
        "response time",
        "time to respond",
        "dispatch",
        "call for service",
        "cfs",
        "non emergent",
        "nonurgent",
    ],
    "emergency_response": [
        "lights and sirens",
        "code 3",
        "code 2",
        "hot call",
        "priority dispatch",
        "urgent response",
    ],
    "search_and_seizure": [
        "search",
        "search warrant",
        "warrant",
        "consent search",
        "probation search",
        "seizure",
        "plain view",
    ],
    "traffic": [
        "traffic stop",
        "vehicle stop",
        "motorist stop",
        "moving violation",
        "citation",
        "warning",
    ],
    "domestic": [
        "domestic violence",
        "dv",
        "family disturbance",
        "intimate partner violence",
        "protection order",
    ],
    "assault": [
        "assault",
        "battery",
        "attack",
        "strike",
        "physical assault",
    ],
    "theft": [
        "theft",
        "larceny",
        "shoplifting",
        "stolen property",
        "property theft",
    ],
    "burglary": [
        "burglary",
        "breaking and entering",
        "unlawful entry",
    ],
    "robbery": [
        "robbery",
        "strong-arm robbery",
        "armed robbery",
    ],
    "weapons": [
        "weapon",
        "firearm",
        "gun",
        "knife",
        "deadly weapon",
    ],
    "controlled_substances": [
        "controlled substance",
        "drug",
        "narcotic",
        "meth",
        "cocaine",
        "opioid",
        "possession",
        "distribution",
    ],
}

# -----------------------------
# BUILD TRIGGERS (simple + effective)
# - If the user query contains one of these triggers, we expand with that key's synonyms.
# - You can add or remove triggers anytime.
# -----------------------------
TRIGGERS: List[Tuple[str, str]] = [
    # key, trigger substring
    ("non_emergency", "non-emergency"),
    ("non_emergency", "non emergency"),
    ("non_emergency", "routine"),
    ("non_emergency", "service call"),
    ("non_emergency", "priority 3"),
    ("emergency", "emergency"),
    ("emergency", "priority 1"),
    ("emergency", "priority 2"),
    ("emergency", "in-progress"),
    ("emergency", "exigent"),
    ("officer_response", "response"),
    ("officer_response", "respond"),
    ("officer_response", "dispatch"),
    ("policy", "policy"),
    ("policy", "sop"),
    ("policy", "general order"),
    ("procedure", "procedure"),
    ("procedure", "protocol"),
    ("statute", "statute"),
    ("statute", "chapter 28"),
    ("crime", "crime"),
    ("felony", "felony"),
    ("misdemeanor", "misdemeanor"),
    ("arrest", "arrest"),
    ("detention", "detain"),
    ("detention", "detention"),
    ("use_of_force", "force"),
    ("evidence", "evidence"),
    ("evidence", "electronic evidence"),
    ("body_camera", "body"),
    ("body_camera", "camera"),
    ("body_camera", "bwc"),
    ("report", "report"),
    ("documentation", "documentation"),
    ("jurisdiction", "jurisdiction"),
    ("probable_cause", "probable cause"),
    ("reasonable_suspicion", "reasonable suspicion"),
    ("search_and_seizure", "search"),
    ("search_and_seizure", "warrant"),
    ("traffic", "traffic"),
    ("traffic", "citation"),
    ("domestic", "domestic"),
    ("assault", "assault"),
    ("theft", "theft"),
    ("burglary", "burglary"),
    ("robbery", "robbery"),
    ("weapons", "weapon"),
    ("weapons", "firearm"),
    ("controlled_substances", "drug"),
    ("controlled_substances", "controlled substance"),
]

# -----------------------------
# HELPERS
# -----------------------------
def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def expand_query(
    query: str,
    max_expansions_per_key: int = 8,
    include_keys: bool = False,
) -> str:
    """
    Expand a user's query with relevant synonyms to improve retrieval.

    Args:
        query: user query
        max_expansions_per_key: limit synonyms appended per matched key
        include_keys: if True, also appends the key name itself (sometimes helps)

    Returns:
        A single expanded string query.
    """
    q_norm = _normalize(query)

    matched_keys = set()
    for key, trig in TRIGGERS:
        if trig in q_norm:
            matched_keys.add(key)

    # If nothing matched, still return original query (no harm)
    if not matched_keys:
        return query

    expansions: List[str] = []
    for key in sorted(matched_keys):
        if include_keys:
            expansions.append(key.replace("_", " "))
        expansions.extend(SYNONYMS.get(key, [])[:max_expansions_per_key])

    # Deduplicate expansions while preserving order
    seen = set()
    uniq = []
    for e in expansions:
        e2 = _normalize(e)
        if e2 and e2 not in seen:
            seen.add(e2)
            uniq.append(e)

    # Return expanded query (original + expansions)
    # NOTE: This is simple lexical expansion; embeddings still do most of the work.
    return query + " " + " ".join(uniq)

def debug_matched_keys(query: str) -> List[str]:
    """Return which synonym keys would be triggered for a given query (for logging/debug)."""
    q_norm = _normalize(query)
    matched = set()
    for key, trig in TRIGGERS:
        if trig in q_norm:
            matched.add(key)
    return sorted(matched)
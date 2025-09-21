from typing import Any, Dict, List, Literal, Optional, Tuple


DEFAULT_ENTITIES = Literal[
    "PRODUCT",
    "MARKET",
    "TECHNOLOGY",
    "EVENT",
    "CONCEPT",
    "ORGANIZATION",
    "PERSON",
    "LOCATION",
    "TIME",
    "MISCELLANEOUS",
]

DEFAULT_RELATIONS = Literal[
    "USED_BY",
    "USED_FOR",
    "LOCATED_IN",
    "PART_OF",
    "WORKED_ON",
    "HAS",
    "IS_A",
    "BORN_IN",
    "DIED_IN",
    "HAS_ALIAS",
]

# Convert the above dict schema into a list of triples
Triple = Tuple[str, str, str]
DEFAULT_VALIDATION_SCHEMA: List[Triple] = [
    ("PRODUCT", "USED_BY", "PRODUCT"),
    ("PRODUCT", "USED_FOR", "MARKET"),
    ("PRODUCT", "HAS", "TECHNOLOGY"),
    ("MARKET", "LOCATED_IN", "LOCATION"),
    ("MARKET", "HAS", "TECHNOLOGY"),
    ("TECHNOLOGY", "USED_BY", "PRODUCT"),
    ("TECHNOLOGY", "USED_FOR", "MARKET"),
    ("TECHNOLOGY", "LOCATED_IN", "LOCATION"),
    ("TECHNOLOGY", "PART_OF", "ORGANIZATION"),
    ("TECHNOLOGY", "IS_A", "PRODUCT"),
    ("EVENT", "LOCATED_IN", "LOCATION"),
    ("EVENT", "PART_OF", "ORGANIZATION"),
    ("CONCEPT", "USED_BY", "TECHNOLOGY"),
    ("CONCEPT", "USED_FOR", "PRODUCT"),
    ("ORGANIZATION", "LOCATED_IN", "LOCATION"),
    ("ORGANIZATION", "PART_OF", "ORGANIZATION"),
    ("ORGANIZATION", "PART_OF", "MARKET"),
    ("PERSON", "BORN_IN", "LOCATION"),
    ("PERSON", "BORN_IN", "TIME"),
    ("PERSON", "DIED_IN", "LOCATION"),
    ("PERSON", "DIED_IN", "TIME"),
    ("PERSON", "WORKED_ON", "EVENT"),
    ("PERSON", "WORKED_ON", "PRODUCT"),
    ("PERSON", "WORKED_ON", "CONCEPT"),
    ("PERSON", "WORKED_ON", "TECHNOLOGY"),
    ("LOCATION", "LOCATED_IN", "LOCATION"),
    ("LOCATION", "PART_OF", "LOCATION"),
]


DEFAULT_SCHEMA_PATH_EXTRACT_PROMPT = (
    "Give the following text, extract the knowledge graph according to the provided schema. "
    "Try to limit to the output {max_triplets_per_chunk} extracted paths.s\n"
    "-------\n"
    "{text}\n"
    "-------\n"
)


DEFAULT_RESPONSE_TEMPLATE = (
    "Generated Cypher query:\n{query}\n\nCypher Response:\n{response}"
)

DEFAULT_SUMMARY_TEMPLATE = (
    """You are an assistant that helps to form nice and human understandable answers.
        The information part contains the provided information you must use to construct an answer.
        The provided information is authoritative, never doubt it or try to use your internal knowledge to correct it.
        If the provided information is empty, say that you don't know the answer.
        Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
        Here is an example:

        Question: How many miles is the flight between the ANC and SEA airports?
        Information:
        [{"r.dist": 1440}]
        Helpful Answer:
        It is 1440 miles to fly between the ANC and SEA airports.

        Follow this example when generating answers.
        Question:
        {question}
        Information:
        {context}
        Helpful Answer:"""
)


DEFAULT_CYPHER_TEMPALTE_STR = """
Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}
"""


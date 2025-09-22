

DEFAULT_SUMMARY_QUERY_STR = """\
What is this table about? Give a very concise summary (imagine you are adding a new caption and summary for this table), \
and output the real/existing table title/caption if context provided.\
and output the real/existing table id if context provided.\
and also output whether or not the table should be kept.\
"""


DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)


DEFAULT_REFINE_PROMPT_TMPL = (
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the query. "
    "If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)


PREFIX = """\
Your goal is to structure the user's query to match the request schema provided below.

<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the \
following schema:

{schema_str}

The query string should contain only text that is expected to match the contents of \
documents. Any conditions in the filter should not be mentioned in the query as well.

Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters take into account the descriptions of attributes.
Make sure that filters are only used as needed. If there are no filters that should be \
applied return [] for the filter value.\

If the user's query explicitly mentions number of documents to retrieve, set top_k to \
that number, otherwise do not set top_k.

"""

EXAMPLES = f"""\
<< Example 1. >>
Data Source:
```json
{example_info.model_dump_json(indent=4)}
```

User Query:
{example_query}

Structured Request:
```json
{example_output.model_dump_json()}


<< Example 2. >>
Data Source:
```json
{example_info_2.model_dump_json(indent=4)}
```

User Query:
{example_query_2}

Structured Request:
```json
{example_output_2.model_dump_json()}

```
""".replace("{", "{{").replace("}", "}}")


SUFFIX = """
<< Example 3. >>
Data Source:
```json
{info_str}
```

User Query:
{query_str}

Structured Request:
"""

DEFAULT_VECTOR_STORE_QUERY_PROMPT_TMPL = PREFIX + EXAMPLES + SUFFIX



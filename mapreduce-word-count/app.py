import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

NUM_REDUCERS = 2


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)


def mapper(text: str) -> list[tuple[str, int]]:
    return [(word, 1) for word in tokenize(text)]


def combine(pairs: list[tuple[str, int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for word, count in pairs:
        counts[word] = counts.get(word, 0) + count
    return counts


def partitioner(word: str, num_reducers: int) -> int:
    return sum(ord(character) for character in word) % num_reducers


def top_k_words(counts: dict[str, int], k: int) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:k]


st.set_page_config(page_title="MapReduce Word Count", page_icon="🔎", layout="wide")
st.title("MapReduce Word Count")
st.write(
    "Upload one or more `.txt` files to follow each step of a word-count "
    "MapReduce job: map, combine, partition, shuffle and sort, reduce, and merge."
)

uploaded_files = st.file_uploader(
    "Choose text files",
    type=["txt"],
    accept_multiple_files=True,
)
st.caption(f"This MapReduce job uses {NUM_REDUCERS} reducers.")

if not uploaded_files:
    st.info("Upload one or more UTF-8 text files to start.")
    st.stop()

documents: list[tuple[str, str]] = []
for uploaded_file in uploaded_files:
    try:
        text = uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError as error:
        st.error(
            f"{uploaded_file.name} is not valid UTF-8 text "
            f"(decode error at byte {error.start}). Please upload a UTF-8 .txt file."
        )
        st.stop()
    documents.append((uploaded_file.name, text))

st.header("1. Input splits")
st.caption(f"{len(documents)} uploaded file(s); each file is one input split.")
input_rows = [
    {
        "Split": index,
        "File": name,
        "Characters": len(text),
        "Tokens": len(tokenize(text)),
    }
    for index, (name, text) in enumerate(documents, start=1)
]
st.dataframe(input_rows, width="stretch", hide_index=True)
with st.expander("View uploaded text"):
    for index, (name, text) in enumerate(documents, start=1):
        st.markdown(f"**Split {index}: {name}**")
        st.text(text if text else "(empty file)")

st.header("2. Map")
with ThreadPoolExecutor() as executor:
    mapper_outputs = list(executor.map(lambda document: mapper(document[1]), documents))

mapped_rows = [
    {"Split": name, "Word": word, "Value": count}
    for (name, _), pairs in zip(documents, mapper_outputs)
    for word, count in pairs
]
st.write(
    f"The mapper emitted **{len(mapped_rows):,}** `(word, 1)` pairs. "
    "Each pair represents one occurrence."
)
with st.expander("View all mapper output pairs"):
    st.dataframe(mapped_rows, width="stretch", hide_index=True)

st.header("3. Combine")
mapper_counts = [combine(pairs) for pairs in mapper_outputs]
combined_rows = [
    {"Split": name, "Word": word, "Subtotal": count}
    for (name, _), counts in zip(documents, mapper_counts)
    for word, count in sorted(counts.items())
]
st.write(
    f"The combiner reduced the mapper output to **{len(combined_rows):,}** "
    "per-file word subtotals."
)
with st.expander("View per-split combined subtotals"):
    st.dataframe(combined_rows, width="stretch", hide_index=True)

st.header("4. Partition, shuffle, and sort")
reducer_inputs: list[dict[str, list[int]]] = [
    defaultdict(list) for _ in range(NUM_REDUCERS)
]
for counts in mapper_counts:
    for word, subtotal in counts.items():
        reducer_index = partitioner(word, NUM_REDUCERS)
        reducer_inputs[reducer_index][word].append(subtotal)

sorted_inputs = [sorted(reducer_input.items()) for reducer_input in reducer_inputs]
for reducer_index, items in enumerate(sorted_inputs):
    with st.expander(f"Reducer {reducer_index} receives {len(items)} unique word(s)"):
        st.dataframe(
            [
                {
                    "Word": word,
                    "Subtotals from splits": ", ".join(map(str, subtotals)),
                }
                for word, subtotals in items
            ],
            width="stretch",
            hide_index=True,
        )

st.header("5. Reduce")
reducer_results = [
    {word: sum(subtotals) for word, subtotals in items}
    for items in sorted_inputs
]
for reducer_index, result in enumerate(reducer_results):
    with st.expander(
        f"Reducer {reducer_index} output ({len(result)} unique word(s))"
    ):
        st.dataframe(
            [{"Word": word, "Count": count} for word, count in result.items()],
            width="stretch",
            hide_index=True,
        )

st.header("6. Merge reducer outputs")
all_counts = {
    word: count
    for reducer_result in reducer_results
    for word, count in reducer_result.items()
}
sorted_counts = sorted(all_counts.items())
output_text = "".join(f"{word}\t{count}\n" for word, count in sorted_counts)
st.write(
    f"Combined **{len(sorted_counts):,}** unique words and "
    f"**{sum(all_counts.values()):,}** total word occurrences."
)
st.dataframe(
    [{"Word": word, "Count": count} for word, count in sorted_counts],
    width="stretch",
    hide_index=True,
)

st.header("7. Top-K words")
max_k = max(1, len(all_counts))
k = st.number_input(
    "Choose how many top words to display (K)",
    min_value=1,
    max_value=max_k,
    value=min(10, max_k),
    step=1,
)
top_k_counts = top_k_words(all_counts, int(k))
if top_k_counts:
    st.dataframe(
        [
            {"Rank": rank, "Word": word, "Count": count}
            for rank, (word, count) in enumerate(top_k_counts, start=1)
        ],
        width="stretch",
        hide_index=True,
    )
else:
    st.info("No words were found in the uploaded files.")

download_columns = st.columns(NUM_REDUCERS + 1)
download_columns[0].download_button(
    "Download output.txt",
    data=output_text,
    file_name="output.txt",
    mime="text/plain",
)
for reducer_index, result in enumerate(reducer_results):
    reducer_text = "".join(
        f"{word}\t{count}\n" for word, count in sorted(result.items())
    )
    column = download_columns[(reducer_index + 1) % len(download_columns)]
    column.download_button(
        f"Download reducer_{reducer_index}.txt",
        data=reducer_text,
        file_name=f"reducer_{reducer_index}.txt",
        mime="text/plain",
        key=f"download-reducer-{reducer_index}",
    )

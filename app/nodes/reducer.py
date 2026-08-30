from app.graphs.blog.states.state import State


def reducer_node(llm):

    from pathlib import Path

    def reducer(state: State) -> dict:
        title = state["plan"].title
        body = "\n\n".join(state["sections"]).strip()

        final_md = f"# {title}\n\n{body}\n"

        # ---- save to file ----
        filename = title.lower().replace(" ", "_") + ".md"
        output_path = Path(filename)
        output_path.write_text(final_md, encoding="utf-8")

        return {"final": final_md}

    return reducer
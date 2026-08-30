from langchain_core.messages import SystemMessage, HumanMessage
def worker_node(llm):

    def worker(payload: dict , ) -> dict:

        # payload contains what we sent
        task = payload["task"]
        topic = payload["topic"]
        plan = payload["plan"]
        blog_title = plan.title
        section_md = llm.invoke(
            [
                SystemMessage(content="Write one clean Markdown section."),
                HumanMessage(
                    content=(
                        f"Blog: {blog_title}\n"
                        f"Topic: {topic}\n\n"
                        f"Section: {task.title}\n"
                        f"Brief: {task.description}\n\n"
                        "Return only the section content in Markdown."
                    )
                ),
            ]
        ).content.strip()

        return {"sections": [section_md]}

    return worker
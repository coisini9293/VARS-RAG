"""Chain-of-thought style instruction block (L4 / L6)."""

COT_INSTRUCTION_BLOCK = (
    "Follow these steps, then give the final label on the last line after 'Final:'.\n"
    "1) Describe what happens in the video.\n"
    "2) Decide if it is a foul under typical football rules.\n"
    "3) Choose Offence, No offence, or Between.\n"
    "Final: <label>\n\n"
)


def prepend_cot(user_message: str) -> str:
    return COT_INSTRUCTION_BLOCK + user_message

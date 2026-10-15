"""Exhibit-evidence explainer (standalone).

Given an auditor's exception description and the auditor's Excel exhibits, produce a JSON that
explains what each attachment demonstrates and pinpoints the specific figures/columns the
auditor likely used to arrive at the exception reasoning and questioned amount.

Self-contained: depends only on PyMuPDF, openpyxl, langchain-openai, pydantic, python-dotenv.
No dependency on the agentic_audit project.
"""

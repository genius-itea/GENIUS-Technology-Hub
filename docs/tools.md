# Tools

This page introduces the tools developed within the GENIUS project.  
Each tool is presented with a short description and contact information.
Additional technical details can be expanded when needed.

---

## Overview

| Tool | Summary | WP | SDLC | TRL | Contacts | Links |
|----|-------|---|----|---|--------|-----|
| [AI Help Assistant](#ai-help-assistant) | Multi-agent AI project help assistant | WP4 | Development | 4 | kamer@dakikyazilim.com,tim@iotiq.de |  |
| [Automated issue tagging](#automated-issue-tagging) | Automated issue tagging by similarity | WP5 | Testing | 6 | valeria.villa@barco.com | N/A |
| [Chat Agent](#chat-agent) | AI powered chat application | WP4 (Task 4.2) | Development | 3 | yuriy@vaadin.com | Link Repo |
| [CoReGraph](#coregraph) | A Knowledge Graph Approach for Software Repository Analysis | WP4 | Development | TRL 4 - TRL5 | egm@isep.ipp.pt | to appear |
| [DETANGLE Architecture Modernization](#detangle-architecture-modernization) | Architecture Refactoring | WP4 | Development, Maintenance | 5 to 6 | wuchner@capeofgoodcode.com | Slides presented at the 1st Plenary and the 2nd Plenary |
| [Diffblue Cover (with LLM test data generation)](#diffblue-cover-with-llm-test-data-generation) | Regression unit test generator | WP5 | Testing | 4 | peter.schrammel@diffblue.com | [https://cover-docs.diffblue.com/features/cover-cli/environment-configuration/llm-configuration](https://cover-docs.diffblue.com/features/cover-cli/environment-configuration/llm-configuration) |
| [Diffblue Cover MCP](#diffblue-cover-mcp) | Actioning testability issue findings | WP5 | Testing, Maintenance | 6 | peter.schrammel@diffblue.com | [https://cover-docs.diffblue.com/features/cover-mcp-server-beta/getting-started-cover-mcp-beta](https://cover-docs.diffblue.com/features/cover-mcp-server-beta/getting-started-cover-mcp-beta) |
| [Diffblue Cover Moderne Recipe](#diffblue-cover-moderne-recipe) | Unit test generation for refactoring | WP4 | Maintenance | 6 | peter.schrammel@diffblue.com | N/A |
| [Diffblue Test Quality Agent](#diffblue-test-quality-agent) | Test quality assessment | WP5 | Testing | 6 | peter.schrammel@diffblue.com | [https://docs.diffblue.com/workflows/test-quality-report](https://docs.diffblue.com/workflows/test-quality-report) |
| [Diffblue Testing Agent](#diffblue-testing-agent) | Regression unit test generator | WP5 | Testing | 7 | peter.schrammel@diffblue.com | [https://docs.diffblue.com/workflows/regression-unit-tests](https://docs.diffblue.com/workflows/regression-unit-tests) |
| [HARE-SM Framework / RE Assistant](#hare-sm-framework-re-assistant) | Human-AI requirements engineering assistant | WP1 | Requirements Engineering / Analysis | Proof of Concept | mateen.a.abbasi@jyu.fi | [https://re-assistant.streamlit.app](https://re-assistant.streamlit.app) |
| [LESS Guaidance](#less-guaidance) | LESS-Based Requirement and Test Case Generation | WP3 | Requirement Engineering |  | abhishek.shrestha@fokus.fraunhofer.de | [https://github.com/Abhishek2271/LESSGuidance](https://github.com/Abhishek2271/LESSGuidance) |
| [MARTA](#marta) | A Decoupled Multi-Agent Architecture for Python Test Generation | WP5 | Developing/testing | TRL 4 - TRL5 | egm@isep.ipp.pt | to appear |
| [ReGEN Tool](#regen-tool) | AI-driven requirement engineering tool | WP3 | Requirements Engineering / Analysis | 4 | kamer@dakikyazilim.com, tim@iotiq.de |  |
| [RestifAI](#restifai) | RestAPI Test Generator | WP5 (Task 5.1 + 5.2) | Testing | 3 | Maximilian.Ehrhart@casablanca.a | [Link Repo](https://github.com/casablancahotelsoftware/RESTifAI) • [Link Paper](https://arxiv.org/abs/2512.08706) |
| [TDD Orchestrator](#tdd-orchestrator) | Test driven code generator | WP4 | Development, Maintenance | Proof of Concept | pyry.kotilainen@jyu.fi | [https://arxiv.org/abs/2604.26615](https://arxiv.org/abs/2604.26615) |
| [TWEASE](#twease) | Test with ease | WP5 | Testing | 3 | andreas.dreschinski@akkodis.com | N/A |

---

## AI Help Assistant
Multi-agent AI help assistant is built to provide intelligent support for project-specific queries. It utilizes an agentic architecture consisting of Input, Controller, Domain, Ranking, Response, and History agents to process requests. It employs RAG to synthesize information from project documentation. Monitored via Langfuse and deployed using Docker, it leverages OpenAI models to offer natural, high-quality technical assistance.  

**Contact:** `kamer@dakikyazilim.com,tim@iotiq.de`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP4

**Approach**  
LLM based workflow , RAG, Agents

**SDLC Stage**  
Development

**TRL**  
4

**Use Cases**  
Dakik, Iotiq

**Links**  
To be added

</details>

---

## Automated issue tagging
A failed testrun must be tagged with a jira ticket. Multiple failed testruns frequently have the same root cause and must be linked to the same jira ticket.
Nowadays, a tester manually checks if there is already a jira ticket that can be linked to the failed test, what is taking a lot of time. If no ticket is found, the tester creates a new one.
This tool automates this process by using semantic similarity to check if there is a historical, already-tagged failed testrun that is similar to the failed testrun and tags the found jira ticket to the failed testrun.  

**Contact:** `valeria.villa@barco.com`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP5

**Approach**  
LLM based workflow, automation

**SDLC Stage**  
Testing

**TRL**  
6

**Use Cases**  
Barco

**Links**  
N/A

</details>

---

## Chat Agent
A standalone AI-powered chat application that understands the context and scope of user projects, enabling natural language interactions to modify existing components or create new parts of an application. Built on a modular architecture, it can be easily extended with new features, such as support for third-party frameworks and libraries.  

**Contact:** `yuriy@vaadin.com`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP4 (Task 4.2)

**Approach**  
LLM based workflow

**SDLC Stage**  
Development

**TRL**  
3

**Use Cases**  
Vaadin

**Links**  
Link Repo

</details>

---

## CoReGraph
extended repository knowledge graph architecture that integrates
technical, organizational, temporal, and provenance-aware context from software repositories.  

**Contact:** `egm@isep.ipp.pt`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP4

**Approach**  
LLM and KG

**SDLC Stage**  
Development

**TRL**  
TRL 4 - TRL5

**Use Cases**  
Loop

**Links**  
to appear

</details>

---

## DETANGLE Architecture Modernization
We aim to restructure/modernize SW systems with no/minimal human support by providing automated decomposition suggestions based on classical and new AI methods:
- Preparing/harmonizing data: Curating & aggregating data from requirements, issue trackers, repositories, and DevOps toolchains.
- Contextual linking: Synthesizing disparate data points into a unified view of system health and in parallel assigning software features to business domains as a target the modernized architecture should reflect.
- Domain-driven partitioning: Simulating and evaluating how to decompose monolithic codebases into modular services based on business domains
- Automated transition: Executing the initial, critical transformation steps to bridge the gap between current state and future architecture.
- Best practice of DDD: Including best practices of Domain-driven Design (DDD) principles and patterns into the transformation steps.  

**Contact:** `wuchner@capeofgoodcode.com`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP4

**Approach**  
LLM based workflow

**SDLC Stage**  
Development, Maintenance

**TRL**  
5 to 6

**Use Cases**  
Cape of Good Code

**Links**  
Slides presented at the 1st Plenary and the 2nd Plenary

</details>

---

## Diffblue Cover (with LLM test data generation)
An extension to the Diffblue Cover tool that allows you to connect it to your LLM. The LLM Is then used to generate context-sensitive test data during unit test generation. The user benefit is to have more relevant test data in the generated unit tests. Requires a Diffblue Cover license (trial licenses available from the website) and an LLM (e.g. OpenAI token) to use.  

**Contact:** `peter.schrammel@diffblue.com`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP5

**Approach**  
RL, mixed tool/LLM based workflow

**SDLC Stage**  
Testing

**TRL**  
4

**Links**  
[https://cover-docs.diffblue.com/features/cover-cli/environment-configuration/llm-configuration](https://cover-docs.diffblue.com/features/cover-cli/environment-configuration/llm-configuration)

</details>

---

## Diffblue Cover MCP
An MCP server to integrate Diffblue Cover with your AI coding agent, in particular for resolving build system configuration and testability issues detected by Diffblue Cover. The user benefit is to automate fixing the build system wrt unit testing configuration as well as improving the testability of the codebase where Diffblue Cover was not able to write tests. Requires a Diffblue Cover license  (trial licenses available from the website) and an AI coding agent (e.g. Github Copilot) to use.  

**Contact:** `peter.schrammel@diffblue.com`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP5

**Approach**  
MCP, mixed tool/LLM based workflow

**SDLC Stage**  
Testing, Maintenance

**TRL**  
6

**Links**  
[https://cover-docs.diffblue.com/features/cover-mcp-server-beta/getting-started-cover-mcp-beta](https://cover-docs.diffblue.com/features/cover-mcp-server-beta/getting-started-cover-mcp-beta)

</details>

---

## Diffblue Cover Moderne Recipe
This is a recipe for the Moderne/OpenRewrite refactoring framework for writing unit tests across a codebase. It forms part of a verified refactoring workflow: first generate unit tests, then run refactoring/upgrade recipes, and finally run the tests again to detect anything that may have been broken.  

**Contact:** `peter.schrammel@diffblue.com`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP4

**Approach**  
RL

**SDLC Stage**  
Maintenance

**TRL**  
6

**Links**  
N/A

</details>

---

## Diffblue Test Quality Agent
A tool that provides a fully automated workflow for performing coverage and mutation analysis on Java and Python code bases. It autonomously configures the analysis tools in the project's build system and performs the measurements without any user interaction. The user benefit is to receive an assessment of the bug-catching ability of their code base without having to run brittle mutation testing tools manually. Requires a Diffblue Agents license  (trial licenses available from the website) and an AI coding agent (e.g. Claude Code) to use.  

**Contact:** `peter.schrammel@diffblue.com`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP5

**Approach**  
mixed tool/LLM based workflow

**SDLC Stage**  
Testing

**TRL**  
6

**Links**  
[https://docs.diffblue.com/workflows/test-quality-report](https://docs.diffblue.com/workflows/test-quality-report)

</details>

---

## Diffblue Testing Agent
A tool that provides a fully automated workflow for writing regression unit tests for entire Java and Python code bases. It autonomously configures the project's build system for unit testing and fills in missing unit tests across the entire codebase without any user interaction. The user benefit is to augment regression unit tests completely automatically without having to babysit an AI coding agent for hours and days. Requires a Diffblue Agents license  (trial licenses available from the website) and an AI coding agent (e.g. Claude Code) to use.  

**Contact:** `peter.schrammel@diffblue.com`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP5

**Approach**  
mixed tool/LLM based workflow

**SDLC Stage**  
Testing

**TRL**  
7

**Links**  
[https://docs.diffblue.com/workflows/regression-unit-tests](https://docs.diffblue.com/workflows/regression-unit-tests)

</details>

---

## HARE-SM Framework / RE Assistant
HARE-SM (Human-AI Requirements Engineering Synergy Model) is a human-in-the-loop framework and prototype for AI-assisted requirements engineering. It supports requirements elicitation, analysis and validation by generating acceptance criteria from user stories, comparing outputs from multiple LLMs, optionally using RAG project context, and allowing engineers to select, edit, regenerate and approve final criteria. The tool logs model outputs, response times, user selections, edits and feedback to support transparency, auditability, bias analysis, trust calibration and later empirical evaluation.  

**Contact:** `mateen.a.abbasi@jyu.fi`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP1

**Approach**  
LLM based workflow, RAG, prompt engineering, human-in-the-loop, multi-model comparison

**SDLC Stage**  
Requirements Engineering / Analysis

**TRL**  
Proof of Concept

**Links**  
[https://re-assistant.streamlit.app](https://re-assistant.streamlit.app)

</details>

---

## LESS Guaidance
The scripts, prompts, and the requirement and test generation Python application in this project support the replicability of the experiments performed in the paper:

LESS is more: Guiding LLMs for Formal Requirement and Test Case Generation (DOI: 10.1007/978-3-032-07244-3_22 )  

**Contact:** `abhishek.shrestha@fokus.fraunhofer.de`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP3

**Approach**  
LLM based workflow, automation

**SDLC Stage**  
Requirement Engineering

**Links**  
[https://github.com/Abhishek2271/LESSGuidance](https://github.com/Abhishek2271/LESSGuidance)

</details>

---

## MARTA
novel multiagent
test generation pipeline tailored specifically for Python  

**Contact:** `egm@isep.ipp.pt`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP5

**Approach**  
LLm

**SDLC Stage**  
Developing/testing

**TRL**  
TRL 4 - TRL5

**Use Cases**  
Loop

**Links**  
to appear

</details>

---

## ReGEN  Tool
ReGEN is a requirement engineering solution featuring a Next.js, FastAPI, parser microservice, and LangGraph-based AI agent. It extracts structured requirements from documents (PDF, DOCX) using extraction pipelines and OCR. It manages phase-based workflows with immutable requirement versioning. It utilizes expert AI agents to process complex requirement operations like merging, splitting, and refining text interactively, maintaining full audit trails.  

**Contact:** `kamer@dakikyazilim.com,
tim@iotiq.de`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP3

**Approach**  
Multi-agent workflows, LLMs (Mistral, Qwen, Ollama)

**SDLC Stage**  
Requirements Engineering / Analysis

**TRL**  
4

**Use Cases**  
Dakik, Iotiq

**Links**  
To be added

</details>

---

## RestifAI
RESTifAI,is  a workflow-LLM-based approach whose novelty derives from automatically generating positive tests (happy-path), which confirm correct system behavior under valid inputs, and systematically deriving negative tests from these happy-paths, that validate robustness under invalid or unexpected conditions.  

**Contact:** `Maximilian.Ehrhart@casablanca.a`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP5 (Task 5.1 + 5.2)

**Approach**  
LLM based workflow

**SDLC Stage**  
Testing

**TRL**  
3

**Use Cases**  
CASABLANCA, UIBK

**Links**  
[Link Repo](https://github.com/casablancahotelsoftware/RESTifAI) • [Link Paper](https://arxiv.org/abs/2512.08706)

</details>

---

## TDD Orchestrator
An Agentic Code generation system that first generates tests, and then generates the code to pass the tests.  

**Contact:** `pyry.kotilainen@jyu.fi`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP4

**Approach**  
LLM based workflow

**SDLC Stage**  
Development, Maintenance

**TRL**  
Proof of Concept

**Links**  
[https://arxiv.org/abs/2604.26615](https://arxiv.org/abs/2604.26615)

</details>

---

## TWEASE
Agentic AI System that imports spec documents and requirements, then analysis knowledge gaps, supports clarification with human domain experts, then generates test case specifications  

**Contact:** `andreas.dreschinski@akkodis.com`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP5

**Approach**  
LLM based workflow , RAG, Agents

**SDLC Stage**  
Testing

**TRL**  
3

**Use Cases**  
Akkodis

**Links**  
N/A

</details>
# Tools

This page introduces the tools developed within the GENIUS project.  
Each tool is presented with a short description and contact information.
Additional technical details can be expanded when needed.

---
## Overview

| Tool | Summary | WP | SDLC | TRL | Contacts | Links |
|----|-------|---|----|---|--------|-----|
| [A multi-agent system for API Recommendation](#a-multi-agent-system-for-api-recommendation) | A multi-agent system for API Recommendation |  |  | Proof of Concept (3) | doruk.tuncel@siemens.com, yongjian.tang@siemens.com | Presented in the 2nd Anwenderforum |
| [A multi-agent system for API validation](#a-multi-agent-system-for-api-validation) | A multi-agent system for API validation |  |  | Proof of Concept (3) | doruk.tuncel@siemens.com, yongjian.tang@siemens.com | Presented in the Plenary in Austria |
| [Agentic Development Workflow Tool](#agentic-development-workflow-tool) | Structured AI-assisted development workflow | WP4 | Developing, Testing | 3 to 4 | valeria.villa@barco.com |  |
| [AI Help Assistant](#ai-help-assistant) | Multi-agent AI project help assistant | WP4 | Development | 4 | kamer@dakikyazilim.com,tim@iotiq.de |  |
| [Analysis Assistant](#analysis-assistant) | AI-powered automated analysis assistant. | WP3 | Development | 4 | ali.irmak@turkcell.com.tr | N/A |
| [Automated issue tagging](#automated-issue-tagging) | Automated issue tagging by similarity | WP5 | Testing | 6 | valeria.villa@barco.com | N/A |
| [Barco Genius](#barco-genius) | Internal RAG knowledge assistant | WP2, WP3 | Maintenance / Support / Knowledge Management | 6 to 7 | valeria.villa@barco.com |  |
| [Chat Agent](#chat-agent) | AI powered chat application | WP4 (Task 4.2) | Development | 3 | yuriy@vaadin.com | Link Repo |
| [CoALA](#coala) | Compositional Active Automata learning | WP5 (Task 5.1) | Testing | 2 to 3 | mohammad.mousavi@kcl.ac.uk | [Repo](https://zenodo.org/records/15170685) • [Paper](https://smrmousavi.github.io/pub/mousavi-concur-2025.pdf) |
| [Code Comment & Documentation](#code-comment-documentation) | Automated Code Documentation | WP 4 | Development | 4 to 5 | salih.dundar@orioninc.com |  |
| [Code Review Bot](#code-review-bot) | Automated Code Review Bot | WP4 (Task 4.2) | Development | 6 | selinsirin.aslangul@beko.com | [Link paper](https://arxiv.org/pdf/2412.18531) |
| [CoReGraph](#coregraph) | A Knowledge Graph Approach for Software Repository Analysis | WP4 | Development | TRL 4 - TRL5 | egm@isep.ipp.pt | to appear |
| [DETANGLE Architecture Modernization](#detangle-architecture-modernization) | Architecture Refactoring | WP4 | Development, Maintenance | 5 to 6 | wuchner@capeofgoodcode.com | Slides presented at the 1st Plenary and the 2nd Plenary |
| [Diffblue Cover (with LLM test data generation)](#diffblue-cover-with-llm-test-data-generation) | Regression unit test generator | WP5 | Testing | 4 | peter.schrammel@diffblue.com | [https://cover-docs.diffblue.com/features/cover-cli/environment-configuration/llm-configuration](https://cover-docs.diffblue.com/features/cover-cli/environment-configuration/llm-configuration) |
| [Diffblue Cover MCP](#diffblue-cover-mcp) | Actioning testability issue findings | WP5 | Testing, Maintenance | 6 | peter.schrammel@diffblue.com | [https://cover-docs.diffblue.com/features/cover-mcp-server-beta/getting-started-cover-mcp-beta](https://cover-docs.diffblue.com/features/cover-mcp-server-beta/getting-started-cover-mcp-beta) |
| [Diffblue Cover Moderne Recipe](#diffblue-cover-moderne-recipe) | Unit test generation for refactoring | WP4 | Maintenance | 6 | peter.schrammel@diffblue.com | N/A |
| [Diffblue Test Quality Agent](#diffblue-test-quality-agent) | Test quality assessment | WP5 | Testing | 6 | peter.schrammel@diffblue.com | [https://docs.diffblue.com/workflows/test-quality-report](https://docs.diffblue.com/workflows/test-quality-report) |
| [Diffblue Testing Agent](#diffblue-testing-agent) | Regression unit test generator | WP5 | Testing | 7 | peter.schrammel@diffblue.com | [https://docs.diffblue.com/workflows/regression-unit-tests](https://docs.diffblue.com/workflows/regression-unit-tests) |
| [HARE-SM Framework / RE Assistant](#hare-sm-framework-re-assistant) | Human-AI requirements engineering assistant | WP1 | Requirements Engineering / Analysis | Proof of Concept | mateen.a.abbasi@jyu.fi | [https://re-assistant.streamlit.app](https://re-assistant.streamlit.app) |
| [LACY](#lacy) | Legacy codebase understanding tool | WP4 (Task 4.2) | Testing | 4 | selinsirin.aslangul@beko.com | Link Repo • Link Paper |
| [LESS Guaidance](#less-guaidance) | LESS-Based Requirement and Test Case Generation | WP3 | Requirement Engineering |  | abhishek.shrestha@fokus.fraunhofer.de | [https://github.com/Abhishek2271/LESSGuidance](https://github.com/Abhishek2271/LESSGuidance) |
| [MARTA](#marta) | A Decoupled Multi-Agent Architecture for Python Test Generation | WP5 | Developing/testing | TRL 4 - TRL5 | egm@isep.ipp.pt | to appear |
| [MASA Codeelevate](#masa-codeelevate) | A Tool Transforming Jupyter Notebooks to Clean Code Architecture by Leveraging multiple GenAI Agents |  | Development, Maintenance | TBD | doruk.tuncel@siemens.com, yongjian.tang@siemens.com |  |
| [ReGEN Tool](#regen-tool) | AI-driven requirement engineering tool | WP3 | Requirements Engineering / Analysis | 4 | kamer@dakikyazilim.com, tim@iotiq.de |  |
| [Requirementor](#requirementor) | Requirement Analysis | WP3 | Requirements | 2 to 3 | selinsirin.aslangul@beko.com | NA |
| [RestifAI](#restifai) | RestAPI Test Generator | WP5 (Task 5.1 + 5.2) | Testing | 3 | Maximilian.Ehrhart@casablanca.a | [Link Repo](https://github.com/casablancahotelsoftware/RESTifAI) • [Link Paper](https://arxiv.org/abs/2512.08706) |
| [TDD Orchestrator](#tdd-orchestrator) | Test driven code generator | WP4 | Development, Maintenance | Proof of Concept | pyry.kotilainen@jyu.fi | [https://arxiv.org/abs/2604.26615](https://arxiv.org/abs/2604.26615) |
| [TWEASE](#twease) | Test with ease | WP5 | Testing | 3 | andreas.dreschinski@akkodis.com | N/A |


---

## A multi-agent system for API Recommendation
**Contact:** `doruk.tuncel@siemens.com, yongjian.tang@siemens.com`

<details>
<summary><strong>Technical details</strong></summary>

**TRL**  
Proof of Concept (3)

**Use Cases**  
Siemens AG

**Links**  
Presented in the 2nd Anwenderforum

</details>

---

## A multi-agent system for API validation
**Contact:** `doruk.tuncel@siemens.com, yongjian.tang@siemens.com`

<details>
<summary><strong>Technical details</strong></summary>

**TRL**  
Proof of Concept (3)

**Use Cases**  
Siemens AG

**Links**  
Presented in the Plenary in Austria

</details>

---

## Agentic Development Workflow Tool
Workflow-driven system orchestrating AI-assisted software development through phases: specification, decomposition, and disciplined TDD execution. Integrated via GitHub Copilot plugin, embedding engineering practices such as test-driven development, adversarial review, and structured refactoring, with human approval gates ensuring accountability and quality control.  

**Contact:** `valeria.villa@barco.com`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP4

**Approach**  
LLM orchestration, agentic workflows

**SDLC Stage**  
Developing, Testing

**TRL**  
3 to 4

**Use Cases**  
Barco

**Links**  
To be added

</details>

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

## Analysis Assistant
Analysis Assistant transforms customer feature requests into high-quality analysis documents by combining AI-driven generation with structured accountability. It automatically scans internal documentation to flag potential gaps and ensure no cross-service impacts are missed. The tool discourages superficial approvals by generating targeted requirements and detailed analysis checklists. Users refine AI-generated drafts with additional instructions, producing standardized, development-ready documentation for team collaboration. By automating repetitive manual tasks, it reduces lead times and improves traceability. The result is a more efficient SDLC where teams focus on high-impact engineering while maintaining consistent documentation standards and quality.  

**Contact:** `ali.irmak@turkcell.com.tr`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP3

**Approach**  
LLM based workflow

**SDLC Stage**  
Development

**TRL**  
4

**Use Cases**  
Turkcell Technology

**Links**  
N/A

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

## Barco Genius
Internal AI assistant enabling natural language access to Barco technical and product knowledge across repositories. Uses a RAG approach to retrieve and ground answers in validated internal documentation, improving knowledge reuse and productivity. Integrated with Microsoft Teams and Copilot, and continuously improved through user feedback loops and iterative validation in operational contexts.  

**Contact:** `valeria.villa@barco.com`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP2, WP3

**Approach**  
RAG, LLMs, semantic retrieval

**SDLC Stage**  
Maintenance / Support / Knowledge Management

**TRL**  
6 to 7

**Use Cases**  
Barco

**Links**  
To be added

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

## CoALA
A tool to learn formal models at scale from black-boxed implementations; this can feed into WP2 (and any other activity targetting spec-driven code generation)  

**Contact:** `mohammad.mousavi@kcl.ac.uk`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP5 (Task 5.1)

**Approach**  
Automata Learning

**SDLC Stage**  
Testing

**TRL**  
2 to 3

**Use Cases**  
Not yet applied to any use-case (have been benchmarked on large research datasets)

**Links**  
[Repo](https://zenodo.org/records/15170685) • [Paper](https://smrmousavi.github.io/pub/mousavi-concur-2025.pdf)

</details>

---

## Code Comment & Documentation
This AI-driven documentation tool streamlines large-scale software ecosystems by generating automated, context-aware comments and technical summaries for Python, Java, and TypeScript. By analyzing class and function structures, it ensures knowledge continuity while minimizing manual documentation. A dedicated VS Code extension enables developers to visualize dependencies directly, accelerating onboarding for dev, QA, and support teams. Ultimately, it fosters seamless information transfer and enhances system comprehension across multi-module projects, significantly reducing the learning curve for new members.  

**Contact:** `salih.dundar@orioninc.com`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP 4

**Approach**  
LLM based workflow, RAG

**SDLC Stage**  
Development

**TRL**  
4 to 5

**Use Cases**  
Orion Innovation

**Links**  
To be added

</details>

---

## Code Review Bot
This tool strengthens code review by combining LLM-based analysis with structured accountability in the pull request workflow. It automatically checks code changes against review checklists, flags potential bugs, highlights standards violations, and generates targeted review comments. By reducing repetitive manual review work and discouraging superficial approvals, it helps teams apply review standards more consistently. The tool also supports comment resolution discipline, improving traceability and responsibility throughout the review process. The result is faster pull request handling, improved code quality, greater adherence to best practices, and a more efficient SDLC with developers focused on higher impact engineering tasks.  

**Contact:** `selinsirin.aslangul@beko.com`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP4 (Task 4.2)

**Approach**  
LLM based workflow

**SDLC Stage**  
Development

**TRL**  
6

**Use Cases**  
Arçelik

**Links**  
[Link paper](https://arxiv.org/pdf/2412.18531)

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

## LACY
LACY is a VS Code extension that supports developer onboarding by providing structured guidance through legacy codebases. It generates navigable code tours that highlight key files, functions, and execution flows. By presenting this information directly in the development environment, LACY reduces the need for manual exploration and helps new developers understand system behavior more efficiently in complex or long-lived software projects.  

**Contact:** `selinsirin.aslangul@beko.com`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP4 (Task 4.2)

**Approach**  
LLM based workflow

**SDLC Stage**  
Testing

**TRL**  
4

**Use Cases**  
Arçelik

**Links**  
Link Repo • Link Paper

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

## MASA Codeelevate
codelevate supports developers in transforming Jupyter notebooks into a productive software system that allows further collaboration, versioning and design guidelines' adherence.  

**Contact:** `doruk.tuncel@siemens.com, yongjian.tang@siemens.com`

<details>
<summary><strong>Technical details</strong></summary>

**Approach**  
LLM based workflow

**SDLC Stage**  
Development, Maintenance

**TRL**  
TBD

**Use Cases**  
TBD

**Links**  
To be added

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

## Requirementor
This tool supports structured requirement analysis by helping teams capture, refine, and standardize project requirements in a clear, traceable format. It applies consistent templates, guided prompts, and root cause analysis frameworks to reduce ambiguity and uncover underlying delivery issues early. By aligning documentation practices across departments, the tool improves collaboration, strengthens shared understanding, and creates a reliable link between business needs and implementation decisions. It is designed to minimize rework, shorten lead times, and reduce defects caused by incomplete or misinterpreted requirements. The result is more consistent delivery, better cross functional alignment, and stronger traceability throughout the development lifecycle.  

**Contact:** `selinsirin.aslangul@beko.com`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP3

**Approach**  
LLM based workflow

**SDLC Stage**  
Requirements

**TRL**  
2 to 3

**Use Cases**  
Arçelik

**Links**  
NA

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

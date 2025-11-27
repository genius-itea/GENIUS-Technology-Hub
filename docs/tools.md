# Tools

This page introduces the tools developed within the GENIUS project.  
Each tool is presented with a short description and contact information.
Additional technical details can be expanded when needed.

---
## Overview

| Tool | Summary | WP | SDLC | TRL | Contacts | Links |
|----|-------|---|----|---|--------|-----|
| [Chat Agent](#chat-agent) | AI powered chat application | WP4 (Task 4.2) | Development | 3 | yuriy@vaadin.com | Link Repo |
| [RestifAI](#restifai) | RestAPI Test Generator | WP5 (Task 5.1 + 5.2) | Testing | 3 | Maximilian.Ehrhart@casablanca.a | Link Repo • Link Paper |
| [UI test generator](#ui-test-generator) | Automatic UI test generation from code and HTML | WP5 (Task 5.1) | Testing | 4 | yuriy@vaadin.com | Link Repo |


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
Link Repo • Link Paper

</details>

---

## UI test generator
UI test generator prototype is available as a Maven plugin, Java library,and Copilot plugin. It creates Gherkin definitions and executable test code from source andrendered HTML, with support for running tests on Playwright (Java/TypeScript) and VaadinTestBench.​  

**Contact:** `yuriy@vaadin.com`

<details>
<summary><strong>Technical details</strong></summary>

**Work Package**  
WP5 (Task 5.1)

**Approach**  
LLM based workflow

**SDLC Stage**  
Testing

**TRL**  
4

**Use Cases**  
Vaadin

**Links**  
Link Repo

</details>

# Contributing to TranslatorHoi4

Thank you for helping improve TranslatorHoi4. This guide explains how to set up the project, make focused changes, and verify them before opening a pull request.

## Project Scope

TranslatorHoi4 is a Python 3.12+ desktop application for translating Paradox game localization files with AI providers. The main areas of the codebase are:

- `translatorhoi4/translator/` - translation engine, provider backends, caching, prompts, and cost tracking.
- `translatorhoi4/parsers/` - Paradox localization parsing and serialization.
- `translatorhoi4/ui/` - PySide6 and Fluent UI windows, controls, localization strings, and review tools.
- `translatorhoi4/utils/` - settings, validation, logging, update checks, and filesystem helpers.
- `tests/` - pytest coverage for parser, cache, settings, validation, UI control behavior, and related logic.
- `docs/` - user-facing and developer documentation.

## Development Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/Locon213/TranslatorHoi4.git
cd TranslatorHoi4
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -U pip wheel setuptools
pip install -r requirements.txt
```

Run the application from source:

```bash
python -m translatorhoi4
```

## Running Tests

Run the full test suite before submitting code changes:

```bash
pytest -q
```

For focused work, run the relevant test file first:

```bash
pytest -q tests/test_parser.py
pytest -q tests/test_ui_batch_controls.py
```

If you change parser behavior, translation masking, settings storage, provider configuration, or batch UI controls, add or update tests that cover the changed behavior.

## Building Locally

TranslatorHoi4 uses Nuitka for distributable builds:

```bash
python build.py
```

Build output is generated under `dist/TranslatorHoi4/`. Build artifacts should not be committed unless the repository explicitly asks for them.

## Code Guidelines

- Keep changes focused on one feature, bug fix, or documentation update.
- Prefer existing project patterns over introducing new abstractions.
- Preserve Paradox localization syntax exactly, including keys, `$VARIABLES$`, `[scripted.macros]`, escaped newlines, and formatting tokens.
- Do not log API keys, provider credentials, mod contents beyond what is necessary for debugging, or other user secrets.
- Keep provider-specific logic inside the relevant backend module when possible.
- Keep UI text in the locale modules under `translatorhoi4/ui/locales/` when adding user-visible strings.
- Use clear names for settings and config fields, and keep backwards compatibility with existing stored settings when practical.

## Localization Contributions

When updating UI translations:

- Update every affected locale file if the string is part of shared UI.
- Keep placeholders, punctuation, and accelerator-style labels consistent across languages.
- Do not translate provider names, model IDs, file extensions, or command names unless the surrounding UI already does so.
- Check that translated text still fits in compact controls and dialogs.

## AI Provider Changes

When adding or changing an AI provider:

- Put provider API calls in `translatorhoi4/translator/backends/`.
- Register the provider through the existing registry/config flow.
- Add settings UI only for options users need to configure directly.
- Handle network errors, authentication failures, rate limits, and malformed provider responses gracefully.
- Add tests for request construction, response parsing, and fallback behavior where feasible.

## Documentation

Documentation changes should be practical and current. If a command, path, or supported provider changes, update the relevant README or file under `docs/` in the same pull request.

## Commit and Pull Request Guidelines

Before committing:

```bash
git status --short
pytest -q
```

Use concise commit messages that describe the change, for example:

```text
Add parser tests for escaped localization values
Fix provider settings persistence
Update user guide for batch translation
```

Pull requests should include:

- A short summary of what changed.
- The reason for the change.
- Tests or manual checks performed.
- Screenshots or screen recordings for visible UI changes.
- Notes about compatibility, migrations, or provider behavior changes when relevant.

## Reporting Issues

When reporting a bug, include:

- TranslatorHoi4 version or commit.
- Operating system and Python version if running from source.
- The game or mod localization format involved.
- Steps to reproduce the problem.
- Expected and actual behavior.
- Relevant logs or screenshots, with API keys and private data removed.

## Security

Do not commit `.env` files, API keys, provider tokens, private mod files, or generated credentials. If a secret is committed by mistake, rotate it immediately and remove it from history before sharing the branch.

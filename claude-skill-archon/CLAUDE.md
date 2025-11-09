# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Claude Code Skills Workshop** - a comprehensive educational repository demonstrating how to create, install, and use Claude Code Skills. Skills are modular capabilities that extend Claude Code's functionality through YAML frontmatter and markdown instructions.

## Key Concepts

### Skills vs Slash Commands
- **Skills**: Model-invoked automatically based on context (defined in `.claude/skills/`)
- **Slash Commands**: User-invoked manually (defined in `.claude/commands/`)

### Project vs Personal Skills
- **Project Skills** (`.claude/skills/`): Team-shared, version-controlled, available in this project
- **Personal Skills** (`~/.claude/skills/`): Individual workflows available across all projects

## Repository Structure

```
claude_code_skills/
├── README.md                    # Complete workshop guide and documentation
├── .claude/
│   ├── skills/
│   │   └── archon/              # Example skill: Archon integration
│   │       ├── SKILL.md         # Main skill definition with YAML frontmatter
│   │       ├── scripts/
│   │       │   └── archon_client.py  # Python client for Archon API
│   │       └── references/
│   │           └── api_reference.md  # Detailed API documentation
│   └── settings.local.json      # Skill permissions configuration
├── personal-skills-guide/       # Guide for creating personal skills
│   └── setup-instructions.md
└── assets/                      # Images and resources for README
```

## Anatomy of a Skill

Every skill must have:

1. **Directory structure**: `.claude/skills/skill-name/`
2. **SKILL.md file** with YAML frontmatter:
   ```yaml
   ---
   name: skill-name                # Required: lowercase-with-hyphens
   description: What it does...    # Required: triggers + capabilities
   allowed-tools: Read, Write      # Optional: tool restrictions
   ---

   # Rest of markdown content
   ```

3. **Optional supporting files**:
   - `examples.md`, `reference.md` - Additional documentation
   - `scripts/` - Helper scripts (Python, shell, etc.)
   - `templates/` - Template files

## Creating Skills

### Minimal Working Skill

```yaml
---
name: my-skill
description: Brief description with trigger keywords
---

# Skill Instructions

Detailed instructions for Claude to follow when this skill is activated.
```

### Installing Community Skills

Use the `npx` command to install pre-built skills from [aitmpl.com/skills](https://www.aitmpl.com/skills):

```bash
# Install a skill (--yes flag skips prompts)
npx claude-code-templates@latest --skill=category/skill-name --yes

# Examples from README:
npx claude-code-templates@latest --skill=development/skill-creator --yes
npx claude-code-templates@latest --skill=document-processing/pdf-processing-pro --yes
npx claude-code-templates@latest --skill=development/webapp-testing --yes
```

Installed skills appear in `.claude/skills/` automatically.

## The Archon Skill Example

This repository includes a production-ready skill demonstrating best practices:

**Location**: `.claude/skills/archon/`

**Purpose**: Integration with Archon knowledge base and project management system

**Structure**:
- `SKILL.md` - Main skill definition with interactive setup flow
- `scripts/archon_client.py` - Complete Python API client
- `references/api_reference.md` - Full REST API documentation

**Key Features**:
- Interactive setup: Prompts user for Archon server URL on first use
- Connection verification before API calls
- Progressive context loading (references loaded only when needed)
- Complete error handling examples
- Production-ready code patterns

## Skill Activation

Skills activate automatically based on their `description` field. To ensure proper activation:

1. **Include trigger keywords** in the description that match user queries
2. **Be specific** about capabilities and use cases
3. **Mention file types** or domain terms (e.g., "PDF", "API", "authentication")

**Good Description**:
```yaml
description: Extract text and tables from PDF files, fill PDF forms, and merge PDF documents. Use when the user mentions PDFs, forms, document extraction, or table parsing.
```

**Bad Description**:
```yaml
description: Helps with documents
```

## Progressive Context Loading

Skills use progressive disclosure to manage context efficiently:

1. **SKILL.md** - Always loaded when skill activates
2. **Referenced files** (examples.md, reference.md) - Loaded only when explicitly needed
3. **Supporting files** (scripts/, templates/) - NOT pre-loaded; accessed directly during execution

This allows skills to include extensive resources without overwhelming context windows.

## Permissions

Configure skill permissions in `.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": ["Skill(pdf-extractor)"],
    "deny": [],
    "ask": []
  }
}
```

## Common Workflows

### Creating a New Skill

1. Create directory: `mkdir -p .claude/skills/my-skill`
2. Create `SKILL.md` with YAML frontmatter
3. Add supporting files if needed (scripts/, examples.md)
4. Test the skill by triggering it with relevant queries
5. Iterate on the description for better activation

### Converting Personal to Project Skill

1. Copy from `~/.claude/skills/my-skill` to `.claude/skills/my-skill`
2. Remove personal preferences, generalize for team use
3. Commit to git: `git add .claude/skills/my-skill && git commit`
4. Team members get it on next `git pull`

### Testing a Skill

1. Make a request that matches the skill's description keywords
2. Verify the skill activates (check for skill-specific instructions being followed)
3. Test error handling and edge cases
4. Refine description if activation isn't working as expected

## Best Practices

### Skill Design
- **Keep skills focused**: One clear purpose per skill
- **Write descriptive names**: Use lowercase-with-hyphens
- **Include trigger keywords**: Help Claude find the right skill
- **Provide examples**: Show expected inputs and outputs
- **Handle errors gracefully**: Include error handling patterns

### Documentation
- **Start with "why"**: Explain the problem the skill solves
- **Show "when to use"**: Clear activation scenarios
- **Include examples**: Real-world usage patterns
- **Document limitations**: Known constraints and edge cases

### Code in Skills
- **Include helper scripts**: Python, shell, etc. in `scripts/`
- **Use `allowed-tools`**: Restrict tools when security matters
- **Show error handling**: Demonstrate proper error patterns
- **Add type hints**: Make Python code clear and maintainable

## Important Resources

- **Main README.md**: Complete workshop guide with 19 community skills listed
- **personal-skills-guide/**: Instructions for creating personal skills
- **Archon skill**: Production example with interactive setup and API client
- **[aitmpl.com/skills](https://www.aitmpl.com/skills)**: Community skill repository

## Skill Development Tips

1. **Start simple**: Begin with minimal SKILL.md, add complexity as needed
2. **Test activation**: Verify the skill triggers with natural language queries
3. **Use progressive loading**: Reference large docs, don't inline them
4. **Share patterns**: Successful personal skills can become project skills
5. **Version control**: Commit skills to share across team via git

## Educational Context

This repository is designed for:
- Learning how Claude Code Skills work
- Understanding project vs personal skills
- Seeing production-ready skill examples (Archon)
- Exploring community skills ecosystem
- Building custom team workflows

When working in this repository, focus on educational clarity and demonstrating best practices for skill creation and usage.

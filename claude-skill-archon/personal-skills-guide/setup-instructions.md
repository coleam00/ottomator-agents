# Personal Skills Setup Guide

This guide explains how to set up and use **personal skills** in Claude Code - skills that are available across all your projects.

## Table of Contents

- [What Are Personal Skills?](#what-are-personal-skills)
- [When to Use Personal Skills](#when-to-use-personal-skills)
- [Setup Instructions](#setup-instructions)
- [Creating Your First Personal Skill](#creating-your-first-personal-skill)
- [Managing Personal Skills](#managing-personal-skills)
- [Example Personal Skills](#example-personal-skills)
- [Syncing Across Machines](#syncing-across-machines)
- [Best Practices](#best-practices)

---

## What Are Personal Skills?

**Personal skills** are skills stored in `~/.claude/skills/` (your home directory) that are available in every project you work on. They're perfect for individual workflows, preferences, and productivity tools.

### Key Characteristics

- **Global Availability**: Work in all your projects automatically
- **Not Version Controlled**: Don't get committed to project repositories
- **Personal Preferences**: Customize to your individual workflow
- **Experimental**: Test new skills before sharing with your team

---

## When to Use Personal Skills

| Use Personal Skills For | Use Project Skills For |
|------------------------|------------------------|
| Your commit message style | Team's commit conventions |
| Your code review checklist | Project testing standards |
| Your documentation preferences | Company API patterns |
| Your debugging workflow | Project code generation templates |
| Experimental skills you're testing | Production-ready team workflows |
| Tools you use across many projects | Project-specific workflows |

### Examples

**Personal Skills:**
- My preferred git commit format
- My code review checklist
- My documentation style guide
- My debugging helpers
- My personal code snippets

**Project Skills:**
- Team's API endpoint patterns
- Company testing standards
- Project-specific code generation
- Shared team workflows

---

## Setup Instructions

### Step 1: Create the Personal Skills Directory

```bash
# Create the directory if it doesn't exist
mkdir -p ~/.claude/skills
```

### Step 2: Verify the Directory

```bash
# Check that the directory was created
ls -la ~/.claude/

# You should see:
# drwxr-xr-x  2 user user 4096 Oct 23 15:30 skills
```

### Step 3: Verify Claude Code Recognizes It

1. Open Claude Code in any project
2. Ask: "What skills are available?"
3. Claude should list both project skills (from `.claude/skills/`) and personal skills (from `~/.claude/skills/`)

---

## Creating Your First Personal Skill

Let's create a simple personal skill for your git commit style.

### Example: Personal Commit Style Skill

#### Step 1: Create Skill Directory

```bash
mkdir -p ~/.claude/skills/my-commit-style
cd ~/.claude/skills/my-commit-style
```

#### Step 2: Create SKILL.md

Create `~/.claude/skills/my-commit-style/SKILL.md`:

```yaml
---
name: my-commit-style
description: Format git commit messages using my personal style. Use when creating commits or user mentions commit messages.
allowed-tools: Read, Bash
---

# My Personal Commit Style

## Purpose

Format git commit messages according to my personal preferences.

## When to Use

Activate when:
- User asks to create a commit
- User mentions "commit message"
- Creating commits as part of a workflow

## My Commit Style

I prefer commits that:
- Start with an emoji that represents the change type
- Have a short, clear subject line (max 50 chars)
- Include "why" in the body, not just "what"
- Reference issue numbers when applicable

### Emoji Guide

- ✨ `:sparkles:` - New feature
- 🐛 `:bug:` - Bug fix
- 📚 `:books:` - Documentation
- 💄 `:lipstick:` - UI/styling
- ♻️ `:recycle:` - Refactoring
- ✅ `:white_check_mark:` - Tests
- 🔧 `:wrench:` - Configuration
- 🚀 `:rocket:` - Performance improvement

## Template

```
<emoji> <subject>

<body explaining why>

<footer with issue references>
```

## Examples

### New Feature
```
✨ Add user profile page

Users requested ability to edit their profiles.
Implemented using the existing form components.

Closes #123
```

### Bug Fix
```
🐛 Fix login redirect loop

Previous logic caused infinite redirect when session expired.
Now properly clears session and redirects to login page.

Fixes #456
```

## Instructions

When creating a commit:
1. Run `git diff` to see changes
2. Determine the appropriate emoji
3. Write a clear subject line (50 chars max)
4. Add body explaining why (not what - code shows what)
5. Reference issues in footer
```

#### Step 3: Restart Claude Code

```bash
# Close and reopen Claude Code to load the new skill
```

#### Step 4: Test It

In any project:

```
Make some changes, then ask Claude:
"Create a commit for these changes"
```

Claude should use your personal commit style with emojis!

---

## Managing Personal Skills

### Listing Personal Skills

```bash
# List all your personal skills
ls -la ~/.claude/skills/

# Or ask Claude:
# "What personal skills do I have?"
```

### Updating a Skill

1. Edit the `SKILL.md` file:
```bash
nano ~/.claude/skills/my-commit-style/SKILL.md
```

2. Save your changes
3. Restart Claude Code to reload

### Removing a Skill

```bash
# Delete the skill directory
rm -rf ~/.claude/skills/unwanted-skill

# Restart Claude Code
```

### Temporarily Disabling a Skill

```bash
# Rename to add .disabled extension
mv ~/.claude/skills/my-skill ~/.claude/skills/my-skill.disabled

# Re-enable by removing .disabled
mv ~/.claude/skills/my-skill.disabled ~/.claude/skills/my-skill
```

---

## Example Personal Skills

### 1. My Code Review Checklist

**File:** `~/.claude/skills/my-code-review/SKILL.md`

```yaml
---
name: my-code-review
description: My personal code review checklist. Use when reviewing code or user asks for code review.
---

# My Code Review Checklist

## Checklist

When reviewing code, I check:

### Functionality
- [ ] Does the code do what it's supposed to?
- [ ] Are edge cases handled?
- [ ] Are errors handled appropriately?

### Code Quality
- [ ] Is the code readable and clear?
- [ ] Are variable names descriptive?
- [ ] Are functions small and focused?
- [ ] Is there unnecessary complexity?

### Testing
- [ ] Are there tests for new functionality?
- [ ] Do existing tests still pass?
- [ ] Are edge cases tested?

### Performance
- [ ] Are there obvious performance issues?
- [ ] Is there unnecessary computation?
- [ ] Are database queries optimized?

### Security
- [ ] Is user input validated?
- [ ] Are there SQL injection risks?
- [ ] Are secrets properly handled?

### Documentation
- [ ] Are functions documented?
- [ ] Are complex sections explained?
- [ ] Is the README updated if needed?
```

### 2. My Documentation Style

**File:** `~/.claude/skills/my-docs-style/SKILL.md`

```yaml
---
name: my-docs-style
description: My preferred documentation style. Use when writing documentation, README files, or docstrings.
---

# My Documentation Style

## Principles

I prefer documentation that:
- Starts with "why" before "how"
- Includes code examples
- Has clear headings and structure
- Uses active voice
- Avoids jargon

## Function Documentation

```python
def my_function(param1: str, param2: int) -> dict:
    """
    Brief one-line description.

    Longer description explaining why this function exists
    and what problem it solves.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Dictionary containing: {
            'key1': description,
            'key2': description
        }

    Raises:
        ValueError: When and why this is raised

    Example:
        >>> result = my_function("test", 42)
        >>> print(result['key1'])
        'expected output'
    """
```

## README Structure

1. **Title & Brief Description**
2. **Why** - Problem this solves
3. **Installation**
4. **Quick Start** - Get running in 2 minutes
5. **Usage Examples**
6. **API Reference** - Detailed documentation
7. **Contributing**
8. **License**
```

### 3. My Debugging Workflow

**File:** `~/.claude/skills/my-debugging/SKILL.md`

```yaml
---
name: my-debugging
description: My systematic debugging workflow. Use when debugging issues or user mentions errors or bugs.
---

# My Debugging Workflow

## Process

### 1. Reproduce the Issue
- Get exact steps to reproduce
- Try to reproduce myself
- Note what's different from expected

### 2. Isolate the Problem
- Is it frontend or backend?
- Is it data-related or logic-related?
- What's the minimal code that reproduces it?

### 3. Gather Information
- Check error messages carefully
- Review logs (app logs, server logs, browser console)
- Check recent changes (git log)
- Review related test failures

### 4. Form Hypothesis
- What do I think is causing this?
- What evidence supports this?
- What would prove/disprove this?

### 5. Test Hypothesis
- Add logging/print statements
- Use debugger to inspect state
- Try the fix
- Verify it's actually fixed

### 6. Prevent Recurrence
- Add test to catch this in future
- Document the issue
- Update validation/error handling

## Common Mistakes to Avoid

- Changing multiple things at once
- Not verifying the fix actually works
- Fixing symptoms instead of root cause
- Not adding tests after fixing
```

---

## Syncing Across Machines

Personal skills are local to each machine. To sync them across multiple machines:

### Option 1: Git Repository (Recommended)

1. **Create a git repository for your personal skills:**

```bash
cd ~/.claude/skills
git init
git add .
git commit -m "Initial commit of personal skills"
```

2. **Push to a private GitHub repository:**

```bash
# Create a private repo on GitHub, then:
git remote add origin https://github.com/yourusername/claude-personal-skills.git
git push -u origin main
```

3. **On other machines, clone the repository:**

```bash
# Backup existing skills first
mv ~/.claude/skills ~/.claude/skills.backup

# Clone your skills
git clone https://github.com/yourusername/claude-personal-skills.git ~/.claude/skills
```

4. **Keep skills in sync:**

```bash
# On machine 1: make changes, commit, and push
cd ~/.claude/skills
git add .
git commit -m "Update my-commit-style skill"
git push

# On machine 2: pull changes
cd ~/.claude/skills
git pull
```

### Option 2: Symbolic Link to Cloud Storage

1. **Move skills to cloud storage:**

```bash
# Example with Dropbox
mv ~/.claude/skills ~/Dropbox/claude-skills
ln -s ~/Dropbox/claude-skills ~/.claude/skills
```

2. **On other machines:**

```bash
# Ensure Dropbox is synced, then create symlink
ln -s ~/Dropbox/claude-skills ~/.claude/skills
```

### Option 3: Manual Copy

For occasional syncing:

```bash
# On machine 1: create archive
tar -czf ~/claude-skills-backup.tar.gz -C ~ .claude/skills

# Transfer file to machine 2, then:
tar -xzf ~/claude-skills-backup.tar.gz -C ~
```

---

## Best Practices

### 1. Start Simple

Begin with one personal skill:
- Choose something you do frequently
- Make it simple and focused
- Test it thoroughly
- Iterate based on usage

### 2. Keep Skills Focused

**Good:**
- `my-commit-style`: Just commit messages
- `my-code-review`: Just code reviews

**Bad:**
- `my-everything`: Commits, reviews, docs, tests, everything

### 3. Document for Future You

Six months from now, you'll forget why you created a skill. Write clear documentation:

```yaml
---
name: my-skill
description: Clear description of what and when
---

# Purpose

Why I created this skill and what problem it solves.

# Usage

How to use this skill with examples.
```

### 4. Version Control Your Skills

Even personal skills benefit from version control:
- Track changes over time
- Roll back if something breaks
- Share individual skills with others

### 5. Regular Review

Schedule quarterly reviews:
- Are you still using all skills?
- Can any be improved?
- Should any be shared with your team?
- Are descriptions still accurate?

### 6. Experiment Freely

Personal skills are your playground:
- Try new ideas
- Test experimental workflows
- Refine before sharing with team
- Delete what doesn't work

---

## Troubleshooting

### Skill Not Appearing

**Problem:** Personal skill doesn't show up in Claude

**Solutions:**
1. Verify directory location: `ls ~/.claude/skills/`
2. Check YAML frontmatter syntax
3. Restart Claude Code
4. Ask Claude: "List my personal skills"

### Skill Not Activating

**Problem:** Skill exists but doesn't activate

**Solutions:**
1. Make description more specific with trigger keywords
2. Explicitly mention the skill: "Use my-commit-style to create a commit"
3. Check for conflicts with project skills
4. Review activation examples in skill's SKILL.md

### Changes Not Taking Effect

**Problem:** Updated skill but changes don't apply

**Solutions:**
1. Restart Claude Code after editing
2. Verify you edited the correct file
3. Check for syntax errors in YAML
4. Clear and restart

---

## Example: Complete Personal Skill

Here's a complete example you can use as a template:

**File:** `~/.claude/skills/my-pr-template/SKILL.md`

```yaml
---
name: my-pr-template
description: My personal pull request description template. Use when creating pull requests or user mentions PR descriptions.
allowed-tools: Read, Bash
---

# My PR Template

## Purpose

Generate pull request descriptions following my preferred format.

## When to Use

- User asks to create a PR
- User mentions "pull request" or "PR description"
- Preparing to push code and create PR

## Template

```markdown
## 🎯 Purpose

[Brief explanation of why this PR exists]

## 📝 Changes

- Change 1
- Change 2
- Change 3

## 🧪 Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manually tested in dev environment
- [ ] Reviewed in staging

## 📸 Screenshots

[If UI changes, include before/after screenshots]

## 🔗 Related

- Closes #[issue number]
- Related to #[issue number]

## ⚠️ Breaking Changes

[List any breaking changes or write "None"]

## 📚 Documentation

- [ ] README updated
- [ ] API docs updated
- [ ] Comments added for complex logic

## ✅ Checklist

- [ ] Code follows project style guidelines
- [ ] Self-reviewed the code
- [ ] Added tests for new functionality
- [ ] All tests pass locally
- [ ] Updated documentation
```

## Instructions

When creating a PR description:

1. Review changes: `git diff main...HEAD`
2. List all commits: `git log main..HEAD --oneline`
3. Identify the purpose (why)
4. List specific changes (what)
5. Describe testing performed
6. Note any breaking changes
7. Reference related issues
8. Complete the checklist

## Example Output

```markdown
## 🎯 Purpose

Adds user authentication to allow secure access to protected resources.

## 📝 Changes

- Implemented JWT-based authentication
- Added login and logout endpoints
- Created user session management
- Added authentication middleware for protected routes

## 🧪 Testing

- [x] Unit tests pass (15 new tests added)
- [x] Integration tests pass
- [x] Manually tested login/logout flow
- [x] Reviewed in staging environment

## 📸 Screenshots

[Login page screenshot]
[Protected dashboard screenshot]

## 🔗 Related

- Closes #234
- Related to #189 (user management)

## ⚠️ Breaking Changes

None - all endpoints are backward compatible

## 📚 Documentation

- [x] README updated with auth setup instructions
- [x] API docs updated with new endpoints
- [x] Added inline comments for JWT handling

## ✅ Checklist

- [x] Code follows project style guidelines
- [x] Self-reviewed the code
- [x] Added tests for new functionality
- [x] All tests pass locally
- [x] Updated documentation
```
```

---

## Next Steps

1. **Create your first personal skill** using the examples above
2. **Test it** in multiple projects to ensure it works everywhere
3. **Iterate** based on your actual usage patterns
4. **Consider syncing** across your machines using git
5. **Share** successful patterns with your team (convert to project skills)

---

## Questions?

**Want to share a personal skill with your team?**
- Copy it to the project's `.claude/skills/` directory
- Commit to git
- Team members will get it on their next pull

**Want to make a project skill personal?**
- Copy from `.claude/skills/` to `~/.claude/skills/`
- Customize to your preferences
- Project version remains unchanged for team

**Having issues?**
- Check the main [README.md](../README.md) for general skills documentation
- Review [GETTING_STARTED.md](../GETTING_STARTED.md) for basic concepts
- Ask Claude: "Help me debug my personal skill"

---

Happy coding with your personalized Claude Code experience!

#!/bin/bash
# askme Dev Loop — Codex ↔ Claude Code iterative improvement
# Alternates: Codex (odd rounds) / Claude Code (even rounds)
# Output logged to dev-loop-ROUND.log

set -e
WORKDIR="D:/inovxio/tools/askme"
LOGDIR="$WORKDIR/dev-loop-logs"
mkdir -p "$LOGDIR"

MAX_ROUNDS=${1:-6}
ROUND=1

cd "$WORKDIR"

while [ $ROUND -le $MAX_ROUNDS ]; do
  LOGFILE="$LOGDIR/round-$ROUND.log"
  echo "=============================" | tee "$LOGFILE"
  echo "🔄 Round $ROUND / $MAX_ROUNDS" | tee -a "$LOGFILE"
  echo "Time: $(date)" | tee -a "$LOGFILE"
  echo "=============================" | tee -a "$LOGFILE"

  # Build context of what's been done so far
  HISTORY=$(git log --oneline -8 2>/dev/null || echo "no commits yet")

  if [ $((ROUND % 2)) -eq 1 ]; then
    # ===== ODD ROUNDS: Codex =====
    echo "🤖 Agent: CODEX" | tee -a "$LOGFILE"
    PROMPT="You are a senior AI engineer doing product improvement on the askme project (v4.0 voice AI for Thunder industrial robots).

Recent git history:
$HISTORY

Your task for Round $ROUND:
1. Read README.md and key source files (askme/brain/, askme/pipeline/, askme/skills/, askme/tools/) to understand the codebase
2. Identify the HIGHEST VALUE improvement not yet done (check git log above to avoid repeating)
3. Implement it completely — no placeholders, real working code
4. Add or update tests if applicable (tests/ directory)
5. git add -A && git commit -m 'feat: [what you improved] (Codex Round $ROUND)'

Focus areas by priority:
- Robot safety: better error recovery, graceful degradation
- Reliability: retry logic, timeout handling, exception coverage
- Intelligence: smarter intent routing, better context handling
- Performance: reduce LLM calls, cache where appropriate
- New skills: useful capabilities for industrial robot use case

Commit before finishing. Do real work."

    codex exec --full-auto "$PROMPT" 2>&1 | tee -a "$LOGFILE"
    EXIT_CODE=${PIPESTATUS[0]}

  else
    # ===== EVEN ROUNDS: Claude Code =====
    echo "🧠 Agent: CLAUDE CODE" | tee -a "$LOGFILE"

    DIFF=$(git diff HEAD~1 --stat 2>/dev/null || echo "no previous commit")
    PROMPT="You are a senior AI engineer doing iterative improvement on the askme project (v4.0 voice AI for Thunder industrial robots).

Recent git history:
$HISTORY

Last change summary:
$DIFF

Your task for Round $ROUND:
1. Review what the previous agent changed (git show HEAD --stat; git diff HEAD~1 -- key files)
2. Fix any bugs, incomplete implementations, or code quality issues in the previous round's work
3. Then implement the NEXT highest-value improvement not yet done
4. Add or update tests for anything you change
5. git add -A && git commit -m 'feat: [what you improved] (Claude Code Round $ROUND)'

Focus areas by priority:
- Fix anything broken or incomplete from previous round
- Robot safety and error handling
- Code quality and test coverage
- New capabilities that make the robot more useful

Do real work, commit before finishing."

    claude --permission-mode bypassPermissions --print "$PROMPT" 2>&1 | tee -a "$LOGFILE"
    EXIT_CODE=${PIPESTATUS[0]}
  fi

  echo "" | tee -a "$LOGFILE"
  echo "Round $ROUND exit code: $EXIT_CODE" | tee -a "$LOGFILE"

  # Show what was committed
  LAST_COMMIT=$(git log --oneline -1 2>/dev/null || echo "no commit")
  echo "Latest commit: $LAST_COMMIT" | tee -a "$LOGFILE"

  # Notify via OpenClaw
  if [ $ROUND -eq $MAX_ROUNDS ]; then
    openclaw system event --text "askme dev loop COMPLETE after $MAX_ROUNDS rounds. Last: $LAST_COMMIT" --mode now 2>/dev/null || true
  else
    openclaw system event --text "askme Round $ROUND done → starting Round $((ROUND+1)). Last: $LAST_COMMIT" --mode now 2>/dev/null || true
  fi

  ROUND=$((ROUND + 1))
  sleep 2
done

echo ""
echo "✅ Dev loop complete! $MAX_ROUNDS rounds done."
echo "Summary:"
git log --oneline -"$MAX_ROUNDS" 2>/dev/null

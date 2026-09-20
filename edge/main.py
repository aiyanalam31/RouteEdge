"""
main.py

Thin launcher only. gesture_detect.py already owns a complete, working
camera loop (capture -> detect -> route -> act -> display), so this file
does NOT duplicate that loop. It exists purely so the repo has a single,
obvious entry point (`python edge/main.py`) matching the project's file
tree, without risking two competing camera loops drifting out of sync.

If you need to add orchestration that's genuinely separate from the
per-frame pipeline (e.g. command-line flags, multiple run modes), add it
here and have it call gesture_detect.main() — don't copy the loop itself.
"""

from gesture_detect import main

if __name__ == "__main__":
    main()

# /team — Shared agile workspace for Ben & Callum

This folder is the operational coordination layer for the project. Both of our Claude sessions read it via the root `CLAUDE.md` pointer, so updates here propagate to both AI workflows automatically.

## Files

| File | Purpose | Update when |
|---|---|---|
| [`TODO.md`](TODO.md) | Master task list with owner + status. The single source of truth for "what is there to do." | A new task is identified, an owner changes, or status moves |
| [`BOARD.md`](BOARD.md) | Kanban-style view (Backlog / In Progress / Review / Recently Done). Visual scan of where work is. | A task moves between states. Mirrors `TODO.md` status field |
| [`MEETING_NOTES.md`](MEETING_NOTES.md) | Append-only log of our ~1pm syncs. Decisions, blockers, action items. | After each 1pm meeting |

## Cadence

We sync **a few times a week** at 1pm. The board doesn't need daily churn — update it whenever you finish a task or pick up a new one, and skim before each meeting.

## Conventions

### Owner field
- `ben` — Ben Gladney
- `callum` — Callum Camazzola
- `shared` — both contributing actively
- `tbd` — not yet assigned

### Status values
- `backlog` — known work, not started
- `in-progress` — actively being worked on right now
- `review` — done from author's perspective; awaiting other's review or ack
- `done` — complete, lives on the board for ~1 cycle then migrates to `docs/MILESTONES.md`

### Priority
- `P0` — blocker / next thing to do
- `P1` — important, this week
- `P2` — backlog, no rush

### How both Claudes should use this

1. **At the start of any non-trivial task**, the agent should read `TODO.md` and `BOARD.md` to know what's already in flight, who owns what, and avoid duplicating work.
2. **When picking up a new task**, set its `status` to `in-progress` in `TODO.md` and move the row to the In Progress column in `BOARD.md`. Mention it briefly in the next meeting notes entry.
3. **When finishing a task**, set status to `review` (so the other person can ack) or `done` if it's been acked. Items marked `done` get migrated to `docs/MILESTONES.md` periodically.
4. **Don't edit work owned by the other person** without flagging it. If Ben's Claude needs to touch something owned `callum`, write a note in `MEETING_NOTES.md` for the next sync rather than acting unilaterally.
5. **Never delete TODO rows** — change status to `done` and let them migrate. We want a record of what was done, not just what's left.

## Where the strategic roadmap lives

The high-level research roadmap (research phases, multi-month direction) is at [`../docs/ROADMAP.md`](../docs/ROADMAP.md). The `/team/` folder is operational ("what are we doing this week"); `/docs/ROADMAP.md` is strategic ("what are we trying to accomplish").

The completed-work history is at [`../docs/MILESTONES.md`](../docs/MILESTONES.md).

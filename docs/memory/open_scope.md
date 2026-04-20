# Open Scope / Future Backlog

Backlog of feature requests, capability gaps, and refactoring ideas for RCC. Items here are logged for future consideration but are not currently in the active sprint or roadmap.

---

## Future Ideas — Not Currently Urgent

### Agent self-testing and integration test harness for RCC refactors

**Description:**
Rook wants the agents to be able to run automated test cycles on changes before those changes ship to the main running RCC. Pain point: when Commander or another worker lands a refactor in raiken/, Rook currently has to manually restart RCC and exercise the change himself to verify it works end-to-end. That is slow and puts the testing burden on him instead of on the agent network.

**Desired capability:**
- A shared test harness the agents can invoke that runs a scratch RCC instance (or equivalent) against a change.
- Agents collaborate on debugging their own output — if a test fails, the originating worker gets the failure back and iterates, or escalates to Raiken Agent.
- Only once the self-test passes does the change get handed to Rook for real-world sign-off and deploy to the live RCC.

**Scope flag:** Not urgent. Rook's exact framing: "that's a thing for another day."

**Related context:**
The Option C / Foreman refactor Shadowling Commander is landing right now is a good first candidate for this kind of automated verification — it touches the Speaker/Foreman wire protocol and has many subtle failure modes that are awkward to test by hand.

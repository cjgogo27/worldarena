Project: Research Problem Outline to Code – Full-stack Scaffold

Notes:
- This repository provides a runnable full-stack scaffold focused on turning a research problem outline into a project structure. It is intentionally lightweight and uses in-memory storage for demonstration purposes.
- It includes a simple web UI served from the backend to exercise the API without needing a separate frontend stack.
- Important: The PDF file you referenced, 研究问题大纲.pdf, cannot be read by this model. If you want me to base the project on content from that document, please provide the text or a summary of its sections, or paste key excerpts here.

What this repo contains:
- backend/: A minimal Express server with JWT-based auth and REST endpoints for projects, outlines, and documents.
- backend/index.html: A tiny static UI to create/login projects and edit outlines.
- README.md: How to run and extend the project.

How to run:
- cd backend
- npm install
- npm start
- Open http://localhost:3000

Next steps (optional):
- Add persistent storage (PostgreSQL, SQLite) and ORM.
- Replace in-memory auth with real user management and password hashing.
- Build a richer frontend (React, Next.js, or Vue) with proper state management.
- Integrate a docs importer to import PDF-derived content after converting to text (outside PDF capture in this MLE).

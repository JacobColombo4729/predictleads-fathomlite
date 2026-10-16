"""The combined demo: the same app, ranked by the combined model (combined/), run on its own.

    combined_demo/run.ps1 (Windows) or combined_demo/run.sh (Mac/Linux): backend on 8002, the page on http://127.0.0.1:5175

Nothing in app/ changes: server.py builds Jesse's app with create_app(repository=..., ranker=...), and ranker.py adapts
the combined model to the app's ranker interface. Its own ports and session database keep it apart from Jesse's app
(8000/5173) and Jacob's app (8001/5174).
"""

import argparse
import sys
import threading

from workflow.app import create_app, start_server


def main():
    parser = argparse.ArgumentParser(description="Anima Workflow Engine")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-gui", action="store_true", help="Run without webview (browser mode)")
    parser.add_argument("--workflows-root", type=str, default=None)
    args = parser.parse_args()

    app = create_app(workflows_root=args.workflows_root)
    port = args.port

    if args.no_gui:
        start_server(app, port)
    else:
        try:
            import webview

            server_thread = threading.Thread(
                target=start_server, args=(app, port), daemon=True
            )
            server_thread.start()
            webview.create_window(
                "Anima Workflow", f"http://localhost:{port}", width=1200, height=800
            )
        except ImportError:
            print("pywebview not installed, falling back to browser mode")
            start_server(app, port)


if __name__ == "__main__":
    main()

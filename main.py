"""
Main Entry Point for LightRAG Knowledge Graph QA System
Provides unified access to CLI and Web UI modes
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv


def main():
    """Main entry point with mode selection"""
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="LightRAG Knowledge Graph QA System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch Web UI
  python main.py --mode webui

  # Build knowledge graph via CLI
  python main.py --mode cli build --docs ./documents

  # Query via CLI
  python main.py --mode cli query --query "What is machine learning?"

  # Interactive mode
  python main.py --mode cli interactive
        """
    )

    parser.add_argument(
        "--mode",
        choices=["cli", "webui"],
        default="webui",
        help="Launch mode: cli (command line) or webui (web interface)"
    )

    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file"
    )

    # Parse known args to allow passing remaining to submodules
    args, remaining = parser.parse_known_args()

    if args.mode == "webui":
        # Launch Gradio Web UI
        from src.webui.webui import launch_webui
        from src.core.config_loader import load_config

        try:
            config = load_config(args.config)
            webui_config = config.webui

            host = webui_config.get("host", "0.0.0.0")
            port = webui_config.get("port", 7860)
            share = webui_config.get("share", False)

            auth = None
            if webui_config.get("auth_enabled", False):
                auth = (
                    webui_config.get("username", "admin"),
                    webui_config.get("password", "")
                )

            print(f"🚀 Launching LightRAG Web UI at http://{host}:{port}")
            print(f"📁 Using config: {args.config}")
            if auth:
                print(f"🔐 Authentication enabled")

            launch_webui(host=host, port=port, share=share, auth=auth)

        except Exception as e:
            print(f"❌ Failed to launch Web UI: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif args.mode == "cli":
        # Launch CLI
        from src.cli.cli import main as cli_main

        # Reconstruct sys.argv for CLI parser
        sys.argv = ["cli.py", "--config", args.config] + remaining

        try:
            cli_main()
        except Exception as e:
            print(f"❌ CLI Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()



# """
# Main Entry Point for LightRAG Knowledge Graph QA System
# Provides unified access to CLI and Web UI modes
# """
#
# import sys
# import argparse
# from pathlib import Path
# from dotenv import load_dotenv
#
#
# def main():
#     """Main entry point with mode selection"""
#     # Load environment variables from .env file
#     load_dotenv()
#
#
#
#
#
#     # Launch Gradio Web UI
#     from src.webui.webui import launch_webui
#     from src.core.config_loader import load_config
#
#     try:
#         config = load_config()
#         webui_config = config.webui
#
#         host = webui_config.get("host", "0.0.0.0")
#         port = webui_config.get("port", 7860)
#         share = webui_config.get("share", False)
#
#         auth = None
#         if webui_config.get("auth_enabled", False):
#             auth = (
#                 webui_config.get("username", "admin"),
#                 webui_config.get("password", "")
#             )
#
#         print(f"🚀 Launching LightRAG Web UI at http://{host}:{port}")
#         # print(f"📁 Using config: {args.config}")
#         if auth:
#             print(f"🔐 Authentication enabled")
#
#         launch_webui(host=host, port=port, share=share, auth=auth)
#
#     except Exception as e:
#         print(f"❌ Failed to launch Web UI: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)
#
#
#
# if __name__ == "__main__":
#     main()


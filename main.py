#!/usr/bin/env python3.10
import os
import sys

# Add the workspace directory to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import app

if __name__ == "__main__":
    app.main()

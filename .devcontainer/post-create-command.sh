#!/bin/bash
chmod -R +x ./.vscode
#poe install_dev_deps
#poe start_shell
poetry install
echo 'source $(poetry env info --path)/bin/activate' >> ~/.bashrc
